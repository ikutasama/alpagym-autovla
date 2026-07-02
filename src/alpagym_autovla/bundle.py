"""AutoVLA policy bundle entry point for AlPaGym.

Implements the PolicyBundle interface to integrate AutoVLA (Qwen2.5-VL + action tokens)
into AlPaGym's Cosmos-RL training framework.
"""

from __future__ import annotations

import functools
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable

import torch

from alpagym_runtime.policies.registry import PolicyBundle

logger = logging.getLogger(__name__)

# Track whether the forward patch is installed
_patch_installed = False


def install_autovla_runtime_bridge() -> None:
    """Wire AutoVLA into cosmos: patch the Qwen VLM forward for trainer log-probs."""
    global _patch_installed
    if _patch_installed:
        return

    # Add AutoVLA repo to sys.path if specified
    autovla_repo_path = os.environ.get("AUTOVLA_REPO_PATH", "")
    if autovla_repo_path and autovla_repo_path not in sys.path:
        sys.path.insert(0, autovla_repo_path)
        logger.info(f"Added AutoVLA repo to sys.path: {autovla_repo_path}")

    _patch_qwen_vlm_forward()
    _patch_installed = True


def _patch_qwen_vlm_forward() -> None:
    """Patch Qwen2.5-VL forward to support trainer-side log-prob computation.

    During training, Cosmos-RL calls model.forward(**forward_kwargs) where
    forward_kwargs comes from build_trainer_model_inputs. The patched forward
    detects the AutoVLA-specific kwargs (prompt_length, completion_mask) and
    computes per-token log-probs, returning {"log_probs": ...}.

    During normal inference (no prompt_length kwarg), the original forward runs.
    """
    from transformers import Qwen2_5_VLForConditionalGeneration

    original_forward = Qwen2_5_VLForConditionalGeneration.forward
    if getattr(original_forward, "_autovla_patched", False):
        return

    def patched_forward(
        self,
        *args,
        prompt_length: int | None = None,
        completion_mask: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        **kwargs,
    ) -> Any:
        # If no prompt_length, this is a normal forward call
        if prompt_length is None:
            # Map pixel_values_videos -> pixel_values for Qwen2.5-VL
            if pixel_values_videos is not None:
                kwargs["pixel_values"] = pixel_values_videos
            if video_grid_thw is not None:
                kwargs["image_grid_thw"] = video_grid_thw
            return original_forward(self, *args, **kwargs)

        # Trainer-side: compute log-probs
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        attention_mask = kwargs.get("attention_mask", None)

        # Map pixel_values_videos -> pixel_values
        if pixel_values_videos is not None:
            kwargs["pixel_values"] = pixel_values_videos
            kwargs.pop("pixel_values_videos", None)
        if video_grid_thw is not None:
            kwargs["image_grid_thw"] = video_grid_thw
            kwargs.pop("video_grid_thw", None)

        # Remove our custom kwargs before calling original
        kwargs.pop("prompt_length", None)
        kwargs.pop("completion_mask", None)

        # Run forward to get logits
        result = original_forward(self, *args, **kwargs)
        logits = result.logits  # [B, L, V]

        # Compute per-token log-probs
        logits = logits[:, :-1, :]  # [B, L-1, V]
        shifted_ids = input_ids[:, 1:]  # [B, L-1]
        log_probs = torch.log_softmax(logits, dim=-1)  # [B, L-1, V]
        per_token_logps = log_probs.gather(2, shifted_ids.unsqueeze(-1)).squeeze(-1)  # [B, L-1]

        # Extract completion part
        comp_logps = per_token_logps[:, prompt_length - 1:]  # [B, L_comp]

        # Determine action_start_id from model config or default
        action_start_id = getattr(self.config, "action_start_id", 151665)

        # Sum action token log-probs -> scalar per sample
        comp_ids = input_ids[:, prompt_length:]  # [B, L_comp]
        action_mask = (comp_ids >= action_start_id).float()
        if completion_mask is not None:
            action_mask = action_mask * completion_mask.float()

        logprob = (comp_logps * action_mask).sum(dim=1)  # [B]

        return {"log_probs": logprob.squeeze(-1)}

    patched_forward._autovla_patched = True  # type: ignore[attr-defined]
    Qwen2_5_VLForConditionalGeneration.forward = patched_forward
    logger.info("Patched Qwen2_5_VLForConditionalGeneration.forward for AutoVLA trainer")


def setup_tokenizer(config: Any) -> Any | None:
    """Set up the Qwen tokenizer with AutoVLA action tokens."""
    install_autovla_runtime_bridge()
    return None


def build_data_packer(run_config: Any, cosmos_role: str | None) -> Any:
    """Build the AutoVLA replay data packer."""
    from alpagym_runtime.cosmos.packer import build_alpagym_data_packer

    install_autovla_runtime_bridge()
    return build_alpagym_data_packer(
        run_config=run_config,
        cosmos_role=cosmos_role,
        build_model_inputs=build_model_inputs(run_config),
    )


def load_inference_model(
    run_config: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> Any:
    """Load an AutoVLA model and return its inference adapter."""
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    install_autovla_runtime_bridge()

    model_config = run_config.policy.model
    model_path = model_config.path

    # Load VLM
    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=str(device),
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(model_path)

    # Load action tokenizer
    autovla_config = getattr(run_config.policy, "autovla", {})
    codebook_path = autovla_config.get(
        "codebook_cache_path",
        os.path.join(os.environ.get("AUTOVLA_REPO_PATH", "."), "codebook_cache/agent_vocab.pkl"),
    )

    # Import AutoVLA's ActionTokenizer
    try:
        from models.action_tokenizer import ActionTokenizer
    except ImportError:
        # Fallback: define inline
        logger.warning("Could not import AutoVLA ActionTokenizer, using inline version")
        ActionTokenizer = _InlineActionTokenizer

    action_tokenizer = ActionTokenizer(processor.tokenizer, model_config={
        "tokens": {"action_start_id": autovla_config.get("action_start_id", 151665)},
        "codebook_cache_path": codebook_path,
    })

    # Resize embeddings for action tokens
    vlm.resize_token_embeddings(len(processor.tokenizer))

    # Store action_start_id on model config for the forward patch
    vlm.config.action_start_id = autovla_config.get("action_start_id", 151665)

    # Build the config dict for the inference model
    inf_config = {
        "model": {
            "use_cot": autovla_config.get("use_cot", False),
            "tokens": {"action_start_id": autovla_config.get("action_start_id", 151665)},
            "trajectory": autovla_config.get("trajectory", {"num_poses": 10}),
            "video": autovla_config.get("video", {"min_pixels": 109760, "max_pixels": 109760}),
        },
        "inference": {
            "sample": autovla_config.get("sample", {
                "max_length": 2048,
                "temperature": 0.01,
                "top_k": 0,
                "top_p": 1.0,
            }),
        },
    }

    from alpagym_autovla.inference_model import AutoVLAInferenceModel
    return AutoVLAInferenceModel(
        vlm=vlm,
        processor=processor,
        action_tokenizer=action_tokenizer,
        config=inf_config,
    )


def build_model_inputs(
    run_config: Any,
) -> Callable:
    """Return the AutoVLA trainer-side replay input builder."""
    from alpagym_autovla.inference_model import AutoVLAInferenceModel
    return AutoVLAInferenceModel.build_trainer_model_inputs


def get_bundle() -> PolicyBundle:
    """Return the AutoVLA runtime hooks."""
    return PolicyBundle(
        setup_tokenizer=setup_tokenizer,
        build_data_packer=build_data_packer,
        install_runtime_bridge=install_autovla_runtime_bridge,
        load_inference_model=load_inference_model,
        build_model_inputs=build_model_inputs,
    )


class _InlineActionTokenizer:
    """Minimal ActionTokenizer when AutoVLA repo is not on path.

    Handles codebook loading and token-to-trajectory decoding.
    Used as fallback when `from models.action_tokenizer import ActionTokenizer` fails.
    """

    def __init__(self, tokenizer, model_config):
        import pickle
        self.action_start_id = model_config["tokens"]["action_start_id"]
        codebook_path = model_config["codebook_cache_path"]
        with open(codebook_path, "rb") as f:
            code_book = pickle.load(f)["token_all"]["veh"]
            self.code_book = torch.tensor(code_book)
        action_len = self.code_book.shape[0]
        tokenizer.add_tokens([f"<action_{i}>" for i in range(action_len)], special_tokens=False)
        self.tokenizer = tokenizer
        self.n_bins = action_len

    def decode_token_ids_to_trajectory(self, token_ids):
        """Decode token IDs to trajectory via codebook lookup."""
        action_token_ids = []
        for i in range(len(token_ids)):
            if token_ids[i] < self.action_start_id:
                action_token_ids.append(0)
            else:
                action = self.tokenizer.decode(token_ids[i])
                action_token_ids.append(int(action.split("_")[1].replace(">", "")))
        action_token_ids = torch.tensor(action_token_ids)
        action_tokens = self.code_book[action_token_ids]
        time_steps = action_tokens.shape[0]
        traj = self._rollout(action_tokens, time_steps)
        return traj

    def _rollout(self, action_tokens, time_steps):
        pos_a = torch.tensor([[[0, 0]]])
        head_a = torch.tensor([[0]])
        for t in range(time_steps):
            next_token_traj = action_tokens[None, t]
            pos_local = next_token_traj.flatten(1, 2)
            pos_now = pos_a[:, t]
            head_now = head_a[:, t]
            cos, sin = head_now.cos(), head_now.sin()
            rot_mat = torch.zeros((head_now.shape[0], 2, 2))
            rot_mat[:, 0, 0] = cos
            rot_mat[:, 0, 1] = sin
            rot_mat[:, 1, 0] = -sin
            rot_mat[:, 1, 1] = cos
            pos_global = torch.bmm(pos_local, rot_mat) + pos_now.unsqueeze(1)
            pos_global = pos_global.view(*next_token_traj.shape)
            pos_a_next = pos_global[:, -1].mean(dim=1)
            diff = pos_global[:, -1, 0] - pos_global[:, -1, 3]
            head_a_next = torch.arctan2(diff[:, 1], diff[:, 0])
            pos_a = torch.cat([pos_a, pos_a_next.unsqueeze(1)], dim=1)
            head_a = torch.cat([head_a, head_a_next.unsqueeze(1)], dim=1)
        trajectory = torch.cat([pos_a, head_a.unsqueeze(-1)], dim=-1)
        return trajectory
