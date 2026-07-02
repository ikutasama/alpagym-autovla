"""AutoVLA inference model adapter for AlPaGym.

Implements the InferenceModel protocol by wrapping Qwen2.5-VL + ActionTokenizer.
Converts AlPaGym's BatchedModelInput (uint8 camera tensors) to AutoVLA's format,
runs VLM generate, decodes action tokens to trajectories, and computes log-probs.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image as PILImage

from alpagym_host.config import SamplingParamsConfig
from alpagym_runtime.inference.types import (
    BatchedModelInput,
    BatchedModelOutput,
    ModelInput,
    ModelOutput,
)
from alpagym_runtime.replay import ActionSelection, PolicyReplayData, require_payload_keys

logger = logging.getLogger(__name__)

# Number of context frames per camera (AutoVLA uses 4 frames at 2Hz)
NUM_CONTEXT_FRAMES = 4
# Camera names AutoVLA expects, in order
CAMERA_NAMES = ["front_camera", "front_left_camera", "front_right_camera"]
# Trajectory horizon (AutoVLA default: 10 poses over 5 seconds at 0.5s interval)
DEFAULT_NUM_POSES = 10


class AutoVLAInferenceModel:
    """Adapter between AlPaGym typed I/O and an AutoVLA model.

    Wraps a Qwen2.5-VL model with an ActionTokenizer. During rollout:
    1. Converts uint8 camera frames to PIL images
    2. Builds Qwen chat messages and processes them
    3. Runs VLM generate() to produce action tokens
    4. Decodes tokens to trajectory (xyz, rot)
    5. Computes per-token log-prob of action tokens for RL
    """

    def __init__(
        self,
        vlm: torch.nn.Module,
        processor: Any,
        action_tokenizer: Any,
        config: dict | None = None,
    ) -> None:
        self._vlm = vlm
        self._processor = processor
        self._action_tokenizer = action_tokenizer
        self._config = config or {}
        self._action_start_id = self._config.get(
            "model", {}
        ).get("tokens", {}).get("action_start_id", 151665)
        self._use_cot = self._config.get("model", {}).get("use_cot", False)
        gen_conf = self._config.get("inference", {}).get("sample", {})
        self._gen_conf = {
            "max_length": gen_conf.get("max_length", 2048),
            "temperature": gen_conf.get("temperature", 0.01),
            "top_k": gen_conf.get("top_k", 0),
            "top_p": gen_conf.get("top_p", 1.0),
        }

    def get_model(self) -> torch.nn.Module:
        return self._vlm

    def set_model(self, model: torch.nn.Module) -> None:
        """Replace the VLM, unwrapping FSDP/cosmos shells if present."""
        if hasattr(model, "_get_fsdp_state"):
            from torch.distributed.fsdp import FSDPModule
            model._get_fsdp_state()._lazy_init()
            for submodule in model.modules():
                if isinstance(submodule, FSDPModule):
                    submodule.unshard()
        self._vlm = model

    # ------------------------------------------------------------------
    # Rollout: sample_trajectories_from_data
    # ------------------------------------------------------------------

    def sample_trajectories_from_data(
        self,
        model_input: BatchedModelInput,
        sampling: SamplingParamsConfig,
        return_trace_for_rl: bool = False,
    ) -> BatchedModelOutput:
        """Run one batched AutoVLA forward and return normalized output.

        AutoVLA generates one trajectory per sample (no multi-candidate sampling).
        We set num_traj_sets=1, num_traj_samples=1 in the output shapes.
        """
        batch_size = model_input.camera_frames.shape[0]
        all_pred_xyz = []
        all_pred_rot = []
        all_logprobs = []
        all_extras = []

        for batch_idx in range(batch_size):
            xyz, rot, logprob, extra = self._sample_single(
                model_input, batch_idx, sampling, return_trace_for_rl
            )
            all_pred_xyz.append(xyz)
            all_pred_rot.append(rot)
            all_logprobs.append(logprob)
            all_extras.append(extra)

        # Stack: [B, 1, 1, T, 3] and [B, 1, 1, T, 3, 3]
        pred_xyz = torch.stack(all_pred_xyz, dim=0).unsqueeze(1).unsqueeze(1)
        pred_rot = torch.stack(all_pred_rot, dim=0).unsqueeze(1).unsqueeze(1)

        if return_trace_for_rl and all_logprobs[0] is not None:
            logprob = torch.stack(all_logprobs, dim=0).unsqueeze(1).unsqueeze(1)
        else:
            logprob = None

        extra: dict[str, Any] = {}
        if return_trace_for_rl and all_extras[0]:
            for key in all_extras[0]:
                extra[key] = torch.stack(
                    [e[key] for e in all_extras], dim=0
                )

        return BatchedModelOutput(
            pred_xyz=pred_xyz,
            pred_rot=pred_rot,
            logprob=logprob,
            extra=extra,
        )

    def _sample_single(
        self,
        model_input: BatchedModelInput,
        batch_idx: int,
        sampling: SamplingParamsConfig,
        return_trace_for_rl: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, dict[str, Any]]:
        """Process one sample in the batch."""
        device = next(self._vlm.parameters()).device

        # 1. Convert camera frames to PIL images and build prompt
        inputs = self._build_prompt_from_tensors(model_input, batch_idx, device)

        # 2. Override sampling params from AlPaGym config
        gen_kwargs = dict(self._gen_conf)
        if sampling.temperature is not None:
            gen_kwargs["temperature"] = sampling.temperature
        if sampling.top_p is not None:
            gen_kwargs["top_p"] = sampling.top_p
        if sampling.top_k is not None:
            gen_kwargs["top_k"] = sampling.top_k
        if sampling.max_generation_length is not None:
            gen_kwargs["max_length"] = sampling.max_generation_length

        # 3. Generate
        with torch.inference_mode():
            prompt_completion_ids = self._vlm.generate(
                **inputs,
                do_sample=True,
                **gen_kwargs,
            )

        prompt_length = inputs["input_ids"].size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # 4. Extract action tokens
        action_tokens = completion_ids[0][completion_ids[0] >= self._action_start_id]
        num_poses = self._config.get("model", {}).get("trajectory", {}).get(
            "num_poses", DEFAULT_NUM_POSES
        )
        if len(action_tokens) > num_poses:
            action_tokens = action_tokens[:num_poses]
        elif len(action_tokens) < num_poses:
            pad = torch.full(
                (num_poses - len(action_tokens),),
                self._action_start_id,
                dtype=action_tokens.dtype,
                device=device,
            )
            action_tokens = torch.cat([action_tokens, pad])

        # 5. Decode to trajectory
        trajectory = self._action_tokenizer.decode_token_ids_to_trajectory(
            action_tokens.cpu()
        )
        if isinstance(trajectory, np.ndarray) or torch.is_tensor(trajectory):
            traj_tensor = torch.as_tensor(trajectory, dtype=torch.float32)
        else:
            traj_tensor = torch.zeros(num_poses, 3)
        # trajectory shape: [T+1, 3] (x, y, heading), skip first (origin)
        traj_tensor = traj_tensor[1:] if traj_tensor.shape[0] > num_poses else traj_tensor
        if traj_tensor.shape[0] < num_poses:
            pad = torch.zeros(num_poses - traj_tensor.shape[0], 3)
            traj_tensor = torch.cat([traj_tensor, pad])

        # 6. Convert to AlPaGym format: xyz [T, 3], rot [T, 3, 3]
        pred_xyz, pred_rot = _traj_to_xyz_rot(traj_tensor, device)

        # 7. Compute logprob if needed
        logprob = None
        extra: dict[str, Any] = {}
        if return_trace_for_rl:
            logprob, token_logps, completion_mask = self._compute_logprob(
                inputs, prompt_completion_ids, prompt_length, device
            )
            extra["generated_ids"] = prompt_completion_ids[0].cpu()
            extra["prompt_length"] = torch.tensor(prompt_length)
            extra["pixel_values_videos"] = inputs.get(
                "pixel_values_videos", inputs.get("pixel_values", None)
            )
            if "video_grid_thw" in inputs:
                extra["video_grid_thw"] = inputs["video_grid_thw"]
            elif "image_grid_thw" in inputs:
                extra["video_grid_thw"] = inputs["image_grid_thw"]
            extra["completion_mask"] = completion_mask
            extra["token_logps"] = token_logps

        return pred_xyz, pred_rot, logprob, extra

    def _build_prompt_from_tensors(
        self,
        model_input: BatchedModelInput,
        batch_idx: int,
        device: torch.device,
    ) -> dict[str, Any]:
        """Build Qwen processor inputs from raw uint8 camera tensors.

        AlPaGym provides camera_frames as uint8 [B, C, T, H, W, 3] (or similar).
        We convert to PIL images and build chat messages.
        """
        from qwen_vl_utils import process_vision_info

        frames = model_input.camera_frames[batch_idx]  # [C, T, H, W, 3] or [C*T, H, W, 3]
        camera_indices = model_input.camera_indices[batch_idx]  # [C]

        # Determine frame layout
        if frames.ndim == 5:  # [C, T, H, W, 3]
            num_cameras, num_frames_per_cam = frames.shape[0], frames.shape[1]
        elif frames.ndim == 4:  # [C*T, H, W, 3]
            num_cameras = len(camera_indices)
            num_frames_per_cam = frames.shape[0] // num_cameras
            frames = frames.reshape(num_cameras, num_frames_per_cam, *frames.shape[1:])
        else:
            raise ValueError(f"Unexpected camera_frames shape: {frames.shape}")

        # Convert to PIL images and build video messages
        camera_images: dict[str, list] = {}
        for cam_idx in range(num_cameras):
            cam_name = CAMERA_NAMES[cam_idx] if cam_idx < len(CAMERA_NAMES) else f"camera_{cam_idx}"
            camera_images[cam_name] = []
            for t in range(num_frames_per_cam):
                img_array = frames[cam_idx, t].cpu().numpy()
                pil_img = PILImage.fromarray(img_array)
                camera_images[cam_name].append(pil_img)

        # Build ego state from model_input
        ego_xyz = model_input.ego_history_xyz[batch_idx]  # [T, 3]
        velocity = float(torch.norm(ego_xyz[-1, :2]).item()) if ego_xyz.numel() > 0 else 0.0
        acceleration = 0.0  # Approximate from history if available

        # Route info
        route_xy = model_input.route_xy[batch_idx]  # [20, 2]
        instruction = "turn left"  # Default; could be derived from route

        # Build chat messages
        user_content = self._build_user_content(camera_images, velocity, acceleration, instruction)

        if self._use_cot:
            system_text = (
                "You are an Advanced Driver Assistance and Full Self-Driving System. "
                "You will receive visual observations from the ego vehicle's cameras and "
                "dynamic information about the vehicle's current state. "
                "Your task is to predict the optimal driving action for the next five seconds."
            )
        else:
            system_text = (
                "You are an Advanced Driver Assistance and Full Self-Driving System. "
                "You will be provided with video observations from the ego vehicle's "
                "surrounding cameras, along with the vehicle's current dynamic states. "
                "Your task is to predict the most appropriate driving action for the next five seconds."
            )

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_text}]},
            {"role": "user", "content": user_content},
        ]

        image_inputs, video_inputs = process_vision_info(messages)
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, add_vision_id=True
        )
        proc_inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        # Move to device
        proc_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in proc_inputs.items()}
        return proc_inputs

    def _build_user_content(
        self,
        camera_images: dict[str, list],
        velocity: float,
        acceleration: float,
        instruction: str,
    ) -> list:
        """Build the user content list with video messages from PIL images."""
        min_pixels = self._config.get("model", {}).get("video", {}).get("min_pixels", 109760)
        max_pixels = self._config.get("model", {}).get("video", {}).get("max_pixels", 109760)

        user_content = [
            {"type": "text", "text": "The autonomous vehicle is equipped with three cameras mounted at the front, left, and right, enabling a comprehensive perception of the surrounding environment."},
        ]

        descriptions = {
            "front_camera": "The first video presents the front view of the vehicle, comprising four sequential frames sampled at 2 Hz.",
            "front_left_camera": "The second video presents the front-left view of the vehicle, comprising four sequential frames sampled at 2 Hz.",
            "front_right_camera": "The third video presents the front-right view of the vehicle, comprising four sequential frames sampled at 2 Hz.",
        }

        for cam_name in CAMERA_NAMES:
            if cam_name not in camera_images:
                continue
            user_content.append({"type": "text", "text": descriptions.get(cam_name, f"Video from {cam_name}.")})
            user_content.append({
                "type": "video",
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
                "video": camera_images[cam_name],
            })

        user_content.append({
            "type": "text",
            "text": (
                f"The current velocity of the vehicle is {velocity:.3f} m/s, "
                f"and the current acceleration is {acceleration:.3f} m/s^2. "
                f"The driving instruction is: {instruction}. "
                "Based on this information, plan the action trajectory for the "
                "autonomous vehicle over the next five seconds."
            ),
        })
        return user_content

    def _compute_logprob(
        self,
        inputs: dict[str, Any],
        prompt_completion_ids: torch.Tensor,
        prompt_length: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute per-token log-probabilities of the generated tokens.

        Returns:
            logprob: scalar (sum of action token log-probs)
            token_logps: [L] per-token log-probs for completion tokens
            completion_mask: [L] mask for valid completion tokens
        """
        # Re-run forward to get logits
        with torch.inference_mode():
            # Use the full sequence (prompt + completion) for forward
            forward_inputs = {k: v for k, v in inputs.items() if isinstance(v, torch.Tensor)}
            # Update input_ids and attention_mask for full sequence
            forward_inputs["input_ids"] = prompt_completion_ids
            forward_inputs["attention_mask"] = torch.ones_like(prompt_completion_ids)

            outputs = self._vlm(**forward_inputs)
            logits = outputs.logits  # [1, L, V]

        # Compute per-token log-probs
        logits = logits[:, :-1, :]  # [1, L-1, V]
        input_ids = prompt_completion_ids[:, 1:]  # [1, L-1]
        log_probs = torch.log_softmax(logits, dim=-1)  # [1, L-1, V]
        per_token_logps = log_probs.gather(2, input_ids.unsqueeze(-1)).squeeze(-1)  # [1, L-1]

        # Extract completion part
        completion_logps = per_token_logps[:, prompt_length - 1:]  # [1, L_comp]
        completion_ids = prompt_completion_ids[:, prompt_length:]  # [1, L_comp]

        # Build completion mask (up to EOS)
        eos_token_id = self._processor.tokenizer.eos_token_id
        is_eos = completion_ids == eos_token_id
        eos_idx = is_eos.size(1)
        if is_eos.any():
            eos_idx = is_eos.int().argmax(dim=1).item()
        seq_indices = torch.arange(is_eos.size(1), device=device)
        completion_mask = (seq_indices <= eos_idx).int().unsqueeze(0)  # [1, L_comp]

        # Sum log-probs of action tokens only
        action_mask = (completion_ids >= self._action_start_id).int() * completion_mask  # [1, L_comp]
        action_logps = completion_logps * action_mask
        logprob = action_logps.sum(dim=1)  # [1] - scalar per sample

        return logprob.squeeze(0), completion_logps.squeeze(0), completion_mask.squeeze(0)

    # ------------------------------------------------------------------
    # Replay: build_policy_replay_data
    # ------------------------------------------------------------------

    def build_policy_replay_data(
        self,
        model_input: ModelInput,
        model_output: ModelOutput,
        action_selection: ActionSelection,
    ) -> PolicyReplayData:
        """Pack AutoVLA replay data for trainer-side log-prob recomputation."""
        if model_output.logprob is None:
            raise ValueError("AutoVLA replay requires rollout logprob")

        old_logprob = model_output.logprob  # scalar

        payload: dict[str, Any] = {
            "model_input": asdict(model_input),
            "generated_ids": model_output.extra.get("generated_ids"),
            "prompt_length": model_output.extra.get("prompt_length"),
            "pixel_values_videos": model_output.extra.get("pixel_values_videos"),
            "video_grid_thw": model_output.extra.get("video_grid_thw"),
            "completion_mask": model_output.extra.get("completion_mask"),
        }

        return PolicyReplayData(
            replay_schema_version=1,
            payload_schema="autovla.trajectory.v1",
            payload_schema_version=1,
            model_family="autovla",
            action_selection=action_selection,
            old_logprob=torch.as_tensor(old_logprob, dtype=torch.float32).reshape(()),
            payload=payload,
        )

    # ------------------------------------------------------------------
    # Trainer: build_trainer_model_inputs
    # ------------------------------------------------------------------

    @classmethod
    def build_trainer_model_inputs(
        cls,
        replay_data: PolicyReplayData,
    ) -> tuple[dict[str, Any], torch.Tensor]:
        """Pack one AutoVLA replay payload into forward kwargs for the trainer.

        The trainer calls the patched VLM forward with these kwargs.
        Returns (forward_kwargs, old_logprob).
        """
        if replay_data.payload_schema != "autovla.trajectory.v1":
            raise ValueError(
                f"autovla replay payload_schema {replay_data.payload_schema!r} "
                f"!= 'autovla.trajectory.v1'"
            )

        payload = replay_data.payload
        require_payload_keys(
            replay_data.model_family,
            payload,
            ("model_input", "generated_ids", "prompt_length"),
        )

        generated_ids = torch.as_tensor(payload["generated_ids"], dtype=torch.int64)
        prompt_length = int(payload["prompt_length"])
        attention_mask = torch.ones_like(generated_ids)

        model_inputs: dict[str, Any] = {
            "input_ids": generated_ids.unsqueeze(0),  # [1, L]
            "attention_mask": attention_mask.unsqueeze(0),  # [1, L]
            "prompt_length": prompt_length,
        }

        # Add pixel values if available
        if payload.get("pixel_values_videos") is not None:
            pv = payload["pixel_values_videos"]
            if not isinstance(pv, torch.Tensor):
                pv = torch.as_tensor(pv)
            model_inputs["pixel_values_videos"] = pv.unsqueeze(0) if pv.ndim < 4 else pv
        if payload.get("video_grid_thw") is not None:
            gt = payload["video_grid_thw"]
            if not isinstance(gt, torch.Tensor):
                gt = torch.as_tensor(gt)
            model_inputs["video_grid_thw"] = gt

        if "completion_mask" in payload and payload["completion_mask"] is not None:
            cm = payload["completion_mask"]
            if not isinstance(cm, torch.Tensor):
                cm = torch.as_tensor(cm)
            model_inputs["completion_mask"] = cm

        if replay_data.old_logprob is None:
            raise ValueError("autovla replay requires old_logprob")
        old_logprob = torch.as_tensor(
            replay_data.old_logprob, dtype=torch.float32
        ).reshape(())

        return model_inputs, old_logprob


def _traj_to_xyz_rot(
    trajectory: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert [T, 3] (x, y, heading) to AlPaGym format.

    Returns:
        pred_xyz: [T, 3] (x, y, z=0)
        pred_rot: [T, 3, 3] rotation matrices from heading angles
    """
    T = trajectory.shape[0]
    xyz = torch.zeros(T, 3, device=device, dtype=trajectory.dtype)
    xyz[:, :2] = trajectory[:, :2].to(device)

    heading = trajectory[:, 2].to(device)
    cos_h = torch.cos(heading)
    sin_h = torch.sin(heading)

    # Rotation matrix around Z-axis: [[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]]
    rot = torch.zeros(T, 3, 3, device=device, dtype=trajectory.dtype)
    rot[:, 0, 0] = cos_h
    rot[:, 0, 1] = -sin_h
    rot[:, 1, 0] = sin_h
    rot[:, 1, 1] = cos_h
    rot[:, 2, 2] = 1.0

    return xyz, rot
