#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HTTP /act server for LeRobot Pi0 / Pi0.5 policies on SO-101.

Exposes the same json-numpy wire format as the MolmoAct2 SO-101 server so the
evaluation UI can switch policies without changing its request loop.

Pi 0.7 weights are not publicly released yet. Use ``--policy-type pi05`` (default)
or ``pi0`` with an open Hugging Face checkpoint such as ``lerobot/pi05_base`` or a
community SO-101 fine-tune (for example ``L7-Robotics/pi05_so101_v6.1``).
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import threading
import time
from typing import Any

import json_numpy
import numpy as np
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from numpy.typing import NDArray

json_numpy.patch()

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame
from lerobot.utils.constants import OBS_IMAGES
from lerobot.utils.feature_utils import hw_to_dataset_features

ACTION_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
STATE_DIM = len(ACTION_NAMES)
PI07_NOTE = (
    "Pi 0.7 weights are not publicly released. This server runs open-source "
    "LeRobot pi05/pi0 checkpoints instead."
)

TOP_CAMERA_ALIASES = ("top_cam", "front_cam")
SIDE_CAMERA_ALIASES = ("side_cam", "wrist_cam")


def is_rocm() -> bool:
    return bool(getattr(torch.version, "hip", None))


def has_amd_strix_gpu() -> bool:
    try:
        out = subprocess.run(
            ["lspci"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return False
    return any(marker in out for marker in ("Radeon 890M", "Radeon 880M", "Strix"))


def resolve_device(requested: str | None) -> torch.device:
    if requested and requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested device {requested!r} but torch.cuda.is_available() is False. "
                "On AMD Strix Halo install the ROCm venv: "
                "./examples/molmoact_so101_eval/setup_amd_pi.sh"
            )
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if has_amd_strix_gpu():
        raise RuntimeError(
            "AMD Strix Halo / Radeon 890M detected, but this Python has no ROCm GPU backend "
            f"(torch={torch.__version__}, hip={getattr(torch.version, 'hip', None)}). "
            "The default LeRobot venv ships CUDA wheels which cannot use the iGPU. "
            "Run: ./examples/molmoact_so101_eval/setup_amd_pi.sh "
            "then relaunch with --policy pi --local."
        )
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    logging.warning("No GPU backend available; falling back to CPU (will be slow).")
    return torch.device("cpu")


def backend_name() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    if is_rocm():
        return "rocm"
    return "cuda"


def short_image_key(full_key: str) -> str:
    prefix = f"{OBS_IMAGES}."
    if full_key.startswith(prefix):
        return full_key[len(prefix) :]
    return full_key


def infer_camera_map(image_keys: list[str]) -> dict[str, str]:
    short_keys = [short_image_key(key) for key in image_keys]
    if not short_keys:
        return {"top_cam": "top", "side_cam": "side"}

    top_priority = ("base_0_rgb", "overhead", "top", "front", "scene", "exterior")
    side_priority = ("left_wrist_0_rgb", "right_wrist_0_rgb", "wrist", "side")

    def pick(candidates: tuple[str, ...], *, exclude: set[str]) -> str | None:
        for token in candidates:
            for key in short_keys:
                if key in exclude:
                    continue
                if token in key:
                    return key
        return None

    top_key = pick(top_priority, exclude=set()) or short_keys[0]
    side_key = pick(side_priority, exclude={top_key}) or (short_keys[1] if len(short_keys) > 1 else top_key)
    return {"top_cam": top_key, "side_cam": side_key}


def extract_camera_frames(payload: dict[str, Any]) -> tuple[NDArray[Any], NDArray[Any]]:
    top_frame = None
    side_frame = None
    for alias in TOP_CAMERA_ALIASES:
        if alias in payload:
            top_frame = np.asarray(payload[alias])
            break
    for alias in SIDE_CAMERA_ALIASES:
        if alias in payload:
            side_frame = np.asarray(payload[alias])
            break
    if top_frame is None:
        raise ValueError(f"Missing top/front camera frame. Expected one of {TOP_CAMERA_ALIASES}.")
    if side_frame is None:
        side_frame = top_frame
    return top_frame, side_frame


class PiPolicyRunner:
    def __init__(
        self,
        *,
        checkpoint: str,
        policy_type: str,
        device: torch.device,
        robot_type: str,
        max_actions: int | None,
        top_image_key: str | None,
        side_image_key: str | None,
        tokenizer_name: str | None = None,
        dtype: str = "auto",
        num_inference_steps: int | None = None,
        warmup: bool = True,
    ) -> None:
        self.checkpoint = checkpoint
        self.policy_type = policy_type
        self.device = device
        self.robot_type = robot_type
        self.dtype = resolve_dtype(dtype, device)

        if policy_type == "pi05":
            from lerobot.policies.pi05.modeling_pi05 import PI05Policy

            self.policy = PI05Policy.from_pretrained(checkpoint)
        elif policy_type == "pi0":
            from lerobot.policies.pi0.modeling_pi0 import PI0Policy

            self.policy = PI0Policy.from_pretrained(checkpoint)
        else:
            raise ValueError(f"Unsupported policy type: {policy_type}")

        if num_inference_steps is not None:
            self.policy.config.num_inference_steps = int(num_inference_steps)
        self.policy.config.dtype = self.dtype

        self.policy.eval()
        self.policy.to(device)
        apply_inference_dtype(self.policy, self.dtype, device)

        preprocessor_overrides: dict[str, Any] = {"device_processor": {"device": str(device)}}
        if tokenizer_name:
            preprocessor_overrides["tokenizer_processor"] = {"tokenizer_name": tokenizer_name}
        try:
            self.preprocess, self.postprocess = make_pre_post_processors(
                self.policy.config,
                checkpoint,
                preprocessor_overrides=preprocessor_overrides,
            )
        except Exception as exc:  # noqa: BLE001
            message = str(exc)
            if "gated" in message.lower() or "403" in message or "paligemma" in message.lower():
                raise RuntimeError(
                    "Pi tokenizer download failed because google/paligemma-3b-pt-224 is gated. "
                    "1) Visit https://huggingface.co/google/paligemma-3b-pt-224 and accept the terms, "
                    "2) run `huggingface-cli login`, "
                    "3) relaunch. Or pass --tokenizer-name /path/to/local/paligemma-tokenizer."
                ) from exc
            raise

        image_keys = sorted(self.policy.config.image_features)
        inferred = infer_camera_map(image_keys)
        self.top_hw_key = top_image_key or inferred["top_cam"]
        self.side_hw_key = side_image_key or inferred["side_cam"]
        self.image_keys = image_keys
        self.max_actions = max_actions or int(self.policy.config.n_action_steps)

        robot_action_features = {f"{name}.pos": float for name in ACTION_NAMES}
        robot_obs_features = {
            **robot_action_features,
            self.top_hw_key: (480, 640, 3),
            self.side_hw_key: (480, 640, 3),
        }
        self.dataset_features = {
            **hw_to_dataset_features(robot_action_features, "action"),
            **hw_to_dataset_features(robot_obs_features, "observation"),
        }
        self.action_feature_names = list(self.dataset_features["action"]["names"])
        self._lock = threading.Lock()

        logging.info(
            "Pi ready: device=%s dtype=%s denoise_steps=%s max_actions=%s",
            device,
            self.dtype,
            self.policy.config.num_inference_steps,
            self.max_actions,
        )
        if warmup and device.type == "cuda":
            self._warmup()

    def _warmup(self) -> None:
        logging.info("Warming up Pi inference (compiles ROCm kernels)...")
        dummy = {
            "top_cam": np.zeros((224, 224, 3), dtype=np.uint8),
            "side_cam": np.zeros((224, 224, 3), dtype=np.uint8),
            "instruction": "warmup",
            "state": np.zeros(len(ACTION_NAMES), dtype=np.float32),
        }
        try:
            self.predict(dummy)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            logging.info("Warmup complete.")
        except Exception as exc:  # noqa: BLE001
            logging.warning("Warmup skipped after error: %s", exc)

    def metadata(self) -> dict[str, Any]:
        gpu = None
        if torch.cuda.is_available():
            try:
                gpu = torch.cuda.get_device_name(0)
            except Exception:  # noqa: BLE001
                gpu = None
        return {
            "status": "ok",
            "backend": f"lerobot-{self.policy_type}-{backend_name()}",
            "repo_id": self.checkpoint,
            "policy_type": self.policy_type,
            "device": str(self.device),
            "dtype": self.dtype,
            "num_inference_steps": int(self.policy.config.num_inference_steps),
            "torch": torch.__version__,
            "hip": getattr(torch.version, "hip", None),
            "gpu": gpu,
            "note": PI07_NOTE,
            "image_keys": self.image_keys,
            "camera_map": {
                "top_cam": self.top_hw_key,
                "side_cam": self.side_hw_key,
            },
            "n_action_steps": self.max_actions,
            "chunk_size": int(self.policy.config.chunk_size),
            "state_dim": STATE_DIM,
            "action_dim": STATE_DIM,
            "camera_order": ["top_cam", "side_cam"],
        }

    def predict(self, payload: dict[str, Any]) -> tuple[NDArray[np.float32], float]:
        start = time.perf_counter()
        top_frame, side_frame = extract_camera_frames(payload)
        # Contiguous copies avoid non-writable NumPy → torch warnings and extra syncs.
        top_frame = np.ascontiguousarray(top_frame)
        side_frame = np.ascontiguousarray(side_frame)
        instruction = str(payload.get("instruction") or payload.get("task") or "")
        state = np.asarray(payload.get("state", np.zeros(STATE_DIM)), dtype=np.float32).reshape(-1)
        if state.shape[0] == int(self.policy.config.max_state_dim):
            logging.warning(
                "Received padded state (%d,); using first %d values as SO-101 joints.",
                state.shape[0],
                STATE_DIM,
            )
            state = state[:STATE_DIM]
        if state.shape[0] != STATE_DIM:
            raise ValueError(f"state must be shape ({STATE_DIM},), got {state.shape}.")

        observation = {f"{name}.pos": float(state[idx]) for idx, name in enumerate(ACTION_NAMES)}
        observation[self.top_hw_key] = top_frame
        observation[self.side_hw_key] = side_frame

        obs_frame = build_inference_frame(
            observation=observation,
            ds_features=self.dataset_features,
            device=self.device,
            task=instruction,
            robot_type=self.robot_type or None,
        )
        with self._lock:
            batch = self.preprocess(obs_frame)
            with torch.inference_mode():
                chunk = self.policy.predict_action_chunk(batch)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
            chunk = chunk[0, : self.max_actions]

            # Postprocessor expects (B, action_dim) per timestep, not the full chunk.
            processed_actions: list[torch.Tensor | np.ndarray] = []
            for step_idx in range(chunk.shape[0]):
                single_action = chunk[step_idx : step_idx + 1]
                processed = self.postprocess(single_action)
                if isinstance(processed, torch.Tensor):
                    processed_actions.append(processed.detach().cpu().squeeze(0))
                else:
                    processed_actions.append(np.asarray(processed, dtype=np.float32).reshape(-1))
            action_matrix = np.stack(processed_actions, axis=0)
        if action_matrix.ndim == 1:
            action_matrix = action_matrix[None, :]
        action_matrix = action_matrix.astype(np.float32, copy=False)

        # Keep only the SO-101 joints in ACTION_NAMES order.
        name_to_index = {name: idx for idx, name in enumerate(self.action_feature_names)}
        columns = [name_to_index[f"{name}.pos"] for name in ACTION_NAMES]
        actions = action_matrix[:, columns]

        elapsed_ms = (time.perf_counter() - start) * 1e3
        return actions, elapsed_ms


def resolve_dtype(requested: str, device: torch.device) -> str:
    requested = (requested or "auto").strip().lower()
    if requested == "auto":
        if device.type == "cuda":
            return "bfloat16"
        return "float32"
    if requested not in {"bfloat16", "float32"}:
        raise ValueError(f"Unsupported dtype {requested!r}; use auto, bfloat16, or float32.")
    if requested == "bfloat16" and device.type == "cpu":
        logging.warning("bfloat16 requested on CPU; using float32 instead.")
        return "float32"
    return requested


def apply_inference_dtype(policy: Any, dtype: str, device: torch.device) -> None:
    if dtype != "bfloat16" or device.type != "cuda":
        return
    model = getattr(policy, "model", None)
    expert = getattr(model, "paligemma_with_expert", None) if model is not None else None
    if expert is not None and hasattr(expert, "to_bfloat16_for_selected_params"):
        expert.to_bfloat16_for_selected_params("bfloat16")
        logging.info("Enabled bfloat16 inference weights for Pi.")
        return
    policy.to(dtype=torch.bfloat16)
    logging.info("Cast Pi policy to bfloat16.")


def _error_response(status: int, message: str) -> Response:
    body = json_numpy.dumps({"error": message})
    return Response(content=body, status_code=status, media_type="application/json")


def build_app(runner: PiPolicyRunner) -> FastAPI:
    app = FastAPI(title="LeRobot Pi SO-101 server", version="0.1.0")

    @app.get("/act")
    async def health() -> JSONResponse:
        return JSONResponse(runner.metadata())

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    @app.post("/act")
    async def act(request: Request) -> Response:
        raw = await request.body()
        try:
            payload = json_numpy.loads(raw.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            return _error_response(400, f"failed to decode json_numpy body: {exc}")

        try:
            top_frame, side_frame = extract_camera_frames(payload)
            instruction = str(payload.get("instruction") or payload.get("task") or "")
            state = payload["state"]
        except KeyError as exc:
            return _error_response(400, f"missing required field: {exc}")
        except ValueError as exc:
            return _error_response(400, str(exc))

        predict_payload = {
            "top_cam": top_frame,
            "side_cam": side_frame,
            "instruction": instruction,
            "state": state,
        }

        t0 = time.perf_counter()
        try:
            actions, _ = runner.predict(predict_payload)
        except Exception as exc:  # noqa: BLE001
            logging.exception("Inference failed")
            return _error_response(500, f"inference failed: {exc}")
        dt_ms = (time.perf_counter() - t0) * 1000.0

        body = json_numpy.dumps({"actions": actions, "dt_ms": dt_ms})
        return Response(content=body, media_type="application/json")

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8102)
    parser.add_argument("--checkpoint", default="lerobot/pi05_base")
    parser.add_argument("--policy-type", choices=["pi05", "pi0"], default="pi05")
    parser.add_argument("--device", default="auto", help="auto, cuda, cpu, or mps")
    parser.add_argument("--robot-type", default="", help="Optional robot type token for multi-embodiment models.")
    parser.add_argument(
        "--max-actions",
        type=int,
        default=None,
        help="Number of actions to return per request (default: policy n_action_steps).",
    )
    parser.add_argument("--top-image-key", default=None, help="Override HW camera key for the top/front frame.")
    parser.add_argument("--side-image-key", default=None, help="Override HW camera key for the side/wrist frame.")
    parser.add_argument(
        "--tokenizer-name",
        default=None,
        help="Override tokenizer path/repo (default: google/paligemma-3b-pt-224 from the checkpoint).",
    )
    parser.add_argument(
        "--dtype",
        default=os.environ.get("PI_DTYPE", "auto"),
        choices=["auto", "bfloat16", "float32"],
        help="Inference dtype (auto → bfloat16 on GPU, float32 on CPU).",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=int(os.environ.get("PI_NUM_INFERENCE_STEPS", "5")),
        help="Flow-matching denoise steps (default 5; policy default is 10).",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip a startup warmup inference that compiles ROCm kernels.",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s:%(name)s:%(message)s")
    device = resolve_device(args.device)
    logging.info(
        "Loading %s policy from %s on %s (torch=%s hip=%s backend=%s)",
        args.policy_type,
        args.checkpoint,
        device,
        torch.__version__,
        getattr(torch.version, "hip", None),
        backend_name(),
    )
    runner = PiPolicyRunner(
        checkpoint=args.checkpoint,
        policy_type=args.policy_type,
        device=device,
        robot_type=args.robot_type,
        max_actions=args.max_actions,
        top_image_key=args.top_image_key,
        side_image_key=args.side_image_key,
        tokenizer_name=args.tokenizer_name or os.environ.get("PI_TOKENIZER_NAME") or None,
        dtype=args.dtype,
        num_inference_steps=args.num_inference_steps,
        warmup=not args.no_warmup,
    )
    app = build_app(runner)
    logging.info("%s", PI07_NOTE)
    logging.info("Pi policy server listening on http://%s:%s/act", args.host, args.port)
    logging.info("Camera map: top_cam -> %s, side_cam -> %s", runner.top_hw_key, runner.side_hw_key)
    logging.info("Wire format: state (%d,) float32, actions (N,%d) float32", STATE_DIM, STATE_DIM)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())


if __name__ == "__main__":
    main()
