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

"""Managed local Pi0 / Pi0.5 inference for the SO-101 evaluation UI.

On AMD Strix Halo this launches ``start_server_pi.sh``, which uses the ROCm
venv created by ``setup_amd_pi.sh`` instead of the CUDA-only LeRobot venv.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import requests

log = logging.getLogger(__name__)

DEFAULT_LOCAL_HOST = "127.0.0.1"
DEFAULT_LOCAL_PORT = 8102
DEFAULT_STARTUP_TIMEOUT_S = 900.0
DEFAULT_CHECKPOINT = "lerobot/pi05_base"


@dataclass(frozen=True)
class LocalPiInferenceConfig:
    example_dir: Path
    host: str = DEFAULT_LOCAL_HOST
    port: int = DEFAULT_LOCAL_PORT
    checkpoint: str = DEFAULT_CHECKPOINT
    policy_type: str = "pi05"
    device: str = "auto"
    robot_type: str = ""
    max_actions: int | None = None
    top_image_key: str | None = None
    side_image_key: str | None = None
    startup_timeout_s: float = DEFAULT_STARTUP_TIMEOUT_S
    python_bin: str | None = None


class LocalPiInferenceServer:
    """Starts and stops ``start_server_pi.sh`` as a child process."""

    def __init__(self, config: LocalPiInferenceConfig) -> None:
        self.config = config
        self.process: subprocess.Popen[bytes] | None = None

    @property
    def endpoint(self) -> str:
        return f"http://{self.config.host}:{self.config.port}/act"

    def start(self) -> dict[str, object]:
        start_script = self.config.example_dir / "start_server_pi.sh"
        host_script = self.config.example_dir / "host_server_pi.py"
        if not start_script.is_file():
            raise FileNotFoundError(f"Pi launcher not found: {start_script}")
        if not host_script.is_file():
            raise FileNotFoundError(f"Pi policy server entrypoint not found: {host_script}")

        rocm_venv = self.config.example_dir / ".venv-rocm" / "bin" / "python"
        if not rocm_venv.is_file() and not (self.config.python_bin or os.environ.get("PI_PYTHON")):
            # Soft check: start_server_pi.sh will hard-fail on Strix without ROCm venv.
            log.warning(
                "ROCm venv not found at %s. On AMD Strix Halo run setup_amd_pi.sh first.",
                rocm_venv.parent.parent,
            )

        env = os.environ.copy()
        env["PI_LOCAL_HOST"] = self.config.host
        env["PI_LOCAL_PORT"] = str(self.config.port)
        env["PI_CHECKPOINT"] = self.config.checkpoint
        env["PI_POLICY_TYPE"] = self.config.policy_type
        env["PI_DEVICE"] = self.config.device
        env["PI_ROBOT_TYPE"] = self.config.robot_type
        env["HF_HUB_ENABLE_HF_TRANSFER"] = env.get("HF_HUB_ENABLE_HF_TRANSFER", "1")
        env["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = env.get(
            "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1"
        )
        if self.config.max_actions is not None:
            env["PI_MAX_ACTIONS"] = str(self.config.max_actions)
        if self.config.top_image_key:
            env["PI_TOP_IMAGE_KEY"] = self.config.top_image_key
        if self.config.side_image_key:
            env["PI_SIDE_IMAGE_KEY"] = self.config.side_image_key
        if self.config.python_bin:
            env["PI_PYTHON"] = self.config.python_bin

        command = [str(start_script)]
        log.info("Starting local Pi inference (%s) from %s", self.config.policy_type, self.config.checkpoint)
        log.info("Command: %s", " ".join(command))
        self.process = subprocess.Popen(
            command,
            cwd=self.config.example_dir,
            env=env,
        )
        metadata = self.wait_until_ready()
        log.info(
            "Local Pi server ready at %s (repo_id=%s, backend=%s, device=%s)",
            self.endpoint,
            metadata.get("repo_id", "?"),
            metadata.get("backend", "?"),
            metadata.get("device", "?"),
        )
        return metadata

    def wait_until_ready(self) -> dict[str, object]:
        deadline = time.monotonic() + self.config.startup_timeout_s
        last_error = "server did not respond"
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(
                    f"Local Pi policy server exited early with code {self.process.returncode}. "
                    "On AMD Strix Halo run: ./examples/molmoact_so101_eval/setup_amd_pi.sh"
                )
            try:
                response = requests.get(self.endpoint, timeout=5)
                if response.status_code == 200:
                    payload = response.json()
                    if payload.get("status") == "ok":
                        return payload
                    last_error = f"unexpected health payload: {payload}"
            except requests.RequestException as exc:
                last_error = str(exc)
            time.sleep(2.0)
        raise TimeoutError(
            f"Timed out after {self.config.startup_timeout_s:.0f}s waiting for {self.endpoint}: {last_error}"
        )

    def stop(self) -> None:
        process = self.process
        self.process = None
        if process is None:
            return
        if process.poll() is not None:
            return
        log.info("Stopping local Pi inference server")
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
