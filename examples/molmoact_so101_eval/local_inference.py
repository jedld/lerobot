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

"""Managed local MolmoAct2 SO-101 inference using the upstream molmoact2 repo."""

from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import requests

log = logging.getLogger(__name__)

DEFAULT_MOLMOACT2_HOME = Path("~/workspace/molmoact2").expanduser()
DEFAULT_LOCAL_HOST = "127.0.0.1"
DEFAULT_LOCAL_PORT = 8101
DEFAULT_STARTUP_TIMEOUT_S = 600.0


@dataclass(frozen=True)
class LocalInferenceConfig:
    molmoact2_home: Path
    host: str = DEFAULT_LOCAL_HOST
    port: int = DEFAULT_LOCAL_PORT
    startup_timeout_s: float = DEFAULT_STARTUP_TIMEOUT_S


class LocalInferenceServer:
    """Starts and stops ``molmoact2/examples/so101/start_server.sh`` as a child process."""

    def __init__(self, config: LocalInferenceConfig) -> None:
        self.config = config
        self.process: subprocess.Popen[bytes] | None = None

    @property
    def endpoint(self) -> str:
        return f"http://{self.config.host}:{self.config.port}/act"

    def start(self) -> dict[str, object]:
        molmoact2_home = self.config.molmoact2_home.resolve()
        start_script = molmoact2_home / "examples/so101/start_server.sh"
        host_script = molmoact2_home / "examples/so101/host_server_so101.py"
        if not start_script.is_file():
            raise FileNotFoundError(f"MolmoAct2 launcher not found: {start_script}")
        if not host_script.is_file():
            raise FileNotFoundError(f"MolmoAct2 server entrypoint not found: {host_script}")

        env = os.environ.copy()
        env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        env.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

        command = [
            str(start_script),
            "--host",
            self.config.host,
            "--port",
            str(self.config.port),
        ]
        log.info("Starting local MolmoAct2 inference from %s", molmoact2_home)
        log.info("Command: %s", " ".join(command))
        self.process = subprocess.Popen(
            command,
            cwd=molmoact2_home,
            env=env,
        )
        metadata = self.wait_until_ready()
        log.info(
            "Local MolmoAct2 server ready at %s (repo_id=%s, backend=%s)",
            self.endpoint,
            metadata.get("repo_id", "?"),
            metadata.get("backend", "?"),
        )
        return metadata

    def wait_until_ready(self) -> dict[str, object]:
        deadline = time.monotonic() + self.config.startup_timeout_s
        last_error = "server did not respond"
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(
                    f"Local MolmoAct2 server exited early with code {self.process.returncode}. "
                    "Check the server logs above."
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
        log.info("Stopping local MolmoAct2 inference server")
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
