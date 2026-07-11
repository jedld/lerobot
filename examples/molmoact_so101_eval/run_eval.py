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

"""Convenience launcher for the MolmoAct2 SO-101 evaluation UI.

Loads ``default.env`` (and optional ``.env.local``), then starts
``server.py`` with those settings. Extra CLI flags are forwarded to the server.

Example::

    ./examples/molmoact_so101_eval/run_eval.sh
    ./examples/molmoact_so101_eval/run_eval.sh --local
    uv run --extra feetech python examples/molmoact_so101_eval/run_eval.py --port 8080
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_ENV = EXAMPLE_DIR / "default.env"
LOCAL_ENV = EXAMPLE_DIR / ".env.local"


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = os.path.expandvars(os.path.expanduser(value.strip().strip("'\"")))
        os.environ.setdefault(key, value)



def load_config() -> dict[str, str]:
    load_env_file(DEFAULT_ENV)
    load_env_file(LOCAL_ENV)

    inference = os.environ.get("MOLMOACT_INFERENCE", "remote").strip().lower()
    local_inference = inference == "local"
    local_host = os.environ.get("MOLMOACT_LOCAL_HOST", "127.0.0.1")
    local_port = os.environ.get("MOLMOACT_LOCAL_PORT", "8101")
    default_endpoint = (
        f"http://{local_host}:{local_port}/act"
        if local_inference
        else os.environ.get("MOLMOACT_ENDPOINT", "http://192.168.0.233:8014/act")
    )

    return {
        "host": os.environ.get("MOLMOACT_UI_HOST", "127.0.0.1"),
        "port": os.environ.get("MOLMOACT_UI_PORT", "7860"),
        "endpoint": default_endpoint,
        "robot_id": os.environ.get("MOLMOACT_ROBOT_ID", "my_awesome_follower_arm"),
        "log_level": os.environ.get("MOLMOACT_LOG_LEVEL", "INFO"),
        "inference": inference,
        "molmoact2_home": os.environ.get("MOLMOACT2_HOME", "~/workspace/molmoact2"),
        "local_host": local_host,
        "local_port": local_port,
        "local_startup_timeout_s": os.environ.get("MOLMOACT_LOCAL_STARTUP_TIMEOUT_S", "600"),
    }


def parse_launcher_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local",
        action="store_true",
        help="Start MolmoAct2 inference from MOLMOACT2_HOME instead of using a remote endpoint.",
    )
    parser.add_argument(
        "--remote",
        action="store_true",
        help="Use a remote MolmoAct2 /act endpoint (overrides MOLMOACT_INFERENCE=local).",
    )
    parser.add_argument(
        "--molmoact2-home",
        default=None,
        help="Path to the molmoact2 repo (default: MOLMOACT2_HOME or ~/workspace/molmoact2).",
    )
    parser.add_argument(
        "--local-port",
        type=int,
        default=None,
        help="Port for the managed local MolmoAct2 server (default: 8101).",
    )
    return parser.parse_known_args(argv)


def resolve_inference_mode(
    *,
    inference_env: str,
    local_flag: bool,
    remote_flag: bool,
) -> bool:
    if remote_flag:
        return False
    if local_flag:
        return True
    return inference_env == "local"


def main() -> None:
    config = load_config()
    launcher_args, forwarded_argv = parse_launcher_args(sys.argv[1:])

    use_local = resolve_inference_mode(
        inference_env=config["inference"],
        local_flag=launcher_args.local,
        remote_flag=launcher_args.remote,
    )
    local_server = None

    if use_local:
        sys.path.insert(0, str(EXAMPLE_DIR))
        from local_inference import LocalInferenceConfig, LocalInferenceServer  # noqa: PLC0415

        molmoact2_home = Path(launcher_args.molmoact2_home or config["molmoact2_home"]).expanduser()
        local_host = config["local_host"]
        local_port = launcher_args.local_port or int(config["local_port"])
        local_server = LocalInferenceServer(
            LocalInferenceConfig(
                molmoact2_home=molmoact2_home,
                host=local_host,
                port=local_port,
                startup_timeout_s=float(config["local_startup_timeout_s"]),
            )
        )
        logging.basicConfig(
            level=getattr(logging, config["log_level"]),
            format="%(levelname)s:%(name)s:%(message)s",
        )
        local_server.start()
        config["endpoint"] = local_server.endpoint
        inference_mode = "local"
        inference_schema = "front_wrist"
    else:
        inference_mode = "remote"
        inference_schema = "top_side"
        config["endpoint"] = os.environ.get("MOLMOACT_ENDPOINT", "http://192.168.0.233:8014/act")

    sys.path.insert(0, str(EXAMPLE_DIR))
    import server  # noqa: PLC0415

    print("Starting MolmoAct2 SO-101 evaluation UI")
    print(f"  UI:       http://{config['host']}:{config['port']}")
    print(f"  Mode:     {inference_mode}")
    print(f"  Endpoint: {config['endpoint']}")
    print(f"  Robot ID: {config['robot_id']}")
    print("Connect cameras and the follower in the browser, then dry-run before evaluating.")

    sys.argv = [
        str(EXAMPLE_DIR / "server.py"),
        "--host",
        config["host"],
        "--port",
        config["port"],
        "--endpoint",
        config["endpoint"],
        "--robot-id",
        config["robot_id"],
        "--log-level",
        config["log_level"],
        "--inference-mode",
        inference_mode,
        "--inference-schema",
        inference_schema,
        *forwarded_argv,
    ]
    try:
        server.main()
    finally:
        if local_server is not None:
            local_server.stop()


if __name__ == "__main__":
    main()
