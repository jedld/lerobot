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

"""Convenience launcher for the SO-101 policy evaluation UI.

Loads ``default.env`` (and optional ``.env.local``), then starts ``server.py``
with those settings. Supports MolmoAct2 and open-source Pi0 / Pi0.5 policies
(Pi 0.7 weights are not public yet — use pi05/pi0 checkpoints instead).

Example::

    ./examples/molmoact_so101_eval/run_eval.sh
    ./examples/molmoact_so101_eval/run_eval.sh --policy pi --local
    uv run --extra feetech --extra pi python examples/molmoact_so101_eval/run_eval.py --port 8080
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

    return {
        "policy": os.environ.get("EVAL_POLICY", "molmoact2").strip().lower(),
        "host": os.environ.get("EVAL_UI_HOST", os.environ.get("MOLMOACT_UI_HOST", "127.0.0.1")),
        "port": os.environ.get("EVAL_UI_PORT", os.environ.get("MOLMOACT_UI_PORT", "7860")),
        "robot_id": os.environ.get("EVAL_ROBOT_ID", os.environ.get("MOLMOACT_ROBOT_ID", "my_awesome_follower_arm")),
        "log_level": os.environ.get("EVAL_LOG_LEVEL", os.environ.get("MOLMOACT_LOG_LEVEL", "INFO")),
        "inference": os.environ.get("EVAL_INFERENCE", os.environ.get("MOLMOACT_INFERENCE", "remote")).strip().lower(),
        "molmoact2_home": os.environ.get("MOLMOACT2_HOME", "~/workspace/molmoact2"),
        "local_host": os.environ.get(
            "EVAL_LOCAL_HOST",
            os.environ.get("PI_LOCAL_HOST", os.environ.get("MOLMOACT_LOCAL_HOST", "127.0.0.1")),
        ),
        "molmoact_local_port": os.environ.get("MOLMOACT_LOCAL_PORT", "8101"),
        "pi_local_port": os.environ.get("PI_LOCAL_PORT", "8102"),
        "molmoact_startup_timeout_s": os.environ.get("MOLMOACT_LOCAL_STARTUP_TIMEOUT_S", "600"),
        "pi_startup_timeout_s": os.environ.get("PI_LOCAL_STARTUP_TIMEOUT_S", "900"),
        "molmoact_endpoint": os.environ.get("MOLMOACT_ENDPOINT", "http://192.168.0.233:8014/act"),
        "pi_endpoint": os.environ.get("PI_ENDPOINT", "http://jedld-lab:8101/act"),
        "pi_checkpoint": os.environ.get("PI_CHECKPOINT", "lerobot/pi05_base"),
        "pi_policy_type": os.environ.get("PI_POLICY_TYPE", "pi05"),
        "pi_device": os.environ.get("PI_DEVICE", "auto"),
        "pi_robot_type": os.environ.get("PI_ROBOT_TYPE", ""),
        "pi_max_actions": os.environ.get("PI_MAX_ACTIONS", ""),
        "pi_top_image_key": os.environ.get("PI_TOP_IMAGE_KEY", ""),
        "pi_side_image_key": os.environ.get("PI_SIDE_IMAGE_KEY", ""),
        "pi_python": os.environ.get("PI_PYTHON", ""),
    }


def parse_launcher_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        choices=["molmoact2", "pi"],
        default=None,
        help="Policy backend: molmoact2 (default) or open-source pi05/pi0 (Pi 0.7 is not public yet).",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Start managed local inference instead of using a remote /act endpoint.",
    )
    parser.add_argument(
        "--remote",
        action="store_true",
        help="Use a remote /act endpoint (overrides EVAL_INFERENCE=local).",
    )
    parser.add_argument(
        "--molmoact2-home",
        default=None,
        help="Path to the molmoact2 repo (MolmoAct2 local mode only).",
    )
    parser.add_argument(
        "--local-port",
        type=int,
        default=None,
        help="Port for the managed local inference server.",
    )
    parser.add_argument(
        "--pi-checkpoint",
        default=None,
        help="Hugging Face repo or local path for Pi policy weights (pi mode only).",
    )
    parser.add_argument(
        "--pi-policy-type",
        choices=["pi05", "pi0"],
        default=None,
        help="LeRobot Pi policy class to load (default: pi05).",
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


def parse_optional_int(value: str) -> int | None:
    value = value.strip()
    if not value:
        return None
    return int(value)


def main() -> None:
    config = load_config()
    launcher_args, forwarded_argv = parse_launcher_args(sys.argv[1:])

    policy = (launcher_args.policy or config["policy"]).strip().lower()
    use_local = resolve_inference_mode(
        inference_env=config["inference"],
        local_flag=launcher_args.local,
        remote_flag=launcher_args.remote,
    )
    local_server = None

    if use_local:
        local_host = config["local_host"]
        if policy == "pi":
            sys.path.insert(0, str(EXAMPLE_DIR))
            from local_inference_pi import LocalPiInferenceConfig, LocalPiInferenceServer  # noqa: PLC0415

            local_port = launcher_args.local_port or int(config["pi_local_port"])
            checkpoint = launcher_args.pi_checkpoint or config["pi_checkpoint"]
            policy_type = launcher_args.pi_policy_type or config["pi_policy_type"]
            local_server = LocalPiInferenceServer(
                LocalPiInferenceConfig(
                    example_dir=EXAMPLE_DIR,
                    host=local_host,
                    port=local_port,
                    checkpoint=checkpoint,
                    policy_type=policy_type,
                    device=config["pi_device"],
                    robot_type=config["pi_robot_type"],
                    max_actions=parse_optional_int(config["pi_max_actions"]),
                    top_image_key=config["pi_top_image_key"] or None,
                    side_image_key=config["pi_side_image_key"] or None,
                    startup_timeout_s=float(config["pi_startup_timeout_s"]),
                    python_bin=config["pi_python"] or None,
                )
            )
            inference_schema = "top_side"
            apply_joint_conversion = False
        else:
            sys.path.insert(0, str(EXAMPLE_DIR))
            from local_inference import LocalInferenceConfig, LocalInferenceServer  # noqa: PLC0415

            molmoact2_home = Path(launcher_args.molmoact2_home or config["molmoact2_home"]).expanduser()
            local_port = launcher_args.local_port or int(config["molmoact_local_port"])
            local_server = LocalInferenceServer(
                LocalInferenceConfig(
                    molmoact2_home=molmoact2_home,
                    host=local_host,
                    port=local_port,
                    startup_timeout_s=float(config["molmoact_startup_timeout_s"]),
                )
            )
            inference_schema = "front_wrist"
            apply_joint_conversion = True

        logging.basicConfig(
            level=getattr(logging, config["log_level"]),
            format="%(levelname)s:%(name)s:%(message)s",
        )
        local_server.start()
        config["endpoint"] = local_server.endpoint
        inference_mode = "local"
    else:
        inference_mode = "remote"
        if policy == "pi":
            inference_schema = "top_side"
            apply_joint_conversion = False
            config["endpoint"] = config["pi_endpoint"]
        else:
            inference_schema = "top_side"
            apply_joint_conversion = True
            config["endpoint"] = config["molmoact_endpoint"]

    sys.path.insert(0, str(EXAMPLE_DIR))
    import server  # noqa: PLC0415

    policy_label = "Pi (pi05/pi0)" if policy == "pi" else "MolmoAct2"
    print(f"Starting SO-101 evaluation UI ({policy_label})")
    print(f"  UI:       http://{config['host']}:{config['port']}")
    print(f"  Policy:   {policy}")
    print(f"  Mode:     {inference_mode}")
    print(f"  Endpoint: {config['endpoint']}")
    print(f"  Robot ID: {config['robot_id']}")
    if policy == "pi":
        print("  Note:     Pi 0.7 weights are not public; using open-source pi05/pi0 checkpoints.")
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
        "--policy",
        policy,
        *(
            ["--default-apply-joint-conversion", "true" if apply_joint_conversion else "false"]
            if not any(arg.startswith("--default-apply-joint-conversion") for arg in forwarded_argv)
            else []
        ),
        *forwarded_argv,
    ]
    try:
        server.main()
    finally:
        if local_server is not None:
            local_server.stop()


if __name__ == "__main__":
    main()
