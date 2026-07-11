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

"""Local web UI for evaluating MolmoAct2-SO100_101 on an SO-ARM101 follower.

The MolmoAct2 model is expected to be hosted separately at a FastAPI-style
`/act` endpoint using the json-numpy wire format. This app owns only local I/O:
webcam capture, SO-101 joint reads/writes, and a small browser UI.
"""

from __future__ import annotations

import argparse
import base64
import glob
import json
import logging
import mimetypes
import threading
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2  # type: ignore[import-untyped]
import numpy as np
import requests
from numpy.typing import NDArray

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

DEFAULT_ENDPOINT = "http://192.168.0.233:8014/act"
DEFAULT_LOCAL_ENDPOINT = "http://127.0.0.1:8101/act"
DEFAULT_ROBOT_ID = "jedld-follower"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7860
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_FPS = 30
DEFAULT_MAX_RELATIVE_TARGET = 10.0
DEFAULT_MAX_STEP_DEG = 15.0
DEFAULT_TIMEOUT_S = 60.0
DEFAULT_SMOOTHING_ALPHA = 0.8
DEFAULT_INTERPOLATION_STEPS = 3

ACTION_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
JOINT_OFFSETS = np.asarray([0.0, 90.0, 90.0, 0.0, 0.0, 0.0], dtype=np.float32)
JOINT_SIGNS = np.asarray([1.0, -1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
HOME_ACTION = {
    "shoulder_pan": 0.0,
    "shoulder_lift": 0.0,
    "elbow_flex": 0.0,
    "wrist_flex": 0.0,
    "wrist_roll": 0.0,
    "gripper": 50.0,
}
PACKED_ACTION = {
    "shoulder_pan": -0.3076923076923077,
    "shoulder_lift": -103.91208791208791,
    "elbow_flex": 97.31868131868131,
    "wrist_flex": 72.65934065934066,
    "wrist_roll": -0.13186813186813187,
    "gripper": 0.7628294036061026,
}
POSE_TOLERANCE = 2.0
POSE_TIMEOUT_S = 10.0


def json_numpy_default(obj: Any) -> dict[str, Any]:
    if isinstance(obj, np.ndarray):
        array = np.ascontiguousarray(obj)
        return {
            "__numpy__": base64.b64encode(array.data).decode("ascii"),
            "dtype": array.dtype.descr if array.dtype.fields else array.dtype.str,
            "shape": array.shape,
        }
    if isinstance(obj, np.generic):
        return json_numpy_default(np.asarray(obj))
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def json_numpy_object_hook(payload: dict[str, Any]) -> Any:
    if "__numpy__" not in payload:
        return payload
    dtype = np.dtype(payload["dtype"])
    array = np.frombuffer(base64.b64decode(payload["__numpy__"]), dtype=dtype)
    shape = payload.get("shape", [])
    if shape:
        return array.reshape(shape)
    return array[0]


def json_dumps(payload: Any) -> bytes:
    return json.dumps(payload, default=json_numpy_default).encode("utf-8")


def read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    content_length = int(handler.headers.get("Content-Length", "0"))
    if content_length <= 0:
        return {}
    return json.loads(handler.rfile.read(content_length).decode("utf-8"))


def parse_optional_positive_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    parsed = int(value)
    if parsed < 1:
        return None
    return parsed


def list_serial_ports() -> list[str]:
    patterns = [
        "/dev/tty.usbmodem*",
        "/dev/tty.usbserial*",
        "/dev/cu.usbmodem*",
        "/dev/cu.usbserial*",
        "/dev/ttyACM*",
        "/dev/ttyUSB*",
    ]
    ports: set[str] = set()
    for pattern in patterns:
        ports.update(glob.glob(pattern))
    return sorted(ports)


def detect_robot_port() -> str:
    ports = list_serial_ports()
    if not ports:
        raise RuntimeError("No USB serial ports found for the SO-ARM follower.")

    preferred_prefixes = (
        "/dev/cu.usbmodem",
        "/dev/cu.usbserial",
        "/dev/ttyACM",
        "/dev/ttyUSB",
        "/dev/tty.usbmodem",
        "/dev/tty.usbserial",
    )
    for prefix in preferred_prefixes:
        matches = [port for port in ports if port.startswith(prefix)]
        if matches:
            return sorted(matches)[0]
    return ports[0]


def list_opencv_cameras() -> list[dict[str, Any]]:
    cameras = []
    for index in range(6):
        cameras.append(
            {
                "id": index,
                "name": f"OpenCV index {index}",
                "readable": None,
                "width": DEFAULT_WIDTH,
                "height": DEFAULT_HEIGHT,
                "fps": DEFAULT_FPS,
                "backend_api": "probe-on-connect",
            }
        )
    return cameras


def read_capture_frame_with_timeout(capture: cv2.VideoCapture, *, timeout_s: float) -> tuple[bool, NDArray[Any] | None]:
    result: dict[str, Any] = {"ok": False, "frame": None}

    def read_frame() -> None:
        result["ok"], result["frame"] = capture.read()

    thread = threading.Thread(target=read_frame, daemon=True)
    thread.start()
    thread.join(timeout=timeout_s)
    if thread.is_alive():
        return False, None
    return bool(result["ok"]), result["frame"]


def probe_opencv_profile(index: int, width: int | None, height: int | None, fps: int | None) -> tuple[int, int, int | None]:
    attempts = [(width, height, fps), (None, None, None)]
    last_error = "no attempts made"
    for attempt_width, attempt_height, attempt_fps in attempts:
        capture = cv2.VideoCapture(index)
        if not capture.isOpened():
            capture.release()
            last_error = "failed to open"
            continue
        try:
            if attempt_width is not None:
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(attempt_width))
            if attempt_height is not None:
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(attempt_height))
            if attempt_fps is not None:
                capture.set(cv2.CAP_PROP_FPS, float(attempt_fps))
            ok = False
            frame = None
            for _ in range(5):
                ok, frame = read_capture_frame_with_timeout(capture, timeout_s=1.0)
                if ok:
                    break
                time.sleep(0.1)
            if ok and frame is not None:
                actual_height, actual_width = frame.shape[:2]
                actual_fps = int(round(capture.get(cv2.CAP_PROP_FPS))) or attempt_fps
                return actual_width, actual_height, actual_fps
            last_error = "did not return a frame"
        finally:
            capture.release()
    raise RuntimeError(f"OpenCV camera index {index} {last_error} while probing.")


class LocalOpenCVCamera:
    """Small OpenCV reader that tolerates AVFoundation resolution renegotiation."""

    def __init__(self, *, index: int, width: int | None, height: int | None, fps: int | None):
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self.capture: cv2.VideoCapture | None = None
        self.frame_lock = threading.Lock()
        self.capture_lock = threading.Lock()
        self.latest_frame: NDArray[Any] | None = None
        self.latest_timestamp: float | None = None
        self.stop_event = threading.Event()
        self.new_frame_event = threading.Event()
        self.thread: threading.Thread | None = None

    @property
    def is_connected(self) -> bool:
        return bool(self.capture and self.capture.isOpened() and self.thread and self.thread.is_alive())

    @property
    def has_fresh_frame(self) -> bool:
        with self.frame_lock:
            timestamp = self.latest_timestamp
        return timestamp is not None and (time.perf_counter() - timestamp) * 1e3 <= 2000

    def status(self) -> dict[str, Any]:
        with self.frame_lock:
            frame = self.latest_frame
            timestamp = self.latest_timestamp
        age_ms = None if timestamp is None else (time.perf_counter() - timestamp) * 1e3
        height = width = None
        if frame is not None:
            height, width = frame.shape[:2]
        return {
            "index": self.index,
            "connected": self.is_connected,
            "fresh": self.has_fresh_frame,
            "frame_age_ms": age_ms,
            "width": width,
            "height": height,
            "requested_width": self.width,
            "requested_height": self.height,
            "requested_fps": self.fps,
        }

    def connect(self) -> None:
        cv2.setNumThreads(1)
        capture = cv2.VideoCapture(self.index)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Failed to open OpenCV camera index {self.index}.")
        if self.width is not None:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        if self.height is not None:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
        if self.fps is not None:
            capture.set(cv2.CAP_PROP_FPS, float(self.fps))
        self.capture = capture
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._read_loop, name=f"LocalOpenCVCamera({self.index})", daemon=True)
        self.thread.start()
        deadline = time.perf_counter() + 2.0
        while time.perf_counter() < deadline:
            with self.frame_lock:
                if self.latest_frame is not None:
                    return
            time.sleep(0.05)
        self.disconnect()
        raise RuntimeError(f"OpenCV camera index {self.index} did not produce frames after connecting.")

    def disconnect(self) -> None:
        self.stop_event.set()
        with self.capture_lock:
            if self.capture is not None:
                self.capture.release()
                self.capture = None
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.thread = None

    def read_latest(self, max_age_ms: int = 1000) -> NDArray[Any]:
        with self.frame_lock:
            frame = self.latest_frame
            timestamp = self.latest_timestamp
        if frame is None or timestamp is None:
            raise RuntimeError(f"OpenCV camera index {self.index} has not captured a frame yet.")
        age_ms = (time.perf_counter() - timestamp) * 1e3
        if age_ms > max_age_ms:
            raise TimeoutError(
                f"OpenCV camera index {self.index} latest frame is too old: {age_ms:.1f} ms "
                f"(max allowed: {max_age_ms} ms)."
            )
        return frame.copy()

    def read(self) -> NDArray[Any]:
        previous_timestamp = self.latest_timestamp
        deadline = time.perf_counter() + 2.0
        while time.perf_counter() < deadline:
            self.new_frame_event.wait(timeout=0.1)
            self.new_frame_event.clear()
            with self.frame_lock:
                if self.latest_frame is not None and self.latest_timestamp != previous_timestamp:
                    return self.latest_frame.copy()
        return self.read_latest(max_age_ms=2000)

    def _read_loop(self) -> None:
        while not self.stop_event.is_set():
            capture = self.capture
            if capture is None:
                return
            ok, frame = capture.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with self.frame_lock:
                self.latest_frame = rgb_frame
                self.latest_timestamp = time.perf_counter()
            self.new_frame_event.set()


def clip_action_to_step(action: NDArray[np.float32], current_state: NDArray[np.float32], max_step_deg: float) -> NDArray[np.float32]:
    delta = action - current_state
    biggest = float(np.max(np.abs(delta)))
    if biggest <= max_step_deg or biggest == 0.0:
        return action
    return current_state + delta * (max_step_deg / biggest)


def build_max_relative_target(max_arm: float | None) -> float | dict[str, float] | None:
    if max_arm is None:
        return None
    return {name: (100.0 if name == "gripper" else max_arm) for name in ACTION_NAMES}


def arm_state_to_model_frame(state: NDArray[np.float32]) -> NDArray[np.float32]:
    return JOINT_SIGNS * state + JOINT_OFFSETS


def model_actions_to_arm_frame(actions: NDArray[np.float32]) -> NDArray[np.float32]:
    return (actions - JOINT_OFFSETS) * JOINT_SIGNS


def build_inference_payload(
    *,
    inference_schema: str,
    top_frame: NDArray[Any],
    side_frame: NDArray[Any],
    instruction: str,
    state: NDArray[np.float32],
) -> dict[str, Any]:
    if inference_schema == "front_wrist":
        return {
            "front_cam": top_frame,
            "wrist_cam": side_frame,
            "instruction": instruction,
            "state": state,
        }
    return {
        "top_cam": top_frame,
        "side_cam": side_frame,
        "instruction": instruction,
        "state": state,
    }


@dataclass
class AppState:
    endpoint: str = DEFAULT_ENDPOINT
    inference_mode: str = "remote"
    inference_schema: str = "top_side"
    instruction: str = "pick up the object"
    robot_id: str = DEFAULT_ROBOT_ID
    action_fps: float = DEFAULT_FPS
    max_step_deg: float = DEFAULT_MAX_STEP_DEG
    actions_per_chunk: int | None = None
    smoothing_alpha: float = DEFAULT_SMOOTHING_ALPHA
    interpolation_steps: int = DEFAULT_INTERPOLATION_STEPS
    apply_joint_conversion: bool = True
    request_timeout_s: float = DEFAULT_TIMEOUT_S
    last_log: list[str] = field(default_factory=list)
    last_action: list[float] | None = None
    last_actions_shape: list[int] | None = None
    last_inference_ms: float | None = None
    last_server_dt_ms: float | None = None
    last_frame_age_ms: float | None = None
    last_image_state_skew_ms: float | None = None
    last_camera_skew_ms: float | None = None
    last_observation_id: int | None = None
    last_action_observation_id: int | None = None
    last_side_camera_source: str | None = None
    last_action_settle_ms: float | None = None
    last_action_error: dict[str, float] | None = None
    smoothed_action_target: NDArray[np.float32] | None = None
    next_observation_id: int = 1
    last_error: str | None = None
    last_joint_positions: dict[str, float] | None = None
    program_mode: bool = False
    server_metadata: dict[str, Any] | None = None
    robot: SO101Follower | None = None
    top_camera: LocalOpenCVCamera | None = None
    side_camera: LocalOpenCVCamera | None = None
    lock: threading.RLock = field(default_factory=threading.RLock)
    hardware_lock: threading.RLock = field(default_factory=threading.RLock)
    command_lock: threading.Lock = field(default_factory=threading.Lock)
    stop_event: threading.Event = field(default_factory=threading.Event)
    eval_thread: threading.Thread | None = None

    def log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        line = f"[{timestamp}] {message}"
        logging.info(line)
        with self.lock:
            self.last_log.append(line)
            self.last_log = self.last_log[-200:]

    @property
    def robot_connected(self) -> bool:
        return bool(self.robot and self.robot.is_connected)

    @property
    def camera_connected(self) -> bool:
        return self.top_camera_connected

    @property
    def top_camera_connected(self) -> bool:
        return bool(self.top_camera and self.top_camera.is_connected and self.top_camera.has_fresh_frame)

    @property
    def side_camera_connected(self) -> bool:
        return bool(self.side_camera and self.side_camera.is_connected and self.side_camera.has_fresh_frame)

    def camera_status(self, *, slot: str) -> dict[str, Any] | None:
        camera = self.top_camera if slot == "top" else self.side_camera
        return camera.status() if camera is not None else None

    @property
    def running(self) -> bool:
        return bool(self.eval_thread and self.eval_thread.is_alive())

    def status(self) -> dict[str, Any]:
        if self.program_mode and self.robot_connected:
            try:
                self.read_joint_positions()
            except Exception as exc:  # noqa: BLE001
                with self.lock:
                    self.last_error = str(exc)
        with self.lock:
            return {
                "endpoint": self.endpoint,
                "inference_mode": self.inference_mode,
                "inference_schema": self.inference_schema,
                "instruction": self.instruction,
                "robot_id": self.robot_id,
                "robot_connected": self.robot_connected,
                "camera_connected": self.camera_connected,
                "top_camera_connected": self.top_camera_connected,
                "side_camera_connected": self.side_camera_connected,
                "top_camera": self.camera_status(slot="top"),
                "side_camera": self.camera_status(slot="side"),
                "running": self.running,
                "program_mode": self.program_mode,
                "action_fps": self.action_fps,
                "max_step_deg": self.max_step_deg,
                "actions_per_chunk": self.actions_per_chunk,
                "smoothing_alpha": self.smoothing_alpha,
                "interpolation_steps": self.interpolation_steps,
                "apply_joint_conversion": self.apply_joint_conversion,
                "joint_positions": self.last_joint_positions,
                "last_action": self.last_action,
                "last_actions_shape": self.last_actions_shape,
                "last_inference_ms": self.last_inference_ms,
                "last_server_dt_ms": self.last_server_dt_ms,
                "last_frame_age_ms": self.last_frame_age_ms,
                "last_image_state_skew_ms": self.last_image_state_skew_ms,
                "last_camera_skew_ms": self.last_camera_skew_ms,
                "last_observation_id": self.last_observation_id,
                "last_action_observation_id": self.last_action_observation_id,
                "last_side_camera_source": self.last_side_camera_source,
                "last_action_settle_ms": self.last_action_settle_ms,
                "last_action_error": self.last_action_error,
                "last_error": self.last_error,
                "server_metadata": self.server_metadata,
                "logs": list(self.last_log),
            }

    def refresh_endpoint_metadata(self) -> dict[str, Any]:
        response = requests.get(self.endpoint, timeout=5)
        response.raise_for_status()
        metadata = response.json()
        with self.lock:
            self.server_metadata = metadata
        return metadata

    def connect_camera(self, *, slot: str, index: int, width: int, height: int, fps: int) -> None:
        if slot not in {"top", "side"}:
            raise ValueError("Camera slot must be 'top' or 'side'.")
        actual_width, actual_height, actual_fps = probe_opencv_profile(index, width, height, fps)
        camera = LocalOpenCVCamera(index=index, width=actual_width, height=actual_height, fps=actual_fps)
        camera.connect()
        old_camera = None
        with self.lock:
            if slot == "top":
                old_camera = self.top_camera
                self.top_camera = camera
            else:
                old_camera = self.side_camera
                self.side_camera = camera
            self.last_error = None
        if old_camera and old_camera.is_connected:
            old_camera.disconnect()
            self.log(f"Disconnected previous {slot} webcam.")
        requested = f"{width}x{height}@{fps}"
        actual = f"{actual_width}x{actual_height}@{actual_fps}"
        self.log(f"Connected {slot} webcam OpenCV index {index} at {actual} (requested {requested}).")

    def disconnect_camera(self, *, slot: str | None = None) -> None:
        if slot == "top":
            cameras = [("top", self.top_camera)]
            self.top_camera = None
        elif slot == "side":
            cameras = [("side", self.side_camera)]
            self.side_camera = None
        else:
            cameras = [("top", self.top_camera), ("side", self.side_camera)]
            self.top_camera = None
            self.side_camera = None
        for camera_slot, camera in cameras:
            if camera and camera.is_connected:
                camera.disconnect()
                self.log(f"Disconnected {camera_slot} webcam.")

    def connect_robot(
        self,
        *,
        port: str,
        robot_id: str,
        max_relative_target: float | None,
        max_step_deg: float,
        calibrate: bool,
    ) -> None:
        with self.lock:
            if self.robot is not None:
                self.disconnect_robot()
            config = SO101FollowerConfig(
                port=port,
                id=robot_id,
                cameras={},
                max_relative_target=build_max_relative_target(max_relative_target),
                use_degrees=True,
            )
            robot = SO101Follower(config)
            with self.hardware_lock:
                robot.connect(calibrate=calibrate)
            self.robot = robot
            self.robot_id = robot_id
            self.max_step_deg = max_step_deg
            self.program_mode = False
            self.last_joint_positions = None
            self.last_error = None
        self.log(f"Connected SO-101 follower {robot_id!r} on {port}.")

    def disconnect_robot(self) -> None:
        robot = self.robot
        self.robot = None
        self.program_mode = False
        self.last_joint_positions = None
        if robot and robot.is_connected:
            with self.hardware_lock:
                robot.disconnect()
            self.log("Disconnected SO-101 follower.")

    def close(self) -> None:
        self.stop()
        self.disconnect_robot()
        self.disconnect_camera()

    def get_frame(self, *, slot: str = "top") -> NDArray[Any]:
        camera = self.top_camera if slot == "top" else self.side_camera
        if camera is None or not camera.is_connected:
            raise RuntimeError(f"{slot.capitalize()} camera is not connected.")
        return camera.read_latest(max_age_ms=1000)

    def read_fresh_frame(self, *, slot: str = "top") -> tuple[NDArray[Any], float, float]:
        camera = self.top_camera if slot == "top" else self.side_camera
        if camera is None or not camera.is_connected:
            raise RuntimeError(f"{slot.capitalize()} camera is not connected.")
        frame = camera.read()
        with camera.frame_lock:
            timestamp = camera.latest_timestamp
        if timestamp is None:
            raise RuntimeError("Camera did not report a frame timestamp.")
        age_ms = (time.perf_counter() - timestamp) * 1e3
        return frame, timestamp, age_ms

    def get_state_vector(self) -> NDArray[np.float32]:
        robot = self.robot
        if robot is None or not robot.is_connected:
            return np.zeros(len(ACTION_NAMES), dtype=np.float32)
        positions = self.read_joint_positions()
        return np.asarray([positions[name] for name in ACTION_NAMES], dtype=np.float32)

    def read_joint_positions(self) -> dict[str, float]:
        robot = self.robot
        if robot is None or not robot.is_connected:
            raise RuntimeError("Robot is not connected.")
        with self.hardware_lock:
            observation = robot.get_observation()
        positions = {name: float(observation[f"{name}.pos"]) for name in ACTION_NAMES}
        with self.lock:
            self.last_joint_positions = positions
        return positions

    def enter_program_mode(self) -> dict[str, Any]:
        robot = self.robot
        if robot is None or not robot.is_connected:
            raise RuntimeError("Robot is not connected.")
        self.stop()
        with self.hardware_lock:
            robot.bus.disable_torque(num_retry=5)
        with self.lock:
            self.program_mode = True
            self.last_error = None
        self.read_joint_positions()
        self.log("Program mode enabled: motor torque disabled; move joints by hand.")
        return self.status()

    def exit_program_mode(self) -> dict[str, Any]:
        robot = self.robot
        if robot is None or not robot.is_connected:
            raise RuntimeError("Robot is not connected.")
        with self.hardware_lock:
            robot.bus.enable_torque(num_retry=5)
        with self.lock:
            self.program_mode = False
            self.last_error = None
        self.read_joint_positions()
        self.log("Program mode disabled: motor torque re-enabled.")
        return self.status()

    def ensure_torque_enabled(self) -> None:
        if self.program_mode:
            self.exit_program_mode()

    def request_actions(self, *, instruction: str | None = None) -> NDArray[np.float32]:
        with self.lock:
            observation_id = self.next_observation_id
            self.next_observation_id += 1
        top_frame, top_timestamp, top_age_ms = self.read_fresh_frame(slot="top")
        side_source = "top_duplicate"
        if (
            self.side_camera_connected
            and self.side_camera is not None
            and self.top_camera is not None
            and self.side_camera.index != self.top_camera.index
        ):
            side_frame, side_timestamp, side_age_ms = self.read_fresh_frame(slot="side")
            side_source = "side"
        else:
            side_frame, side_timestamp, side_age_ms = top_frame, top_timestamp, top_age_ms
        arm_state = self.get_state_vector()
        state_timestamp = time.perf_counter()
        image_state_skew_ms = (state_timestamp - min(top_timestamp, side_timestamp)) * 1e3
        camera_skew_ms = abs(top_timestamp - side_timestamp) * 1e3
        frame_age_ms = max(top_age_ms, side_age_ms)
        state = arm_state_to_model_frame(arm_state) if self.apply_joint_conversion else arm_state
        task = instruction if instruction is not None else self.instruction
        payload = build_inference_payload(
            inference_schema=self.inference_schema,
            top_frame=top_frame,
            side_frame=side_frame,
            instruction=task,
            state=state,
        )
        start = time.perf_counter()
        response = requests.post(
            self.endpoint,
            data=json.dumps(payload, default=json_numpy_default),
            headers={"Content-Type": "application/json"},
            timeout=self.request_timeout_s,
        )
        elapsed_ms = (time.perf_counter() - start) * 1e3
        response.raise_for_status()
        decoded = json.loads(response.text, object_hook=json_numpy_object_hook)
        actions = np.asarray(decoded["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != len(ACTION_NAMES):
            raise RuntimeError(f"Expected actions shaped (N, 6), got {actions.shape}.")
        if self.apply_joint_conversion:
            actions = model_actions_to_arm_frame(actions).astype(np.float32, copy=False)
        with self.lock:
            self.last_action = actions[0].astype(float).tolist()
            self.last_actions_shape = list(actions.shape)
            self.last_inference_ms = elapsed_ms
            self.last_server_dt_ms = float(decoded["dt_ms"]) if decoded.get("dt_ms") is not None else None
            self.last_frame_age_ms = frame_age_ms
            self.last_image_state_skew_ms = image_state_skew_ms
            self.last_camera_skew_ms = camera_skew_ms
            self.last_observation_id = observation_id
            self.last_action_observation_id = observation_id
            self.last_side_camera_source = side_source
            self.last_error = None
        return actions

    def execute_actions(self, actions: NDArray[np.float32]) -> None:
        robot = self.robot
        if robot is None or not robot.is_connected:
            raise RuntimeError("Robot is not connected.")
        control_period_s = 1.0 / max(self.action_fps * self.interpolation_steps, 1.0)
        actions_to_execute = actions if self.actions_per_chunk is None else actions[: self.actions_per_chunk]
        smoothed_target = self.smoothed_action_target
        for action in actions_to_execute:
            if self.stop_event.is_set():
                break
            current = self.get_state_vector()
            if smoothed_target is None:
                smoothed_target = current
            alpha = float(np.clip(self.smoothing_alpha, 0.05, 1.0))
            smoothed_target = (alpha * action + (1.0 - alpha) * smoothed_target).astype(np.float32)
            start_state = current
            sent_target = start_state
            step_start = time.perf_counter()
            for step_idx in range(1, self.interpolation_steps + 1):
                if self.stop_event.is_set():
                    break
                target = start_state + (smoothed_target - start_state) * (step_idx / self.interpolation_steps)
                current = self.get_state_vector()
                target = clip_action_to_step(target, current, self.max_step_deg)
                robot_action = {f"{name}.pos": float(value) for name, value in zip(ACTION_NAMES, target, strict=True)}
                with self.hardware_lock:
                    sent_action = robot.send_action(robot_action)
                sent_target = np.asarray([float(sent_action[f"{name}.pos"]) for name in ACTION_NAMES], dtype=np.float32)
                with self.lock:
                    self.last_action = sent_target.astype(float).tolist()
                    self.last_action_error = {
                        name: float(error)
                        for name, error in zip(ACTION_NAMES, np.abs(current - sent_target), strict=True)
                    }
                sleep_s = control_period_s - (time.perf_counter() - step_start)
                if sleep_s > 0:
                    time.sleep(sleep_s)
                step_start = time.perf_counter()
            with self.lock:
                self.last_action = sent_target.astype(float).tolist()
                self.last_action_settle_ms = None
                self.smoothed_action_target = smoothed_target

    def infer_once(self, *, execute: bool, instruction: str | None = None) -> dict[str, Any]:
        actions = self.request_actions(instruction=instruction)
        if execute:
            self.execute_actions(actions)
            self.log(f"Executed {len(actions)} MolmoAct2 actions.")
        else:
            self.log(f"Dry-run inference returned {len(actions)} actions.")
        return self.status()

    def reset_home(self) -> dict[str, Any]:
        return self.move_to_pose("home", HOME_ACTION)

    def reset_packed(self) -> dict[str, Any]:
        return self.move_to_pose("packed", PACKED_ACTION)

    def move_to_pose(self, pose_name: str, pose: dict[str, float]) -> dict[str, Any]:
        robot = self.robot
        if robot is None or not robot.is_connected:
            raise RuntimeError("Robot is not connected.")
        self.stop()
        with self.command_lock:
            self.ensure_torque_enabled()
            with self.lock:
                self.smoothed_action_target = None
            self.log(f"Moving robot to {pose_name} position.")
            deadline = time.perf_counter() + POSE_TIMEOUT_S
            control_period_s = 1.0 / max(self.action_fps, 1.0)
            sent_action = {f"{name}.pos": pose[name] for name in ACTION_NAMES}
            while time.perf_counter() < deadline:
                target = np.asarray([pose[name] for name in ACTION_NAMES], dtype=np.float32)
                clipped_target = clip_action_to_step(target, self.get_state_vector(), self.max_step_deg)
                with self.hardware_lock:
                    sent_action = robot.send_action(
                        {f"{name}.pos": float(value) for name, value in zip(ACTION_NAMES, clipped_target, strict=True)}
                    )
                with self.lock:
                    self.last_action = [float(sent_action[f"{name}.pos"]) for name in ACTION_NAMES]
                    self.last_error = None
                time.sleep(control_period_s)
                state = self.get_state_vector()
                if float(np.max(np.abs(state - target))) <= POSE_TOLERANCE:
                    break
            self.log(
                f"{pose_name.capitalize()} command sent: "
                + ", ".join(f"{name}={float(sent_action[f'{name}.pos']):.1f}" for name in ACTION_NAMES)
            )
        return self.status()

    def start(
        self,
        *,
        instruction: str,
        endpoint: str,
        action_fps: float,
        max_step_deg: float,
        actions_per_chunk: int | None,
        smoothing_alpha: float,
        interpolation_steps: int,
        apply_joint_conversion: bool,
    ) -> None:
        if not self.camera_connected:
            raise RuntimeError("Connect the webcam before starting evaluation.")
        if not self.robot_connected:
            raise RuntimeError("Connect the robot before starting evaluation.")
        if self.running:
            raise RuntimeError("Evaluation is already running.")
        if self.command_lock.locked():
            raise RuntimeError("Robot is busy finishing another command. Wait for it to complete before starting evaluation.")
        self.ensure_torque_enabled()
        with self.lock:
            self.instruction = instruction
            self.endpoint = endpoint
            self.action_fps = action_fps
            self.max_step_deg = max_step_deg
            self.actions_per_chunk = actions_per_chunk
            self.smoothing_alpha = float(np.clip(smoothing_alpha, 0.05, 1.0))
            self.interpolation_steps = max(1, interpolation_steps)
            self.apply_joint_conversion = apply_joint_conversion
            self.smoothed_action_target = None
            self.stop_event.clear()
        thread = threading.Thread(target=self._eval_loop, name="molmoact_eval_loop", daemon=True)
        self.eval_thread = thread
        thread.start()
        self.log("Started closed-loop evaluation.")

    def _eval_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                if not self.command_lock.acquire(timeout=0.1):
                    continue
                try:
                    actions = self.request_actions()
                    self.execute_actions(actions)
                finally:
                    self.command_lock.release()
            except Exception as exc:  # noqa: BLE001
                with self.lock:
                    self.last_error = str(exc)
                self.log(f"Evaluation stopped after error: {exc}")
                self.stop_event.set()
                break
        self.log("Evaluation loop stopped.")

    def stop(self) -> None:
        self.stop_event.set()
        thread = self.eval_thread
        if thread and thread.is_alive():
            thread.join(timeout=3.0)
        with self.lock:
            self.smoothed_action_target = None


class MolmoActHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], handler: type[BaseHTTPRequestHandler], state: AppState):
        super().__init__(server_address, handler)
        self.state = state


class Handler(BaseHTTPRequestHandler):
    server: MolmoActHTTPServer

    def log_message(self, fmt: str, *args: Any) -> None:
        logging.debug("%s - %s", self.client_address[0], fmt % args)

    def do_GET(self) -> None:  # noqa: N802
        try:
            parsed_url = urlparse(self.path)
            path = parsed_url.path
            if path in {"/", "/index.html"}:
                self.send_file(Path(__file__).with_name("static") / "index.html")
            elif path == "/api/status":
                self.send_json(self.server.state.status())
            elif path == "/api/devices":
                serial_ports = list_serial_ports()
                detected_robot_port = detect_robot_port() if serial_ports else None
                self.send_json(
                    {
                        "serial_ports": serial_ports,
                        "detected_robot_port": detected_robot_port,
                        "opencv_cameras": list_opencv_cameras(),
                    }
                )
            elif path == "/api/frame.jpg":
                self.send_frame(slot="side" if "slot=side" in parsed_url.query else "top")
            else:
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
        except Exception as exc:  # noqa: BLE001
            self.send_json_error(exc)

    def do_POST(self) -> None:  # noqa: N802
        try:
            path = urlparse(self.path).path
            body = read_json_body(self)
            state = self.server.state
            if path == "/api/endpoint":
                state.endpoint = str(body.get("endpoint") or state.endpoint)
                self.send_json(state.refresh_endpoint_metadata())
            elif path == "/api/connect_camera":
                state.connect_camera(
                    slot=str(body.get("slot") or "top"),
                    index=int(body.get("camera_index", 0)),
                    width=int(body.get("width") or DEFAULT_WIDTH),
                    height=int(body.get("height") or DEFAULT_HEIGHT),
                    fps=int(body.get("fps") or DEFAULT_FPS),
                )
                self.send_json(state.status())
            elif path == "/api/detect_robot_port":
                self.send_json({"port": detect_robot_port(), "serial_ports": list_serial_ports()})
            elif path == "/api/connect_robot":
                max_relative_target = body.get("max_relative_target", DEFAULT_MAX_RELATIVE_TARGET)
                state.connect_robot(
                    port=str(body.get("port") or detect_robot_port()),
                    robot_id=str(body.get("robot_id") or DEFAULT_ROBOT_ID),
                    max_relative_target=None
                    if max_relative_target in {None, ""}
                    else float(max_relative_target),
                    max_step_deg=float(body.get("max_step_deg", body.get("max_relative_target", DEFAULT_MAX_STEP_DEG))),
                    calibrate=bool(body.get("calibrate", False)),
                )
                self.send_json(state.status())
            elif path == "/api/disconnect":
                state.close()
                self.send_json(state.status())
            elif path == "/api/dry_run":
                self.update_runtime_settings(body)
                self.send_json(state.infer_once(execute=False, instruction=state.instruction))
            elif path == "/api/start":
                state.start(
                    instruction=str(body["instruction"]),
                    endpoint=str(body.get("endpoint") or state.endpoint),
                    action_fps=float(body.get("action_fps", DEFAULT_FPS)),
                    max_step_deg=float(body.get("max_step_deg", body.get("max_relative_target", DEFAULT_MAX_STEP_DEG))),
                    actions_per_chunk=parse_optional_positive_int(body.get("actions_per_chunk")),
                    smoothing_alpha=float(body.get("smoothing_alpha", DEFAULT_SMOOTHING_ALPHA)),
                    interpolation_steps=max(1, int(body.get("interpolation_steps", DEFAULT_INTERPOLATION_STEPS))),
                    apply_joint_conversion=bool(body.get("apply_joint_conversion", True)),
                )
                self.send_json(state.status())
            elif path == "/api/stop":
                state.stop()
                self.send_json(state.status())
            elif path == "/api/program_mode":
                self.send_json(state.enter_program_mode())
            elif path == "/api/exit_program_mode":
                self.send_json(state.exit_program_mode())
            elif path == "/api/reset_home":
                self.send_json(state.reset_home())
            elif path == "/api/reset_packed":
                self.send_json(state.reset_packed())
            else:
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
        except Exception as exc:  # noqa: BLE001
            self.server.state.last_error = str(exc)
            self.server.state.log(f"Request failed: {exc}")
            self.send_json_error(exc)

    def update_runtime_settings(self, body: dict[str, Any]) -> None:
        state = self.server.state
        with state.lock:
            state.endpoint = str(body.get("endpoint") or state.endpoint)
            state.instruction = str(body.get("instruction") or state.instruction)
            state.action_fps = float(body.get("action_fps", state.action_fps))
            state.max_step_deg = float(body.get("max_step_deg", body.get("max_relative_target", state.max_step_deg)))
            state.actions_per_chunk = parse_optional_positive_int(body.get("actions_per_chunk"))
            state.smoothing_alpha = float(np.clip(float(body.get("smoothing_alpha", state.smoothing_alpha)), 0.05, 1.0))
            state.interpolation_steps = max(1, int(body.get("interpolation_steps", state.interpolation_steps)))
            state.apply_joint_conversion = bool(body.get("apply_joint_conversion", state.apply_joint_conversion))

    def send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json_dumps(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_json_error(self, exc: Exception) -> None:
        self.send_json({"error": str(exc), "status": self.server.state.status()}, HTTPStatus.BAD_REQUEST)

    def send_file(self, path: Path) -> None:
        if not path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(path)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_frame(self, *, slot: str) -> None:
        frame = self.server.state.get_frame(slot=slot)
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".jpg", bgr)
        if not ok:
            raise RuntimeError("Failed to encode camera frame.")
        data = encoded.tobytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host interface for the local web UI.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port for the local web UI.")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="MolmoAct2 /act endpoint URL.")
    parser.add_argument(
        "--inference-mode",
        choices=["remote", "local"],
        default="remote",
        help="Remote HTTP endpoint (default) or locally managed MolmoAct2 inference.",
    )
    parser.add_argument(
        "--inference-schema",
        choices=["top_side", "front_wrist"],
        default="top_side",
        help="Camera field names sent to the inference endpoint.",
    )
    parser.add_argument("--robot-id", default=DEFAULT_ROBOT_ID, help="LeRobot calibration id for the SO-101.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s:%(name)s:%(message)s")
    state = AppState(
        endpoint=args.endpoint,
        robot_id=args.robot_id,
        inference_mode=args.inference_mode,
        inference_schema=args.inference_schema,
    )
    server = MolmoActHTTPServer((args.host, args.port), Handler, state)
    if args.inference_mode == "local":
        state.log(f"Local MolmoAct2 inference via {args.endpoint} ({args.inference_schema} camera schema).")
    state.log(f"Open http://{args.host}:{args.port} to evaluate MolmoAct2 on SO-ARM101.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        state.close()
        server.server_close()


if __name__ == "__main__":
    main()
