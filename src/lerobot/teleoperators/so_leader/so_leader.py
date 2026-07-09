# !/usr/bin/env python

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

import contextlib
import logging
import time

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .config_so_leader import SOLeaderTeleopConfig
from .force_feedback import GripperFeedbackState, apply_gripper_force_feedback

logger = logging.getLogger(__name__)


class SOLeader(Teleoperator):
    """Generic SO leader base for SO-100/101/10X teleoperators."""

    config_class = SOLeaderTeleopConfig
    name = "so_leader"

    def __init__(self, config: SOLeaderTeleopConfig):
        super().__init__(config)
        self.config = config
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "wrist_roll": Motor(5, "sts3215", norm_mode_body),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )

        # Gripper force feedback state (only used when config.gripper_force_feedback is True).
        self._gripper_feedback_state = GripperFeedbackState()

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def feedback_features(self) -> dict[str, type]:
        return self.action_features

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        if self.calibration:
            # Calibration file exists, ask user whether to use it or run new calibration
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                self.bus.write_calibration(self.calibration)
                return

        logger.info(f"\nRunning calibration of {self}")
        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        input(f"Move {self} to the middle of its range of motion and press ENTER....")
        homing_offsets = self.bus.set_half_turn_homings()

        full_turn_motor = "wrist_roll"
        unknown_range_motors = [motor for motor in self.bus.motors if motor != full_turn_motor]
        print(
            f"Move all joints except '{full_turn_motor}' sequentially through their "
            "entire ranges of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion(unknown_range_motors)
        range_mins[full_turn_motor] = 0
        range_maxes[full_turn_motor] = 4095

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print(f"Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        self.bus.disable_torque()
        self.bus.configure_motors()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        # Configure gripper motor for force feedback mode if enabled.
        if self.config.gripper_force_feedback:
            self._setup_leader_gripper_for_force_feedback()
            self._gripper_feedback_state.reset()
            logger.info("Gripper force feedback enabled (gain=%.2f)", self.config.gripper_force_feedback_gain)

    def enable_torque(self) -> None:
        self.bus.enable_torque()

    def disable_torque(self) -> None:
        self.bus.disable_torque()

    def setup_motors(self) -> None:
        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    @check_if_not_connected
    def get_action(self) -> dict[str, float]:
        start = time.perf_counter()
        action = self.bus.sync_read("Present_Position")
        action = {f"{motor}.pos": val for motor, val in action.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    @check_if_not_connected
    def send_feedback(self, feedback: dict[str, float]) -> None:
        goals = {k.removesuffix(".pos"): v for k, v in feedback.items() if k.endswith(".pos")}
        if not goals:
            return

        # --- Gripper force feedback ---
        if self.config.gripper_force_feedback:
            goals = self._apply_gripper_force_feedback(goals)

        self.bus.sync_write("Goal_Position", goals)

    def _setup_leader_gripper_for_force_feedback(self) -> None:
        """Configure the leader gripper motor for compliant force feedback.

        Sets PD coefficients so the gripper can be pushed open by the operator
        while the controller applies resisting torque.
        """
        gripper_motor = "gripper"
        self.bus.write("P_Coefficient", gripper_motor, 8)
        self.bus.write("I_Coefficient", gripper_motor, 0)
        self.bus.write("D_Coefficient", gripper_motor, 20)
        self.bus.disable_torque(motors=[gripper_motor])

    def _apply_gripper_force_feedback(self, goals: dict[str, float]) -> dict[str, float]:
        """Apply gripper force feedback and return (possibly modified) goals.

        Reads follower state via the robot interface, computes virtual-wall
        goal and torque-limit for the leader gripper, and dispatches bus
        writes only on state transitions.
        """
        try:
            # Read follower state — we need a reference to the follower robot.
            # The follower is stored as _follower on SOLeader when set up by
            # a pairing context (e.g. rollout strategies, lerobot-teleoperate).
            follower = getattr(self, "_follower", None)
            if follower is None:
                # No follower reference available; skip force feedback.
                return goals

            follower_bus = getattr(follower, "bus", None)
            if follower_bus is None:
                return goals

            try:
                follower_load = follower_bus.sync_read("Present_Load", ["gripper"])["gripper"]
                follower_pos = follower_bus.sync_read("Present_Position", ["gripper"])["gripper"]
            except Exception as e:
                logger.debug("Gripper force feedback read failed: %s", e)
                return goals

            leader_present = goals.get("gripper", 0.0)
            goal = apply_gripper_force_feedback(
                feedback_state=self._gripper_feedback_state,
                follower_load=follower_load,
                follower_pos=float(follower_pos),
                leader_present=leader_present,
                gain=self.config.gripper_force_feedback_gain,
                config=self.config,
                bus=self.bus,
            )
            goals["gripper"] = goal

        except Exception as e:  # noqa: BLE001
            logger.debug("Gripper force feedback error: %s", e)

        return goals

    @check_if_not_connected
    def disconnect(self) -> None:
        # Disable torque on the gripper motor before disconnecting.
        if self.config.gripper_force_feedback:
            with contextlib.suppress(Exception):
                self.bus.disable_torque(motors=["gripper"])
            self._gripper_feedback_state.reset()

        self.bus.disconnect()
        logger.info(f"{self} disconnected.")


SO100Leader = SOLeader
SO101Leader = SOLeader
