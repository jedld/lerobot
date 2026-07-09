#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import logging
from functools import cached_property

from lerobot.types import RobotAction
from lerobot.utils.bimanual import BimanualMixin
from lerobot.utils.decorators import check_if_not_connected

from ..so_leader import SOLeader, SOLeaderTeleopConfig
from ..teleoperator import Teleoperator
from .config_bi_so_leader import BiSOLeaderConfig

logger = logging.getLogger(__name__)


class BiSOLeader(BimanualMixin, Teleoperator):
    """
    [Bimanual SO Leader Arms](https://github.com/TheRobotStudio/SO-ARM100) designed by TheRobotStudio
    """

    config_class = BiSOLeaderConfig
    name = "bi_so_leader"

    def __init__(self, config: BiSOLeaderConfig):
        super().__init__(config)
        self.config = config

        left_arm_config = SOLeaderTeleopConfig(
            id=f"{config.id}_left" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.left_arm_config.port,
            use_degrees=config.left_arm_config.use_degrees,
            gripper_force_feedback=config.left_arm_config.gripper_force_feedback,
            gripper_force_feedback_gain=config.left_arm_config.gripper_force_feedback_gain,
            gripper_force_feedback_position_margin=config.left_arm_config.gripper_force_feedback_position_margin,
            gripper_force_feedback_load_deadband=config.left_arm_config.gripper_force_feedback_load_deadband,
            gripper_force_feedback_max_overshoot=config.left_arm_config.gripper_force_feedback_max_overshoot,
            gripper_force_feedback_release_hysteresis=config.left_arm_config.gripper_force_feedback_release_hysteresis,
            gripper_force_feedback_torque_limit_min=config.left_arm_config.gripper_force_feedback_torque_limit_min,
            gripper_force_feedback_torque_limit_max=config.left_arm_config.gripper_force_feedback_torque_limit_max,
        )

        right_arm_config = SOLeaderTeleopConfig(
            id=f"{config.id}_right" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.right_arm_config.port,
            use_degrees=config.right_arm_config.use_degrees,
            gripper_force_feedback=config.right_arm_config.gripper_force_feedback,
            gripper_force_feedback_gain=config.right_arm_config.gripper_force_feedback_gain,
            gripper_force_feedback_position_margin=config.right_arm_config.gripper_force_feedback_position_margin,
            gripper_force_feedback_load_deadband=config.right_arm_config.gripper_force_feedback_load_deadband,
            gripper_force_feedback_max_overshoot=config.right_arm_config.gripper_force_feedback_max_overshoot,
            gripper_force_feedback_release_hysteresis=config.right_arm_config.gripper_force_feedback_release_hysteresis,
            gripper_force_feedback_torque_limit_min=config.right_arm_config.gripper_force_feedback_torque_limit_min,
            gripper_force_feedback_torque_limit_max=config.right_arm_config.gripper_force_feedback_torque_limit_max,
        )

        self.left_arm = SOLeader(left_arm_config)
        self.right_arm = SOLeader(right_arm_config)

    @cached_property
    def action_features(self) -> dict[str, type]:
        left_arm_features = self.left_arm.action_features
        right_arm_features = self.right_arm.action_features

        return {
            **{f"left_{k}": v for k, v in left_arm_features.items()},
            **{f"right_{k}": v for k, v in right_arm_features.items()},
        }

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

    def setup_motors(self) -> None:
        self.left_arm.setup_motors()
        self.right_arm.setup_motors()

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        action_dict = {}

        # Add "left_" prefix
        left_action = self.left_arm.get_action()
        action_dict.update({f"left_{key}": value for key, value in left_action.items()})

        # Add "right_" prefix
        right_action = self.right_arm.get_action()
        action_dict.update({f"right_{key}": value for key, value in right_action.items()})

        return action_dict

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # Namespace feedback keys back to per-arm format and delegate to each arm.
        left_feedback = {k.removeprefix("left_"): v for k, v in feedback.items() if k.startswith("left_")}
        right_feedback = {k.removeprefix("right_"): v for k, v in feedback.items() if k.startswith("right_")}
        self.left_arm.send_feedback(left_feedback)
        self.right_arm.send_feedback(right_feedback)
