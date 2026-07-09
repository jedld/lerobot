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

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@dataclass
class SOLeaderConfig:
    """Base configuration class for SO Leader teleoperators."""

    # Port to connect to the arm
    port: str

    # Whether to use degrees for angles
    use_degrees: bool = True

    # Whether to enable gripper-only force feedback (virtual wall when leader gripper
    # tries to close past the follower's achieved grasp position).
    gripper_force_feedback: bool = False

    # Gain multiplier for the force feedback torque limit (0.0 = no resistance,
    # 1.0 = full torque limit). Defaults to 1.0 for maximum feel.
    gripper_force_feedback_gain: float = 1.0

    # Position margin (in gripper units 0–100) within which no resistance is applied.
    # Allows the leader gripper to approach the follower's position without sudden onset.
    gripper_force_feedback_position_margin: float = 3.0

    # Deadband for follower load reading (Feetech Present_Load is signed 16-bit,
    # centered at 0; values below this are considered "no object grasped").
    gripper_force_feedback_load_deadband: int = 50

    # Maximum overshoot beyond follower position that triggers full torque limit.
    gripper_force_feedback_max_overshoot: float = 30.0

    # Hysteresis for release detection to prevent chattering near the boundary.
    gripper_force_feedback_release_hysteresis: float = 2.0

    # Minimum torque limit (in Feetech torque-limit units, 0–1024).
    gripper_force_feedback_torque_limit_min: int = 0

    # Maximum torque limit (in Feetech torque-limit units, 0–1024).
    gripper_force_feedback_torque_limit_max: int = 1024


@TeleoperatorConfig.register_subclass("so101_leader")
@TeleoperatorConfig.register_subclass("so100_leader")
@dataclass
class SOLeaderTeleopConfig(TeleoperatorConfig, SOLeaderConfig):
    pass


SO100LeaderConfig = SOLeaderTeleopConfig
SO101LeaderConfig = SOLeaderTeleopConfig
