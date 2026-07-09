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

"""
Gripper-only force feedback for SO leader teleoperators.

When the follower grasps an object, the leader gripper resists squeezing past
the follower's achieved position (virtual wall).  Releasing -- leader moving
back toward the follower's position -- disables feedback immediately.

Torque writes are gated on state transitions only (Copilot suggestion), so
the serial bus isn't flooded every loop iteration.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GripperFeedbackState:
    """Tracks the active/inactive torque state to gate bus writes."""

    # Was torque enabled on the previous send_feedback() call?
    was_active: bool = False

    # Was the torque-limit value written last time we were active?
    last_torque_limit: int | None = None

    # Last leader gripper position (for release-direction detection).
    last_leader_pos: float | None = None

    def reset(self) -> None:
        """Clear all state (call during setup/teardown)."""
        self.was_active = False
        self.last_torque_limit = None
        self.last_leader_pos = None


def _compute_feedback_targets(
    follower_pos: float,
    leader_pos: float,
    follower_load: float,
    gain: float,
    prev_leader_pos: float | None,
    config,  # SOLeaderConfig
) -> tuple[float, int, bool]:
    """Compute the leader gripper goal position and torque limit for force feedback.

    Args:
        follower_pos: The follower gripper's current position (0–100 scale).
        leader_pos: The leader gripper's current position (0–100 scale).
        follower_load: The follower's load reading (from Present_Load).
        gain: User-configurable gain multiplier (0.0–1.0+, clamped internally).
        prev_leader_pos: Leader position on the previous iteration (or None).
        config: SOLeaderConfig with force feedback parameters.

    Returns:
        (goal_pos, torque_limit, active):
            - goal_pos: Goal position for the leader gripper (clamped 0–100).
            - torque_limit: Torque limit to write (or None if inactive).
            - active: True if torque should be enabled.
    """
    load = abs(follower_load)
    diff = leader_pos - follower_pos

    # --- Release detection: if leader is moving BACK toward follower,
    #     disable feedback immediately (no resistance while opening). ---
    if prev_leader_pos is not None:
        prev_diff = prev_leader_pos - follower_pos
        if abs(diff) < abs(prev_diff) - config.gripper_force_feedback_release_hysteresis:
            return leader_pos, config.gripper_force_feedback_torque_limit_min, False

    # --- No load on follower = nothing grasped = no feedback. ---
    if load < config.gripper_force_feedback_load_deadband:
        return leader_pos, config.gripper_force_feedback_torque_limit_min, False

    # --- Check if leader has overshot past the follower's position. ---
    overshoot = abs(diff)
    if overshoot <= config.gripper_force_feedback_position_margin:
        return leader_pos, config.gripper_force_feedback_torque_limit_min, False

    # --- Compute torque limit proportional to overshoot fraction. ---
    span = max(
        config.gripper_force_feedback_max_overshoot - config.gripper_force_feedback_position_margin,
        1.0,
    )
    overshoot_frac = min(1.0, (overshoot - config.gripper_force_feedback_position_margin) / span)

    # Virtual wall: hold leader at follower's achieved position.
    goal = min(100.0, max(0.0, follower_pos))

    torque_limit = int(
        config.gripper_force_feedback_torque_limit_min
        + overshoot_frac
        * gain
        * (config.gripper_force_feedback_torque_limit_max - config.gripper_force_feedback_torque_limit_min)
    )
    torque_limit = min(
        config.gripper_force_feedback_torque_limit_max,
        max(config.gripper_force_feedback_torque_limit_min, torque_limit),
    )

    return goal, torque_limit, True


def apply_gripper_force_feedback(
    feedback_state: GripperFeedbackState,
    follower_load: float,
    follower_pos: float,
    leader_present: float,
    gain: float,
    config,  # SOLeaderConfig
    bus,
) -> float:
    """Apply gripper force feedback, returning the (possibly modified) leader goal position.

    This function is called every teleoperation loop iteration.  It reads the
    follower's state and computes the leader goal, but **only issues torque
    bus writes on state transitions** (active↔inactive, torque-limit change).

    Args:
        feedback_state: Shared state tracker.
        follower_load: Current follower gripper load.
        follower_pos: Current follower gripper position (0–100).
        leader_present: Current leader gripper position (0–100).
        gain: Force feedback gain multiplier.
        config: SOLeaderConfig with force feedback parameters.
        bus: FeetechMotorsBus instance for torque-limit writes.

    Returns:
        The goal position to send to the leader gripper (may be the original
        position if feedback is inactive, or the virtual-wall position).
    """
    gripper_motor = "gripper"

    goal, torque_limit, active = _compute_feedback_targets(
        follower_pos,
        leader_present,
        follower_load,
        gain,
        feedback_state.last_leader_pos,
        config,
    )

    feedback_state.last_leader_pos = leader_present

    try:
        if not active:
            # Transition: active → inactive. Disable torque on the gripper motor.
            if feedback_state.was_active:
                logger.debug("Gripper force feedback: deactivating (release or no load)")
                bus.disable_torque(motors=[gripper_motor])
                feedback_state.was_active = False
                feedback_state.last_torque_limit = None
            return goal

        # We are active.
        if not feedback_state.was_active:
            # Transition: inactive → active. Enable torque, then set limit.
            logger.debug("Gripper force feedback: activating (follower grasped, leader overshot)")
            bus.enable_torque(motors=[gripper_motor])
            bus.write("Torque_Limit", gripper_motor, torque_limit)
            feedback_state.was_active = True
            feedback_state.last_torque_limit = torque_limit
            return goal

        # Already active: only write if torque limit changed.
        if feedback_state.last_torque_limit != torque_limit:
            bus.write("Torque_Limit", gripper_motor, torque_limit)
            feedback_state.last_torque_limit = torque_limit

    except Exception as e:  # noqa: BLE001
        logger.debug("Gripper force feedback bus write failed: %s", e)

    return goal
