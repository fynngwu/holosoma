"""Whole Body Tracking-specific termination terms."""

from __future__ import annotations

from typing import Any, List

from holosoma.config_types.termination import TerminationTermCfg
from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager
from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.managers.observation.terms.wbt import gravity_vector
from holosoma.managers.termination.base import TerminationTermBase
from holosoma.utils.rotations import (
    quat_error_magnitude,
    quat_rotate_inverse,
)
from holosoma.utils.safe_torch_import import torch


#########################################################################################################
## Termination terms
#########################################################################################################
def motion_ends(env, **_) -> torch.Tensor:
    """Terminate if the motion ends."""
    motion_command = env.command_manager.get_state("motion_command")
    return motion_command.time_steps >= motion_command.motion.time_step_total - 2


class BadTracking(TerminationTermBase):
    """Terminate if the tracking is bad.

    - bad ref pos
    - bad ref ori
    - bad motion body pos
    if has object:
        - bad object pos
        - bad object ori

    When bad tracking is detected, the motion_commmand.AdaptiveTimestepsSampler will be updated.
    """

    def __init__(self, cfg: TerminationTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)

        self.bad_ref_pos_threshold = cfg.params["bad_ref_pos_threshold"]
        self.bad_ref_ori_threshold = cfg.params["bad_ref_ori_threshold"]

        self.bad_motion_body_pos_body_names = cfg.params["bad_motion_body_pos_body_names"]

        # NOTE: body_names_to_track is shared with command_manager
        self.body_names_to_track = cfg.params["body_names_to_track"]
        self.bad_motion_body_pos_threshold = cfg.params["bad_motion_body_pos_threshold"]
        self.bad_motion_body_pos_body_indexes = self._get_index_of_a_in_b(
            self.bad_motion_body_pos_body_names, self.body_names_to_track, self.env.device
        )

        self.bad_object_pos_threshold = cfg.params["bad_object_pos_threshold"]
        self.bad_object_ori_threshold = cfg.params["bad_object_ori_threshold"]

    def __call__(self, env: Any, **kwargs) -> torch.Tensor:
        motion_command = self.env.command_manager.get_state("motion_command")
        assert motion_command.motion_cfg.body_names_to_track == self.body_names_to_track, (
            "body_names_to_track in motion_command and termination.params are not the same"
            f"motion_command.motion_cfg.body_names_to_track: {motion_command.motion_cfg.body_names_to_track}"
            f"termination.params['body_names_to_track']: {self.body_names_to_track}"
        )

        # return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        bad_ref_pos = self.bad_ref_pos(motion_command)
        bad_ref_ori = self.bad_ref_ori(motion_command)
        bad_motion_body_pos = self.bad_motion_body_pos(motion_command)
        bad_tracking = bad_ref_pos | bad_ref_ori | bad_motion_body_pos

        if motion_command.motion.has_object:
            bad_object_pos = self.bad_object_pos(motion_command)
            bad_object_ori = self.bad_object_ori(motion_command)
            bad_tracking |= bad_object_pos | bad_object_ori

        return bad_tracking

    def bad_ref_pos(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate if the reference position is too far from the robot's position."""
        return torch.norm(motion_command.ref_pos_w - motion_command.robot_ref_pos_w, dim=1) > self.bad_ref_pos_threshold

    def bad_ref_ori(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate if the reference orientation is too far from the robot's orientation."""
        motion_projected_gravity_b = quat_rotate_inverse(
            motion_command.ref_quat_w, gravity_vector(self.env), w_last=True
        )
        robot_projected_gravity_b = quat_rotate_inverse(
            motion_command.robot_ref_quat_w, gravity_vector(self.env), w_last=True
        )
        return (
            torch.abs(motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]) > self.bad_ref_ori_threshold
        )

    def bad_motion_body_pos(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate if the motion body position is too far from the robot's body position."""
        body_idx = self.bad_motion_body_pos_body_indexes
        error = torch.norm(
            motion_command.body_pos_relative_w[:, body_idx] - motion_command.robot_body_pos_w[:, body_idx], dim=-1
        )
        return torch.any(error > self.bad_motion_body_pos_threshold, dim=-1)

    def bad_object_pos(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate if the object position is too far from the simulator's object position."""
        return (
            torch.norm(motion_command.object_pos_w - motion_command.simulator_object_pos_w, dim=-1)
            > self.bad_object_pos_threshold
        )

    def bad_object_ori(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate if the object orientation is too far from the simulator's object orientation."""
        return (
            quat_error_magnitude(motion_command.object_quat_w, motion_command.simulator_object_quat_w)
            > self.bad_object_ori_threshold
        )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset internal state for specified environments."""

    #########################################################################################################
    ## Internal Helper functions
    #########################################################################################################
    def _get_index_of_a_in_b(self, a_names: List[str], b_names: List[str], device: str = "cpu") -> torch.Tensor:
        indexes = []
        for name in a_names:
            assert name in b_names, f"The specified name ({name}) doesn't exist: {b_names}"
            indexes.append(b_names.index(name))
        return torch.tensor(indexes, dtype=torch.long, device=device)


class BadTrackingZOnly(BadTracking):
    """BadTracking variant using z-axis-only position checks for parity with BM Wo-State-Estimation."""

    def bad_ref_pos(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate if the reference z position is too far from the robot's z position."""
        z_err = torch.abs(motion_command.ref_pos_w[:, -1] - motion_command.robot_ref_pos_w[:, -1])
        return z_err > self.bad_ref_pos_threshold

    def bad_motion_body_pos(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate if tracked bodies have too much z-axis position error."""
        body_idx = self.bad_motion_body_pos_body_indexes
        error = torch.abs(
            motion_command.body_pos_relative_w[:, body_idx, -1] - motion_command.robot_body_pos_w[:, body_idx, -1]
        )
        return torch.any(error > self.bad_motion_body_pos_threshold, dim=-1)


class BadTrackingSonicStyle(BadTracking):
    """SONIC-style adaptive termination — relaxes thresholds during low-pelvis sections.

    Goal: stop false-positive terminations on choreography that puts the
    reference pelvis on the floor (breakdance, kneeling, handstand-to-bridge,
    supine), which previously caused all such clips to terminate around the
    floor-touch frame regardless of actual tracking quality.

    Borrowed from gear_sonic's `exceeded_anchor_height` + `exceeded_body_height`
    + `exceeded_anchor_ori` + `exceeded_body_pos`. Key idea: gate threshold
    relaxation on the EMA of the reference's pelvis world-z
    (``motion_command.running_ref_root_height``); when that EMA drops below
    ``root_height_threshold``, the termination thresholds switch from default
    (tight) to ``down_*`` (loose) per-env. This lets the policy track through
    floor sections without termination while still catching real falls during
    upright dance.

    Replaces the brittle `bad_ref_ori` gravity-projection check (false-fires on
    any handstand) with a much looser quaternion-error magnitude squared check
    (``threshold ≈ 1.0 rad² ≈ 57°²``), and replaces the per-body world-frame
    position check with z-height-only on a small set of bodies (wrists+ankles).

    Required cfg.params (in addition to the base BadTracking params):
        bad_ref_pos_threshold (float): default anchor z-error tolerance, m. Defaults to 0.5.
        bad_ref_pos_threshold_low (float): looser tolerance when ref pelvis is low. Defaults to 0.75.
        bad_motion_body_pos_threshold (float): default per-body z-error tolerance, m. Defaults to 0.5.
        bad_motion_body_pos_threshold_low (float): looser tolerance when ref pelvis is low. Defaults to 0.75.
        bad_ref_ori_threshold (float): squared quat-error threshold, rad². Defaults to 1.0.
        ref_root_height_threshold (float): pelvis-z below this triggers low-pose mode. Defaults to 0.5.
        bad_motion_body_pos_body_names: bodies to z-check (wrists + ankles only).

    Note: ``bad_motion_body_pos_threshold`` semantics differ from BadTrackingZOnly:
    here it's a default that can adapt per-env, not a fixed value.
    """

    def __init__(self, cfg: TerminationTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self.bad_ref_pos_threshold_low = cfg.params.get("bad_ref_pos_threshold_low", 0.75)
        self.bad_motion_body_pos_threshold_low = cfg.params.get("bad_motion_body_pos_threshold_low", 0.75)
        self.ref_root_height_threshold = cfg.params.get("ref_root_height_threshold", 0.5)

    def bad_ref_pos(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate iff |ref_z - robot_z| > adaptive threshold.

        Threshold defaults to ``bad_ref_pos_threshold`` (e.g. 0.5m) and relaxes
        to ``bad_ref_pos_threshold_low`` (e.g. 0.75m) per-env when the
        running EMA of reference pelvis z is below ``ref_root_height_threshold``.
        """
        z_err = torch.abs(motion_command.ref_pos_w[:, -1] - motion_command.robot_ref_pos_w[:, -1])
        running_ref_z = getattr(motion_command, "running_ref_root_height", None)
        if running_ref_z is None:
            return z_err > self.bad_ref_pos_threshold
        thresh = torch.full_like(z_err, self.bad_ref_pos_threshold)
        thresh[running_ref_z < self.ref_root_height_threshold] = self.bad_ref_pos_threshold_low
        return z_err > thresh

    def bad_ref_ori(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate iff squared quaternion-error magnitude > threshold.

        Replaces gravity-projection difference (which false-fires on any
        ref inversion) with a symmetric, inversion-tolerant quat-error² check.
        Default threshold 1.0 rad² ≈ 57°² — only fires on truly divergent
        orientation, not on cartwheels / handstands.
        """
        angular_err = quat_error_magnitude(motion_command.ref_quat_w, motion_command.robot_ref_quat_w)
        return angular_err.square() > self.bad_ref_ori_threshold

    def bad_motion_body_pos(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate iff any tracked body's z-error exceeds adaptive threshold.

        Per-body z-height only (not full xyz). Adaptive threshold same as
        bad_ref_pos: relaxed when running ref pelvis z is low.
        """
        body_idx = self.bad_motion_body_pos_body_indexes
        z_err = torch.abs(
            motion_command.body_pos_relative_w[:, body_idx, -1] - motion_command.robot_body_pos_w[:, body_idx, -1]
        )
        running_ref_z = getattr(motion_command, "running_ref_root_height", None)
        if running_ref_z is None:
            return torch.any(z_err > self.bad_motion_body_pos_threshold, dim=-1)
        thresh = torch.full_like(z_err, self.bad_motion_body_pos_threshold)
        # Per-env: when ref is low, relax all body thresholds for that env.
        low_mask = (running_ref_z < self.ref_root_height_threshold).unsqueeze(1)
        thresh = torch.where(
            low_mask.expand_as(thresh),
            torch.full_like(thresh, self.bad_motion_body_pos_threshold_low),
            thresh,
        )
        return torch.any(z_err > thresh, dim=-1)
