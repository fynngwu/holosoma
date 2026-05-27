from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, List

import numpy as np
import torch
from loguru import logger

from holosoma.config_types.command import MotionConfig, NoiseToInitialPoseConfig
from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager
from holosoma.managers.command.base import CommandTermBase
from holosoma.utils.file_cache import cached_open
from holosoma.utils.path import resolve_data_file_path
from holosoma.utils.rotations import (
    get_euler_xyz,
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inverse,
    quat_mul,
    slerp,
    yaw_quat,
)
from holosoma.utils.simulator_config import SimulatorType


#########################################################################################################
## MotionLoader and AdaptiveTimestepsSampler
#########################################################################################################
class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        robot_body_names: list[str],
        robot_joint_names: list[str],
        device: str = "cpu",
        recompute_velocities_from_positions: bool = False,
    ):
        # Resolve the motion file path using importlib.resources
        motion_file = resolve_data_file_path(motion_file)

        logger.info(f"Loading motion file: {motion_file}")
        self._recompute_velocities = recompute_velocities_from_positions
        body_names_in_motion_data, joint_names_in_motion_data = self._load_data_from_motion_npz(motion_file, device)
        body_indexes = self._get_index_of_a_in_b(robot_body_names, body_names_in_motion_data, device)
        joint_indexes = self._get_index_of_a_in_b(robot_joint_names, joint_names_in_motion_data, device)

        self._joint_indexes = joint_indexes
        self._body_indexes = body_indexes
        self.time_step_total = self._joint_pos.shape[0]

    def _get_index_of_a_in_b(self, a_names: List[str], b_names: List[str], device: str = "cpu") -> torch.Tensor:
        indexes = []
        for name in a_names:
            assert name in b_names, f"The specified name ({name}) doesn't exist: {b_names}"
            indexes.append(b_names.index(name))
        return torch.tensor(indexes, dtype=torch.long, device=device)

    # Expected holosoma NPZ keys
    _REQUIRED_KEYS = {
        "fps",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
        "body_names",
        "joint_names",
    }

    def _load_data_from_motion_npz(self, motion_file: str, device: str) -> tuple[list[str], list[str]]:
        with cached_open(motion_file, "rb") as f, np.load(f) as data:
            # Sanity check: warn if not in expected holosoma format
            keys = set(data.files)
            missing = self._REQUIRED_KEYS - keys
            if missing:
                logger.warning(
                    f"Motion NPZ '{motion_file}' is missing expected holosoma keys: {missing}. "
                    f"All motion data should be in holosoma format (with body_names, joint_names, "
                    f"and root DOFs in joint_pos). Convert from TML/BeyondMimic first."
                )
                raise ValueError(
                    f"Unsupported motion format in '{motion_file}': missing keys {missing}. "
                    f"Please convert to holosoma format."
                )

            self.fps = data["fps"]

            body_names = data["body_names"].tolist()
            joint_names = data["joint_names"].tolist()

            joint_pos_raw = data["joint_pos"]
            joint_vel_raw = data["joint_vel"]
            body_pos_w_raw = data["body_pos_w"]
            body_quat_w_raw = data["body_quat_w"]
            body_lin_vel_w_raw = data["body_lin_vel_w"]
            body_ang_vel_w_raw = data["body_ang_vel_w"]

            # Holosoma format: joint_pos includes root DOFs [xyz, wxyz] as first 7 values
            # joint_vel includes root velocity [vel_xyz, vel_wxyz] as first 6 values
            num_joint_cols = joint_pos_raw.shape[1]
            num_vel_cols = joint_vel_raw.shape[1]
            num_bodies = body_pos_w_raw.shape[1]

            if num_joint_cols != len(joint_names) + 7:
                logger.warning(
                    f"Unexpected joint_pos columns: got {num_joint_cols}, expected {len(joint_names) + 7} "
                    f"(= {len(joint_names)} joints + 7 root DOFs). File: {motion_file}"
                )
            if num_vel_cols != len(joint_names) + 6:
                logger.warning(
                    f"Unexpected joint_vel columns: got {num_vel_cols}, expected {len(joint_names) + 6} "
                    f"(= {len(joint_names)} joints + 6 root DOFs). File: {motion_file}"
                )
            if num_bodies != len(body_names):
                logger.warning(
                    f"Body count mismatch: body_pos_w has {num_bodies} bodies but body_names has "
                    f"{len(body_names)}. File: {motion_file}"
                )

            # Strip root DOFs
            self._joint_pos = torch.tensor(joint_pos_raw[:, 7:], dtype=torch.float32, device=device)
            self._joint_vel = torch.tensor(joint_vel_raw[:, 6:], dtype=torch.float32, device=device)

            assert len(joint_names) == self._joint_pos.shape[1], (
                f"Joint names ({len(joint_names)}) != joint_pos columns ({self._joint_pos.shape[1]}) in {motion_file}"
            )
            assert len(body_names) == body_pos_w_raw.shape[1], (
                f"Body names ({len(body_names)}) != body_pos_w bodies ({body_pos_w_raw.shape[1]}) in {motion_file}"
            )

            self._body_pos_w = torch.tensor(body_pos_w_raw, dtype=torch.float32, device=device)

            # NOTE: wxyz after loading from npz
            body_quat_w_wxyz = torch.tensor(body_quat_w_raw, dtype=torch.float32, device=device)  # This is wxyz
            self._body_quat_w = body_quat_w_wxyz[:, :, [1, 2, 3, 0]]  # Change to xyzw

            self._body_lin_vel_w = torch.tensor(body_lin_vel_w_raw, dtype=torch.float32, device=device)
            self._body_ang_vel_w = torch.tensor(body_ang_vel_w_raw, dtype=torch.float32, device=device)

            if getattr(self, "_recompute_velocities", False):
                self._recompute_velocity_fields(motion_file)

            # Load motion_ends / motion_idxs for multi-clip files (from combine_motions).
            # Ref consumes these in step() to resample envs at each source-clip boundary;
            # without them, the concatenated motion is treated as one giant clip and
            # the target pose jumps discontinuously across source-clip borders,
            # leaving the robot unable to track and terminating early.
            num_frames = self._joint_pos.shape[0]
            if "motion_ends" in data.files:
                self.motion_ends = torch.tensor(data["motion_ends"], dtype=torch.bool, device=device)
            else:
                self.motion_ends = torch.zeros(num_frames, dtype=torch.bool, device=device)
                self.motion_ends[-1] = True
            if "motion_idxs" in data.files:
                self.motion_idxs = torch.tensor(data["motion_idxs"], dtype=torch.long, device=device)
            else:
                self.motion_idxs = torch.zeros(num_frames, dtype=torch.long, device=device)

            # add object pos and quat
            self.has_object = "object_pos_w" in data
            if self.has_object:
                self._object_pos_w = torch.tensor(data["object_pos_w"], dtype=torch.float32, device=device)
                # NOTE: wxyz after loading from npz
                object_quat_w = torch.tensor(data["object_quat_w"], dtype=torch.float32, device=device)
                self._object_quat_w = object_quat_w[:, [1, 2, 3, 0]]  # Change to xyzw
                self._object_lin_vel_w = torch.tensor(data["object_lin_vel_w"], dtype=torch.float32, device=device)
            else:
                self._object_pos_w = torch.zeros(0, 3, device=device)
                self._object_quat_w = torch.zeros(0, 4, device=device)
                self._object_lin_vel_w = torch.zeros(0, 3, device=device)
        return body_names, joint_names

    @property
    def joint_pos(self) -> torch.Tensor:
        return self._joint_pos[:, self._joint_indexes]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._joint_vel[:, self._joint_indexes]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]

    # Time-step-first accessors: index time_steps on raw data BEFORE body_indexes
    # to avoid creating full-dataset intermediates (600x memory reduction for 5K motions)
    def get_joint_pos(self, time_steps: torch.Tensor) -> torch.Tensor:
        return self._joint_pos[time_steps][:, self._joint_indexes]

    def get_joint_vel(self, time_steps: torch.Tensor) -> torch.Tensor:
        return self._joint_vel[time_steps][:, self._joint_indexes]

    def get_body_pos_w(self, time_steps: torch.Tensor) -> torch.Tensor:
        return self._body_pos_w[time_steps][:, self._body_indexes]

    def get_body_quat_w(self, time_steps: torch.Tensor) -> torch.Tensor:
        return self._body_quat_w[time_steps][:, self._body_indexes]

    def get_body_lin_vel_w(self, time_steps: torch.Tensor) -> torch.Tensor:
        return self._body_lin_vel_w[time_steps][:, self._body_indexes]

    def get_body_ang_vel_w(self, time_steps: torch.Tensor) -> torch.Tensor:
        return self._body_ang_vel_w[time_steps][:, self._body_indexes]

    def get_object_pos_w(self, time_steps: torch.Tensor) -> torch.Tensor:
        return self._object_pos_w[time_steps]

    def get_object_quat_w(self, time_steps: torch.Tensor) -> torch.Tensor:
        return self._object_quat_w[time_steps]

    def get_object_lin_vel_w(self, time_steps: torch.Tensor) -> torch.Tensor:
        return self._object_lin_vel_w[time_steps]

    @property
    def object_pos_w(self) -> torch.Tensor:
        return self._object_pos_w[:]

    @property
    def object_quat_w(self) -> torch.Tensor:
        return self._object_quat_w[:]

    @property
    def object_lin_vel_w(self) -> torch.Tensor:
        return self._object_lin_vel_w[:]

    @property
    def num_motions(self) -> int:
        return 1

    @property
    def motion_filenames(self) -> List[str]:
        return ["<single_motion>"]

    def _recompute_velocity_fields(self, motion_file: str) -> None:
        """Replace stored velocity arrays with finite-difference estimates from positions.

        Defensive sanitization: when the stored velocity field disagrees with the
        position trajectory (e.g. retargeted clips where joint_vel was computed
        in a different frame than joint_pos, leaving spikes mid-clip), the
        position trajectory is the source of truth. This recomputes
        joint_vel / body_lin_vel_w / body_ang_vel_w from joint_pos / body_pos_w /
        body_quat_w via central-difference. Endpoints use forward/backward diff.

        Note: this does NOT smooth the positions themselves. If the position
        trajectory contains real discontinuities (e.g. IK ambiguity flips
        producing 50+ rad/s implied velocity), the recomputed velocities will
        still reflect them. This is the desired behavior — preserve the
        trajectory; let the sampler / termination handle hard clips.
        """
        from holosoma.utils.rotations import quat_conjugate, quat_mul, quat_to_angle_axis

        fps = float(self.fps)

        # Joint vel: central diff of joint_pos (already root-stripped, shape (T, J))
        jp = self._joint_pos
        jv_recomputed = torch.zeros_like(self._joint_vel)
        jv_recomputed[1:-1] = (jp[2:] - jp[:-2]) * (fps / 2.0)
        jv_recomputed[0] = (jp[1] - jp[0]) * fps
        jv_recomputed[-1] = (jp[-1] - jp[-2]) * fps

        # Body lin vel: central diff of body_pos_w (T, B, 3)
        bp = self._body_pos_w
        blv_recomputed = torch.zeros_like(self._body_lin_vel_w)
        blv_recomputed[1:-1] = (bp[2:] - bp[:-2]) * (fps / 2.0)
        blv_recomputed[0] = (bp[1] - bp[0]) * fps
        blv_recomputed[-1] = (bp[-1] - bp[-2]) * fps

        # Body ang vel from relative quaternion / dt
        bq = self._body_quat_w  # xyzw, (T, B, 4)
        T = bq.shape[0]
        bav_recomputed = torch.zeros_like(self._body_ang_vel_w)
        if T >= 2:
            q_next = bq[2:]
            q_prev_conj = quat_conjugate(bq[:-2], w_last=True)
            q_rel_central = quat_mul(q_next, q_prev_conj, w_last=True)
            angle_c, axis_c = quat_to_angle_axis(q_rel_central)
            bav_recomputed[1:-1] = axis_c * (angle_c.unsqueeze(-1) * (fps / 2.0))
            q_rel_fwd = quat_mul(bq[1], quat_conjugate(bq[0], w_last=True), w_last=True)
            angle_f, axis_f = quat_to_angle_axis(q_rel_fwd)
            bav_recomputed[0] = axis_f * (angle_f.unsqueeze(-1) * fps)
            q_rel_bwd = quat_mul(bq[-1], quat_conjugate(bq[-2], w_last=True), w_last=True)
            angle_b, axis_b = quat_to_angle_axis(q_rel_bwd)
            bav_recomputed[-1] = axis_b * (angle_b.unsqueeze(-1) * fps)

        old_jv_max = float(self._joint_vel.abs().max())
        new_jv_max = float(jv_recomputed.abs().max())
        if old_jv_max > new_jv_max * 2 + 5:
            logger.info(
                f"[recompute_velocities] {motion_file}: stored joint_vel disagreed with "
                f"joint_pos delta; max|joint_vel| {old_jv_max:.2f} -> {new_jv_max:.2f} rad/s"
            )

        self._joint_vel = jv_recomputed
        self._body_lin_vel_w = blv_recomputed
        self._body_ang_vel_w = bav_recomputed

    @property
    def motion_start_idx(self) -> torch.Tensor:
        return torch.tensor([0], dtype=torch.long, device=self._joint_pos.device)

    @property
    def motion_end_idx(self) -> torch.Tensor:
        return torch.tensor([self.time_step_total], dtype=torch.long, device=self._joint_pos.device)

    def extend_with_segments(self, segments: dict[str, torch.Tensor], prepend: bool) -> MotionLoader:
        """Merge interpolated segments with motion data, mutating this MotionLoader."""
        concat_targets = [
            ("joint_pos", "_joint_pos"),
            ("joint_vel", "_joint_vel"),
            ("body_pos", "_body_pos_w"),
            ("body_quat", "_body_quat_w"),
            ("body_lin_vel", "_body_lin_vel_w"),
            ("body_ang_vel", "_body_ang_vel_w"),
        ]
        if self.has_object:
            concat_targets.extend(
                [
                    ("object_pos", "_object_pos_w"),
                    ("object_quat", "_object_quat_w"),
                    ("object_lin_vel", "_object_lin_vel_w"),
                ]
            )

        for seg_key, attr_name in concat_targets:
            existing = getattr(self, attr_name)
            tensors = (segments[seg_key], existing) if prepend else (existing, segments[seg_key])
            setattr(self, attr_name, torch.cat(tensors, dim=0))

        self.time_step_total = self._joint_pos.shape[0]
        return self


class MultiMotionLoader:
    """Loads multiple NPZ motion files from a directory and concatenates them at runtime.

    Tracks per-motion boundaries so environments can sample within individual clips.
    Compatible with the same interface as MotionLoader.
    """

    def __init__(
        self,
        motion_dir: str,
        robot_body_names: list[str],
        robot_joint_names: list[str],
        device: str = "cpu",
        shard_rank: int = 0,
        shard_world_size: int = 1,
        recompute_velocities_from_positions: bool = False,
        exclude_filename_substrings: tuple = (),
    ):
        # Support comma-separated directories for combining multiple datasets
        dirs = [d.strip() for d in motion_dir.split(",")]
        motion_files = []
        for d in dirs:
            expanded = os.path.expanduser(d)
            files = sorted(str(p) for p in Path(expanded).glob("*.npz"))
            logger.info(f"MultiMotionLoader: found {len(files)} .npz files in {expanded}")
            motion_files.extend(files)
        assert len(motion_files) > 0, f"No .npz files found in {motion_dir}"

        # Apply substring blacklist (e.g. exclude known-pathological clips like
        # fightAndSports1_subject4 that dominate the failure-weighted sampler).
        if exclude_filename_substrings:
            patterns = tuple(s for s in exclude_filename_substrings if s)
            n_before = len(motion_files)
            motion_files = [f for f in motion_files if not any(pat in Path(f).stem for pat in patterns)]
            n_excluded = n_before - len(motion_files)
            # Always log the parsed exclusion list so silent no-ops (e.g. tyro
            # parsing the literal Python tuple syntax `("foo",)` as a single
            # string element rather than a real tuple) become visible at startup.
            logger.info(
                f"MultiMotionLoader: motion_exclude_filename_substrings={patterns}, "
                f"matched and excluded {n_excluded} of {n_before} files"
            )
            if patterns and n_excluded == 0:
                logger.warning(
                    f"MultiMotionLoader: exclusion patterns {patterns} matched 0 files. "
                    "Check that you used positional syntax (e.g. "
                    "`--motion-exclude-filename-substrings fightAndSports1_subject4`) "
                    "and not Python-literal syntax `('fightAndSports1_subject4',)` — "
                    "tyro parses the latter as a single literal string."
                )

        # Shard motion files across GPUs when shard_world_size > 1
        if shard_world_size > 1:
            total_files = len(motion_files)
            motion_files = [f for i, f in enumerate(motion_files) if i % shard_world_size == shard_rank]
            logger.info(
                f"MultiMotionLoader: shard {shard_rank}/{shard_world_size}, "
                f"loading {len(motion_files)}/{total_files} motion files"
            )
        else:
            logger.info(f"MultiMotionLoader: loading {len(motion_files)} total motion files")

        loaders = []
        loader_basenames: list[str] = []
        skipped = 0
        for mf in motion_files:
            try:
                loader = MotionLoader(
                    mf,
                    robot_body_names,
                    robot_joint_names,
                    device=device,
                    recompute_velocities_from_positions=recompute_velocities_from_positions,
                )
            except (KeyError, AssertionError, ValueError) as e:
                # Skip files with incompatible format (e.g., missing body_names, wrong body count)
                skipped += 1
                if skipped <= 3:
                    logger.warning(f"MultiMotionLoader: skipping {mf}: {e}")
                continue
            loaders.append(loader)
            loader_basenames.append(Path(mf).stem)
        if skipped > 3:
            logger.warning(f"MultiMotionLoader: skipped {skipped} files total due to format issues")
        assert len(loaders) > 0, f"No compatible motion files found (skipped {skipped})"

        # Track per-motion boundaries
        lengths = [loader.time_step_total for loader in loaders]
        cumulative = torch.tensor(lengths, dtype=torch.long, device=device).cumsum(dim=0)
        self._motion_start_idx = torch.cat([torch.tensor([0], dtype=torch.long, device=device), cumulative[:-1]])
        self._motion_end_idx = cumulative
        self._num_motions = len(loaders)

        # Per-motion-id basename list — built in the same loop as loaders so the
        # i-th name maps to the i-th loader / motion_id. This is the diagnostic the
        # failure-weighted sampler uses to identify which clip is being concentrated on.
        # NOTE: motion_files is the per-rank shard, so motion_id i in this loader maps
        # to motion_files[i] for THIS rank's sampler. Different ranks can collapse onto
        # different clips when shard_motions=True.
        self._motion_filenames = loader_basenames
        assert len(self._motion_filenames) == self._num_motions, (
            f"motion_filenames ({len(self._motion_filenames)}) != num_motions ({self._num_motions}) — "
            "internal accounting bug in MultiMotionLoader"
        )

        # Concatenate all motion data
        self._joint_pos = torch.cat([ld._joint_pos for ld in loaders], dim=0)
        self._joint_vel = torch.cat([ld._joint_vel for ld in loaders], dim=0)
        self._body_pos_w = torch.cat([ld._body_pos_w for ld in loaders], dim=0)
        self._body_quat_w = torch.cat([ld._body_quat_w for ld in loaders], dim=0)
        self._body_lin_vel_w = torch.cat([ld._body_lin_vel_w for ld in loaders], dim=0)
        self._body_ang_vel_w = torch.cat([ld._body_ang_vel_w for ld in loaders], dim=0)

        # Per-frame clip-end markers and per-frame source-clip index, matching
        # the single-file layout used by ref. Consumed by step() for clip-
        # boundary resampling (resample_on_motion_end=True).
        self.motion_ends = torch.cat([ld.motion_ends for ld in loaders], dim=0)
        self.motion_idxs = torch.cat(
            [torch.full((ld.time_step_total,), i, dtype=torch.long, device=device) for i, ld in enumerate(loaders)],
            dim=0,
        )

        # Use indexes from first loader (all loaders share the same robot)
        self._joint_indexes = loaders[0]._joint_indexes
        self._body_indexes = loaders[0]._body_indexes
        self.fps = loaders[0].fps
        self.time_step_total = self._joint_pos.shape[0]

        # Object support: only if ALL motions have objects
        self.has_object = all(ld.has_object for ld in loaders)
        if self.has_object:
            self._object_pos_w = torch.cat([ld._object_pos_w for ld in loaders], dim=0)
            self._object_quat_w = torch.cat([ld._object_quat_w for ld in loaders], dim=0)
            self._object_lin_vel_w = torch.cat([ld._object_lin_vel_w for ld in loaders], dim=0)
        else:
            self._object_pos_w = torch.zeros(0, 3, device=device)
            self._object_quat_w = torch.zeros(0, 4, device=device)
            self._object_lin_vel_w = torch.zeros(0, 3, device=device)

        logger.info(f"MultiMotionLoader: {self._num_motions} motions, {self.time_step_total} total frames")

    @property
    def num_motions(self) -> int:
        return self._num_motions

    @property
    def motion_filenames(self) -> List[str]:
        """Per-motion-id filename stems (without extension), aligned with sampler IDs.

        Used by adaptive samplers (Layer-1 / Two-layer) for top-K diagnostic logging:
        identify the actual clip the failure-weighted sampler is concentrating on.
        """
        return self._motion_filenames

    @property
    def motion_start_idx(self) -> torch.Tensor:
        return self._motion_start_idx

    @property
    def motion_end_idx(self) -> torch.Tensor:
        return self._motion_end_idx

    @property
    def joint_pos(self) -> torch.Tensor:
        return self._joint_pos[:, self._joint_indexes]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._joint_vel[:, self._joint_indexes]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]

    # Time-step-first accessors: index time_steps on raw data BEFORE body_indexes
    # to avoid creating full-dataset intermediates (600x memory reduction for 5K motions)
    def get_joint_pos(self, time_steps: torch.Tensor) -> torch.Tensor:
        return self._joint_pos[time_steps][:, self._joint_indexes]

    def get_joint_vel(self, time_steps: torch.Tensor) -> torch.Tensor:
        return self._joint_vel[time_steps][:, self._joint_indexes]

    def get_body_pos_w(self, time_steps: torch.Tensor) -> torch.Tensor:
        return self._body_pos_w[time_steps][:, self._body_indexes]

    def get_body_quat_w(self, time_steps: torch.Tensor) -> torch.Tensor:
        return self._body_quat_w[time_steps][:, self._body_indexes]

    def get_body_lin_vel_w(self, time_steps: torch.Tensor) -> torch.Tensor:
        return self._body_lin_vel_w[time_steps][:, self._body_indexes]

    def get_body_ang_vel_w(self, time_steps: torch.Tensor) -> torch.Tensor:
        return self._body_ang_vel_w[time_steps][:, self._body_indexes]

    def get_object_pos_w(self, time_steps: torch.Tensor) -> torch.Tensor:
        return self._object_pos_w[time_steps]

    def get_object_quat_w(self, time_steps: torch.Tensor) -> torch.Tensor:
        return self._object_quat_w[time_steps]

    def get_object_lin_vel_w(self, time_steps: torch.Tensor) -> torch.Tensor:
        return self._object_lin_vel_w[time_steps]

    @property
    def object_pos_w(self) -> torch.Tensor:
        return self._object_pos_w[:]

    @property
    def object_quat_w(self) -> torch.Tensor:
        return self._object_quat_w[:]

    @property
    def object_lin_vel_w(self) -> torch.Tensor:
        return self._object_lin_vel_w[:]

    def extend_with_segments(self, segments: dict[str, torch.Tensor], prepend: bool) -> MultiMotionLoader:
        """Merge interpolated segments with motion data, mutating this MultiMotionLoader."""
        concat_targets = [
            ("joint_pos", "_joint_pos"),
            ("joint_vel", "_joint_vel"),
            ("body_pos", "_body_pos_w"),
            ("body_quat", "_body_quat_w"),
            ("body_lin_vel", "_body_lin_vel_w"),
            ("body_ang_vel", "_body_ang_vel_w"),
        ]
        if self.has_object:
            concat_targets.extend(
                [
                    ("object_pos", "_object_pos_w"),
                    ("object_quat", "_object_quat_w"),
                    ("object_lin_vel", "_object_lin_vel_w"),
                ]
            )

        added_frames = 0
        for seg_key, attr_name in concat_targets:
            existing = getattr(self, attr_name)
            tensors = (segments[seg_key], existing) if prepend else (existing, segments[seg_key])
            setattr(self, attr_name, torch.cat(tensors, dim=0))
            if added_frames == 0:
                added_frames = segments[seg_key].shape[0]

        # Update boundaries — shift all motion boundaries if prepending
        if prepend:
            self._motion_start_idx = self._motion_start_idx + added_frames
            self._motion_end_idx = self._motion_end_idx + added_frames
            dev = self._motion_start_idx.device
            self._motion_start_idx = torch.cat(
                [torch.tensor([0], dtype=torch.long, device=dev), self._motion_start_idx]
            )
            self._motion_end_idx = torch.cat(
                [torch.tensor([added_frames], dtype=torch.long, device=dev), self._motion_end_idx]
            )
        else:
            old_total = self.time_step_total
            dev = self._motion_start_idx.device
            self._motion_start_idx = torch.cat(
                [self._motion_start_idx, torch.tensor([old_total], dtype=torch.long, device=dev)]
            )
            self._motion_end_idx = torch.cat(
                [self._motion_end_idx, torch.tensor([old_total + added_frames], dtype=torch.long, device=dev)]
            )

        self.time_step_total = self._joint_pos.shape[0]
        self._num_motions = len(self._motion_start_idx)
        return self


class AdaptiveTimestepsSampler:
    """Prioritizes training on motion segments where the robot fails most often."""

    def __init__(
        self,
        motion_time_step_total: int,
        device: str,
        env_fps: int,
        adaptive_kernel_size: int = 1,
        adaptive_lambda: float = 0.8,
        adaptive_uniform_ratio: float = 0.1,
        adaptive_alpha: float = 0.001,
    ):
        self.device = device
        # length of the motion in rl environment time steps
        self.motion_time_step_total = motion_time_step_total
        # fps of the rl environment
        self.env_fps = env_fps

        self.adaptive_kernel_size = adaptive_kernel_size
        self.adaptive_lambda = adaptive_lambda
        self.adaptive_uniform_ratio = adaptive_uniform_ratio
        self.adaptive_alpha = adaptive_alpha

        # Match BeyondMimic binning: ~1 second bins at env FPS, with +1 tail bin.
        self.num_bins = int(self.motion_time_step_total // max(self.env_fps, 1)) + 1

        # Match BeyondMimic non-causal kernel.
        self.kernel = torch.tensor(
            [self.adaptive_lambda**i for i in range(self.adaptive_kernel_size)],
            device=self.device,
        )
        self.kernel = self.kernel / self.kernel.sum()

        # key data: failure counts
        self.init_buffers()
        # metrics
        self.metrics: dict[str, torch.Tensor] = {}

    def init_buffers(self):
        self.current_bin_failed_count = torch.zeros(self.num_bins, dtype=torch.float, device=self.device)
        self.bin_failed_count = torch.zeros(self.num_bins, dtype=torch.float, device=self.device)

    def update_current_bin_failed_count(self, failed_at_time_step: torch.Tensor):
        """Update the current bin failed count with terminated time steps."""
        failed_bin = torch.clamp(
            (failed_at_time_step * self.num_bins) // max(self.motion_time_step_total, 1),
            0,
            self.num_bins - 1,
        ).long()
        assert failed_bin.min() >= 0 and failed_bin.max() < self.num_bins, "Failed bin is out of range"
        self.current_bin_failed_count[:] = torch.bincount(failed_bin, minlength=self.num_bins)

    def update_bin_failed_count(self):
        """At every rl environment step, update the failed count with the current bin failed count."""
        self.bin_failed_count = (self.adaptive_alpha * self.current_bin_failed_count) + (
            1 - self.adaptive_alpha
        ) * self.bin_failed_count
        self.current_bin_failed_count.zero_()

    @property
    def sampling_probabilities(self) -> torch.Tensor:
        sampling_probabilities = self.bin_failed_count + 1e-6
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.adaptive_kernel_size - 1),  # Non-causal kernel
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(sampling_probabilities, self.kernel.view(1, 1, -1)).view(-1)
        sampling_probabilities += 0.01  # Uniform floor post-convolution
        return sampling_probabilities / sampling_probabilities.sum()

    def sample(self, num_samples: int) -> torch.Tensor:
        sampled_bins = torch.multinomial(self.sampling_probabilities, num_samples, replacement=True)
        # inside of each bin, randomly sample a time step, ignoring the borders
        return (sampled_bins + torch.rand(num_samples, device=self.device)) / self.num_bins

    def get_stats(self):
        # Metrics
        prob = self.sampling_probabilities
        H = -(prob * (prob + 1e-12).log()).sum()
        H_norm = H / np.log(self.num_bins)
        pmax, imax = prob.max(dim=0)
        self.metrics["sampling_entropy"] = H_norm
        self.metrics["sampling_top1_prob"] = pmax
        self.metrics["sampling_top1_bin"] = imax.float() / self.num_bins


class TwoLayerAdaptiveSampler:
    """Per-motion adaptive sampler: failure rates tracked per (motion_id, bin_within_motion).

    Layer 1: choose motion_id by per-clip failure EMA. Length-imbalanced clips no longer dilute
    each other (a 6574-frame dance and a 358-frame flip both contribute one EMA scalar).

    Layer 2: choose phase ∈ [0,1) conditioned on the chosen motion_id, using per-motion bins.
    The phase is then applied as start_idx + phase * (motion_len-1) — and now both training and
    application share the per-clip semantic, no more global-vs-per-clip mismatch.

    Storage: dense (num_motions, max_K) tensor. Bins beyond a motion's actual K_i are masked.

    NOTE: motion_start_idx / motion_end_idx are passed in; this sampler does not own the loader.
    """

    def __init__(
        self,
        motion_start_idx: torch.Tensor,
        motion_end_idx: torch.Tensor,
        device: str,
        env_fps: int,
        adaptive_lambda: float = 0.8,
        adaptive_uniform_ratio: float = 0.1,
        adaptive_alpha: float = 0.001,
        motion_filenames: list[str] | None = None,
        top1_prob_cap: float = 0.0,
        sonic_style: bool = False,
        sonic_failure_rate_max_over_mean: float = 200.0,
    ):
        self.device = device
        self.env_fps = max(env_fps, 1)
        self.adaptive_alpha = adaptive_alpha
        self.adaptive_uniform_ratio = adaptive_uniform_ratio
        self.adaptive_lambda = adaptive_lambda
        # Hard per-clip cap on max sampling probability (0 = no cap).
        # When set (e.g. 0.30), no single clip can exceed this fraction of
        # the total sampling mass — applied via water-fill redistribution
        # post-mixture in motion_sampling_probabilities.
        self.top1_prob_cap = float(top1_prob_cap)
        # SONIC-style sampling: use per-bin failure RATE (not count) with a
        # max-over-mean cap. Long clips don't dominate because the rate is
        # length-independent.
        self.sonic_style = bool(sonic_style)
        self.sonic_failure_rate_max_over_mean = float(sonic_failure_rate_max_over_mean)

        self.motion_start_idx = motion_start_idx.to(device).long()
        self.motion_end_idx = motion_end_idx.to(device).long()
        self.num_motions = int(self.motion_start_idx.numel())
        # Per-motion-id filename stems for top-K diagnostic logging. None = unavailable.
        self.motion_filenames = motion_filenames
        self.motion_lengths = (self.motion_end_idx - self.motion_start_idx).clamp(min=1)
        # Per-motion bin count: 1 second granularity (matches single-layer sampler).
        self.K_per_motion = (self.motion_lengths // self.env_fps + 1).long()
        self.K_max = int(self.K_per_motion.max().item())

        # Per-motion failure EMA (Layer 1) — initialized to ones so first reset is uniform.
        self.motion_failed_ema = torch.ones(self.num_motions, dtype=torch.float, device=device)

        # Per-(motion, bin) failure EMA (Layer 2) — initialized to ones in valid bins, 0 in masked bins.
        valid_mask = torch.arange(self.K_max, device=device).unsqueeze(0) < self.K_per_motion.unsqueeze(
            1
        )  # (num_motions, K_max), bool
        self.bin_valid_mask = valid_mask
        self.bin_failed_ema = valid_mask.float().clone()

        # SONIC-style: also track episode-count EMA per (motion, bin) so we can compute
        # failure_rate = failed / episodes. Init to ones (matches SONIC's pseudo-count
        # init that prevents NaN at start).
        if self.sonic_style:
            self.bin_episode_ema = valid_mask.float().clone()

        self.metrics: dict[str, torch.Tensor] = {}

    def update_episodes_and_failures(
        self,
        reset_motion_ids: torch.Tensor,
        reset_local_t: torch.Tensor,
        failed_mask: torch.Tensor,
    ) -> None:
        """Inject reset-time episode + failure events into both layer EMAs.

        Mirrors gear_sonic ``update_adaptive_sampling`` (motion_lib_base.py:2462):
        SONIC credits BOTH the episode counter and the failure counter at the
        END time-step bin (i.e., the bin where the env terminated/reset). All
        reset envs (failed or timed-out) bump the episode counter; only the
        failed-mask subset bumps the failure counter. This keeps the per-bin
        rate ``failed/episode`` properly aligned: numerator and denominator
        index the same bin.

        Decay is NOT applied here — it runs every env step inside
        ``MotionCommand.step()`` (parity with the legacy ``AdaptiveTimestepsSampler``).
        This method only adds the alpha-weighted instantaneous count.

        Args:
            reset_motion_ids: (n_reset,) clip index for every reset env.
            reset_local_t: (n_reset,) frame within that clip at reset time
                (i.e., time_step - motion_start_idx).
            failed_mask: (n_reset,) bool, True for reset envs that terminated
                due to failure (vs timeout / episode-length limit).
        """
        if reset_motion_ids.numel() == 0:
            return

        # Per-(motion, bin_within_motion) bin index for each reset env.
        K_per_reset = self.K_per_motion[reset_motion_ids]
        len_per_reset = self.motion_lengths[reset_motion_ids]
        reset_bins = torch.minimum(
            (reset_local_t.long() * K_per_reset) // len_per_reset.clamp(min=1),
            K_per_reset - 1,
        )
        reset_flat_idx = reset_motion_ids * self.K_max + reset_bins
        reset_inc = (
            torch.bincount(reset_flat_idx, minlength=self.num_motions * self.K_max)
            .float()
            .view(self.num_motions, self.K_max)
            * self.bin_valid_mask.float()
        )

        # SONIC-style: credit episode count for ALL reset envs at the end-bin.
        # Required so the per-bin rate denominator tracks bins actually reached
        # (otherwise easy clips that timeout-but-never-fail get rate→1, making
        # them look hard).
        if self.sonic_style and hasattr(self, "bin_episode_ema"):
            self.bin_episode_ema = self.bin_episode_ema + self.adaptive_alpha * reset_inc

        # Failure-only updates (subset of reset envs).
        failed_motion_ids = reset_motion_ids[failed_mask]
        if failed_motion_ids.numel() == 0:
            return

        # Layer 1: per-motion failure count → alpha-weighted increment.
        layer1_inc = torch.bincount(failed_motion_ids, minlength=self.num_motions).float()
        self.motion_failed_ema = self.motion_failed_ema + self.adaptive_alpha * layer1_inc

        # Layer 2: per-(motion, bin) failure count, restricted to failed envs.
        failed_flat_idx = reset_flat_idx[failed_mask]
        layer2_inc = (
            torch.bincount(failed_flat_idx, minlength=self.num_motions * self.K_max)
            .float()
            .view(self.num_motions, self.K_max)
            * self.bin_valid_mask.float()
        )
        self.bin_failed_ema = self.bin_failed_ema + self.adaptive_alpha * layer2_inc

    # Backward-compat shim: callers that only have failure data (no reset_mask)
    # can still funnel through update_episodes_and_failures by treating every
    # reset as a failure. Used by Layer-1-only callers and as a safety net for
    # external code paths that haven't been updated. SONIC-style mode requires
    # the new entry point because the episode denominator is meaningful.
    def update_failures(
        self,
        failed_motion_ids: torch.Tensor,
        failed_local_t: torch.Tensor,
    ) -> None:
        if failed_motion_ids.numel() == 0:
            return
        failed_mask = torch.ones(failed_motion_ids.numel(), dtype=torch.bool, device=self.device)
        self.update_episodes_and_failures(failed_motion_ids, failed_local_t, failed_mask)

    @property
    def motion_sampling_probabilities(self) -> torch.Tensor:
        """Per-motion-id sampling distribution.

        Default mode: mixture-form on per-clip failure COUNT EMA. Bug: long clips
        (e.g. fightAndSports1_subject4 at 12257 frames vs typical 200) accumulate
        proportionally more failure events even at uniform per-frame failure rate,
        so they dominate. Mitigated (but not eliminated) by mixture floor + hard cap.

        SONIC-style mode (sonic_style=True): aggregate per-bin failure RATE — i.e.
        ``failure_count / episode_count`` averaged across bins of each motion.
        Length-independent. Cap per-motion rate at ``sonic_failure_rate_max_over_mean
        x mean_rate`` (default 200x) before normalizing. Mixture floor applied on top.
        """
        n = self.num_motions
        uniform_p = torch.full((n,), 1.0 / max(n, 1), device=self.device)
        r = float(self.adaptive_uniform_ratio)

        if self.sonic_style and hasattr(self, "bin_episode_ema"):
            # SONIC reference: per-BIN failure rate is failed/episode (motion_lib_base.py:2531),
            # then capped at mean x max_over_mean (default 200), normalized to a per-bin
            # distribution, and finally aggregated to per-motion by SUMMING the (capped,
            # uniform-mixed) per-bin probabilities over each motion's bins
            # (motion_lib_base.py:2730-2734). This makes the per-motion sampling probability
            # length-independent: a long clip's bins each get a cap-bounded share, and their
            # sum ∝ (clipped_rate x num_bins / total_clipped_rate x some_norm). Critically,
            # the rate is NOT averaged over a clip's bins — it is summed (after capping),
            # so a clip with one truly hard bin and many easy bins is still sampled
            # proportionally. Mirroring this here:
            valid = self.bin_valid_mask.float()
            per_bin_rate = self.bin_failed_ema / self.bin_episode_ema.clamp(min=1e-6)
            per_bin_rate = per_bin_rate * valid  # zero invalid bins

            # Step 1: cap per-bin rate at mean (across valid bins) x max_over_mean.
            max_over_mean = self.sonic_failure_rate_max_over_mean
            valid_total = valid.sum().clamp(min=1.0)
            mean_rate = per_bin_rate.sum() / valid_total  # mean over valid bins
            rate_cap = mean_rate.clamp(min=1e-12) * max_over_mean
            per_bin_rate_clipped = per_bin_rate.clamp(max=rate_cap) * valid

            # Step 2: aggregate to per-motion by summing each motion's bin rates.
            # Length-independent in the same sense as SONIC: capping bounds each bin's
            # contribution; longer clips with more bins still get more probability mass
            # but only proportional to bin count, not proportional to length x per-bin-rate.
            per_motion_rate = per_bin_rate_clipped.sum(dim=1)  # (num_motions,)
            failure_p = per_motion_rate / per_motion_rate.sum().clamp(min=1e-12)
        else:
            ema = self.motion_failed_ema
            failure_p = ema / ema.sum().clamp(min=1e-12)

        p = (1.0 - r) * failure_p + r * uniform_p
        # Hard per-clip prob cap (water-fill). Layered on top of failure-rate cap above.
        if self.top1_prob_cap > 0:
            p = _apply_prob_cap(p, self.top1_prob_cap)
        return p

    def per_motion_phase_distribution(self, motion_ids: torch.Tensor) -> torch.Tensor:
        """Return (n, K_max) phase-bin distribution conditioned on each env's chosen motion_id.

        Mixture form: (1 - r) * (bin_ema-normalized over valid bins) + r * uniform_over_valid_bins.
        Masked (invalid) bins receive zero probability via valid_mask multiply at the end.
        """
        bin_ema = self.bin_failed_ema[motion_ids]  # (n, K_max)
        valid = self.bin_valid_mask[motion_ids].float()  # (n, K_max), 0/1
        per_env_valid_count = valid.sum(dim=1, keepdim=True).clamp(min=1.0)  # (n, 1)
        # Failure-driven part, normalized over valid bins.
        masked_ema = bin_ema * valid
        masked_sum = masked_ema.sum(dim=1, keepdim=True).clamp(min=1e-12)
        failure_p = masked_ema / masked_sum  # (n, K_max), zeros on invalid bins
        # Uniform-over-valid-bins part.
        uniform_p = valid / per_env_valid_count
        r = float(self.adaptive_uniform_ratio)
        return (1.0 - r) * failure_p + r * uniform_p

    def sample(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (motion_ids, phase) jointly sampled from the two-layer distribution.

        phase ∈ [0, 1) is the per-clip fraction (consistent with the apply step).
        """
        # Layer 1: motion ids
        motion_probs = self.motion_sampling_probabilities
        motion_ids = torch.multinomial(motion_probs, n, replacement=True)

        # Layer 2: phase, conditioned on motion_ids
        phase_probs = self.per_motion_phase_distribution(motion_ids)  # (n, K_max)
        sampled_bins = torch.multinomial(phase_probs, 1, replacement=True).squeeze(1)  # (n,)
        # Each motion has its own K_i; clamp sampled_bins (already in [0, K_max-1] but
        # could exceed K_per_motion - 1 if floor produced an invalid hit).
        K_per_env = self.K_per_motion[motion_ids]
        sampled_bins = torch.minimum(sampled_bins, K_per_env - 1)
        # NOTE: bin_episode_ema is credited inside update_episodes_and_failures at the
        # END-time-step bin (where the env actually terminated/reset), NOT here at the
        # start-time-step bin. Mirrors SONIC reference (motion_lib_base.py:2479-2487):
        # both num_episodes and num_failures are bumped at the same bin so per-bin rate
        # ``failed/episode`` stays aligned. Crediting episodes here would put numerator
        # and denominator at different bins.
        # Convert bin index to phase ∈ [0, 1).
        phase = (sampled_bins.float() + torch.rand(n, device=self.device)) / K_per_env.float()
        phase = phase.clamp(0.0, 1.0 - 1e-6)
        return motion_ids, phase

    def get_stats(self):
        # Layer 1 entropy.
        p = self.motion_sampling_probabilities
        H = -(p * (p + 1e-12).log()).sum()
        H_norm_motion = H / max(1.0, np.log(max(self.num_motions, 1)))
        # Layer 1 top-1.
        p_max, idx_max = p.max(dim=0)
        # Layer 2: average entropy across motions of their per-motion bin distribution.
        all_motion_ids = torch.arange(self.num_motions, device=self.device)
        bin_p = self.per_motion_phase_distribution(all_motion_ids)  # (num_motions, K_max)
        bin_H = -(bin_p * (bin_p + 1e-12).log()).sum(dim=1).mean()
        bin_H_norm = bin_H / max(1.0, np.log(max(self.K_max, 1)))
        self.metrics["motion_sampling_entropy"] = H_norm_motion
        self.metrics["motion_sampling_top1_prob"] = p_max
        self.metrics["motion_sampling_top1_id"] = idx_max.float() / max(self.num_motions, 1)
        self.metrics["motion_phase_avg_entropy"] = bin_H_norm

        # Top-K motion clip diagnostics: which clips is the failure-weighted sampler
        # concentrating on? Logged as scalars (top1..top10 probs and integer ids) plus
        # a single comma-joined name string in metrics_str (consumed by wbt_manager
        # and pushed to wandb as a Table) for easy spotting of pathological clips.
        K = min(10, int(self.num_motions))
        topk_probs, topk_ids = torch.topk(p, K)
        self.metrics["motion_sampling_top1_id_int"] = idx_max.float()  # raw integer for clarity
        for k in range(K):
            self.metrics[f"motion_sampling_top{k + 1}_prob"] = topk_probs[k]
            self.metrics[f"motion_sampling_top{k + 1}_id_int"] = topk_ids[k].float()
        if self.motion_filenames is not None and len(self.motion_filenames) >= self.num_motions:
            top_names = [self.motion_filenames[int(i)] for i in topk_ids.cpu().tolist()]
            top_pcts = [f"{float(topk_probs[k]):.3f}" for k in range(K)]
            # String concat — consumers of metrics_str (e.g. wbt_manager) can
            # log this as a wandb.Html / wandb.Table for richer display.
            self.metrics_str = getattr(self, "metrics_str", {})
            self.metrics_str["motion_sampling_top10_names"] = ", ".join(
                f"{n}({p})" for n, p in zip(top_names, top_pcts)
            )


#########################################################################################################
## Helper functions
#########################################################################################################
def _apply_prob_cap(p: torch.Tensor, cap: float, max_passes: int = 50) -> torch.Tensor:
    """Redistribute probability mass so no entry exceeds ``cap``.

    Iteratively clip entries above ``cap`` and add the excess uniformly to the rest,
    until convergence. Robust water-fill — terminates after at most max_passes iterations
    or when no entry exceeds the cap. Cap must be > 1/N or the operation is impossible
    (returns uniform in that degenerate case).

    Used by adaptive samplers to enforce a hard ceiling on any single clip's sampling
    probability, e.g. preventing one pathologically-hard clip from dominating training.
    """
    if cap <= 0 or cap >= 1.0:
        return p
    n = p.numel()
    if cap * n <= 1.0:
        # Cap is below the uniform floor → return uniform (best feasible)
        return torch.full_like(p, 1.0 / max(n, 1))
    p = p.clone()
    for _ in range(max_passes):
        over_mask = p > cap
        n_over = int(over_mask.sum())
        if n_over == 0:
            break
        excess = (p[over_mask] - cap).sum()
        p[over_mask] = cap
        below_mask = ~over_mask
        n_below = int(below_mask.sum())
        if n_below == 0:
            break
        p[below_mask] = p[below_mask] + excess / n_below
    return p


FAKE_BODY_NAME_ALIASES: dict[str, str] = {
    # Fake foot contact bodies are authored in the URDF purely for height computation.
    # They do not exist in the motion-capture dataset, so we alias them back to the
    # closest real body when indexing into motion data. These are not actually used in training.
    "left_foot_contact_point": "left_ankle_roll_link",
    "right_foot_contact_point": "right_ankle_roll_link",
}


def get_filtered_body_names(body_list: List[str], pattern: str) -> List[str]:
    return [body_name for body_name in body_list if re.match(pattern, body_name)]


class MotionCommand(CommandTermBase):
    def __init__(self, cfg: Any, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)

        self._env = env
        # self.motion_cfg: MotionConfig = cfg.params["motion_config"]
        # TODO(jchen):temporary fix for motion_config being a dict after tyro.cli
        if isinstance(cfg.params["motion_config"], MotionConfig):
            self.motion_cfg = cfg.params["motion_config"]
        else:
            self.motion_cfg = MotionConfig(**cfg.params["motion_config"])
        self.init_pose_cfg: NoiseToInitialPoseConfig = self.motion_cfg.noise_to_initial_pose
        # Ref semantic: when False, reset root-velocity noise falls back to push
        # randomizer max_push_vel. When True, use init_pose_cfg.root_lin/ang_vel.
        self.use_configured_root_velocity_noise = bool(cfg.params.get("use_configured_root_velocity_noise", False))

    def setup(self) -> None:
        self.num_envs = self._env.num_envs
        self.device = self._env.device

        robot_body_names = self._env.simulator._body_list  # type: ignore[attr-defined]
        robot_body_names_alias = [FAKE_BODY_NAME_ALIASES.get(bn, bn) for bn in robot_body_names]

        robot_joint_names = self._env.simulator.dof_names  # type: ignore[attr-defined]

        # 1. load motion data
        assert self.motion_cfg.motion_file or self.motion_cfg.motion_dir, (
            "Either motion_file or motion_dir must be set in MotionConfig"
        )
        self.motion: MotionLoader | MultiMotionLoader
        if self.motion_cfg.motion_dir:
            # Determine sharding params from torch distributed if enabled
            shard_rank = 0
            shard_world_size = 1
            if self.motion_cfg.shard_motions and torch.distributed.is_initialized():
                shard_rank = torch.distributed.get_rank()
                shard_world_size = torch.distributed.get_world_size()
            self.motion = MultiMotionLoader(
                self.motion_cfg.motion_dir,
                robot_body_names_alias,
                robot_joint_names,
                device=self.device,
                shard_rank=shard_rank,
                shard_world_size=shard_world_size,
                recompute_velocities_from_positions=getattr(
                    self.motion_cfg, "recompute_velocities_from_positions", False
                ),
                exclude_filename_substrings=tuple(getattr(self.motion_cfg, "motion_exclude_filename_substrings", ())),
            )
        else:
            self.motion = MotionLoader(
                self.motion_cfg.motion_file,
                robot_body_names_alias,
                robot_joint_names,
                device=self.device,
                recompute_velocities_from_positions=getattr(
                    self.motion_cfg, "recompute_velocities_from_positions", False
                ),
            )

        # Store body and joint indexes for interpolation
        self._body_indexes_in_motion = self.motion._body_indexes
        self._joint_indexes_in_motion = self.motion._joint_indexes

        # Cache default pose roll/pitch from config (constant, avoids per-reset tensor creation)
        if self.motion_cfg.default_pose_transition_strategy == "learned":
            init_state = self._env.robot_config.init_state
            init_quat = torch.tensor(init_state.rot, dtype=torch.float32, device=self.device).unsqueeze(0)
            self._default_init_roll, self._default_init_pitch, _ = get_euler_xyz(init_quat, w_last=True)

        # Maybe prepend interpolated transition from default pose
        self._maybe_add_default_pose_transition(prepend=True)

        # Maybe append interpolated transition back to default pose
        self._maybe_add_default_pose_transition(prepend=False)

        # 2. get the indexes of the root link and the tracked links
        self.ref_body_index = robot_body_names.index(self.motion_cfg.body_name_ref[0])  # int
        self.tracked_body_indexes = self._get_index_of_a_in_b(
            self.motion_cfg.body_names_to_track, robot_body_names, self.device
        )
        # Pre-compose: motion raw body dim -> tracked bodies (skips intermediate)
        self._tracked_body_indexes_in_motion = self.motion._body_indexes[self.tracked_body_indexes]

        # 3. get the name of the object, or indices of the object
        if self.motion.has_object:
            # cache the object_index_in_simulator
            self.object_name = "object"  # hardcoded object name
            self.object_indices_in_simulator = self._env.simulator.get_actor_indices(self.object_name, env_ids=None)

            assert self._env.simulator.get_simulator_type() == SimulatorType.ISAACSIM, (
                "Object is only supported in IsaacSim"
            )

        # 4. get the adaptive timesteps sampler
        # Validate adaptive sampling flags loudly at setup time. typos in the
        # weighting string would otherwise silently fall through to uniform.
        _adaptive_motion_weighting = getattr(self.motion_cfg, "adaptive_motion_weighting", "uniform")
        if _adaptive_motion_weighting not in ("uniform", "failure"):
            raise ValueError(
                f"motion_cfg.adaptive_motion_weighting must be 'uniform' or 'failure', "
                f"got {_adaptive_motion_weighting!r}. Check for typos / case mismatch."
            )
        _adaptive_phase_per_motion = bool(getattr(self.motion_cfg, "adaptive_phase_per_motion", False))
        # Warn if user enabled new flags but forgot to enable the global adaptive sampler
        # (the new code paths only activate when use_adaptive_timesteps_sampler is True).
        if not self.motion_cfg.use_adaptive_timesteps_sampler and (
            _adaptive_motion_weighting != "uniform" or _adaptive_phase_per_motion
        ):
            logger.warning(
                "adaptive_motion_weighting=%s and adaptive_phase_per_motion=%s set, but "
                "use_adaptive_timesteps_sampler=False — both new flags are silently ignored.",
                _adaptive_motion_weighting,
                _adaptive_phase_per_motion,
            )

        if self.motion_cfg.use_adaptive_timesteps_sampler:
            self.adaptive_timesteps_sampler = AdaptiveTimestepsSampler(
                self.motion.time_step_total,
                self.device,
                int(1 / (self._env.dt)),
                adaptive_kernel_size=self.motion_cfg.adaptive_kernel_size,
                adaptive_lambda=self.motion_cfg.adaptive_lambda,
                adaptive_uniform_ratio=self.motion_cfg.adaptive_uniform_ratio,
                adaptive_alpha=self.motion_cfg.adaptive_alpha,
            )
            # Two-layer adaptive sampler — opt-in via config. Co-exists with the single-layer
            # sampler above so we can keep that path's bin metrics for monitoring.
            if _adaptive_phase_per_motion:
                self.two_layer_sampler = TwoLayerAdaptiveSampler(
                    motion_start_idx=self.motion.motion_start_idx,
                    motion_end_idx=self.motion.motion_end_idx,
                    device=self.device,
                    env_fps=int(1 / (self._env.dt)),
                    adaptive_lambda=self.motion_cfg.adaptive_lambda,
                    adaptive_uniform_ratio=self.motion_cfg.adaptive_uniform_ratio,
                    adaptive_alpha=self.motion_cfg.adaptive_alpha,
                    motion_filenames=getattr(self.motion, "motion_filenames", None),
                    top1_prob_cap=float(getattr(self.motion_cfg, "motion_sampling_top1_prob_cap", 0.0)),
                    sonic_style=bool(getattr(self.motion_cfg, "sonic_style_sampler", False)),
                    sonic_failure_rate_max_over_mean=float(
                        getattr(self.motion_cfg, "sonic_failure_rate_max_over_mean", 200.0)
                    ),
                )
            else:
                self.two_layer_sampler = None  # type: ignore[assignment]
            # Per-motion failure EMA - used when adaptive_motion_weighting='failure' but
            # adaptive_phase_per_motion=False (Layer-1-only mode).
            if _adaptive_motion_weighting == "failure" and not _adaptive_phase_per_motion:
                self._motion_failed_ema_layer1_only = torch.ones(
                    int(self.motion.num_motions), dtype=torch.float, device=self.device
                )
            else:
                self._motion_failed_ema_layer1_only = None

        # 5. metrics
        self.metrics: dict[str, torch.Tensor] = {}

        self.init_buffers()

        # 6. visualization markers for isaacsim
        if self._env.viewer and self._env.simulator.get_simulator_type() == SimulatorType.ISAACSIM:
            self._setup_visualization_markers_for_isaacsim()

    def reset(self, env_ids: torch.Tensor | None) -> None:
        """called per reset_idx, reset timesteps and robot/object poses."""
        env_ids = self._ensure_index_tensor(env_ids)
        if env_ids.numel() == 0:
            return

        n = env_ids.numel()
        num_motions = self.motion.num_motions

        # 0. Sample the time steps (three modes: uniform, failure-weighted Layer-1 only,
        #    or full two-layer per-motion phase).
        use_two_layer = (
            self.motion_cfg.use_adaptive_timesteps_sampler and getattr(self, "two_layer_sampler", None) is not None
        )
        use_layer1_only = (
            self.motion_cfg.use_adaptive_timesteps_sampler
            and not use_two_layer
            and getattr(self, "_motion_failed_ema_layer1_only", None) is not None
        )

        if self.motion_cfg.use_adaptive_timesteps_sampler:
            # Failure attribution from envs that terminated since last reset.
            # Gated on NOT is_evaluating so eval episodes don't pollute the
            # training-time EMAs (legacy single-layer + two-layer + layer1-only).
            if not self._env.is_evaluating:
                episode_failed = self._env.termination_manager.terminated[env_ids]
            else:
                episode_failed = torch.zeros_like(env_ids, dtype=torch.bool)

            # Two-layer sampler (incl. SONIC-style): credit episode count for ALL
            # reset envs at end-bin, and credit failure count for the failed-mask
            # subset at the same bin. Required during training (not eval) and only
            # when there's anything to credit.
            if use_two_layer and not self._env.is_evaluating and env_ids.numel() > 0:
                reset_at_time_step = self.time_steps[env_ids]
                clamped_ts = reset_at_time_step.clamp(0, self.motion.time_step_total - 1)
                reset_motion_ids = self.motion.motion_idxs[clamped_ts].to(self.device)
                reset_local_t = (reset_at_time_step - self.motion.motion_start_idx[reset_motion_ids]).clamp(min=0)
                self.two_layer_sampler.update_episodes_and_failures(reset_motion_ids, reset_local_t, episode_failed)

            if torch.any(episode_failed):
                failed_at_time_step = self.time_steps[env_ids][episode_failed]
                # Always update the global single-layer sampler's bins (kept for metrics
                # and backward-compat eval).
                self.adaptive_timesteps_sampler.update_current_bin_failed_count(failed_at_time_step)
                if use_layer1_only:
                    failed_motion_ids = self.motion.motion_idxs[
                        failed_at_time_step.clamp(0, self.motion.time_step_total - 1)
                    ].to(self.device)
                    inc = torch.bincount(failed_motion_ids, minlength=int(num_motions)).float()
                    # Inject only; decay runs every env step inside step().
                    self._motion_failed_ema_layer1_only = (
                        self._motion_failed_ema_layer1_only + self.motion_cfg.adaptive_alpha * inc
                    )
            # Note: no decay-only branch here. Per-step decay is handled inside
            # MotionCommand.step() (parity with the legacy single-layer sampler).

            # Now draw motion_ids and phase.
            if use_two_layer:
                sampled_motion_ids, phase = self.two_layer_sampler.sample(n)
                self.motion_ids[env_ids] = sampled_motion_ids
            else:
                phase = self.adaptive_timesteps_sampler.sample(n)
                if use_layer1_only:
                    # Mixture form: (1 - r) * failure_p + r * uniform, plus optional cap.
                    ema = self._motion_failed_ema_layer1_only
                    failure_p = ema / ema.sum().clamp(min=1e-12)
                    n_motions = ema.numel()
                    uniform_p = torch.full_like(ema, 1.0 / max(n_motions, 1))
                    r = float(self.motion_cfg.adaptive_uniform_ratio)
                    weights = (1.0 - r) * failure_p + r * uniform_p
                    cap = float(getattr(self.motion_cfg, "motion_sampling_top1_prob_cap", 0.0))
                    if cap > 0:
                        weights = _apply_prob_cap(weights, cap)
                    self.motion_ids[env_ids] = torch.multinomial(weights, n, replacement=True)
                else:
                    self.motion_ids[env_ids] = torch.randint(0, num_motions, (n,), device=self.device)
        else:
            phase = torch.rand(n, device=self.device)
            self.motion_ids[env_ids] = torch.randint(0, num_motions, (n,), device=self.device)

        if self._env.is_evaluating:
            phase = torch.zeros_like(phase)
            # Force uniform clip coverage in eval, regardless of training-time weighting.
            self.motion_ids[env_ids] = torch.arange(n, dtype=torch.long, device=self.device) % num_motions

        start_idx = self.motion.motion_start_idx[self.motion_ids[env_ids]]
        end_idx = self.motion.motion_end_idx[self.motion_ids[env_ids]]
        motion_len = end_idx - start_idx

        self.time_steps[env_ids] = start_idx + (phase * (motion_len - 1).float()).long()

        # Handle start_at_timestep_zero_prob (reset to start of assigned motion)
        prob = self.motion_cfg.start_at_timestep_zero_prob
        if prob >= 1.0:
            self.time_steps[env_ids] = start_idx
        elif prob > 0.0:
            subset = self.time_steps[env_ids]
            rand_vals = torch.rand_like(subset, dtype=torch.float32)
            subset = torch.where(rand_vals < prob, start_idx, subset)
            self.time_steps[env_ids] = subset

        # If the motion is at the last timestep, set it to the second last timestep;
        # Otherwise, update_tasks_callback will advance the timestep to the next timestep -> out of bounds error.
        already_last_timestep_mask = self.time_steps[env_ids] >= end_idx - 1
        self.time_steps[env_ids] = torch.where(already_last_timestep_mask, end_idx - 2, self.time_steps[env_ids])

        # 1. Get the root/body poses from the motion data
        # Index only reset-env timesteps (raw body index 0 = root, same as root_pos_w property)
        reset_ts = self.time_steps[env_ids]
        env_origins = self._env.simulator.scene.env_origins[env_ids]
        root_pos = self.motion._body_pos_w[reset_ts, 0] + env_origins
        root_rot = self.motion._body_quat_w[reset_ts, 0]
        root_lin_vel = self.motion._body_lin_vel_w[reset_ts, 0]
        root_ang_vel = self.motion._body_ang_vel_w[reset_ts, 0]

        # Snap running_ref_root_height EMA to the freshly-sampled timestep's
        # ref pelvis z so SONIC-style adaptive terminations don't carry stale
        # cross-clip history at episode start.
        if hasattr(self, "running_ref_root_height"):
            self.running_ref_root_height[env_ids] = (  # type: ignore[has-type]
                self.motion._body_pos_w[reset_ts, 0, 2]
            )

        dof_pos = self.motion.get_joint_pos(reset_ts)
        dof_vel = self.motion.get_joint_vel(reset_ts)

        # "learned" strategy: for envs starting at frame 0, init robot in default standing
        # pose. Motion command stays at frame 0 — policy learns the transition autonomously.
        if self.motion_cfg.default_pose_transition_strategy == "learned":
            at_start_mask = self.time_steps[env_ids] == start_idx
            if torch.any(at_start_mask):
                n_start = int(at_start_mask.sum().item())
                # Default joint angles
                dof_pos[at_start_mask] = self._env.default_dof_pos_base.expand(n_start, -1)
                dof_vel[at_start_mask] = 0.0
                # Default root: keep motion x/y + env_origin, use config z/roll/pitch
                init_state = self._env.robot_config.init_state
                root_pos[at_start_mask, 2] = env_origins[at_start_mask, 2] + init_state.pos[2]
                root_lin_vel[at_start_mask] = 0.0
                root_ang_vel[at_start_mask] = 0.0
                # Root rotation: keep motion yaw, replace roll/pitch with config defaults
                init_roll, init_pitch = self._default_init_roll, self._default_init_pitch
                _, _, motion_yaw = get_euler_xyz(root_rot[at_start_mask], w_last=True)
                default_rot = quat_from_euler_xyz(
                    init_roll.expand(n_start),
                    init_pitch.expand(n_start),
                    motion_yaw,
                )
                root_rot[at_start_mask] = default_rot

        # 2. Adding noise
        # 2.1 prepare the noise scale
        dof_pos_noise = self.init_pose_cfg.dof_pos * self.init_pose_cfg.overall_noise_scale  # float
        root_pos_noise = (
            torch.tensor(
                self.init_pose_cfg.root_pos,
                device=self.device,
            )
            * self.init_pose_cfg.overall_noise_scale
        )  # (3,)
        root_rot_noise_rpy = (
            torch.tensor(
                self.init_pose_cfg.root_rot,
                device=self.device,
            )
            * self.init_pose_cfg.overall_noise_scale
        )  # (3,)
        if self.use_configured_root_velocity_noise:
            root_vel_noise = (
                torch.tensor(
                    self.init_pose_cfg.root_lin_vel,
                    device=self.device,
                )
                * self.init_pose_cfg.overall_noise_scale
            )  # (3,)
            root_ang_vel_noise_rpy = (
                torch.tensor(
                    self.init_pose_cfg.root_ang_vel,
                    device=self.device,
                )
                * self.init_pose_cfg.overall_noise_scale
            )  # (3,)
        else:
            # Reuse push randomizer velocity limits for reset-state velocity noise
            # . Falls back to zero when the push
            # randomizer is disabled.
            push_state = self._env.randomization_manager.get_state("push_randomizer_state")
            _push_vel = getattr(push_state, "max_push_vel", None) if push_state is not None else None
            if _push_vel is None:
                max_push_vel = torch.zeros(6, device=self.device)
            else:
                max_push_vel = torch.abs(_push_vel.to(self.device))
            root_vel_noise = max_push_vel[:3]
            root_ang_vel_noise_rpy = max_push_vel[3:6]

        # 2.2 Adding noise to dof_pos, root_pos, root_vel, root_ang_vel, root_rot
        # 1.2.1 dof_pos
        target_dof_pos = (
            dof_pos + (torch.rand(dof_pos.shape, device=self.device) - 0.5) * 2 * dof_pos_noise
        )  # (num_envs, num_dofs)
        soft_joint_pos_limits = self._env.simulator.dof_pos_limits  # type: ignore[attr-defined]  # (num_dofs, 2)
        target_dof_pos = torch.clip(target_dof_pos, soft_joint_pos_limits[:, 0], soft_joint_pos_limits[:, 1])

        # 1.2.2 dof_vel no noise
        target_dof_vel = dof_vel

        # 1.2.3 root_pos
        target_root_pos = root_pos + (
            torch.rand(root_pos.shape, device=self.device) - 0.5
        ) * 2 * root_pos_noise.unsqueeze(0)  # (num_envs, 3)

        # 1.2.4 root_rot
        rand_sample_rpy = (torch.rand((len(env_ids), 3), device=self.device) - 0.5) * 2 * root_rot_noise_rpy
        orientations_delta = quat_from_euler_xyz(
            rand_sample_rpy[:, 0], rand_sample_rpy[:, 1], rand_sample_rpy[:, 2]
        )  # (num_envs, 4), xyzw
        target_root_rot = quat_mul(orientations_delta, root_rot, w_last=True)  # (num_envs, 4), xyzw

        # 1.2.5 root_lin_vel
        target_root_lin_vel = root_lin_vel + (
            torch.rand(root_lin_vel.shape, device=self.device) - 0.5
        ) * 2 * root_vel_noise.unsqueeze(0)  # (num_envs, 3)

        # 1.2.6 root_ang_vel
        target_root_ang_vel = root_ang_vel + (
            torch.rand(root_ang_vel.shape, device=self.device) - 0.5
        ) * 2 * root_ang_vel_noise_rpy.unsqueeze(0)  # (num_envs, 3)

        # 3. Set the robot states in simulator
        self._env.simulator.dof_pos[env_ids] = target_dof_pos
        self._env.simulator.dof_vel[env_ids] = target_dof_vel

        self._env.simulator.robot_root_states[env_ids, :3] = target_root_pos
        self._env.simulator.robot_root_states[env_ids, 3:7] = target_root_rot
        self._env.simulator.robot_root_states[env_ids, 7:10] = target_root_lin_vel
        self._env.simulator.robot_root_states[env_ids, 10:13] = target_root_ang_vel

        # 4. Set the object states in simulator
        if self.motion.has_object:
            obj_pos = self.motion.get_object_pos_w(reset_ts) + env_origins
            obj_ori = self.motion.get_object_quat_w(reset_ts)
            obj_lin_vel = self.motion.get_object_lin_vel_w(reset_ts)

            # 4.2 add noise to the object states
            obj_pos_noise = torch.tensor(
                [self.init_pose_cfg.object_pos],
                device=self.device,
            )
            obj_pos_noise = obj_pos_noise * self.init_pose_cfg.overall_noise_scale  # (3,)
            target_obj_pos = obj_pos + (torch.rand(obj_pos.shape, device=self.device) - 0.5) * 2 * obj_pos_noise

            object_states = torch.cat(
                [target_obj_pos, obj_ori, obj_lin_vel, torch.zeros_like(obj_lin_vel)], dim=-1
            )  # (num_envs, 7)
            # 4.3 set the object states in simulator
            self._env.simulator.set_actor_states([self.object_name], env_ids, object_states)

    def step(self) -> None:
        """called in _update_tasks_callback of the environment. (after compute_reward, before compute_observations)"""
        # 0. update time steps, all motion joint/body poses are updated automatically with the time steps.
        advance_mask = torch.ones_like(self.time_steps, dtype=torch.bool)

        # Handle freeze_at_timestep_zero_prob: for envs at their motion's start, randomly decide whether to advance
        freeze_prob = self.motion_cfg.freeze_at_timestep_zero_prob
        if freeze_prob > 0.0:
            zero_mask = self.time_steps == self.motion.motion_start_idx[self.motion_ids]
            if zero_mask.any():
                rand_vals = torch.rand(self.num_envs, device=self.device)
                freeze_mask = (rand_vals < freeze_prob) & zero_mask
                advance_mask = advance_mask & ~freeze_mask

        self.time_steps += advance_mask.long()

        # BeyondMimic-style behavior: when any source clip ends (boundary
        # between concatenated clips, or end of the NPZ), resample motion and
        # reset robot/object state without terminating the whole episode.
        # Uses per-frame motion_ends bool tensor (matches ref) so source-clip
        # boundaries inside a combined NPZ are detected, not just NPZ end.
        self.motion_end_reset.zero_()
        if self.motion_cfg.resample_on_motion_end:
            clamped = self.time_steps.clamp(0, self.motion.time_step_total - 1)
            at_end = self.motion.motion_ends[clamped]
            ended_env_ids = torch.where(at_end)[0]
        else:
            per_motion_end = self.motion.motion_end_idx[self.motion_ids]
            ended_env_ids = torch.where(self.time_steps >= per_motion_end)[0]
        if ended_env_ids.numel() > 0:
            self.motion_end_reset[ended_env_ids] = True
            self.reset(ended_env_ids)
            # Flush the mutated root/dof state into the simulator so that
            # rigid-body positions are up-to-date for downstream consumers
            # (termination checks, observations, rewards).
            sim = self._env.simulator
            sim.set_actor_root_state_tensor_robots(ended_env_ids, sim.robot_root_states)
            sim.set_dof_state_tensor_robots(ended_env_ids, sim.dof_state)  # type: ignore[attr-defined]
            sim.refresh_sim_tensors()

        # Update running_ref_root_height EMA (alpha=0.1, matches gear_sonic).
        # Read the reference pelvis world-z at the current per-env time_step
        # (motion._body_pos_w is the raw motion-frame body positions, body 0 = root).
        # Used by SONIC-style adaptive terminations to relax thresholds during
        # low-pelvis sections (breakdance, kneeling, handstand).
        ema_alpha_root = 0.1
        ref_root_z_now = self.motion._body_pos_w[self.time_steps, 0, 2]
        prev_running = self.running_ref_root_height  # type: ignore[has-type]
        self.running_ref_root_height = ema_alpha_root * ref_root_z_now + (1.0 - ema_alpha_root) * prev_running
        # On motion boundary resets, snap the EMA to the new motion's root z
        # so we don't carry stale state across clip transitions.
        if ended_env_ids.numel() > 0:
            self.running_ref_root_height[ended_env_ids] = ref_root_z_now[ended_env_ids]

        # 1. update body_pos_relative_w and body_quat_relative_w
        # definition of body_pos/quat_relative_w:
        # If I take this motion data and adapt it to where my robot currently is
        # (accounting for position(x, y) offset and yaw difference of a reference body),
        # what should each body part's target pose be?

        ## 1.0 get the reference body poses

        # Issue (This is a isaacgym only issue.):
        # ------------------------------------------------------------
        # In isaacgym, immediately after reset (self._env.episode_length_buf == 0), calling
        # simulator.set_actor_root_state_tensor and simulator.set_dof_state_tensor will reset
        # the robot_root_pos_w and robot_root_quat_w successfully.
        # However, the robot_body_pos_w and robot_body_quat_w are not updated successfully,
        # (since kinematic forward has not been applied yet).
        # Therefore, using robot_ref_pos_w and robot_ref_quat_w as reference body poses is not resetted correctly.

        # Solution:
        # ------------------------------------------------------------
        # if episode_length_buf == 0 OR motion just wrapped (motion_end_reset),
        # use robot_root on the ROBOT side only. Motion side always uses
        # ref_pos_w/ref_quat_w so tracking targets stay on the configured
        # reference body (torso_link), not the motion pelvis.
        use_root = ((self._env.episode_length_buf == 0) | self.motion_end_reset).unsqueeze(1).float()
        robot_ref_pos_w = self.robot_root_pos_w * use_root + self.robot_ref_pos_w * (1 - use_root)
        robot_ref_quat_w = self.robot_root_quat_w * use_root + self.robot_ref_quat_w * (1 - use_root)

        ## 1.1 expand to match the number of body parts (no memory copy, unlike repeat)
        _nb = len(self.motion_cfg.body_names_to_track)  # type: ignore[arg-type]
        ref_pos_w_repeat = self.ref_pos_w[:, None, :].expand(-1, _nb, -1)
        ref_quat_w_repeat = self.ref_quat_w[:, None, :].expand(-1, _nb, -1)
        robot_ref_pos_w_repeat = robot_ref_pos_w[:, None, :].expand(-1, _nb, -1)
        robot_ref_quat_w_repeat = robot_ref_quat_w[:, None, :].expand(-1, _nb, -1)

        ## 1.2 compute the relative body poses
        delta_quat_w = yaw_quat(
            quat_mul(robot_ref_quat_w_repeat, quat_inverse(ref_quat_w_repeat, w_last=True), w_last=True), w_last=True
        )
        ### 1.2.1 body_quat_relative_w
        self.body_quat_relative_w = quat_mul(delta_quat_w, self.body_quat_w, w_last=True)
        ### 1.2.2 body_pos_relative_w
        delta_pos_w_height = ref_pos_w_repeat - robot_ref_pos_w_repeat
        delta_pos_w_height[..., :2] = 0.0  # adjusting for height differences
        self.body_pos_relative_w = (
            robot_ref_pos_w_repeat
            + delta_pos_w_height
            + quat_apply(delta_quat_w, self.body_pos_w - ref_pos_w_repeat, w_last=True)
        )

        ### 1.3 update the adaptive timesteps sampler
        if self.motion_cfg.use_adaptive_timesteps_sampler:
            self.adaptive_timesteps_sampler.update_bin_failed_count()
            # Per-step EMA decay for the two-layer / layer1-only samplers, mirroring
            # the legacy single-layer sampler's per-step decay above. Without this,
            # the new samplers' EMAs only decay at reset boundaries — making them
            # ~Lx more sticky than the legacy sampler at the same alpha (where L is
            # the average episode length in env steps).
            tl = getattr(self, "two_layer_sampler", None)
            if tl is not None:
                decay = 1.0 - tl.adaptive_alpha
                tl.motion_failed_ema *= decay
                tl.bin_failed_ema *= decay
                # SONIC-style: episode-counter denominator must decay symmetrically
                # with bin_failed_ema, otherwise per-bin rate = failed/episode collapses
                # over training (numerator decays, denominator grows).
                if getattr(tl, "sonic_style", False) and hasattr(tl, "bin_episode_ema"):
                    tl.bin_episode_ema *= decay
            ema1 = getattr(self, "_motion_failed_ema_layer1_only", None)
            if ema1 is not None:
                ema1 *= 1.0 - self.motion_cfg.adaptive_alpha

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    #########################################################################################
    ## Robot from motion data
    #########################################################################################
    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.get_joint_pos(self.time_steps)

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.get_joint_vel(self.time_steps)

    @property
    def body_pos_w(self) -> torch.Tensor:
        return (
            self.motion._body_pos_w[self.time_steps][:, self._tracked_body_indexes_in_motion]
            + self._env.simulator.scene.env_origins[:, None, :]
        )

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion._body_quat_w[self.time_steps][:, self._tracked_body_indexes_in_motion]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion._body_lin_vel_w[self.time_steps][:, self._tracked_body_indexes_in_motion]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion._body_ang_vel_w[self.time_steps][:, self._tracked_body_indexes_in_motion]

    @property
    def ref_pos_w(self) -> torch.Tensor:
        return self.motion._body_pos_w[self.time_steps, self.ref_body_index] + self._env.simulator.scene.env_origins

    @property
    def ref_quat_w(self) -> torch.Tensor:
        return self.motion._body_quat_w[self.time_steps, self.ref_body_index]

    @property
    def ref_lin_vel_w(self) -> torch.Tensor:
        return self.motion._body_lin_vel_w[self.time_steps, self.ref_body_index]

    @property
    def ref_ang_vel_w(self) -> torch.Tensor:
        return self.motion._body_ang_vel_w[self.time_steps, self.ref_body_index]

    @property
    def root_pos_w(self) -> torch.Tensor:
        return self.motion._body_pos_w[self.time_steps, 0] + self._env.simulator.scene.env_origins

    @property
    def root_quat_w(self) -> torch.Tensor:
        return self.motion._body_quat_w[self.time_steps, 0]

    @property
    def root_lin_vel_w(self) -> torch.Tensor:
        return self.motion._body_lin_vel_w[self.time_steps, 0]

    @property
    def root_ang_vel_w(self) -> torch.Tensor:
        return self.motion._body_ang_vel_w[self.time_steps, 0]

    #########################################################################################
    ## Robot from simulator
    #########################################################################################
    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self._env.simulator.dof_pos  # (num_envs, num_dofs)

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self._env.simulator.dof_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_pos[:, self.tracked_body_indexes, :]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_rot[:, self.tracked_body_indexes, :]  # xyzw

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_vel[:, self.tracked_body_indexes, :]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_ang_vel[:, self.tracked_body_indexes, :]

    @property
    def robot_root_pos_w(self) -> torch.Tensor:
        return self._env.simulator.robot_root_states[:, :3]  # type: ignore[attr-defined]

    @property
    def robot_root_quat_w(self) -> torch.Tensor:
        return self._env.simulator.robot_root_states[:, 3:7]  # type: ignore[attr-defined]

    @property
    def robot_root_lin_vel_w(self) -> torch.Tensor:
        return self._env.simulator.robot_root_states[:, 7:10]  # type: ignore[attr-defined]

    @property
    def robot_root_ang_vel_w(self) -> torch.Tensor:
        return self._env.simulator.robot_root_states[:, 10:13]  # type: ignore[attr-defined]

    @property
    def robot_ref_pos_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_pos[:, self.ref_body_index, :]

    @property
    def robot_ref_quat_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_rot[:, self.ref_body_index, :]  # xyzw

    @property
    def robot_ref_lin_vel_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_vel[:, self.ref_body_index, :]

    @property
    def robot_ref_ang_vel_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_ang_vel[:, self.ref_body_index, :]

    #########################################################################################
    ## Object from motion data
    #########################################################################################
    @property
    def object_pos_w(self) -> torch.Tensor:
        return self.motion.get_object_pos_w(self.time_steps) + self._env.simulator.scene.env_origins

    @property
    def object_quat_w(self) -> torch.Tensor:
        return self.motion.get_object_quat_w(self.time_steps)

    @property
    def object_lin_vel_w(self) -> torch.Tensor:
        return self.motion.get_object_lin_vel_w(self.time_steps)

    #########################################################################################
    ## Object from simulator
    #########################################################################################
    @property
    def simulator_object_pos_w(self) -> torch.Tensor:
        return self._env.simulator.all_root_states[self.object_indices_in_simulator][:, :3]

    @property
    def simulator_object_quat_w(self) -> torch.Tensor:
        return self._env.simulator.all_root_states[self.object_indices_in_simulator][:, 3:7]

    @property
    def simulator_object_lin_vel_w(self) -> torch.Tensor:
        return self._env.simulator.all_root_states[self.object_indices_in_simulator][:, 7:10]

    #########################################################################################
    ## Methods that does not fit into setup/step/reset pattern
    #########################################################################################

    def init_buffers(self):
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.body_pos_relative_w = torch.zeros(
            self.num_envs, len(self.motion_cfg.body_names_to_track), 3, device=self.device
        )  # type: ignore[arg-type]
        self.body_quat_relative_w = torch.zeros(
            self.num_envs, len(self.motion_cfg.body_names_to_track), 4, device=self.device
        )  # type: ignore[arg-type]
        self.body_quat_relative_w[:, :, 0] = 1.0

        # Per-env flag signalling that a motion clip just ended and the env was resampled.
        # Consumed by _update_body_pos_relative_w to fall back on robot_root on the
        # resample step, since body tensors lag one frame after teleport.
        self.motion_end_reset = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # EMA of the reference's pelvis world-z, per env. Used by SONIC-style
        # adaptive terminations to relax thresholds during low-pelvis sections
        # (breakdance, kneeling, handstand-to-bridge, supine). alpha=0.1 — same
        # smoothing as gear_sonic's `running_ref_root_height`. Initialized to 0.78m
        # (rough G1 standing pelvis height) so the first step doesn't miscategorize
        # everything as "low".
        self.running_ref_root_height = torch.full((self.num_envs,), 0.78, dtype=torch.float, device=self.device)

        if self.motion_cfg.use_adaptive_timesteps_sampler:
            self.adaptive_timesteps_sampler.init_buffers()
            # Reset two-layer / layer1-only EMAs back to their uniform priors so
            # reset_all() (called e.g. on training restart) yields a clean state.
            tl = getattr(self, "two_layer_sampler", None)
            if tl is not None:
                tl.motion_failed_ema.fill_(1.0)
                tl.bin_failed_ema.copy_(tl.bin_valid_mask.float())
            ema1 = getattr(self, "_motion_failed_ema_layer1_only", None)
            if ema1 is not None:
                ema1.fill_(1.0)

    def update_metrics(self):
        """Update the metrics. After action, before step() is called."""
        self.metrics["motion/error_ref_pos"] = torch.norm(self.ref_pos_w - self.robot_ref_pos_w, dim=-1)
        self.metrics["motion/error_ref_rot"] = quat_error_magnitude(self.ref_quat_w, self.robot_ref_quat_w)
        self.metrics["motion/error_ref_lin_vel"] = torch.norm(self.ref_lin_vel_w - self.robot_ref_lin_vel_w, dim=-1)
        self.metrics["motion/error_ref_ang_vel"] = torch.norm(self.ref_ang_vel_w - self.robot_ref_ang_vel_w, dim=-1)

        self.metrics["motion/error_body_pos"] = torch.norm(
            self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
        ).mean(dim=-1)

        self.metrics["motion/error_body_rot"] = quat_error_magnitude(
            self.body_quat_relative_w, self.robot_body_quat_w
        ).mean(dim=-1)

        self.metrics["motion/error_body_lin_vel"] = torch.norm(
            self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
        ).mean(dim=-1)
        self.metrics["motion/error_body_ang_vel"] = torch.norm(
            self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
        ).mean(dim=-1)

        self.metrics["motion/error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["motion/error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

        if self.motion_cfg.use_adaptive_timesteps_sampler:
            self.adaptive_timesteps_sampler.get_stats()
            self.metrics["motion/adaptive_timesteps_sampler_entropy"] = self.adaptive_timesteps_sampler.metrics[
                "sampling_entropy"
            ]
            self.metrics["motion/adaptive_timesteps_sampler_top1_prob"] = self.adaptive_timesteps_sampler.metrics[
                "sampling_top1_prob"
            ]
            self.metrics["motion/adaptive_timesteps_sampler_top1_bin"] = self.adaptive_timesteps_sampler.metrics[
                "sampling_top1_bin"
            ]

            # Two-layer sampler metrics (only present when adaptive_phase_per_motion=True).
            tl = getattr(self, "two_layer_sampler", None)
            if tl is not None:
                tl.get_stats()
                self.metrics["motion/two_layer_motion_entropy"] = tl.metrics["motion_sampling_entropy"]
                self.metrics["motion/two_layer_motion_top1_prob"] = tl.metrics["motion_sampling_top1_prob"]
                self.metrics["motion/two_layer_motion_top1_id"] = tl.metrics["motion_sampling_top1_id"]
                self.metrics["motion/two_layer_phase_avg_entropy"] = tl.metrics["motion_phase_avg_entropy"]
                # Top-K diagnostics: per-rank top-10 clip probs + integer ids.
                # The named string version (clip filenames + probs) lands in
                # self.metrics_str so wbt_manager can push it to wandb as text.
                for k in range(1, 11):
                    pk_key = f"motion_sampling_top{k}_prob"
                    ik_key = f"motion_sampling_top{k}_id_int"
                    if pk_key in tl.metrics:
                        self.metrics[f"motion/two_layer_top{k}_prob"] = tl.metrics[pk_key]
                    if ik_key in tl.metrics:
                        self.metrics[f"motion/two_layer_top{k}_id"] = tl.metrics[ik_key]
                tl_str = getattr(tl, "metrics_str", None)
                if tl_str is not None and "motion_sampling_top10_names" in tl_str:
                    self.metrics_str = getattr(self, "metrics_str", {})
                    self.metrics_str["motion/two_layer_top10_names"] = tl_str["motion_sampling_top10_names"]

            # Layer-1-only failure-weighted motion sampling metrics.
            ema1 = getattr(self, "_motion_failed_ema_layer1_only", None)
            if ema1 is not None:
                p_layer1 = ema1 / ema1.sum().clamp(min=1e-12)
                # Apply same mixture form + optional cap to the wandb-logged probability
                # so the metric reflects the actual sampling distribution.
                r_floor = float(getattr(self.motion_cfg, "adaptive_uniform_ratio", 0.5))
                n_mot = int(self.motion.num_motions)
                uniform_p_layer1 = torch.full_like(p_layer1, 1.0 / max(n_mot, 1))
                p_mix_layer1 = (1.0 - r_floor) * p_layer1 + r_floor * uniform_p_layer1
                cap = float(getattr(self.motion_cfg, "motion_sampling_top1_prob_cap", 0.0))
                if cap > 0:
                    p_mix_layer1 = _apply_prob_cap(p_mix_layer1, cap)
                H = -(p_mix_layer1 * (p_mix_layer1 + 1e-12).log()).sum()
                self.metrics["motion/layer1_motion_entropy"] = H / max(1.0, np.log(max(n_mot, 1)))
                self.metrics["motion/layer1_motion_top1_prob"] = p_mix_layer1.max()
                # Top-K for layer-1-only mode (clip name lookup via motion_filenames).
                K = min(10, n_mot)
                tk_probs, tk_ids = torch.topk(p_mix_layer1, K)
                self.metrics["motion/layer1_motion_top1_id"] = tk_ids[0].float() / max(n_mot, 1)
                for k in range(K):
                    self.metrics[f"motion/layer1_top{k + 1}_prob"] = tk_probs[k]
                    self.metrics[f"motion/layer1_top{k + 1}_id"] = tk_ids[k].float()
                names_attr = getattr(self.motion, "motion_filenames", None)
                if names_attr is not None and len(names_attr) >= n_mot:
                    top_names = [names_attr[int(i)] for i in tk_ids.cpu().tolist()]
                    top_pcts = [f"{float(tk_probs[k]):.3f}" for k in range(K)]
                    self.metrics_str = getattr(self, "metrics_str", {})
                    self.metrics_str["motion/layer1_top10_names"] = ", ".join(
                        f"{n}({p})" for n, p in zip(top_names, top_pcts)
                    )

    #########################################################################################
    ## Internal helpers
    #########################################################################################
    def _maybe_add_default_pose_transition(self, *, prepend: bool) -> None:
        """Shared path for optionally inserting default-pose interpolation before/after the clip."""
        if prepend:
            enabled = (
                self.motion_cfg.enable_default_pose_prepend
                or self.motion_cfg.default_pose_transition_strategy == "interpolation"
            )
        else:
            enabled = self.motion_cfg.enable_default_pose_append
        if not enabled:
            return

        duration = (
            self.motion_cfg.default_pose_prepend_duration_s
            if prepend
            else self.motion_cfg.default_pose_append_duration_s
        )
        if duration <= 0.0:
            return

        num_steps = round(duration / self._env.dt)
        if num_steps <= 1:
            logger.warning(
                "Default pose {} duration {}s is too short for dt {}; skipping augmentation.",
                "prepend" if prepend else "append",
                duration,
                self._env.dt,
            )
            return

        default_state = self._build_default_pose_state(use_motion_end=not prepend)

        action = "prepend" if prepend else "append"
        log_str = f"{action} {num_steps} interpolated frames ({duration}s) from default pose to motion"
        try:
            self._add_transition_to_motion(default_state, num_steps, prepend=prepend)
            logger.info(log_str)
        except Exception as exc:
            logger.error(f"Failed to {action} default pose transition: {exc}")
            raise RuntimeError(
                f"Critical error during motion interpolation setup: {exc}\n"
                "This indicates a mismatch in tensor dimensions during interpolation. "
                "Please check that the motion file and robot configuration are compatible."
            ) from exc

    def _build_default_pose_state(self, use_motion_end: bool = False) -> dict[str, torch.Tensor]:
        """Build the state dict representing the robot's default standing pose.

        By default, anchor root pos/yaw to the motion start; when use_motion_end is True, anchor to motion end.
        """
        init_state = self._env.robot_config.init_state
        joint_pos = self._env.default_dof_pos_base.squeeze(0).to(self.device)
        joint_vel = torch.zeros_like(joint_pos)

        init_root_quat = torch.tensor(init_state.rot, dtype=torch.float32, device=self.device).unsqueeze(0)
        init_roll, init_pitch, _ = get_euler_xyz(init_root_quat, w_last=True)

        motion_idx = -1 if use_motion_end else 0

        # Assume the pelvis is the first in robot_body_names
        motion_root_pos = self.motion._body_pos_w[motion_idx, 0].to(self.device)
        motion_root_quat = self.motion._body_quat_w[motion_idx, 0].to(self.device).unsqueeze(0)
        _, _, motion_yaw = get_euler_xyz(motion_root_quat, w_last=True)

        # Keep z from init config but adopt the clip's x,y at the chosen anchor frame.
        default_root_pos = torch.tensor(
            [motion_root_pos[0], motion_root_pos[1], init_state.pos[2]],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        # Keep roll/pitch from init config but adopt the clip's yaw at the chosen anchor frame.
        default_root_quat = quat_from_euler_xyz(
            init_roll.squeeze(0),
            init_pitch.squeeze(0),
            motion_yaw.squeeze(0),
        )
        default_root_lin_vel = torch.tensor(init_state.lin_vel, dtype=torch.float32, device=self.device)
        default_root_ang_vel = torch.tensor(init_state.ang_vel, dtype=torch.float32, device=self.device)

        body_states = self._capture_body_states(
            joint_pos,
            joint_vel,
            default_root_pos,
            default_root_quat,
            default_root_lin_vel,
            default_root_ang_vel,
        )

        default_body_pos = self._map_robot_bodies_to_motion_order(body_states["pos"])
        default_body_quat = self._map_robot_bodies_to_motion_order(body_states["quat"])
        default_body_lin_vel = self._map_robot_bodies_to_motion_order(body_states["lin_vel"])
        default_body_ang_vel = self._map_robot_bodies_to_motion_order(body_states["ang_vel"])

        if self.motion.has_object:
            object_pos = self.motion._object_pos_w[motion_idx].to(self.device)
            object_quat = self.motion._object_quat_w[motion_idx].to(self.device)
            object_lin_vel = self.motion._object_lin_vel_w[motion_idx].to(self.device)
        else:
            object_pos = torch.zeros(0, 3, device=self.device, dtype=torch.float32)
            object_quat = torch.zeros(0, 4, device=self.device, dtype=torch.float32)
            object_lin_vel = torch.zeros(0, 3, device=self.device, dtype=torch.float32)

        return {
            "joint_pos": joint_pos.clone(),
            "joint_vel": joint_vel,
            "root_pos": default_root_pos,
            "root_quat": default_root_quat,
            "root_lin_vel": default_root_lin_vel,
            "root_ang_vel": default_root_ang_vel,
            "body_pos": default_body_pos,
            "body_quat": default_body_quat,
            "body_lin_vel": default_body_lin_vel,
            "body_ang_vel": default_body_ang_vel,
            "object_pos": object_pos,
            "object_quat": object_quat,
            "object_lin_vel": object_lin_vel,
        }

    def _add_transition_to_motion(self, default_state: dict[str, torch.Tensor], num_steps: int, prepend: bool) -> None:
        """Add interpolated frames either before or after the motion data."""
        assert self._body_indexes_in_motion is not None
        assert self._joint_indexes_in_motion is not None

        if num_steps <= 0:
            return

        device = self.device
        dtype = self.motion._joint_pos.dtype

        default_motion_state = self._default_motion_state(default_state, dtype=dtype, device=device)
        motion_state = self._motion_state(0 if prepend else -1, dtype=dtype, device=device)

        start_state = default_motion_state if prepend else motion_state
        target_state = motion_state if prepend else default_motion_state
        drop_first, drop_last = (False, True) if prepend else (True, False)

        self._build_and_apply_transition(
            start_state=start_state,
            target_state=target_state,
            num_steps=num_steps,
            prepend=prepend,
            drop_first=drop_first,
            drop_last=drop_last,
            dtype=dtype,
            device=device,
        )

    def _slerp_quat_sequence(self, start: torch.Tensor, end: torch.Tensor, alphas: torch.Tensor) -> torch.Tensor:
        """Spherically interpolate quaternions across multiple time steps."""
        if alphas.numel() == 0:
            return start.new_zeros((0,) + start.shape)

        num_steps = alphas.shape[0]
        start_expand = start.unsqueeze(0).expand(num_steps, -1, -1)
        end_expand = end.unsqueeze(0).expand(num_steps, -1, -1)
        alpha_flat = alphas.repeat_interleave(start.shape[0]).unsqueeze(-1)
        blended = slerp(
            start_expand.reshape(-1, 4),
            end_expand.reshape(-1, 4),
            alpha_flat,
        )
        return blended.view(num_steps, start.shape[0], 4)

    def _capture_body_states(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
        root_lin_vel: torch.Tensor,
        root_ang_vel: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Capture body states by temporarily setting the robot state in the simulator."""
        simulator = self._env.simulator
        assert simulator.get_simulator_type() == SimulatorType.ISAACSIM, (
            "Default-pose interpolation only supports IsaacSim; IsaacGym write_state_updates does not run FK."
        )
        env_id = 0
        env_origin = simulator.scene.env_origins[env_id].to(self.device)

        root_backup = simulator.robot_root_states[env_id].clone()
        dof_pos_backup = simulator.dof_pos[env_id].clone()
        dof_vel_backup = simulator.dof_vel[env_id].clone()

        try:
            simulator.robot_root_states[env_id, :3] = root_pos + env_origin
            simulator.robot_root_states[env_id, 3:7] = root_quat
            simulator.robot_root_states[env_id, 7:10] = root_lin_vel
            simulator.robot_root_states[env_id, 10:13] = root_ang_vel
            simulator.dof_pos[env_id] = joint_pos
            simulator.dof_vel[env_id] = joint_vel

            simulator.set_actor_root_state_tensor_robots()
            simulator.set_dof_state_tensor_robots()
            simulator.write_state_updates()
            simulator.refresh_sim_tensors()

            body_pos = (simulator._rigid_body_pos[env_id] - env_origin).clone()
            body_quat = simulator._rigid_body_rot[env_id].clone()
            body_lin_vel = simulator._rigid_body_vel[env_id].clone()
            body_ang_vel = simulator._rigid_body_ang_vel[env_id].clone()
        finally:
            simulator.robot_root_states[env_id] = root_backup
            simulator.dof_pos[env_id] = dof_pos_backup
            simulator.dof_vel[env_id] = dof_vel_backup
            simulator.set_actor_root_state_tensor_robots()
            simulator.set_dof_state_tensor_robots()
            simulator.write_state_updates()
            simulator.refresh_sim_tensors()

        return {
            "pos": body_pos,
            "quat": body_quat,
            "lin_vel": body_lin_vel,
            "ang_vel": body_ang_vel,
        }

    def _map_robot_bodies_to_motion_order(self, robot_tensor: torch.Tensor) -> torch.Tensor:
        """Map robot body tensor to motion data order using body indexes."""
        assert self._body_indexes_in_motion is not None
        num_motion_bodies = self.motion._body_pos_w.shape[1]
        motion_shape = (num_motion_bodies,) + robot_tensor.shape[1:]
        motion_tensor = torch.zeros(motion_shape, device=robot_tensor.device, dtype=robot_tensor.dtype)
        motion_tensor[self._body_indexes_in_motion] = robot_tensor
        return motion_tensor

    def _map_robot_joints_to_motion_order(
        self, robot_tensor: torch.Tensor, num_motion_joints: int | None = None
    ) -> torch.Tensor:
        """Map robot joint tensor to motion data order using joint indexes."""
        assert self._joint_indexes_in_motion is not None
        if num_motion_joints is None:
            num_motion_joints = self.motion._joint_pos.shape[1]
        motion_shape = robot_tensor.shape[:-1] + (num_motion_joints,)
        motion_tensor = torch.zeros(motion_shape, device=robot_tensor.device, dtype=robot_tensor.dtype)
        motion_tensor[..., self._joint_indexes_in_motion] = robot_tensor
        return motion_tensor

    def _motion_state(self, idx: int, dtype: torch.dtype, device: torch.device) -> dict[str, torch.Tensor]:
        """Slice motion tensors at a given index into a state dict."""
        state = {
            "joint_pos": self.motion._joint_pos[idx].to(device=device, dtype=dtype),
            "joint_vel": self.motion._joint_vel[idx].to(device=device, dtype=dtype),
            "body_pos": self.motion._body_pos_w[idx].to(device=device, dtype=dtype),
            "body_quat": self.motion._body_quat_w[idx].to(device=device, dtype=dtype),
            "body_lin_vel": self.motion._body_lin_vel_w[idx].to(device=device, dtype=dtype),
            "body_ang_vel": self.motion._body_ang_vel_w[idx].to(device=device, dtype=dtype),
        }
        if self.motion.has_object:
            state["object_pos"] = self.motion._object_pos_w[idx].to(device=device, dtype=dtype)
            state["object_quat"] = self.motion._object_quat_w[idx].to(device=device, dtype=dtype)
            state["object_lin_vel"] = self.motion._object_lin_vel_w[idx].to(device=device, dtype=dtype)
        return state

    def _default_motion_state(
        self, default_state: dict[str, torch.Tensor], dtype: torch.dtype, device: torch.device
    ) -> dict[str, torch.Tensor]:
        """Map default robot-state tensors into motion order for interpolation."""
        state = {
            "joint_pos": self._map_robot_joints_to_motion_order(
                default_state["joint_pos"].to(device=device, dtype=dtype),
                num_motion_joints=self.motion._joint_pos.shape[1],
            ),
            "joint_vel": self._map_robot_joints_to_motion_order(
                default_state["joint_vel"].to(device=device, dtype=dtype),
                num_motion_joints=self.motion._joint_vel.shape[1],
            ),
            "body_pos": default_state["body_pos"].to(device=device, dtype=dtype),
            "body_quat": default_state["body_quat"].to(device=device, dtype=dtype),
            "body_lin_vel": default_state["body_lin_vel"].to(device=device, dtype=dtype),
            "body_ang_vel": default_state["body_ang_vel"].to(device=device, dtype=dtype),
        }
        if self.motion.has_object:
            state["object_pos"] = default_state["object_pos"].to(device=device, dtype=dtype)
            state["object_quat"] = default_state["object_quat"].to(device=device, dtype=dtype)
            state["object_lin_vel"] = default_state["object_lin_vel"].to(device=device, dtype=dtype)
        return state

    def _build_transition_segments(
        self,
        start: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
        alphas: torch.Tensor,
        alphas_joint: torch.Tensor,
        alphas_body: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Linearly/spherically interpolate between start and target states."""

        def _lerp(a: torch.Tensor, b: torch.Tensor, view: torch.Tensor) -> torch.Tensor:
            return a.unsqueeze(0) + view * (b - a).unsqueeze(0)

        segments = {
            "joint_pos": _lerp(start["joint_pos"], target["joint_pos"], alphas_joint),
            "joint_vel": _lerp(start["joint_vel"], target["joint_vel"], alphas_joint),
            "body_pos": _lerp(start["body_pos"], target["body_pos"], alphas_body),
            "body_lin_vel": _lerp(start["body_lin_vel"], target["body_lin_vel"], alphas_body),
            "body_ang_vel": _lerp(start["body_ang_vel"], target["body_ang_vel"], alphas_body),
            "body_quat": self._slerp_quat_sequence(start["body_quat"], target["body_quat"], alphas),
        }

        if self.motion.has_object:
            segments["object_pos"] = _lerp(start["object_pos"], target["object_pos"], alphas_joint)
            segments["object_lin_vel"] = _lerp(start["object_lin_vel"], target["object_lin_vel"], alphas_joint)
            segments["object_quat"] = self._slerp_quat_sequence(
                start["object_quat"].unsqueeze(0), target["object_quat"].unsqueeze(0), alphas
            ).squeeze(1)

        return segments

    def _apply_transition_segments(self, segments: dict[str, torch.Tensor], prepend: bool) -> None:
        """Splice interpolated segments into motion data, either prepending or appending."""
        self.motion = self.motion.extend_with_segments(segments, prepend=prepend)

    def _build_and_apply_transition(
        self,
        start_state: dict[str, torch.Tensor],
        target_state: dict[str, torch.Tensor],
        num_steps: int,
        prepend: bool,
        drop_first: bool,
        drop_last: bool,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Shared interpolation path for prepend/append transitions."""
        if num_steps <= 0:
            return

        alphas = torch.linspace(0.0, 1.0, steps=num_steps + 1, device=device, dtype=dtype)
        if drop_first:
            alphas = alphas[1:]
        if drop_last:
            alphas = alphas[:-1]
        if alphas.numel() == 0:
            return

        alphas_joint = alphas.view(num_steps, 1)
        alphas_body = alphas.view(num_steps, 1, 1)

        segments = self._build_transition_segments(start_state, target_state, alphas, alphas_joint, alphas_body)
        self._apply_transition_segments(segments, prepend=prepend)

    def _setup_visualization_markers_for_isaacsim(self):
        from isaaclab.markers import VisualizationMarkers
        from isaaclab.markers.config import FRAME_MARKER_CFG, RAY_CASTER_MARKER_CFG

        visualization_markers_cfg = FRAME_MARKER_CFG.replace(
            prim_path="/Visuals/Command/real_robot",
        )
        visualization_markers_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
        real_robot_visualizer = VisualizationMarkers(visualization_markers_cfg)

        visualization_markers_cfg = FRAME_MARKER_CFG.replace(
            prim_path="/Visuals/Command/motion_robot",
        )
        visualization_markers_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
        motion_robot_visualizer = VisualizationMarkers(visualization_markers_cfg)
        self.visualization_markers = {
            "real_robot": real_robot_visualizer,
            "motion_robot": motion_robot_visualizer,
        }

        for body_names in self.motion_cfg.body_names_to_track:
            visualization_markers_cfg = RAY_CASTER_MARKER_CFG.replace(
                prim_path=f"/Visuals/Command/motion_robot_body/motion_{body_names}",
            )
            visualization_markers_cfg.markers["hit"].radius = 0.03
            visualization_markers_cfg.markers["hit"].visual_material.diffuse_color = (0.0, 1.0, 0.0)
            self.visualization_markers[f"motion_{body_names}"] = VisualizationMarkers(visualization_markers_cfg)

        if self.motion.has_object:
            visualization_markers_cfg = FRAME_MARKER_CFG.replace(
                prim_path="/Visuals/Command/real_object",
            )
            visualization_markers_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
            real_object_visualizer = VisualizationMarkers(visualization_markers_cfg)

            visualization_markers_cfg = FRAME_MARKER_CFG.replace(
                prim_path="/Visuals/Command/motion_object",
            )
            visualization_markers_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
            motion_object_visualizer = VisualizationMarkers(visualization_markers_cfg)

            self.visualization_markers["real_object"] = real_object_visualizer
            self.visualization_markers["motion_object"] = motion_object_visualizer

    def _ensure_index_tensor(self, env_ids: torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

    def _get_index_of_a_in_b(self, a_names: List[str], b_names: List[str], device: str = "cpu") -> torch.Tensor:
        indexes = []
        for name in a_names:
            assert name in b_names, f"The specified name ({name}) doesn't exist: {b_names}"
            indexes.append(b_names.index(name))
        return torch.tensor(indexes, dtype=torch.long, device=device)
