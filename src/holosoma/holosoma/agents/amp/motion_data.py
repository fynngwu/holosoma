"""Motion data manager for AMP — loads pre-computed .pkl files and samples.

All computation (DOF reorder, 50Hz resample, body-frame transform, frame commands)
is done offline by scripts/resample_amp_motion.py. This class only loads and samples.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# Category classification thresholds
_TH_VX_LOW = 0.15
_TH_VY = 0.15
_TH_WZ = 0.15
_TH_VX_HIGH = 0.5
_TH_WZ_TURN = 0.3

CATEGORY_NAMES = ["直走", "前进+转弯", "原地转", "侧移", "静止"]


def classify_command(vx: float, vy: float, wz: float) -> int:
    """Classify a command into one of 5 categories. Returns category index."""
    if vx > _TH_VX_HIGH and abs(wz) > _TH_WZ_TURN:
        return 1  # 前进+转弯
    if abs(vx) < _TH_VX_HIGH and abs(vy) < _TH_VX_HIGH and abs(wz) > _TH_WZ_TURN:
        return 2  # 原地转
    if abs(vx) > _TH_VX_LOW and abs(vy) < _TH_VY and abs(wz) < _TH_WZ:
        return 0  # 直走
    if abs(vy) > _TH_VY and abs(vx) < _TH_VX_LOW and abs(wz) < _TH_WZ:
        return 3  # 侧移
    if abs(vx) < _TH_VX_LOW and abs(vy) < _TH_VY and abs(wz) < _TH_WZ:
        return 4  # 静止
    # Fallback: nearest category by L2 distance to prototypes
    prototypes = torch.tensor([
        [0.5, 0.0, 0.0],     # 直走
        [0.9, 0.0, 0.6],     # 前进+转弯
        [0.0, 0.0, -0.6],    # 原地转
        [0.0, 0.5, 0.0],     # 侧移
        [0.0, 0.0, 0.0],     # 静止
    ])
    dist = ((prototypes - torch.tensor([vx, vy, wz])) ** 2).sum(dim=-1)
    return int(dist.argmin())


@dataclass
class MotionDataConfig:
    motion_dir: str = "amp_data/g1_29dof/amp/walk"
    motion_weights: dict[str, float] | None = None
    history: int = 2


class MotionDataManager:
    def __init__(
        self,
        cfg: MotionDataConfig,
        device: str | torch.device = "cpu",
    ):
        self.cfg = cfg
        self.device = torch.device(device)
        self.history = cfg.history
        self._load_all()
        self._build_categories()

    # ------------------------------------------------------------------ #
    # Data loading
    # ------------------------------------------------------------------ #

    def _load_all(self) -> None:
        pkl_dir = Path(self.cfg.motion_dir)
        if not pkl_dir.exists():
            raise FileNotFoundError(f"Motion data directory not found: {pkl_dir}")
        pkl_paths = sorted(pkl_dir.glob("*.pkl"))
        if not pkl_paths:
            raise FileNotFoundError(f"No .pkl files found in {pkl_dir}")

        self.motion_names: list[str] = []
        self._dof_pos: list[torch.Tensor] = []
        self._dof_vel: list[torch.Tensor] = []
        self._root_vel_b: list[torch.Tensor] = []
        self._root_ang_vel_b: list[torch.Tensor] = []
        self._key_body_pos_b: list[torch.Tensor] = []
        self._root_pos: list[torch.Tensor] = []
        self._frame_commands: list[torch.Tensor] = []
        self._num_frames: list[int] = []

        cfg = self.cfg
        for path in sorted(pkl_paths):
            with open(path, "rb") as f:
                raw = pickle.load(f)

            name = path.stem
            n = raw["dof_pos"].shape[0]

            self.motion_names.append(name)
            self._dof_pos.append(torch.as_tensor(raw["dof_pos"], device=self.device, dtype=torch.float))
            self._dof_vel.append(torch.as_tensor(raw["dof_vel"], device=self.device, dtype=torch.float))
            self._root_vel_b.append(torch.as_tensor(raw["root_vel_b"], device=self.device, dtype=torch.float))
            self._root_ang_vel_b.append(torch.as_tensor(raw["root_ang_vel_b"], device=self.device, dtype=torch.float))
            self._key_body_pos_b.append(torch.as_tensor(raw["key_body_pos_b"], device=self.device, dtype=torch.float))
            self._root_pos.append(torch.as_tensor(raw["root_pos"], device=self.device, dtype=torch.float))
            self._num_frames.append(n)

            if "frame_commands" in raw:
                self._frame_commands.append(torch.as_tensor(raw["frame_commands"], device=self.device, dtype=torch.float))
            else:
                fc = torch.zeros((n, 3), device=self.device, dtype=torch.float)
                fc[:, :2] = self._root_vel_b[-1][:, :2]
                fc[:, 2] = self._root_ang_vel_b[-1][:, 2]
                self._frame_commands.append(fc)

        self.num_dofs = self._dof_pos[0].shape[1]
        self.num_key_bodies = self._key_body_pos_b[0].shape[1]
        self.num_motions = len(self.motion_names)

    # ------------------------------------------------------------------ #
    # Category bucketing
    # ------------------------------------------------------------------ #

    def _build_categories(self) -> None:
        """Classify all consecutive 2-frame pairs into 5 motion categories.

        Each pair (i, i+1) is represented by the first frame's command.
        Stores (motion_id, frame_id) per category for fast sampling.
        """
        h = self.history
        self._cat_pairs: list[list[tuple[int, int]]] = [[] for _ in range(5)]
        self._cat_pair_count: list[int] = [0] * 5

        for mid in range(self.num_motions):
            nf = self._num_frames[mid]
            if nf < h:
                continue
            cmds = self._frame_commands[mid]
            for i in range(nf - h + 1):
                vx, vy, wz = cmds[i].tolist()
                cat = classify_command(vx, vy, wz)
                self._cat_pairs[cat].append((mid, i))

        self._cat_pair_count = [len(p) for p in self._cat_pairs]
        self._cat_total = sum(self._cat_pair_count)
        self._cat_probs = torch.tensor(
            [c / self._cat_total for c in self._cat_pair_count],
            device=self.device, dtype=torch.float,
        )

        # Pre-build index tensors per category for fast sampling
        self._cat_motion_ids: list[torch.Tensor] = []
        self._cat_frame_starts: list[torch.Tensor] = []
        for cat in range(5):
            pairs = self._cat_pairs[cat]
            if pairs:
                mids, starts = zip(*pairs)
                self._cat_motion_ids.append(torch.tensor(mids, device=self.device, dtype=torch.long))
                self._cat_frame_starts.append(torch.tensor(starts, device=self.device, dtype=torch.long))
            else:
                self._cat_motion_ids.append(torch.empty(0, device=self.device, dtype=torch.long))
                self._cat_frame_starts.append(torch.empty(0, device=self.device, dtype=torch.long))

    # ------------------------------------------------------------------ #
    # Public sampling API
    # ------------------------------------------------------------------ #

    def sample_categories(self, num_samples: int) -> torch.Tensor:
        """Sample category indices by their natural proportions.

        Returns:
            Tensor (num_samples,) of category indices [0..4].
        """
        return torch.multinomial(self._cat_probs, num_samples, replacement=True)

    def sample_commands(self, num_samples: int) -> torch.Tensor:
        """Sample training commands from the data distribution.

        Picks a random pair from the pool and returns its frame_command.
        This ensures commands always match real demo data.

        Returns:
            Tensor (num_samples, 3) as [vx, vy, wz].
        """
        idx = torch.randint(self._cat_total, (num_samples,), device=self.device)
        return self._command_at_flat_index(idx)

    def sample_commands_by_category(self, categories: torch.Tensor) -> torch.Tensor:
        """Sample commands for specific categories.

        Args:
            categories: (num_samples,) category index per sample.

        Returns:
            Tensor (num_samples, 3) as [vx, vy, wz].
        """
        B = categories.shape[0]
        cmds = torch.zeros((B, 3), device=self.device, dtype=torch.float)
        for cat in range(5):
            mask = categories == cat
            n = int(mask.sum())
            if n == 0:
                continue
            idx = torch.randint(len(self._cat_pairs[cat]), (n,), device=self.device)
            for j, global_j in enumerate(mask.nonzero().squeeze(-1)):
                mi = self._cat_motion_ids[cat][idx[j]].item()
                si = self._cat_frame_starts[cat][idx[j]].item()
                cmds[global_j] = self._frame_commands[mi][si]
        return cmds

    def sample_demo_obs(
        self, num_samples: int,
        categories: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample discriminator observations from demo data.

        Args:
            num_samples: Number of samples.
            categories: Optional (num_samples,) category indices.
                        If None, categories are sampled by natural proportion.

        Returns:
            Tensor (num_samples, 80 * history). With history=2 → 160-dim.
        """
        h = self.history
        if categories is not None:
            cat_idx = categories
        else:
            cat_idx = self.sample_categories(num_samples)

        motion_ids = torch.empty(num_samples, device=self.device, dtype=torch.long)
        frame_starts = torch.empty(num_samples, device=self.device, dtype=torch.long)

        for cat in range(5):
            mask = cat_idx == cat
            n = int(mask.sum())
            if n == 0:
                continue
            pick = torch.randint(len(self._cat_pairs[cat]), (n,), device=self.device)
            motion_ids[mask] = self._cat_motion_ids[cat][pick]
            frame_starts[mask] = self._cat_frame_starts[cat][pick]

        frame_ids = frame_starts.unsqueeze(-1) + torch.arange(h, device=self.device)
        return self._build_obs(motion_ids, frame_ids)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _command_at_flat_index(self, idx: torch.Tensor) -> torch.Tensor:
        """Map flat index to frame_command."""
        cumsum = torch.tensor(self._num_frames, device=self.device, dtype=torch.long)
        cumsum = torch.cat([torch.zeros(1, device=self.device, dtype=torch.long), cumsum.cumsum(0)])
        motion_ids = torch.searchsorted(cumsum[1:], idx, right=True)
        local = idx - cumsum[motion_ids]
        out = torch.empty((idx.shape[0], 3), device=self.device, dtype=torch.float)
        for mid in range(self.num_motions):
            mask = motion_ids == mid
            if not mask.any():
                continue
            out[mask] = self._frame_commands[mid][local[mask]]
        return out

    def _build_obs(
        self, motion_ids: torch.Tensor, frame_ids: torch.Tensor
    ) -> torch.Tensor:
        """Build 160-dim discriminator obs from (motion_id, frame_id) pairs."""
        B, H = frame_ids.shape
        obs_list = []
        for key_list, key_name in [
            (self._root_ang_vel_b, "root_ang_vel"),
            (self._dof_pos, "dof_pos"),
            (self._dof_vel, "dof_vel"),
            (self._key_body_pos_b, "key_body"),
            (self._root_pos, "root_height"),
        ]:
            feat_dim = 18 if key_name == "key_body" else (1 if key_name == "root_height" else key_list[0].shape[-1])
            out = torch.empty((B, H, feat_dim), device=self.device, dtype=torch.float)
            for mid in range(self.num_motions):
                mask = motion_ids == mid
                if not mask.any():
                    continue
                gathered = key_list[mid][frame_ids[mask]]
                if key_name == "key_body":
                    gathered = gathered.reshape(gathered.shape[0], gathered.shape[1], -1)
                elif key_name == "root_height":
                    gathered = gathered[..., 2:3]
                out[mask] = gathered
            obs_list.append(out)

        obs_per_frame = torch.cat(obs_list, dim=-1)
        return obs_per_frame.reshape(B, -1)

    # ------------------------------------------------------------------ #
    # Info
    # ------------------------------------------------------------------ #

    @property
    def category_info(self) -> dict[str, dict]:
        """Return category statistics for logging."""
        info = {}
        for cat in range(5):
            info[CATEGORY_NAMES[cat]] = {
                "count": self._cat_pair_count[cat],
                "ratio": self._cat_pair_count[cat] / max(self._cat_total, 1),
            }
        return info
