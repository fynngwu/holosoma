"""Offline pre-computation: AMP motion .pkl files.

1. Load original .pkl (lab_dof_names order, ~30Hz)
2. Reorder DOF: lab_dof_names → gmr_dof_names
3. Compute velocities (dof_vel, root_lin_vel, root_ang_vel)
4. Compute body-frame data (key_body_pos_b, root_vel_b, root_ang_vel_b)
5. Compute frame_commands from velocities
6. Resample to target_fps (50Hz) via interpolation
7. Save all fields pre-computed — MotionDataManager loads ready-to-use data

Usage:
    source scripts/source_mujoco_uv_setup.sh
    python scripts/resample_amp_motion.py
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch

try:
    import mujoco as mj
except ImportError:
    mj = None

from holosoma.agents.amp.math import (
    ang_vel_from_quat_diff,
    quat_slerp,
    vel_forward_diff,
)

LAB_DOF_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]

GMR_DOF_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

DOF_REORDER = [LAB_DOF_NAMES.index(name) for name in GMR_DOF_NAMES]

# MuJoCo model for torso FK (loaded once in main)
_MJ_MODEL: mj.MjModel | None = None
_MJ_DATA: mj.MjData | None = None
_TORSO_BODY_ID: int | None = None


def init_mujoco(xml_path: str) -> tuple[mj.MjModel, mj.MjData, int]:
    """Load MuJoCo model and get torso body ID."""
    model = mj.MjModel.from_xml_path(xml_path)
    data = mj.MjData(model)
    torso_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "torso_link")
    return model, data, torso_id


def compute_torso_fk(
    root_pos: np.ndarray,
    root_rot: np.ndarray,
    dof_pos: np.ndarray,
    model: mj.MjModel,
    data: mj.MjData,
    torso_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run MuJoCo FK to get torso world position and rotation for all frames."""
    N = len(root_pos)
    torso_pos = np.zeros((N, 3), dtype=np.float64)
    torso_rot = np.zeros((N, 4), dtype=np.float64)  # [w,x,y,z]
    for i in range(N):
        data.qpos[:3] = root_pos[i]
        data.qpos[3:7] = root_rot[i]
        data.qpos[7:] = dof_pos[i]
        mj.mj_forward(model, data)
        torso_pos[i] = data.xpos[torso_id]
        torso_rot[i] = data.xquat[torso_id]
    return torso_pos, torso_rot


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return q * torch.tensor([1, -1, -1, -1], device=q.device)


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    qvec = q[..., 1:]
    uv = torch.linalg.cross(qvec, v, dim=-1)
    uuv = torch.linalg.cross(qvec, uv, dim=-1)
    return v + 2 * (q[..., :1] * uv + uuv)


def quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return quat_apply(quat_conjugate(q), v)


def process_one_motion(
    src_path: Path,
    dst_path: Path,
    target_fps: float,
    device: torch.device,
    model: mj.MjModel | None = None,
    data: mj.MjData | None = None,
    torso_id: int | None = None,
) -> None:
    with open(src_path, "rb") as f:
        raw = pickle.load(f)

    src_fps = float(np.asarray(raw["fps"]).reshape(-1)[0])
    src_dt = 1.0 / src_fps
    loop_mode = int(raw.get("loop_mode", 0))

    # Load raw data
    root_pos = torch.as_tensor(raw["root_pos"], device=device, dtype=torch.float)
    root_rot = torch.as_tensor(raw["root_rot"], device=device, dtype=torch.float)  # [w,x,y,z]
    dof_pos = torch.as_tensor(raw["dof_pos"], device=device, dtype=torch.float)
    key_body_pos = torch.as_tensor(raw["key_body_pos"], device=device, dtype=torch.float)  # (N, 6, 3)

    N = dof_pos.shape[0]
    num_kb = key_body_pos.shape[1]

    # Fix quaternion sign flips: ensure dot(q[i], q[i+1]) > 0
    # q and -q represent the same rotation, but sign flips cause
    # ang_vel_from_quat_diff to compute huge spurious velocities.
    for _ in range(3):
        flipped = False
        for i in range(1, N):
            if (root_rot[i - 1] * root_rot[i]).sum() < 0:
                root_rot[i] = -root_rot[i]
                flipped = True
        if not flipped:
            break

    # Step 1: DOF reorder (lab → gmr)
    dof_pos = dof_pos[:, DOF_REORDER]

    # Step 1b: Torso FK — compute torso world position and rotation
    # Torso is more stable than pelvis during gait, giving cleaner body-frame
    # velocities for frame_commands (especially vy, which oscillates ±0.5 in
    # pelvis frame but should be ~0 for straight-line motion).
    torso_pos = root_pos
    torso_rot = root_rot
    if model is not None and data is not None and torso_id is not None:
        torso_pos_np, torso_rot_np = compute_torso_fk(
            root_pos.cpu().numpy(),
            root_rot.cpu().numpy(),
            dof_pos.cpu().numpy(),
            model, data, torso_id,
        )
        torso_pos = torch.as_tensor(torso_pos_np, device=device, dtype=torch.float)
        torso_rot = torch.as_tensor(torso_rot_np, device=device, dtype=torch.float)

    # Step 2: Compute velocities in world frame (from raw fps)
    dof_vel = vel_forward_diff(dof_pos, src_dt)
    root_lin_vel_w = vel_forward_diff(root_pos, src_dt)
    root_ang_vel_w = ang_vel_from_quat_diff(root_rot, src_dt, in_frame="world")

    # Step 3: Body-frame transforms
    root_vel_b = quat_apply_inverse(root_rot, root_lin_vel_w)
    root_ang_vel_b = quat_apply_inverse(root_rot, root_ang_vel_w)
    root_rot_exp = root_rot.unsqueeze(1).expand(-1, num_kb, -1)
    key_body_pos_b = quat_apply_inverse(root_rot_exp, key_body_pos - root_pos.unsqueeze(1))

    # Step 4: Frame commands [vx, vy, wz]
    # vx, vy from TORSO body-frame linear velocity (smoothed).
    #   Torso is more stable than pelvis during gait — pelvis tilts/rotates
    #   with each step, creating false lateral velocity (±0.5 m/s) in
    #   pelvis body frame that doesn't reflect actual trajectory curvature.
    # wz from body yaw rate with net-change gating.
    #   Gait-cycle yaw oscillations (±10-15°) are periodic — net yaw change
    #   over one gait cycle is near zero. Sustained turning has large net change.
    #   By gating with a 15° threshold over a 1s window, we suppress gait
    #   oscillations while preserving real turns.

    # Torso body-frame velocity for vx/vy
    torso_lin_vel_w = vel_forward_diff(torso_pos, src_dt)
    torso_vel_t = quat_apply_inverse(torso_rot, torso_lin_vel_w)

    # Yaw from torso quaternion [w,x,y,z]
    qw, qx, qy, qz = torso_rot.unbind(-1)
    yaw = torch.atan2(2 * (qx * qy + qw * qz), qw * qw + qx * qx - qy * qy - qz * qz)
    # Unwrap
    for i in range(1, N):
        diff = yaw[i] - yaw[i - 1]
        if diff > torch.pi:
            yaw[i:] -= 2 * torch.pi
        elif diff < -torch.pi:
            yaw[i:] += 2 * torch.pi

    def smooth_tensor(x: torch.Tensor, window: int = 11) -> torch.Tensor:
        kernel = torch.ones(window, device=x.device, dtype=torch.float) / window
        pad = window // 2
        padded = torch.cat([x[:1].expand(pad, -1), x, x[-1:].expand(pad, -1)])
        return torch.nn.functional.conv1d(
            padded.T.unsqueeze(0), kernel.view(1, 1, -1), padding=0
        ).squeeze(0).T

    # Yaw rate from central diff, then smooth
    yaw_rate = torch.zeros(N, device=device, dtype=torch.float)
    yaw_rate[1:-1] = (yaw[2:] - yaw[:-2]) / (2 * src_dt)
    yaw_rate_smooth = smooth_tensor(yaw_rate.unsqueeze(-1), window=31)[:, 0]

    # Net yaw change over a 1s window (31 frames @ ~30Hz).
    # Gait oscillations cancel out (net < 15°); sustained turns pass (net > 15°).
    WINDOW = 31
    half = WINDOW // 2
    net_change = torch.zeros(N, device=device, dtype=torch.float)
    for i in range(half, N - half):
        net_change[i] = (yaw[i + half] - yaw[i - half]).abs()
    yaw_rate_smooth[net_change < (15.0 * torch.pi / 180.0)] = 0.0

    vx_smooth = smooth_tensor(torso_vel_t[:, 0:1], window=11)[:, 0]
    vy_smooth = smooth_tensor(torso_vel_t[:, 1:2], window=11)[:, 0]
    frame_commands = torch.stack([vx_smooth, vy_smooth, yaw_rate_smooth], dim=-1)

    # Step 5: Resample to target_fps
    target_dt = 1.0 / target_fps
    original_duration = N * src_dt
    new_N = int(round(original_duration * target_fps)) + 1
    new_times = torch.arange(new_N, device=device, dtype=torch.float) * target_dt
    new_times = new_times.clamp(max=original_duration)

    # Frame interpolation
    src_times = torch.arange(N, device=device, dtype=torch.float) * src_dt
    src_indices = torch.arange(N, device=device, dtype=torch.long)

    def lerp_1d(src: torch.Tensor) -> torch.Tensor:
        """Interpolate (N, D) → (new_N, D) using nearest-neighbor-lerp."""
        idx = (new_times / src_dt).long().clamp(max=N - 2)
        alpha = ((new_times - idx.float() * src_dt) / src_dt).unsqueeze(-1)
        return torch.lerp(src[idx], src[idx + 1], alpha)

    def slerp_quat(src: torch.Tensor) -> torch.Tensor:
        idx = (new_times / src_dt).long().clamp(max=N - 2)
        alpha = (new_times - idx.float() * src_dt) / src_dt
        return quat_slerp(src[idx], src[idx + 1], alpha)

    new_dof_pos = lerp_1d(dof_pos)
    new_dof_vel = lerp_1d(dof_vel)
    new_root_pos = lerp_1d(root_pos)
    new_root_rot = slerp_quat(root_rot)
    new_root_lin_vel_w = lerp_1d(root_lin_vel_w)
    new_root_ang_vel_w = lerp_1d(root_ang_vel_w)
    new_key_body_pos = lerp_1d(key_body_pos.view(N, -1)).view(new_N, num_kb, 3)
    new_key_body_pos_b = lerp_1d(key_body_pos_b.view(N, -1)).view(new_N, num_kb, 3)
    new_root_vel_b = lerp_1d(root_vel_b)
    new_root_ang_vel_b = lerp_1d(root_ang_vel_b)
    new_frame_commands = lerp_1d(frame_commands)

    # Step 6: Build output
    out = {
        "fps": target_fps,
        "dof_names": GMR_DOF_NAMES,
        "loop_mode": loop_mode,
        # World-frame (for GMR visualization)
        "root_pos": new_root_pos.cpu().numpy(),
        "root_rot": new_root_rot.cpu().numpy(),  # [w,x,y,z]
        "key_body_pos_w": new_key_body_pos.cpu().numpy(),
        # Pre-computed body-frame (for training)
        "dof_pos": new_dof_pos.cpu().numpy(),
        "dof_vel": new_dof_vel.cpu().numpy(),
        "root_vel_b": new_root_vel_b.cpu().numpy(),
        "root_ang_vel_b": new_root_ang_vel_b.cpu().numpy(),
        "key_body_pos_b": new_key_body_pos_b.cpu().numpy(),
        "frame_commands": new_frame_commands.cpu().numpy(),
    }

    with open(dst_path, "wb") as f:
        pickle.dump(out, f)

    print(f"  {src_path.stem}: {src_fps:.1f}Hz {N}f → {target_fps}Hz {new_N}f  →  {dst_path}")


def convert_for_gmr_viewer(src_pkl: str, dst_pkl: str) -> None:
    """Create GMR-viewer-compatible copy: root_rot [w,x,y,z] → [x,y,z,w]."""
    with open(src_pkl, "rb") as f:
        data = pickle.load(f)
    data["root_rot"] = data["root_rot"][:, [1, 2, 3, 0]]
    data["local_body_pos"] = None
    data["link_body_list"] = None
    with open(dst_pkl, "wb") as f:
        pickle.dump(data, f)
    print(f"  GMR viz → {dst_pkl}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="MotionData/g1_29dof/amp/walk_and_run")
    parser.add_argument("--dst", default="amp_data/g1_29dof/amp/walk_and_run")
    parser.add_argument("--target-fps", type=float, default=50.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    src_dir = Path(args.src)
    dst_dir = Path(args.dst)
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Initialize MuJoCo for torso FK
    model = data = torso_id = None
    if mj is not None:
        xml_path = "GMR/assets/unitree_g1/g1_mocap_29dof.xml"
        model, data, torso_id = init_mujoco(xml_path)
        print("  MuJoCo FK: enabled (torso body-frame for vx/vy)")
    else:
        print("  MuJoCo FK: disabled (using pelvis body-frame)")

    pkl_paths = sorted(src_dir.glob("*.pkl"))
    print(f"Processing {len(pkl_paths)} motions:")
    print(f"  DOF: lab_dof_names → gmr_dof_names")
    print(f"  FPS: ~30Hz → {args.target_fps}Hz")
    print(f"  Output: {dst_dir}")
    print()

    for src_path in pkl_paths:
        dst_path = dst_dir / f"{src_path.stem}.pkl"
        process_one_motion(src_path, dst_path, args.target_fps, device, model, data, torso_id)

    # GMR viewer copies
    print()
    gmrviz_dir = dst_dir.parent / (dst_dir.name + "_gmrviz")
    gmrviz_dir.mkdir(parents=True, exist_ok=True)
    for pkl_path in sorted(dst_dir.glob("*.pkl")):
        convert_for_gmr_viewer(str(pkl_path), str(gmrviz_dir / pkl_path.name))

    print(f"\nDone. Training data: {dst_dir}")
    print(f"  GMR visualization: {gmrviz_dir}")


if __name__ == "__main__":
    main()
