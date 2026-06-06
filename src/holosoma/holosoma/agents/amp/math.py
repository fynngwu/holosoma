"""Standalone math utilities for AMP, ported from legged_lab without Isaac Lab dependency."""

from __future__ import annotations

import torch


@torch.jit.script
def vel_forward_diff(data: torch.Tensor, dt: float) -> torch.Tensor:
    """Compute forward-difference velocities.

    Args:
        data: Input tensor of shape (N, dim).
        dt: Time step duration.

    Returns:
        Velocity tensor of shape (N, dim). Last row copies the second-last.
    """
    N = data.shape[0]
    vel = torch.zeros_like(data)
    vel[:-1] = (data[1:] - data[:-1]) / dt
    if N >= 2:
        vel[-1] = vel[-2]
    return vel


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Multiply two quaternions. Both in [w, x, y, z] format."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Conjugate of quaternion [w, x, y, z]."""
    return q * torch.tensor([1, -1, -1, -1], device=q.device)


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Apply rotation q to vector v. q is [w, x, y, z], v is [x, y, z]."""
    qvec = q[..., 1:]
    uv = torch.linalg.cross(qvec, v, dim=-1)
    uuv = torch.linalg.cross(qvec, uv, dim=-1)
    return v + 2 * (q[..., :1] * uv + uuv)


def quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Apply inverse of rotation q to vector v. q is [w, x, y, z]."""
    return quat_apply(quat_conjugate(q), v)


def axis_angle_from_quat(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion [w, x, y, z] to axis-angle vector [x, y, z].

    The magnitude of the output is the rotation angle in radians.
    """
    angle = 2 * torch.acos(q[..., :1].clamp(-1.0, 1.0))
    sin_half = torch.sin(angle / 2)
    axis = q[..., 1:] / sin_half.clamp(min=1e-8)
    return axis * angle


def matrix_from_quat(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix.

    Args:
        q: Quaternion tensor of shape (..., 4).

    Returns:
        Rotation matrix of shape (..., 3, 3).
    """
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def yaw_quat(q: torch.Tensor) -> torch.Tensor:
    """Extract yaw-only quaternion [w, x, y, z] from full quaternion.

    Zeroes out roll and pitch components.
    """
    rotm = matrix_from_quat(q)
    yaw = torch.atan2(rotm[..., 1, 0], rotm[..., 0, 0])
    cos_half = torch.cos(yaw / 2)
    sin_half = torch.sin(yaw / 2)
    zeros = torch.zeros_like(cos_half)
    return torch.stack([cos_half, zeros, zeros, sin_half], dim=-1)


@torch.jit.script
def ang_vel_from_quat_diff(quat: torch.Tensor, dt: float, in_frame: str = "body") -> torch.Tensor:
    """Compute angular velocity from quaternion differences.

    Args:
        quat: Quaternion tensor of shape (N, 4) in [w, x, y, z] format,
              representing rotation from world to body frame.
        dt: Time step duration.
        in_frame: "body" or "world" - the frame for the angular velocity.

    Returns:
        Angular velocity tensor of shape (N, 3).
    """
    N = quat.shape[0]
    ang_vel = torch.zeros((N, 3), dtype=torch.float32, device=quat.device)
    for i in range(N - 1):
        q1 = quat[i].unsqueeze(0)
        q2 = quat[i + 1].unsqueeze(0)
        diff_quat = quat_mul(quat_conjugate(q1), q2)
        diff_angle_axis = axis_angle_from_quat(diff_quat)
        if in_frame == "world":
            diff_angle_axis = quat_apply(q1, diff_angle_axis)
        ang_vel[i, :] = diff_angle_axis.squeeze() / dt
    if N >= 2:
        ang_vel[-1, :] = ang_vel[-2, :]
    return ang_vel


@torch.jit.script
def quat_slerp(q0: torch.Tensor, q1: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    """Spherical linear interpolation between two quaternions.

    Args:
        q0: Start quaternion (..., 4), scalar-first [w, x, y, z].
        q1: End quaternion (..., 4), scalar-first [w, x, y, z].
        blend: Interpolation factor (...,), 0.0 = q0, 1.0 = q1.

    Returns:
        Interpolated quaternion (..., 4).
    """
    blend = blend.unsqueeze(-1)
    cos_half_theta = (q0 * q1).sum(dim=-1, keepdim=True)
    neg_mask = cos_half_theta < 0
    q1_s = torch.where(neg_mask, -q1, q1)
    cos_half_theta = cos_half_theta.abs()
    half_theta = torch.acos(cos_half_theta.clamp(-1.0, 1.0))
    sin_half_theta = torch.sqrt((1.0 - cos_half_theta.square()).clamp(min=1e-8))
    ratio_a = torch.sin((1.0 - blend) * half_theta) / sin_half_theta
    ratio_b = torch.sin(blend * half_theta) / sin_half_theta
    result = ratio_a * q0 + ratio_b * q1_s
    near_zero = sin_half_theta < 0.001
    result = torch.where(near_zero, 0.5 * q0 + 0.5 * q1_s, result)
    near_one = cos_half_theta.abs() >= 1.0
    result = torch.where(near_one, q0, result)
    return result


@torch.jit.script
def calc_phase(times: torch.Tensor, duration: torch.Tensor, loop_mode: torch.Tensor) -> torch.Tensor:
    """Calculate motion playback phase given times and loop mode.

    Args:
        times: Query times in seconds.
        duration: Motion duration in seconds.
        loop_mode: 0 = CLAMP, 1 = WRAP.

    Returns:
        Phase in [0, 1].
    """
    phase = times / duration
    wrap_mask = loop_mode == 1
    phase = torch.where(wrap_mask, phase - torch.floor(phase), phase)
    return phase.clamp(0.0, 1.0)
