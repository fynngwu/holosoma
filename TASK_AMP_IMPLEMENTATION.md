# AMP Training Implementation Task

## Overview

Integrate AMP (Adversarial Motion Priors) training into the FastSAC agent for the locomotion task. The implementation creates a new `FastSACAMPAgent` subclass with a discriminator network, motion data loading, category-based command allocation, and style reward computation.

PRD: `docs/prd/amp-training.md`

---

## Files to Create

### 1. `src/holosoma/holosoma/agents/amp/discriminator.py` — AMP Discriminator Module

**Reference**: `legged_lab/rsl_rl/rsl_rl/modules/amp.py` (AMPDiscriminator class)

A standalone discriminator module adapted from legged_lab, without RSL-RL dependency. It should:

- Take 160-dim input (80-dim per frame × 2 history steps)
- Use the holosoma `EmpiricalNormalization` (from `fast_sac_utils.py`)
- Use LSGAN loss by default
- Support gradient penalty computation

```python
class LossType(Enum):
    GAN = 0
    LSGAN = 1
    WGAN = 2

class AMPDiscriminator(nn.Module):
    def __init__(self, disc_obs_dim=80, disc_obs_steps=2, loss_type=LossType.LSGAN,
                 hidden_dims=[256, 256, 256], activation="elu", device="cpu"):
        # disc_obs_dim=80, disc_obs_steps=2 → input=160
        # disc_obs_normalizer: EmpiricalNormalization(shape=80)
        # Network: Linear(160,256)+ELU → Linear(256,256)+ELU → Linear(256,256)+ELU → Linear(256,1)
    
    def forward(self, x) -> torch.Tensor  # x: (N, 160) → score: (N, 1)
    def compute_grad_penalty(self, demo_data, scale=10.0)  # gradient penalty on demo
    def predict_style_reward(self, disc_obs_160, dt)  # LSGAN reward computation
    def normalize_disc_obs(self, disc_obs_160)
    def update_normalization(self, disc_obs_160)
```

**Key details**:
- The normalizer normalizes at the **per-frame** level (80-dim), not the stacked 160-dim level
- `predict_style_reward` is called with `torch.no_grad()` during rollout
- LSGAN reward formula: `clamp(1 - 0.25 * (score - 1)^2, min=0)`
- Style reward = `dt * 5.0 * lsgan_reward` (the 5.0 is the style_reward_scale)

### 2. `src/holosoma/holosoma/agents/fast_sac/fast_sac_amp_agent.py` — FastSACAMP Agent

**Inherits from**: `FastSACAgent` (in `fast_sac_agent.py`)

A subclass that overrides `setup()` and `learn()` to add AMP functionality.

#### Setup additions (in `setup()` override):

1. Initialize `MotionDataManager` with `MotionDataConfig(motion_dir="amp_data/g1_29dof/amp/walk", history=2)`
2. Initialize `AMPDiscriminator(disc_obs_dim=80, disc_obs_steps=2, ...)`
3. Create `disc_optimizer` (Adam, lr=1e-5, separate parameters for trunk and linear with weight_decay)
4. Add `prev_disc_obs` buffer `(num_envs, 80)` for rolling history
5. Create `category_ids` buffer `(num_envs,)` with fixed category allocation
6. Modify the `SimpleReplayBuffer` or create subclass to store disc_obs

#### SimpleReplayBuffer changes (in `fast_sac_utils.py`):

Add `disc_obs` field to `SimpleReplayBuffer`:
- `__init__`: add `n_disc_obs=160` parameter, create `self.disc_obs = zeros(n_env, buffer_size, n_disc_obs)`
- `extend()`: extract `disc_obs` from tensor_dict and store at ptr
- `sample()`: include `disc_obs` in the returned TensorDict

#### Rollout loop changes (in `learn()` override):

After each env step, construct 160-dim disc_obs from raw sim state:

```python
def _compute_disc_obs(self) -> torch.Tensor:
    """Build 160-dim discriminator obs from simulator state."""
    sim = self.unwrapped_env.simulator
    
    # 1. root_ang_vel_b (3) — already in body frame
    root_ang_vel = sim.base_ang_vel  # (num_envs, 3)
    
    # 2. dof_pos (29) — reorder to GMR order
    dof_pos = sim.dof_pos[:, self.dof_reorder_idx]  # (num_envs, 29)
    
    # 3. dof_vel (29) — reorder to GMR order
    dof_vel = sim.dof_vel[:, self.dof_reorder_idx]  # (num_envs, 29)
    
    # 4. key_body_pos_b (18) — select 6 bodies, convert to body frame
    # key_body_indices maps: left_ankle_roll_link, right_ankle_roll_link,
    #   left_wrist_yaw_link, right_wrist_yaw_link,
    #   left_shoulder_roll_link, right_shoulder_roll_link
    body_pos_w = sim._rigid_body_pos[:, self.key_body_indices, :]  # (num_envs, 6, 3)
    root_pos = sim.robot_root_states[:, :3]  # (num_envs, 3)
    root_quat = sim.base_quat  # (num_envs, 4)
    body_pos_b = quat_apply_inverse(
        root_quat.unsqueeze(1).expand(-1, 6, -1),
        body_pos_w - root_pos.unsqueeze(1)
    )  # (num_envs, 6, 3)
    key_body_pos_b = body_pos_b.reshape(num_envs, -1)  # (num_envs, 18)
    
    # 5. root_height (1)
    root_height = root_pos[:, 2:3]  # (num_envs, 1)
    
    # Concatenate: (num_envs, 80)
    disc_obs_80 = torch.cat([root_ang_vel, dof_pos, dof_vel, key_body_pos_b, root_height], dim=1)
    
    # Stack with previous: (num_envs, 160)
    disc_obs_160 = torch.cat([self.prev_disc_obs, disc_obs_80], dim=1)
    
    # Update prev
    self.prev_disc_obs = disc_obs_80
    
    return disc_obs_160
```

**On done/reset**: `self.prev_disc_obs[env_ids] = disc_obs_80[env_ids]`

#### Style reward injection:

After `env.step()`:
```python
with torch.no_grad(), self._maybe_amp():
    style_rew = self.amp_discriminator.predict_style_reward(disc_obs_160_cloned, dt=self.env.dt)
    rewards = rewards + style_rew
```
**Important**: The style reward should be included in `rewards` stored in the SAC replay buffer.

#### Discriminator training (added alongside actor update):

```python
def _update_disc(self, disc_obs_batch, disc_demo_obs_batch):
    # disc_obs_batch: from replay buffer (policy rollouts)
    # disc_demo_obs_batch: from MotionDataManager.sample_demo_obs()
    
    # Normalize
    normed_policy = self.amp_discriminator.normalize_disc_obs(disc_obs_batch)
    normed_demo = self.amp_discriminator.normalize_disc_obs(disc_demo_obs_batch)
    
    # Forward
    policy_score = self.amp_discriminator(normed_policy.reshape(...))  # (N, 1)
    demo_score = self.amp_discriminator(normed_demo.reshape(...))  # (N, 1)
    
    # LSGAN loss
    disc_loss = 0.5 * (torch.mean((policy_score) ** 2) + torch.mean((demo_score - 1) ** 2))
    
    # Gradient penalty
    grad_penalty = self.amp_discriminator.compute_grad_penalty(normed_demo.reshape(...), scale=10.0)
    
    total_loss = disc_loss + grad_penalty
    # Optimize discriminator
```

Training schedule:
- Called after each `update_pol()`, same number of times
- Current config: `num_updates=8, policy_frequency=4` → actor updates 2x → discriminator updates 2x

### 3. `src/holosoma/holosoma/config_types/algo.py` — Add FastSACAMPConfig

Add a new config dataclass:

```python
@dataclass(frozen=True)
class FastSACAMPConfig(FastSACConfig):
    """FastSAC with AMP (Adversarial Motion Priors)."""
    # Inherits all FastSACConfig fields
    
    # AMP specific fields
    amp_disc_obs_dim: int = 80
    amp_disc_obs_steps: int = 2
    amp_hidden_dims: list[int] = field(default_factory=lambda: [256, 256, 256])
    amp_activation: str = "elu"
    amp_learning_rate: float = 1e-5
    amp_grad_penalty_scale: float = 10.0
    amp_loss_type: str = "LSGAN"
    amp_style_reward_scale: float = 5.0
    amp_weight_decay: float = 1e-4
    amp_disc_obs_key: str = "disc_obs"
    amp_motion_dir: str = "amp_data/g1_29dof/amp/walk"
    # Category ratios
    amp_category_ratios: list[float] = field(default_factory=lambda: [0.3, 0.3, 0.1, 0.2, 0.1])
    amp_command_resample_time: float = 7.0

@dataclass(frozen=True)
class FastSACAMPAlgoConfig(FastSACAlgoConfig):
    _target_: str = "holosoma.agents.fast_sac.fast_sac_amp_agent.FastSACAMPAgent"
    config: FastSACAMPConfig = field(default_factory=FastSACAMPConfig)
```

Also add the Union type:
```python
AlgoConfig = Union[PPOAlgoConfig, FastSACAlgoConfig, FastSACAMPAlgoConfig]
```

### 4. `src/holosoma/holosoma/config_values/algo.py` — Add FastSACAMP Default Config

Add a new default config value:

```python
fast_sac_amp = FastSACAMPAlgoConfig(
    _target_="holosoma.agents.fast_sac.fast_sac_amp_agent.FastSACAMPAgent",
    _recursive_=False,
    config=FastSACAMPConfig(
        # Inherit from fast_sac but override AMP-specific fields
        num_learning_iterations=50000,
        # ... same as fast_sac ...
        amp=True,  # mixed precision still on
        amp_dtype="bf16",
        # AMP specific overrides if needed
    ),
)

DEFAULTS = {
    "ppo": ppo,
    "fast_sac": fast_sac,
    "fast_sac_amp": fast_sac_amp,  # ADD THIS
}
```

### 5. `src/holosoma/holosoma/config_values/loco/g1/experiment.py` — Add G1 FastSACAMP Experiment

```python
g1_29dof_fast_sac_amp = ExperimentConfig(
    env_class="holosoma.envs.locomotion.locomotion_manager.LeggedRobotLocomotionManager",
    training=TrainingConfig(project="hv-g1-manager", name="g1_29dof_fast_sac_amp_manager"),
    algo=replace(algo.fast_sac_amp, config=replace(
        algo.fast_sac_amp.config,
        num_learning_iterations=100000,
        use_symmetry=True,
    )),
    # ... rest same as g1_29dof_fast_sac ...
)

__all__ = ["g1_29dof", "g1_29dof_fast_sac", "g1_29dof_fast_sac_amp"]
```

### 6. `src/holosoma/holosoma/config_values/experiment.py` — Register Experiment

Add the new experiment to `DEFAULTS`:

```python
from holosoma.config_values.loco.g1.experiment import (
    g1_29dof, g1_29dof_fast_sac, g1_29dof_fast_sac_amp
)
DEFAULTS = {
    # ... existing ...
    "g1_29dof_fast_sac_amp": g1_29dof_fast_sac_amp,
}
```

---

## Files to Modify

### 1. `src/holosoma/holosoma/agents/fast_sac/fast_sac_utils.py` — SimpleReplayBuffer

**Changes**:
- `__init__`: Add `n_disc_obs: int = 0` parameter (0 = disabled). If > 0, create `self.disc_obs` tensor of shape `(n_env, buffer_size, n_disc_obs)`.
- `extend()`: Look for `"disc_obs"` key in the tensor_dict and store it at ptr.
- `sample()`: If disc_obs is stored, include `"disc_obs"` in the returned TensorDict.

This is a backwards-compatible change — when `n_disc_obs=0` (default), the buffer behaves exactly as before.

### 2. `src/holosoma/holosoma/agents/fast_sac/fast_sac.py` — No changes needed

Actor and Critic networks remain the same.

---

## Category-Based Command System

The `LocomotionCommand` in `managers/command/terms/locomotion.py` needs an AMP-aware variant.

Strategy: In `FastSACAMPAgent`, after the env's standard command is sampled, override the commands tensor with category-based commands. Alternatively, create a new `AMPCommand` term class.

**Simplest approach**: In `FastSACAMPAgent.setup()`, directly set `self.env.simulator.commands` tensor with category-based sampling logic, and in `learn()` rollout, resample commands every 7s.

**Category allocation**:
```python
num_envs = env.num_envs
ratios = [0.3, 0.3, 0.1, 0.2, 0.1]
# Deterministic split: first 30% → cat0, next 30% → cat1, etc.
counts = [int(r * num_envs) for r in ratios]
# Adjust last category to match num_envs exactly
counts[-1] = num_envs - sum(counts[:-1])
category_ids = torch.cat([torch.full((c,), i) for i, c in enumerate(counts)])
```

**Command sampling per category**:
```python
def _sample_amp_commands(self, env_ids, category_ids):
    commands = torch.zeros(len(env_ids), 3, device=self.device)
    for cat_id in range(5):
        mask = category_ids[env_ids] == cat_id
        n = mask.sum()
        if n == 0: continue
        if cat_id == 0:  # 直走
            commands[mask, 0] = uniform(0.5, 1.5, (n,))    # vx
            commands[mask, 1] = 0                            # vy
            commands[mask, 2] = uniform(-0.15, 0.15, (n,))  # wz
        elif cat_id == 1:  # 前进+转弯
            vx = torch.cat([uniform(0.3, 1.5, (n//2+1,)), uniform(-1.5, -0.3, (n//2+1,))])[:n]
            commands[mask, 0] = vx
            commands[mask, 1] = 0
            # wz: sample from [-2, -0.3] ∪ [0.3, 2]
            wz = torch.cat([uniform(0.3, 2.0, (n//2+1,)), uniform(-2.0, -0.3, (n//2+1,))])[:n]
            commands[mask, 2] = wz
        elif cat_id == 2:  # 原地转
            commands[mask, 0] = 0
            commands[mask, 1] = 0
            wz = torch.cat([uniform(0.3, 2.0, (n//2+1,)), uniform(-2.0, -0.3, (n//2+1,))])[:n]
            commands[mask, 2] = wz
        elif cat_id == 3:  # 侧移
            commands[mask, 0] = 0
            vy = torch.cat([uniform(0.3, 1.0, (n//2+1,)), uniform(-1.0, -0.3, (n//2+1,))])[:n]
            commands[mask, 1] = vy
            commands[mask, 2] = uniform(-0.15, 0.15, (n,))
        elif cat_id == 4:  # 静止
            commands[mask] = 0
    return commands
```

---

## DOF Reorder Mapping

From the `resample_amp_motion.py` script, the GMR DOF order is:

```python
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
```

**Create mapping** in `FastSACAMPAgent.setup()`:

```python
robot_dof_names = self.env.robot_config.dof_names  # URDF order
gmr_dof_names = [ ... ]  # from above
self.dof_reorder_idx = torch.tensor(
    [robot_dof_names.index(name) for name in gmr_dof_names],
    dtype=torch.long, device=self.device
)
```

---

## Key Body Index Mapping

The 6 key bodies used in legged_lab AMP:
```python
KEY_BODY_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
]
```

In `FastSACAMPAgent.setup()`, find these body indices:
```python
sim = self.unwrapped_env.simulator
body_names = sim.body_names  # list of str, URDF body order
self.key_body_indices = torch.tensor(
    [body_names.index(name) for name in KEY_BODY_NAMES],
    dtype=torch.long, device=self.device
)
```

---

## Training Loop Pseudocode

```python
def learn(self):
    args = self.config
    super().learn() would not work — need to override completely
    # Instead, copy learn() from FastSACAgent and add AMP logic
    
    # Setup
    obs, critic_obs = env.reset_with_critic_obs()
    self.prev_disc_obs = torch.zeros(num_envs, 80, device=device)
    # Set initial commands based on categories
    self._resample_amp_commands(torch.arange(num_envs, device=device), self.category_ids)
    
    while global_step <= args.num_learning_iterations:
        # Rollout
        with torch.no_grad(), self._maybe_amp():
            # ... standard SAC rollout ...
            actions = policy(norm_obs, dones)
            next_obs, rewards, dones, infos = env.step(actions)
            
            # Compute disc_obs from sim state
            disc_obs_160 = self._compute_disc_obs()  # (num_envs, 160)
            
            # Compute style reward
            style_rew = self.amp_discriminator.predict_style_reward(
                disc_obs_160.unsqueeze(1).reshape(num_envs, 1, 80, 2)...,
                dt=self.env.dt
            )
            rewards = rewards + style_rew
            
            # Handle dones: fill prev_disc_obs with current for reset envs
            self.prev_disc_obs[dones.bool()] = disc_obs_80[dones.bool()]
            
            # Store transition with disc_obs
            transition["disc_obs"] = disc_obs_160
            rb.extend(transition)
        
        # Resample commands every 7s
        self._resample_amp_commands_if_needed()
        
        # Training
        if global_step > learning_starts:
            # Sample and prepare batches (includes disc_obs from buffer)
            prepared_batches = self._sample_and_prepare_batches(...)
            
            for i, data in enumerate(prepared_batches):
                # SAC critic update
                update_main(data)
                
                if i % policy_frequency == 1:
                    # SAC actor update
                    update_pol(data)
                    
                    # AMP discriminator update (×2)
                    for _ in range(2):
                        # Sample disc_obs from buffer & demo from MotionDataManager
                        disc_batch = data["disc_obs"]
                        demo_batch = self.motion_data_manager.sample_demo_obs(disc_batch.shape[0])
                        self._update_disc(disc_batch, demo_batch)
        
        global_step += 1
```

---

## Reference Links

- **AMPDiscriminator (legged_lab)**: `legged_lab/rsl_rl/rsl_rl/modules/amp.py`
- **PPOAMP training (legged_lab)**: `legged_lab/rsl_rl/rsl_rl/algorithms/ppo_amp.py` (lines ~315-407 for AMP loss)
- **AMP env config**: `legged_lab/source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1/g1_amp_env_cfg.py`
- **MotionDataManager**: `src/holosoma/holosoma/agents/amp/motion_data.py`
- **FastSACAgent**: `src/holosoma/holosoma/agents/fast_sac/fast_sac_agent.py`
- **SimpleReplayBuffer**: `src/holosoma/holosoma/agents/fast_sac/fast_sac_utils.py`
- **FastSACConfig**: `src/holosoma/holosoma/config_types/algo.py`
- **Algo defaults**: `src/holosoma/holosoma/config_values/algo.py`
- **Experiment configs**: `src/holosoma/holosoma/config_values/loco/g1/experiment.py`
- **Experiment defaults**: `src/holosoma/holosoma/config_values/experiment.py`
- **GMR DOF names**: `scripts/resample_amp_motion.py` (lines 51-61)
- **AMP math utils**: `src/holosoma/holosoma/agents/amp/math.py`
- **PRD**: `docs/prd/amp-training.md`

---

## Implementation Order

1. `discriminator.py` — AMPDiscriminator module
2. `fast_sac_utils.py` — SimpleReplayBuffer disc_obs support
3. `algo.py` (config_types) — FastSACAMPConfig
4. `algo.py` (config_values) — fast_sac_amp default
5. `fast_sac_amp_agent.py` — Main agent subclass
6. `loco/g1/experiment.py` — Experiment config
7. `experiment.py` (config_values) — Register experiment

## pkl Data Note

The existing `.pkl` files in `amp_data/g1_29dof/amp/walk/` do NOT contain body name metadata. The current `MotionDataManager` uses `key_body_pos_b` with shape `(N, 6, 3)` but doesn't know which bodies they are.

**Two options**:
- **Option A** (simpler): Accept that the 6 key body indices are hardcoded from legged_lab (`KEY_BODY_NAMES`) and assume the pkl's 6 bodies match
- **Option B**: Regenerate pkl files with explicit `key_body_names` metadata

The PRD states Option B (regenerate), but Option A can be used initially for development.
