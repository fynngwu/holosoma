# PRD: AMP (Adversarial Motion Priors) Training Integration for FastSAC

## Problem Statement

当前 FastSAC agent 缺乏运动先验能力，训练出的 locomotion 策略虽然能跟踪速度指令，但运动风格僵硬、不自然。需要引入 AMP (Adversarial Motion Priors) 对抗式运动先验训练，让策略学习更自然、类人的运动风格。

AMP 通过一个 **discriminator** 来区分"策略产生的运动"和"真实运动数据（demo）"，策略在优化任务奖励的同时，也受到 discriminator 的"风格奖励"约束，从而生成既完成任务又风格自然的运动。

## Solution

在 FastSAC agent 的基础上新建 `FastSACAMPAgent` 子类，集成 AMP 训练循环：

1. **MotionDataManager**: 加载预处理的 `.pkl` 运动数据，提供 discriminator 的 demo 正样本
2. **AMPDiscriminator**: MLP 网络，区分 demo 和 policy rollout 的运动
3. **Style Reward**: discriminator 输出作为额外奖励（权重 5.0），直接加到任务奖励上
4. **类别化命令分配**: 将 envs 按 5 个运动类别固定分配，每 7s 同类内重采样命令

## User Stories

1. As a **training system**, I want to load pre-processed motion demo data from `.pkl` files, so that the discriminator has positive examples of natural motion.
2. As a **training system**, I want to construct 160-dim discriminator observations from the simulator state during rollout, so that the discriminator can compare policy motions to demo motions.
3. As a **training system**, I want to train an AMP discriminator alongside the SAC actor/critic, so that the policy receives style rewards that encourage natural motion.
4. As a **training system**, I want the style reward to be added directly to the task reward with weight 5.0, so that total_reward = task_reward + 5.0 * style_reward.
5. As a **training system**, I want the discriminator to use LSGAN loss with gradient penalty, so that training is stable.
6. As a **training system**, I want envs to be partitioned into 5 command categories with fixed ratios (0.3/0.3/0.1/0.2/0.1), so that the policy is trained evenly across different motion types.
7. As a **training system**, I want commands to be resampled every 7 seconds within each category's range, so that the policy learns to respond to varying commands within the same motion type.
8. As a **training system**, I want the discriminator observation to be stored in the SAC replay buffer, so that training data for the discriminator is sampled alongside SAC transitions.
9. As a **training system**, I want the discriminator to update at the same frequency as the actor (2 times per global step with current config), so that the discriminator and policy co-adapt.
10. As a **training system**, I want the DOF order between pkl motion data and simulator to be unified, so that discriminator observations from both sources are directly comparable.

## Implementation Decisions

### 1. Code Organization: New `FastSACAMPAgent` Subclass

新建 `FastSACAMPAgent(FastSACAgent)` 子类，在 `agents/fast_sac/fast_sac_amp_agent.py` 中实现。不修改现有的 `FastSACAgent` 代码。

子类需要 override:
- `setup()`: 初始化 MotionDataManager、AMPDiscriminator、disc_obs buffer
- `learn()`: 主训练循环中增加 AMP 相关逻辑

### 2. Discriminator Observation Format (160-dim)

每帧 80-dim，堆叠 2 帧 (history=2) 得到 160-dim:

| 字段 | Per-frame 维度 | 来源 (pkl) | 来源 (simulator) |
|------|---------------|------------|-----------------|
| root_ang_vel_b | 3 | pkl 预计算 | `sim.base_ang_vel` |
| dof_pos | 29 | pkl 预计算 (GMR order) | `sim.dof_pos` (需 reorder) |
| dof_vel | 29 | pkl 预计算 (GMR order) | `sim.dof_vel` (需 reorder) |
| key_body_pos_b | 18 (6×3) | pkl 预计算 | `sim._rigid_body_pos` 选取 6 个 body → body 系转换 |
| root_height | 1 | `root_pos[..., 2]` | `robot_root_states[:, 2]` |
| **Total** | **80** | | |

**6 key bodies** (与 legged_lab 一致):
- `left_ankle_roll_link`
- `right_ankle_roll_link`
- `left_wrist_yaw_link`
- `right_wrist_yaw_link`
- `left_shoulder_roll_link`
- `right_shoulder_roll_link`

**Root body**: `pelvis` (pkl 和仿真器一致)

### 3. Motion Data Manager

使用现有 `agents/amp/motion_data.py` 中的 `MotionDataManager`。

- 数据目录: `amp_data/g1_29dof/amp/walk/`
- pkl 数据需重新生成，包含 body name 元信息
- DOF 顺序统一为 GMR_DOF_NAMES 顺序

### 4. AMP Discriminator Network

```
AMPDiscriminator(
  input: 160-dim
  → Linear(160, 256) + ELU
  → Linear(256, 256) + ELU
  → Linear(256, 256) + ELU
  → Linear(256, 1)
  → 输出: score (logit)
)
```

- Loss: LSGAN
- Learning rate: 1e-5
- Gradient penalty scale: 10.0
- Optimizer: Adam (separate from actor/critic)
- Discriminator obs normalizer: EmpiricalNormalization (160-dim)

### 5. Style Reward Computation

```
style_reward = dt * 5.0 * LSGAN_reward(score)
total_reward = task_reward + style_reward
```

LSGAN reward formula:
```
style_reward = dt * 5.0 * clamp(1 - 0.25 * (score - 1)^2, min=0)
```

### 6. Discriminator Observation Storage

在 `SimpleReplayBuffer` 中新增 `disc_obs` 字段:
- `self.disc_obs`: `(n_env, buffer_size, 160)` float tensor
- `extend()` 时存入当前的 160-dim disc_obs
- `sample()` 时返回 `disc_obs` 作为采样结果的一部分

不需存储 `next_disc_obs`，discriminator 只需要当前步的观测。

### 7. Rollout 时的 160-dim 构造

维护 `prev_disc_obs` (num_envs, 80) 历史缓冲区:

```
每步 rollout:
  1. 从 simulator raw data 计算当前帧 disc_obs_80
  2. 堆叠: disc_obs_160 = [prev_disc_obs, disc_obs_80].reshape(num_envs, -1) → (num_envs, 160)
  3. 存 disc_obs_160 到 replay buffer
  4. prev_disc_obs = disc_obs_80

当 env reset/done:
  prev_disc_obs[env_ids] = disc_obs_80[env_ids]  # 用当前帧填充 history
```

### 8. Command Categories & Allocation

**5 个类别及占比**:

| # | 名称 | 比例 | 命令范围 |
|---|------|------|---------|
| 0 | 直走 (forward) | 0.3 | vx~U[0.5, 1.5], vy=0, wz~U[-0.15, 0.15] |
| 1 | 前进+转弯 (forward+turn) | 0.3 | vx~U[-1.5,-0.3]∪[0.3,1.5], vy=0, wz~U[-2,-0.3]∪[0.3,2] |
| 2 | 原地转 (spin) | 0.1 | vx=0, vy=0, wz~U[-2,-0.3]∪[0.3,2] |
| 3 | 侧移 (sideways) | 0.2 | vx=0, vy~U[-1,-0.3]∪[0.3,1], wz~U[-0.15, 0.15] |
| 4 | 静止 (stand) | 0.1 | vx=0, vy=0, wz=0 |

**分配策略**: 一次性固定分配。训练开始时按比例将 envs 分配到 5 个类别，每个 env 的类别在训练过程中不改变。

**命令重采样**: 每 7 秒（通过 `episode_length_buf % interval == 0` 判断），env 从所属类别命令范围内均匀采样新的 (vx, vy, wz)。

### 9. 类别索引存储

在环境中维护 `category_ids` 缓冲区:
- 形状: `(num_envs,)`，值为 0-4
- 创建 envs 时按比例分配
- `reset()` 时重分配（保持类别不变）

### 10. Discriminator Training Loop

```
每 global step:
  1. Rollout (同 FastSAC)
  2. 从 replay buffer 采样 (batch_size * num_updates)
  3. 对每个 mini-batch:
     a. update_main()  # SAC critic 更新
     b. 每 policy_frequency 次:
        - update_pol()  # SAC actor 更新
        - update_disc()  # AMP discriminator 更新 (×2)
  4. 每 policy_frequency 次:
     - 从 buffer 采样 disc_obs
     - 从 MotionDataManager 采样 disc_demo_obs
     - 计算 discriminator loss + gradient penalty
     - 更新 discriminator
```

### 11. DOF 顺序统一

URDF 中的 DOF 顺序需要与 GMR_DOF_NAMES 一致。建立 `dof_reorder_idx` 映射表:
- 从 `robot_config.dof_names` 获取 URDF DOF 顺序
- 对照 `GMR_DOF_NAMES` 计算出 reorder 索引
- 在 rollout 中 `dof_pos[:, dof_reorder_idx]` / `dof_vel[:, dof_reorder_idx]` 对齐到 GMR 顺序

## Testing Decisions

### Verification Strategy

1. **Dimension check**: 验证构造的 disc_obs 维度为 (num_envs, 160)
2. **Buffer I/O**: 验证 disc_obs 正确存入和取出 replay buffer
3. **Discriminator forward**: 验证 AMPDiscriminator 输入 160-dim 输出 (num_envs, 1)
4. **Command allocation**: 验证 envs 按正确比例分配到 5 个类别
5. **Overfit test**: 在小规模（如 64 envs，少量 iteration）上验证训练流程能跑通，loss 不 NaN

### Existing Seams

- `agents/fast_sac/tests/` — 可添加 FastSACAMP 的单元测试
- `config_types/algo.py` — 需添加 FastSACAMPConfig 类型

## Out of Scope

- 不支持 AMP discriminator 的 CNN 编码器
- 不支持多运动数据源混合（仅使用 walk 目录下的 pkl）
- 不涉及 legged_lab 中的 symmetry augmentation
- 不涉及 WBT 任务（仅限于 locomotion）
- 不涉及 legged_lab 中的 task_style_lerp 混合模式

## Further Notes

### Key References

- legged_lab AMP 实现: `legged_lab/rsl_rl/rsl_rl/modules/amp.py` (AMPDiscriminator)
- legged_lab AMP 训练: `legged_lab/rsl_rl/rsl_rl/algorithms/ppo_amp.py` (PPOAMP.update())
- legged_lab AMP 配置: `legged_lab/source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1/g1_amp_env_cfg.py`
- 当前 MotionDataManager: `src/holosoma/holosoma/agents/amp/motion_data.py`
- 当前 FastSACAgent: `src/holosoma/holosoma/agents/fast_sac/fast_sac_agent.py`

### pkl 数据需求

需要重新生成 pkl 文件，添加 body name 元信息：
- 原始 pkl 中 `key_body_pos` 无 body name → 新 pkl 需包含 `key_body_names`
- DOF 顺序固定为 `GMR_DOF_NAMES`

### 命令系统交互

AMP 类别化命令需要替代现有的 `LocomotionCommand` 采样逻辑。在 AMP 模式下，`LocomotionCommand` 被 AMP-specific command term 替代，后者按类别进行限制范围内的均匀采样。
