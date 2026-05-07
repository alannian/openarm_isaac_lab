# 双臂托盘举升强化学习任务深度解析

本文档面向希望**深入理解**当前双臂托盘举升（Bimanual Tray Lift）强化学习任务的开发者，覆盖仿真环境配置、数据输入输出、奖励函数设计、训练流程等全部核心内容，并精确标注每项配置所在的源文件。

---

## 目录

1. [整体任务概述](#1-整体任务概述)
2. [文件结构总览](#2-文件结构总览)
3. [仿真环境配置](#3-仿真环境配置)
   - 3.1 物理仿真参数
   - 3.2 场景布局（SceneCfg）
   - 3.3 机器人资产（Robot Asset）
   - 3.4 托盘物体
4. [数据输入：观测空间](#4-数据输入观测空间)
5. [数据输出：动作空间](#5-数据输出动作空间)
6. [奖励函数（分阶段详解）](#6-奖励函数分阶段详解)
7. [终止条件](#7-终止条件)
8. [重置机制与随机化](#8-重置机制与随机化)
9. [课程学习](#9-课程学习)
10. [PPO 算法配置](#10-ppo-算法配置)
11. [完整训练流程](#11-完整训练流程)
12. [输出文件说明](#12-输出文件说明)
13. [推理/回放流程](#13-推理回放流程)

---

## 1. 整体任务概述

**目标**：训练一个策略，使双臂机器人（OpenArm）用两只手臂从两端夹住一个长条形托盘，协同将其从支架上举高到目标高度（约 0.52 m），并保持托盘水平、双手对称。

任务分为 4 个隐式阶段：

| 阶段 | 行为描述 |
|---|---|
| Phase 1 | 双臂末端执行器（EE）分别靠近托盘两端，并与托盘高度对齐 |
| Phase 2 | 双手同时进入各自抓握点附近，并闭合夹爪 |
| Phase 3 | 双臂协同将托盘举离支架，推向目标高度 |
| Phase 4 | 维持对称抓握、保持托盘水平，持续精确跟踪目标高度 |

---

## 2. 文件结构总览

```
source/openarm/openarm/tasks/manager_based/openarm_manipulation/
├── assets/
│   └── openarm_bimanual.py          # 机器人 USD 资产 + 执行器参数
├── bimanual/
│   └── lift/
│       ├── lift_env_cfg.py          # ★ 环境核心配置（场景/观测/动作/奖励/终止/课程）
│       ├── config/
│       │   ├── joint_pos_env_cfg.py  # ★ 具体实例化配置（托盘/EE/关节动作参数）
│       │   └── agents/
│       │       ├── rsl_rl_ppo_cfg.py  # ★ RSL-RL PPO 超参数
│       │       └── rl_games_ppo_cfg.yaml  # rl_games PPO 超参数
│       └── mdp/
│           ├── observations.py      # 自定义观测函数
│           ├── rewards.py           # ★ 全部奖励函数实现
│           └── terminations.py      # 自定义终止条件

scripts/reinforcement_learning/rsl_rl/
├── train.py                         # ★ 训练入口脚本
└── play.py                          # 推理/回放脚本

logs/rsl_rl/openarm_bi_tray_lift/    # 训练输出目录
```

---

## 3. 仿真环境配置

### 3.1 物理仿真参数

**文件**：[lift_env_cfg.py](../source/openarm/openarm/tasks/manager_based/openarm_manipulation/bimanual/lift/lift_env_cfg.py)（`BimanualTrayLiftEnvCfg.__post_init__`）

| 参数 | 值 | 含义 |
|---|---|---|
| `sim.dt` | 0.01 s | 物理引擎步长，即 100 Hz 物理仿真 |
| `decimation` | 2 | 策略每 2 个物理步执行一次，即策略频率 = 50 Hz |
| `episode_length_s` | 10.0 s | 每个 episode 最长 10 秒 = 500 个策略步 |
| `scene.num_envs` | 2048 | 并行环境数量（训练时） |
| `scene.env_spacing` | 3.0 m | 相邻环境间距 |
| `physx.bounce_threshold_velocity` | 0.01 | 弹跳速度阈值（减少抖动） |

**每次策略 step 对应的实际物理时间**：$\Delta t_\text{policy} = \text{decimation} \times \text{sim.dt} = 2 \times 0.01 = 0.02\,\text{s}$

### 3.2 场景布局（SceneCfg）

**文件**：[lift_env_cfg.py](../source/openarm/openarm/tasks/manager_based/openarm_manipulation/bimanual/lift/lift_env_cfg.py)（`TrayLiftSceneCfg`）

场景包含以下实体：

```
World
├── /World/GroundPlane          地面（无限平面）
├── /World/light                穹顶灯光（强度 3000，灰白色）
└── {ENV_REGEX_NS}/             每个并行环境的命名空间
    ├── Robot                   双臂机器人（ArticulationCfg）
    ├── Tray                    托盘（RigidObjectCfg）
    └── Stand                   支架（纯视觉/碰撞静态方块）
```

**支架几何**（用于支撑托盘，作为起始台）：

```python
table = AssetBaseCfg(
    init_state=AssetBaseCfg.InitialStateCfg(pos=[0.28, 0.0, 0.18]),
    spawn=sim_utils.CuboidCfg(size=(0.10, 0.20, 0.36), ...)
)
```

- 支架尺寸：x=10cm, y=20cm, z=36cm
- 支架顶面高度：z = 0.18 + 0.18 = **0.36 m**（几何中心在 z=0.18，半高 0.18）
- 位于 x=0.28m 处（机器人前方），y 方向仅 20cm 宽，不遮挡双臂从侧面接近托盘端部

### 3.3 机器人资产（Robot Asset）

**文件**：[openarm_bimanual.py](../source/openarm/openarm/tasks/manager_based/openarm_manipulation/assets/openarm_bimanual.py)

机器人模型从 USD 文件加载：`usds/openarm_bimanual/openarm_bimanual.usd`

#### 关节结构

| 组别 | 关节名称 | 数量 |
|---|---|---|
| 左臂 | `openarm_left_joint1` ~ `joint7` | 7 |
| 右臂 | `openarm_right_joint1` ~ `joint7` | 7 |
| 左夹爪 | `openarm_left_finger_joint.*` | 2（左右指） |
| 右夹爪 | `openarm_right_finger_joint.*` | 2（左右指） |

**合计 18 个可控关节**，均初始化为 0（手臂）或 0.044 rad（夹爪张开）。

#### 执行器参数（ImplicitActuatorCfg）

```python
# 手臂执行器
"openarm_arm": ImplicitActuatorCfg(
    joint_names_expr=["openarm_left_joint[1-7]", "openarm_right_joint[1-7]"],
    stiffness=80.0,   # PD 控制刚度
    damping=4.0,      # PD 控制阻尼
    effort_limit_sim={
        "..._joint[1-2]": 40.0,   # 肩部关节力矩限制
        "..._joint[3-4]": 27.0,   # 肘部关节力矩限制
        "..._joint[5-7]": 7.0,    # 腕部关节力矩限制
    },
    velocity_limit_sim={
        "..._joint[1-4]": 2.175,  # rad/s
        "..._joint[5-7]": 2.61,   # rad/s
    }
)

# 夹爪执行器（高刚度保证夹紧）
"openarm_gripper": ImplicitActuatorCfg(
    stiffness=2000.0,
    damping=100.0,
    effort_limit_sim=333.33,
    velocity_limit_sim=0.2,
)
```

> **隐式执行器**意味着 PhysX 直接用 PD 控制求解关节力矩，策略输出的是目标位置而非力矩。

### 3.4 托盘物体

**文件**：[joint_pos_env_cfg.py](../source/openarm/openarm/tasks/manager_based/openarm_manipulation/bimanual/lift/config/joint_pos_env_cfg.py)

```python
self.scene.tray = RigidObjectCfg(
    spawn=sim_utils.CuboidCfg(
        size=(0.12, 0.60, 0.03),   # x=12cm宽, y=60cm长, z=3cm厚
        mass_props=sim_utils.MassPropertiesCfg(mass=1.5),  # 1.5 kg
        ...
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.28, 0.0, 0.375),    # 初始放在支架顶面上方
    ),
)
```

**托盘的空间关系**：

```
z轴（高度）
0.000  ←── 地面
0.360  ←── 支架顶面（0.18 + 0.18）
0.375  ←── 托盘质心初始高度（放在支架上，托盘厚3cm，底面在0.36，质心+0.015）
0.400  ←── 举离阈值（tray_is_lifted 的 minimal_height）
0.520  ←── 目标举升高度（target_height）
```

---

## 4. 数据输入：观测空间

**文件**：[lift_env_cfg.py](../source/openarm/openarm/tasks/manager_based/openarm_manipulation/bimanual/lift/lift_env_cfg.py)（`ObservationsCfg.PolicyCfg`）、[mdp/observations.py](../source/openarm/openarm/tasks/manager_based/openarm_manipulation/bimanual/lift/mdp/observations.py)

策略每步接收一个**拼接好的一维向量**作为观测，所有观测在训练时开启加性均匀噪声（`enable_corruption=True`）。推理时关闭噪声（`OpenArmTrayLiftEnvCfg_PLAY`）。

观测项按顺序拼接，合计 **57 维**：

| # | 观测项名 | 维度 | 来源函数 | 噪声 | 说明 |
|---|---|---|---|---|---|
| 1 | `left_joint_pos` | 7 | `mdp.joint_pos_rel` | U(-0.01, 0.01) | 左臂 7 关节相对位置 |
| 2 | `left_finger_pos` | 2 | `mdp.joint_pos_rel` | U(-0.01, 0.01) | 左夹爪 2 指关节位置 |
| 3 | `right_joint_pos` | 7 | `mdp.joint_pos_rel` | U(-0.01, 0.01) | 右臂 7 关节相对位置 |
| 4 | `right_finger_pos` | 2 | `mdp.joint_pos_rel` | U(-0.01, 0.01) | 右夹爪 2 指关节位置 |
| 5 | `left_joint_vel` | 7 | `mdp.joint_vel_rel` | U(-0.01, 0.01) | 左臂 7 关节相对速度 |
| 6 | `left_finger_vel` | 2 | `mdp.joint_vel_rel` | 无 | 左夹爪速度 |
| 7 | `right_joint_vel` | 7 | `mdp.joint_vel_rel` | U(-0.01, 0.01) | 右臂 7 关节相对速度 |
| 8 | `right_finger_vel` | 2 | `mdp.joint_vel_rel` | 无 | 右夹爪速度 |
| 9 | `tray_position` | 3 | `tray_position_in_robot_root_frame` | 无 | 托盘在机器人基座坐标系下的 xyz |
| 10 | `tray_tilt` | 2 | `tray_roll_pitch` | 无 | 托盘 roll/pitch 欧拉角，感知倾斜 |
| 11 | `left_actions` | 7 | `mdp.last_action` | 无 | 上一步左臂动作 |
| 12 | `right_actions` | 7 | `mdp.last_action` | 无 | 上一步右臂动作 |
| 13 | `left_gripper_action_obs` | 1 | `mdp.last_action` | 无 | 上一步左夹爪动作 |
| 14 | `right_gripper_action_obs` | 1 | `mdp.last_action` | 无 | 上一步右夹爪动作 |

**注意**：`joint_pos_rel` / `joint_vel_rel` 表示相对于默认配置的偏差值，而不是绝对值。

**`tray_position_in_robot_root_frame` 的计算**（`mdp/observations.py`）：

```python
tray_pos_b, _ = subtract_frame_transforms(
    robot.data.root_pos_w, robot.data.root_quat_w, tray_pos_w
)
```

将世界坐标系下的托盘位置变换到机器人基座坐标系，使策略对机器人底座位置具有不变性。

**`tray_roll_pitch` 的计算**：

从四元数 $[w, x, y, z]$ 计算：

$$\text{roll} = \arctan2\left(2(wx + yz),\, 1 - 2(x^2+y^2)\right)$$

$$\text{pitch} = \arcsin\left(\text{clamp}(2(wy - zx), -1, 1)\right)$$

---

## 5. 数据输出：动作空间

**文件**：[joint_pos_env_cfg.py](../source/openarm/openarm/tasks/manager_based/openarm_manipulation/bimanual/lift/config/joint_pos_env_cfg.py)

策略每步输出一个向量，经过动作管理器（`ActionsCfg`）分发到各关节：

| 动作项 | 类型 | 维度 | scale | 含义 |
|---|---|---|---|---|
| `left_arm_action` | `JointPositionActionCfg` | 7 | 0.5 | 左臂 7 关节目标位置偏量 |
| `right_arm_action` | `JointPositionActionCfg` | 7 | 0.5 | 右臂 7 关节目标位置偏量 |
| `left_gripper_action` | `BinaryJointPositionActionCfg` | 1 | — | 左夹爪 开(0.044)/关(0.0) |
| `right_gripper_action` | `BinaryJointPositionActionCfg` | 1 | — | 右夹爪 开(0.044)/关(0.0) |

**合计动作维度：16**（7+7+1+1）

#### 关节位置动作的执行逻辑

策略网络输出的手臂动作 $a \in \mathbb{R}^7$ 经过以下变换才成为关节目标：

$$\theta_\text{target} = \theta_\text{default} + \text{scale} \times a$$

其中 `scale=0.5`，`use_default_offset=True` 表示以默认关节位置为基准偏移。这意味着策略输出 $a=0$ 时机器人保持在默认姿态，±1 的输出对应 ±0.5 rad 的关节偏移。

#### 夹爪动作的执行逻辑

`BinaryJointPositionActionCfg` 将连续动作做二值化：

- 策略输出 $a > 0$：夹爪张开，目标位置 = 0.044 rad
- 策略输出 $a \leq 0$：夹爪闭合，目标位置 = 0.0 rad（夹紧）

---

## 6. 奖励函数（分阶段详解）

**文件**：[lift_env_cfg.py](../source/openarm/openarm/tasks/manager_based/openarm_manipulation/bimanual/lift/lift_env_cfg.py)（`RewardsCfg`）、[mdp/rewards.py](../source/openarm/openarm/tasks/manager_based/openarm_manipulation/bimanual/lift/mdp/rewards.py)

总奖励是所有奖励项的加权和：$R = \sum_i w_i \cdot r_i(s)$

### Phase 1：末端接近托盘

#### 奖励 1/2：`left_reach_tray` / `right_reach_tray`（权重 = 1.5）

```python
func=mdp.ee_reach_tray_end, weight=1.5
params={"std": 0.1, "ee_frame_cfg": ..., "side": "left"/"right", "half_length": 0.22}
```

**实现**（`rewards.py::ee_reach_tray_end`）：

托盘长轴沿局部 Y 轴，`half_length=0.22`m 表示从托盘中心到抓握点的距离（0.22m，在托盘端部向内约 8cm 处）。

$$\text{left\_end} = \text{tray\_center} + 0.22 \times \hat{y}_\text{world}$$

$$r = 1 - \tanh\left(\frac{\|\text{ee\_pos} - \text{target}\|}{0.1}\right)$$

当 EE 与目标点距离为 0 时奖励 = 1.0，距离约 0.1m 时奖励约 0.76，距离 >> 0.1m 时趋向 0。

#### 奖励 3/4：`left_ee_height` / `right_ee_height`（权重 = 2.0）

```python
func=mdp.ee_height_align_reward, weight=2.0
params={"std": 0.05, "ee_frame_cfg": ...}
```

**目的**：防止机械臂从托盘下方托推（应从侧面夹住）。

$$r = 1 - \tanh\left(\frac{|z_\text{ee} - z_\text{tray}|}{0.05}\right)$$

当 EE 与托盘高度差 ≤ 0.05m 时奖励接近最大值。

---

### Phase 2：双手抓握

#### 奖励 5：`grasp_both_ends`（权重 = 5.0）

```python
func=mdp.grasp_both_ends, weight=5.0
params={"distance_threshold": 0.06, ...}
```

**实现**：二值联合奖励，**只有左右手同时**进入各自端点 6cm 范围内才触发：

$$r = \mathbf{1}[\|\text{left\_ee} - \text{left\_end}\| < 0.06 \;\wedge\; \|\text{right\_ee} - \text{right\_end}\| < 0.06]$$

权重 5.0 较大，是第二阶段的核心激励。

#### 奖励 6/7：`left_finger_closure` / `right_finger_closure`（权重 = 0.5）

```python
func=mdp.finger_closure_reward, weight=0.5
```

**逻辑**（特意设计为不惩罚远离时开爪）：

| EE 位置 | 夹爪状态 | 奖励 |
|---|---|---|
| 靠近 (< 6cm) | 闭合 (< 0.015 rad) | +1 |
| 靠近 (< 6cm) | 张开 | 0 |
| 远离 | 张开 | +1（不干扰探索） |
| 远离 | 闭合 | 0 |

---

### Phase 3：举起托盘

#### 奖励 8：`tray_lifted`（权重 = 20.0）⭐ 最重要奖励

```python
func=mdp.tray_is_lifted, weight=20.0
params={"minimal_height": 0.40}
```

**二值奖励**：托盘质心高于 0.40m 时得 1.0。权重 20.0 是所有奖励项中最高的，确保举起托盘是策略的主要优化目标。

#### 奖励 9：`tray_goal_height`（权重 = 16.0）

```python
func=mdp.tray_goal_height_tracking, weight=16.0
params={"target_height": 0.52, "std": 0.1, "minimal_height": 0.40}
```

**条件性 tanh 奖励**：只有托盘已举起时才激活：

$$r = \mathbf{1}[z_\text{tray} > 0.40] \times \left(1 - \tanh\left(\frac{|z_\text{tray} - 0.52|}{0.1}\right)\right)$$

#### 奖励 10：`tray_goal_height_fine`（权重 = 5.0）

同上但 `std=0.03`，提供更精细的高度跟踪奖励信号，引导策略精确停在 0.52m。

---

### Phase 4：协同约束

#### 奖励 11：`grasp_symmetry`（权重 = **-2.0**，惩罚）

```python
func=mdp.grasp_symmetry_penalty, weight=-2.0
```

惩罚左右臂到托盘中心距离不对称：

$$r = |d_\text{left} - d_\text{right}|$$

#### 奖励 12：`tray_tilt`（权重 = **-3.0**，惩罚）

```python
func=mdp.tray_tilt_penalty, weight=-3.0
params={"max_tilt_rad": 0.1}
```

仅在托盘被举起后惩罚倾斜，容许 0.1 rad（≈5.7°）以内的倾斜：

$$\text{tilt} = 1 - (1 - 2(q_x^2 + q_y^2))$$

$$r = \mathbf{1}[z_\text{tray}>0.40] \times \max(0,\, \text{tilt} - 0.1)$$

#### 奖励 13：`hand_distance`（权重 = 0.5）

```python
func=mdp.hand_distance_reward, weight=0.5
params={"target_distance": 0.44, "std": 0.2, ...}
```

奖励双手间距维持在 0.44m（即 $2 \times 0.22$m，恰好对应两个抓握点的距离）：

$$r = 1 - \tanh\left(\frac{|d_\text{current} - 0.44|/0.44}{0.2}\right)$$

---

### 平滑性惩罚

| 奖励项 | 权重 | 函数 | 说明 |
|---|---|---|---|
| `action_rate` | -1e-4 | `mdp.action_rate_l2` | 相邻两步动作的 L2 变化 |
| `left_joint_vel` | -1e-4 | `mdp.joint_vel_l2` | 左臂关节速度 L2 范数 |
| `right_joint_vel` | -1e-4 | `mdp.joint_vel_l2` | 右臂关节速度 L2 范数 |

权重初始很小（-1e-4），课程学习会逐步增大（见第 9 节）。

---

### 奖励权重汇总表

| 奖励项 | 权重 | 类型 |
|---|---|---|
| `left_reach_tray` | +1.5 | Phase 1 接近 |
| `right_reach_tray` | +1.5 | Phase 1 接近 |
| `left_ee_height` | +2.0 | Phase 1 高度对齐 |
| `right_ee_height` | +2.0 | Phase 1 高度对齐 |
| `grasp_both_ends` | +5.0 | Phase 2 双手接触 |
| `left_finger_closure` | +0.5 | Phase 2 夹爪闭合 |
| `right_finger_closure` | +0.5 | Phase 2 夹爪闭合 |
| `tray_lifted` | **+20.0** | Phase 3 举起 |
| `tray_goal_height` | **+16.0** | Phase 3 高度追踪（粗） |
| `tray_goal_height_fine` | +5.0 | Phase 3 高度追踪（精） |
| `grasp_symmetry` | **-2.0** | Phase 4 对称性 |
| `tray_tilt` | **-3.0** | Phase 4 水平姿态 |
| `hand_distance` | +0.5 | Phase 4 间距维持 |
| `action_rate` | -1e-4 → -5e-3 | 平滑性 |
| `left_joint_vel` | -1e-4 → -1e-3 | 平滑性 |
| `right_joint_vel` | -1e-4 → -1e-3 | 平滑性 |

---

## 7. 终止条件

**文件**：[lift_env_cfg.py](../source/openarm/openarm/tasks/manager_based/openarm_manipulation/bimanual/lift/lift_env_cfg.py)（`TerminationsCfg`）、[mdp/terminations.py](../source/openarm/openarm/tasks/manager_based/openarm_manipulation/bimanual/lift/mdp/terminations.py)

| 条件名 | 触发条件 | 是否为超时 |
|---|---|---|
| `time_out` | episode 步数达到 `episode_length_s / step_dt = 500` 步 | 是（bootstrap）|
| `tray_dropped` | 托盘质心 z < 0.33m（低于支架顶面 3cm） | 否（failure）|

**`tray_dropped` 的参数**：`minimum_height=0.33`，意味着托盘从支架上滑落到低于 0.33m 时立即终止。托盘初始在 0.375m，所以该条件在正常状态下不触发。

`time_out=True` 对强化学习有特殊意义：超时终止不视为真正失败，算法在计算 value bootstrap 时会使用估计的未来价值，而不是将 episode 价值截断为零。

---

## 8. 重置机制与随机化

**文件**：[lift_env_cfg.py](../source/openarm/openarm/tasks/manager_based/openarm_manipulation/bimanual/lift/lift_env_cfg.py)（`EventCfg`）

每次 episode 结束后触发以下重置事件（`mode="reset"`）：

### 事件 1：`reset_all`

```python
func=mdp.reset_scene_to_default
```

将所有场景实体（包括机器人和托盘）重置到配置文件中定义的默认状态。这是第一步，后续事件在此基础上叠加扰动。

### 事件 2：`reset_tray_position`

```python
func=mdp.reset_root_state_uniform
params={
    "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.0, 0.0)},
    "velocity_range": {},
}
```

在托盘初始位置基础上添加 x、y 方向各 ±5cm 的均匀随机扰动，迫使策略学会泛化到托盘位置的轻微变化，而不是记住固定位置。

### 事件 3：`reset_robot_joints`

```python
func=mdp.reset_joints_by_offset
params={"position_range": (-0.2, 0.2), "velocity_range": (0.0, 0.0)}
```

在机器人默认关节位置基础上添加 ±0.2 rad 的均匀随机偏移，防止策略过拟合到单一起始姿态，同时也能加速早期探索。

---

## 9. 课程学习

**文件**：[lift_env_cfg.py](../source/openarm/openarm/tasks/manager_based/openarm_manipulation/bimanual/lift/lift_env_cfg.py)（`CurriculumCfg`）

课程学习通过逐步提高平滑性惩罚权重，实现先学会完成任务、再学优雅完成任务的两阶段策略。

| 课程项 | 初始权重 | 最终权重 | 步数 | 说明 |
|---|---|---|---|---|
| `action_rate` | -1e-4 | -5e-3 | 20000 | 动作变化率惩罚，50倍增长 |
| `left_joint_vel` | -1e-4 | -1e-3 | 20000 | 左臂速度惩罚，10倍增长 |
| `right_joint_vel` | -1e-4 | -1e-3 | 20000 | 右臂速度惩罚，10倍增长 |

权重的变化是**线性插值**，在 20000 训练步内从初始值过渡到最终值。这意味着：

- **前期（0~20000 步）**：平滑性惩罚很小，策略可以用任意动作完成举升任务
- **后期（>20000 步）**：平滑性惩罚增大，策略被迫在完成任务的同时避免剧烈运动

---

## 10. PPO 算法配置

**文件**：[config/agents/rsl_rl_ppo_cfg.py](../source/openarm/openarm/tasks/manager_based/openarm_manipulation/bimanual/lift/config/agents/rsl_rl_ppo_cfg.py)

使用 **RSL-RL** 框架的 `OnPolicyRunner`，算法为标准 PPO（Proximal Policy Optimization）。

### 网络结构

```python
policy = RslRlPpoActorCriticCfg(
    init_noise_std=1.0,
    actor_hidden_dims=[256, 128, 64],
    critic_hidden_dims=[256, 128, 64],
    activation="elu",
)
```

- **Actor（策略网络）**：MLP，输入 57 维观测，输出 16 维动作均值（+学习到的对数标准差）
- **Critic（价值网络）**：MLP，输入 57 维观测，输出 1 维价值估计
- **激活函数**：ELU（指数线性单元），在负值处平滑，适合强化学习
- **初始噪声标准差**：1.0（较大，鼓励早期探索）

### 算法超参数

```python
algorithm = RslRlPpoAlgorithmCfg(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.2,           # PPO clip 范围
    entropy_coef=0.005,       # 熵正则化系数
    num_learning_epochs=8,    # 每批数据更新 8 次
    num_mini_batches=8,       # 每次更新拆成 8 个 mini-batch
    learning_rate=3.0e-4,
    schedule="adaptive",      # 学习率自适应调整
    gamma=0.99,               # 折扣因子
    lam=0.95,                 # GAE lambda
    desired_kl=0.01,          # 目标 KL 散度（触发 lr 调整）
    max_grad_norm=1.0,        # 梯度裁剪
)
```

### 训练规模参数

```python
num_steps_per_env = 48      # 每个环境每轮收集 48 步数据
max_iterations = 3000       # 最多 3000 轮迭代
```

**每轮数据总量**：

$$N_\text{total} = \text{num\_envs} \times \text{num\_steps} = 2048 \times 48 = 98304 \;\text{transitions}$$

**每轮更新总次数**：

$$N_\text{updates} = \text{num\_learning\_epochs} \times \text{num\_mini\_batches} = 8 \times 8 = 64$$

**每个 mini-batch 大小**：

$$B = \frac{98304}{8} = 12288 \;\text{samples}$$

`empirical_normalization=True` 表示使用经验均值/方差对观测和价值进行归一化（运行时统计，而非固定参数）。

---

## 11. 完整训练流程

**入口文件**：[scripts/reinforcement_learning/rsl_rl/train.py](../scripts/reinforcement_learning/rsl_rl/train.py)

### 启动命令

```bash
cd /home/jintao/isaac_ws/openarm_isaac_lab
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Tray-Lift-OpenArm-v0 \
    --num_envs 2048 \
    --max_iterations 3000
```

### 训练流程图

```
启动 Isaac Sim (AppLauncher)
        │
        ▼
加载任务配置 (hydra_task_config)
  ├─ 环境配置: OpenArmTrayLiftEnvCfg
  └─ 算法配置: OpenArmTrayLiftPPORunnerCfg
        │
        ▼
创建仿真环境 gym.make(task, cfg=env_cfg)
  ├─ 初始化 2048 个并行环境
  ├─ 加载 USD 场景（机器人 + 托盘 + 支架 + 地面）
  └─ 设置 PhysX 物理引擎
        │
        ▼
包装环境 RslRlVecEnvWrapper
  ├─ 统一观测/动作格式为向量
  └─ 可选视频录制
        │
        ▼
创建 OnPolicyRunner
  ├─ 构建 Actor-Critic 网络（2× MLP [256,128,64]）
  └─ 初始化 PPO 优化器 (Adam, lr=3e-4)
        │
        ▼
训练循环 (3000 次迭代)
  │
  ├─ [数据采集] 2048 个环境并行运行 48 步
  │    ├─ obs → Actor → 采样动作
  │    ├─ 执行动作 → 物理步进 2 次 (decimation=2)
  │    ├─ 计算各奖励项 → 求和得 r
  │    ├─ 检查终止条件（timeout/tray_dropped）
  │    └─ 收集 (obs, action, reward, done, value, log_prob)
  │
  ├─ [GAE 计算] 用 Critic 估计优势函数
  │    └─ A(s,a) = Σ (γλ)^k δ_{t+k}，其中 δ = r + γV(s') - V(s)
  │
  ├─ [PPO 更新] 对收集的数据更新 8 轮，每轮分 8 个 mini-batch
  │    ├─ Actor loss = -min(r·A, clip(r,1-ε,1+ε)·A) - 0.005·H(π)
  │    ├─ Critic loss = (V - V_target)² × 1.0
  │    ├─ 自适应调整学习率（目标 KL = 0.01）
  │    └─ 梯度裁剪 (max_norm=1.0)
  │
  ├─ [课程更新] 每步线性调整平滑性惩罚权重（20000步内）
  │
  └─ [存档] 每 100 次迭代保存 model_{iter}.pt
        │
        ▼
训练结束，关闭仿真器
```

---

## 12. 输出文件说明

训练产生的所有文件保存在：

```
logs/rsl_rl/openarm_bi_tray_lift/{YYYY-MM-DD_HH-MM-SS}/
├── params/
│   ├── env.yaml              # 完整环境配置快照
│   └── agent.yaml            # 完整算法配置快照
├── events.out.tfevents.*     # TensorBoard 日志
├── model_0.pt                # 初始模型（iteration 0）
├── model_100.pt              # 第 100 轮检查点
├── model_200.pt              # ...
└── model_{N}.pt              # 每 100 轮保存一次
```

**`model_X.pt` 文件内容**（PyTorch 格式）：

```python
{
    "model_state_dict": {...},   # Actor-Critic 网络权重
    "optimizer_state_dict": {...},  # Adam 优化器状态
    "iter": X,                   # 当前迭代数
    "infos": {...},              # 运行时统计信息
}
```

**TensorBoard 监控指标**：

| 指标 | 含义 |
|---|---|
| `Train/mean_reward` | 当前轮次平均 episode 总奖励 |
| `Train/mean_episode_length` | 平均 episode 长度（步数） |
| `Loss/value_function` | Critic 损失 |
| `Loss/surrogate` | Actor PPO 损失 |
| `Loss/entropy` | 策略熵（探索程度） |
| `Policy/mean_noise_std` | 动作噪声标准差 |
| `Perf/learning_rate` | 当前学习率 |

查看命令：

```bash
tensorboard --logdir logs/rsl_rl/openarm_bi_tray_lift
```

---

## 13. 推理/回放流程

**入口文件**：[scripts/reinforcement_learning/rsl_rl/play.py](../scripts/reinforcement_learning/rsl_rl/play.py)

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Tray-Lift-OpenArm-Play-v0 \
    --num_envs 16 \
    --checkpoint logs/rsl_rl/openarm_bi_tray_lift/2026-05-06_03-04-39/model_2000.pt
```

推理模式使用 `OpenArmTrayLiftEnvCfg_PLAY`，与训练模式的差异：

| 配置项 | 训练 | 推理 |
|---|---|---|
| `num_envs` | 2048 | 16 |
| `env_spacing` | 3.0m | 3.5m |
| `enable_corruption` | True（有噪声） | **False**（无噪声） |

**推理时额外操作**：`play.py` 会自动将加载的模型导出为：

```
logs/rsl_rl/openarm_bi_tray_lift/{run}/exported/
├── policy.pt     # TorchScript 格式（可用于 C++ 部署）
└── policy.onnx   # ONNX 格式（跨平台部署）
```

---

## 附录：观测维度计算验证

```
left_joint_pos:          7
left_finger_pos:         2
right_joint_pos:         7
right_finger_pos:        2
left_joint_vel:          7
left_finger_vel:         2
right_joint_vel:         7
right_finger_vel:        2
tray_position:           3
tray_tilt:               2
left_actions:            7
right_actions:           7
left_gripper_action_obs: 1
right_gripper_action_obs:1
────────────────────────────
总计:                   57
```

## 附录：坐标系说明

- **世界坐标系**（World）：全局固定，z 轴朝上
- **机器人基座坐标系**（Robot Root）：固定在机器人底座，`tray_position` 观测在此坐标系下，使策略对机器人绝对位置具有不变性
- **托盘局部坐标系**：Y 轴为托盘长轴，`_get_tray_ends()` 使用四元数旋转将局部 Y 轴变换到世界系，确保托盘旋转后抓握点计算仍然正确
