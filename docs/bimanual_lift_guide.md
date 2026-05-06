# 双臂协同抬举任务 — 实现指南

## 一、任务总览

### 1.1 目标
工作台上放置一个长形重物（托盘），需要双臂各抓住物体的一端，协同将其平稳抬起至目标高度。

### 1.2 训练价值
- **对称协同**：双臂需保持合适的相对距离——太远物体会滑脱，太近会碰撞
- **姿态约束**：抬起过程中托盘需保持水平（模拟上面放了一个不能滚落的球）

### 1.3 与现有任务的关系
| 现有任务 | 复用什么 | 新增什么 |
|---------|---------|---------|
| `bimanual/reach` | 双臂机器人模型、双臂动作结构、双 EE 观测 | - |
| `unimanual/lift` | 桌面 + 物体场景、物体观测、抬升奖励 | 双臂协调 + 水平约束 |

---

## 二、整体设计思路

### 2.1 动作空间（20 维）

```
左臂关节位置 × 7  （openarm_left_joint1 ~ joint7，每维 scale=0.5）
右臂关节位置 × 7  （openarm_right_joint1 ~ joint7，每维 scale=0.5）
左手爪开合 × 1    （BinaryJointPositionAction: open=0.044, close=0.0）
右手爪开合 × 1    （同上）
左手爪关节 × 2    （实际 finger joints 仍为 2，但 BinaryAction 统一控制）
右手爪关节 × 2    （实际 finger joints 仍为 2）
```

**注意**：手爪控制复用 unimanual lift 的 `BinaryJointPositionActionCfg` 模式。虽然每个手爪有 2 个 finger joint，但 BinaryAction 将它们视为一组开/关。

### 2.2 观测空间（约 62 维）

| 观测项 | 维度 | 说明 |
|--------|------|------|
| 左臂关节位置 | 7 | `openarm_left_joint[1-7]` |
| 右臂关节位置 | 7 | `openarm_right_joint[1-7]` |
| 左手爪关节位置 | 2 | `openarm_left_finger_joint.*` |
| 右手爪关节位置 | 2 | `openarm_right_finger_joint.*` |
| 左臂关节速度 | 7 | |
| 右臂关节速度 | 7 | |
| 左手爪关节速度 | 2 | |
| 右手爪关节速度 | 2 | |
| 物体位置（机器人基座系）| 3 | 物体在 robot root frame 下的坐标 |
| 物体姿态（四元数）| 4 | 物体在世界系下的朝向 |
| 目标物体位置（机器人基座系）| 3 | command 中的目标 xyz |
| 上一步左臂动作 | 7 | |
| 上一步右臂动作 | 7 | |
| 上一步手爪动作 | 2 | left + right gripper action |
| **总计** | **62** | |

### 2.3 奖励函数设计（核心）

奖励函数分为四个阶段，每阶段有不同的主导奖励项：

```
阶段 1: 靠近物体  →  reaching_object (双手)
阶段 2: 抓取两端  →  grasping_object
阶段 3: 抬升物体  →  lifting_object
阶段 4: 到达目标  →  object_goal_tracking + levelness_constraint
```

#### 奖励项详情

**(1) `left_reaching_object` — 左手靠近物体左端**

```python
def body_reaching_object(env, std, body_name, object_cfg, robot_cfg):
    """tanh-shaped reward for a specific body reaching the object."""
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    # 获取指定 body（如 "openarm_left_hand"）在世界系下的位置
    body_pos_w = robot.data.body_pos_w[:, body_ids]
    obj_pos_w = obj.data.root_pos_w[:, :3]
    distance = torch.norm(body_pos_w - obj_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)
```

| 参数 | 值 | 说明 |
|------|-----|------|
| `weight` | **+1.5** | 比单臂 lift 的 1.1 稍高，双臂各有一份 |
| `std` | 0.1 | 距离 > 0.15m 时奖励接近 0 |

**(2) `right_reaching_object` — 右手靠近物体右端**

同上，`body_name="openarm_right_hand"`，`weight=+1.5`。

**(3) `grasp_both_ends` — 双手同时与物体两端接触的抓握奖励**

```python
def grasp_both_ends(env, left_body, right_body, object_cfg, robot_cfg):
    """Reward when both hands are simultaneously near the object."""
    # 左手到物体距离
    left_dist = norm(left_hand_pos - obj_pos)
    # 右手到物体距离
    right_dist = norm(right_hand_pos - obj_pos)
    # 同时满足：左右手都在 5cm 以内
    both_close = (left_dist < 0.05) & (right_dist < 0.05)
    return both_close.float()
```

| 参数 | 值 | 说明 |
|------|-----|------|
| `weight` | **+5.0** | 一次性奖励，鼓励双手同时靠近 |

**(4) `left_finger_closure` / `right_finger_closure` — 靠近物体时闭合手爪**

```python
def finger_closure_reward(env, body_name, object_cfg, robot_cfg, finger_joint_names):
    # 获取手指关节位置
    finger_pos = robot.data.joint_pos[:, finger_indices].mean(dim=1)
    # 手到物体距离
    dist = norm(body_pos - obj_pos)
    # 距离近 + 手指闭合 → 高分；距离近 + 手指张开 → 低分
    near = (dist < 0.05).float()
    closed = (finger_pos < 0.01).float()  # 手指位置绝对值接近 0 = 闭合
    return near * closed + (1 - near) * (1 - closed)  # 远时不强求闭合
```

| 参数 | 值 | 说明 |
|------|-----|------|
| `weight` | **+0.5** | 每只手，柔和引导 |

**(5) `lifting_object` — 物体被抬离桌面**

与单臂 lift 完全相同，直接从 `unimanual/lift/mdp/rewards.py` 导入：

```python
def object_is_lifted(env, minimal_height, object_cfg):
    obj = env.scene[object_cfg.name]
    return torch.where(obj.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)
```

| 参数 | 值 | 说明 |
|------|-----|------|
| `weight` | **+20.0** | 比单臂 lift (15.0) 更高，因为这是核心目标 |
| `minimal_height` | 0.06m | 物体 z > 6cm 视为抬离桌面 |

**(6) `object_goal_tracking` — 物体跟踪目标高度**

```python
# 直接复用 unimanual/lift 的 object_goal_distance
def object_goal_distance(env, std, minimal_height, command_name, robot_cfg, object_cfg):
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    distance = torch.norm(des_pos_w - obj.data.root_pos_w, dim=1)
    return (obj.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))
```

| 参数 | 值 | 说明 |
|------|-----|------|
| `weight` | **+16.0** | 粗粒度目标跟踪 |
| `std` | 0.3 | |
| `minimal_height` | 0.04m | |

**(7) `object_goal_tracking_fine_grained` — 细粒度目标跟踪**

| 参数 | 值 | 说明 |
|------|-----|------|
| `weight` | **+5.0** | |
| `std` | 0.05 | 精确到达时主导 |

**(8) `levelness_constraint` — 水平姿态约束（新增核心奖励）**

```python
def levelness_constraint(env, object_cfg):
    """Penalize object tilt: compute roll and pitch from quaternion.
    
    Assuming the object's local Z should align with world Z:
    - Transforms world Z-axis into object local frame
    - Measures the angle between object Z and world Z
    - Returns negative penalty proportional to tilt angle.
    """
    obj = env.scene[object_cfg.name]
    quat = obj.data.root_quat_w  # (num_envs, 4), wxyz
    
    # 将世界 Z 轴 [0,0,1] 旋转到物体局部坐标系
    # object_Z_in_world = rotate_vector_by_quat([0,0,1], quat)
    # 简化：计算物体 Z 轴与世界 Z 轴之间的夹角
    # 物体局部 Z 在世界系下的方向 = quat_rotate(quat, [0,0,1])
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    
    # quaternion rotate v=[0,0,1]: 
    #   v' = v + 2*q_vec × (q_vec × v + w*v)
    #   result_z_component = 1 - 2*(x² + y²)
    # 所以我们得到物体 Z 轴在世界系下的 z 分量
    obj_z_in_world_z = 1 - 2 * (x**2 + y**2)
    
    # 如果完全水平（无倾斜），object_Z 与世界 Z 重合，z 分量 = 1
    # 倾斜越大，z 分量 < 1
    # 用 arccos 得到倾斜角，但用 z 分量本身作为近似即可
    # tilt = |1 - z_component| 在 [0, 2] 之间，0 = 完美水平
    tilt = 1.0 - obj_z_in_world_z  # 0 = 水平, >0 = 倾斜
    return -tilt  # 负奖励
```

| 参数 | 值 | 说明 |
|------|-----|------|
| `weight` | **-2.0** | 惩罚倾斜 |
| 激活条件 | 物体被抬离桌面后 | 仅当 `object_z > minimal_height` 时生效 |

**补充**：也可改为 tanh 塑形，或仅在物体高于阈值时才施加。

**(9) `relative_distance_penalty` — 双手相对距离约束（新增核心奖励）**

```python
def relative_distance_penalty(env, target_distance, std, robot_cfg):
    """Penalize when hands are too close or too far from the ideal grasping width.
    
    The tray has a specific length. Hands should be at roughly target_distance apart.
    """
    robot = env.scene[robot_cfg.name]
    left_body_ids = robot_cfg.body_ids[0]  # left_hand
    right_body_ids = robot_cfg.body_ids[1]  # right_hand
    
    left_pos = robot.data.body_pos_w[:, left_body_ids]
    right_pos = robot.data.body_pos_w[:, right_body_ids]
    current_dist = torch.norm(left_pos - right_pos, dim=1)
    
    # 使用 soft constraint：距离偏差的平方
    deviation = (current_dist - target_distance) / target_distance
    # tanh shaping: 偏差小于 10% 时接近满分
    return 1 - torch.tanh(torch.abs(deviation) / std)
```

| 参数 | 值 | 说明 |
|------|-----|------|
| `weight` | **+0.5** | |
| `target_distance` | 0.25m | 取决于托盘长度 |
| `std` | 0.2 | |

**(10) 正则化项**

| 奖励项 | 权重 | 说明 |
|--------|------|------|
| `action_rate_l2` | **-0.0001** → **-0.005** (curriculum) | 平滑性 |
| `left_joint_vel_l2` | **-0.0001** → **-0.001** (curriculum) | 左臂关节速度 |
| `right_joint_vel_l2` | **-0.0001** → **-0.001** (curriculum) | 右臂关节速度 |

### 2.4 终止条件

| 终止条件 | 说明 |
|---------|------|
| `time_out` | episode 超时（如 10 秒）|
| `object_dropping` | 物体坠落到桌面以下 (z < -0.05m) |

### 2.5 Command 设计

物体目标位置在机器人基座坐标系下随机采样：

```python
object_pose = mdp.UniformPoseCommandCfg(
    asset_name="robot",
    body_name=MISSING,  # 在具体 config 中设为 openarm_left_hand（参考点）
    resampling_time_range=(5.0, 5.0),  # 每 5 秒重采样
    debug_vis=True,
    ranges=mdp.UniformPoseCommandCfg.Ranges(
        pos_x=(0.15, 0.35),   # 物体 x 范围（前方）
        pos_y=(-0.05, 0.05),  # 物体 y 范围（中间，双臂共享）
        pos_z=(0.15, 0.35),   # 目标高度
        roll=(0.0, 0.0),       # 不允许旋转
        pitch=(0.0, 0.0),
        yaw=(0.0, 0.0),
    ),
)
```

### 2.6 物体参数

使用比 DexCube 更长的形状。可以用 `isaaclab.sim.spawners.GroundPlaneCfg` 替换为简单的 box primitive，或者缩放 DexCube：

```python
self.scene.object = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Object",
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=[0.35, 0, 0.04],  # 放在桌面上
        rot=[1, 0, 0, 0],
    ),
    spawn=UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
        scale=(1.5, 0.3, 0.15),  # 长条形：x方向1.5倍，y方向0.3倍，z方向0.15倍
        rigid_props=RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        ),
        mass_props=MassPropertiesCfg(mass=2.0),  # 2kg，模拟重物
    ),
)
```

---

## 三、实操步骤

### Step 1：创建目录结构

在 `bimanual/` 下新建 `lift/` 任务：

```bash
cd /home/jintao/isaac_ws/openarm_isaac_lab/source/openarm/openarm/tasks/manager_based/openarm_manipulation/bimanual/

mkdir -p lift/mdp
mkdir -p lift/config/agents
```

最终目录结构：

```
bimanual/
  __init__.py
  lift/
    __init__.py              # 空文件（license only）
    lift_env_cfg.py           # 基类：场景 + MDP 抽象配置
    mdp/
      __init__.py             # re-export isaaclab mdp + 本地函数
      rewards.py              # 自定义奖励函数
      observations.py         # 自定义观测函数
      terminations.py         # 自定义终止条件
    config/
      __init__.py             # gym.register()
      joint_pos_env_cfg.py    # 具体 robot + object 配置
      agents/
        __init__.py           # 空文件
        rsl_rl_ppo_cfg.py     # PPO 超参数
        rl_games_ppo_cfg.yaml # rl_games PPO 配置
        skrl_ppo_cfg.yaml     # skrl PPO 配置
```

### Step 2：编写 `lift/mdp/rewards.py`

创建文件 `bimanual/lift/mdp/rewards.py`：

```python
# Copyright 2025 Enactic, Inc.
# ... license header ...

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def body_reaching_object(
    env: ManagerBasedRLEnv,
    std: float,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward for a specific body (hand) reaching the object using tanh-kernel."""
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    body_pos_w = robot.data.body_pos_w[:, robot_cfg.body_ids[0]]
    obj_pos_w = obj.data.root_pos_w[:, :3]
    distance = torch.norm(body_pos_w - obj_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)


def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    obj: RigidObject = env.scene[object_cfg.name]
    return torch.where(obj.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward for tracking the goal pose using tanh-kernel, only when lifted."""
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b
    )
    distance = torch.norm(des_pos_w - obj.data.root_pos_w, dim=1)
    return (obj.data.root_pos_w[:, 2] > minimal_height) * (
        1 - torch.tanh(distance / std)
    )


def levelness_constraint(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Penalize tray tilt. 0 = perfectly flat, negative = tilted.

    Computes how much the object's local Z axis deviates from world Z.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    quat = obj.data.root_quat_w  # (num_envs, 4) w,x,y,z order
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    # z-component of object's local Z in world frame = 1 - 2*(x^2 + y^2)
    obj_z_in_world_z = 1 - 2 * (x ** 2 + y ** 2)
    tilt = 1.0 - obj_z_in_world_z  # 0.0 = level, up to 2.0 = fully flipped
    return -tilt


def relative_distance_constraint(
    env: ManagerBasedRLEnv,
    target_distance: float,
    std: float,
    left_body_cfg: SceneEntityCfg,
    right_body_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for keeping hands at roughly target_distance apart."""
    robot: RigidObject = env.scene[robot_cfg.name]
    left_pos = robot.data.body_pos_w[:, left_body_cfg.body_ids[0]]
    right_pos = robot.data.body_pos_w[:, right_body_cfg.body_ids[0]]
    current_dist = torch.norm(left_pos - right_pos, dim=1)
    deviation = (current_dist - target_distance) / target_distance
    return 1 - torch.tanh(torch.abs(deviation) / std)


def grasp_both_ends(
    env: ManagerBasedRLEnv,
    distance_threshold: float,
    left_body_cfg: SceneEntityCfg,
    right_body_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Binary bonus: 1.0 when both hands are within threshold of the object."""
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    obj_pos = obj.data.root_pos_w[:, :3]
    left_pos = robot.data.body_pos_w[:, left_body_cfg.body_ids[0]]
    right_pos = robot.data.body_pos_w[:, right_body_cfg.body_ids[0]]
    left_close = torch.norm(left_pos - obj_pos, dim=1) < distance_threshold
    right_close = torch.norm(right_pos - obj_pos, dim=1) < distance_threshold
    return (left_close & right_close).float()


def finger_closure_reward(
    env: ManagerBasedRLEnv,
    distance_threshold: float,
    body_cfg: SceneEntityCfg,
    finger_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward closing fingers when the hand is near the object."""
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    hand_pos = robot.data.body_pos_w[:, body_cfg.body_ids[0]]
    obj_pos = obj.data.root_pos_w[:, :3]
    near = torch.norm(hand_pos - obj_pos, dim=1) < distance_threshold
    # finger position: lower value = more closed
    finger_pos = robot.data.joint_pos[:, finger_cfg.joint_ids].mean(dim=1)
    closed = finger_pos < 0.01
    return (near.float() * closed.float()) + ((1 - near.float()) * (1 - closed.float()))
```

### Step 3：编写 `lift/mdp/observations.py`

```python
# ... license ...

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    obj_pos_w = obj.data.root_pos_w[:, :3]
    obj_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, obj_pos_w
    )
    return obj_pos_b


def object_orientation_in_world(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The orientation of the object in world frame (quaternion w,x,y,z)."""
    obj: RigidObject = env.scene[object_cfg.name]
    return obj.data.root_quat_w
```

### Step 4：编写 `lift/mdp/terminations.py`

```python
# ... license ...

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_dropped(
    env: ManagerBasedRLEnv,
    minimum_height: float = -0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Terminate if the object falls below the minimum height."""
    obj: RigidObject = env.scene[object_cfg.name]
    return obj.data.root_pos_w[:, 2] < minimum_height
```

### Step 5：编写 `lift/mdp/__init__.py`

```python
# ... license ...

from isaaclab.envs.mdp import *  # noqa: F401, F403

from .observations import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403
from .terminations import *  # noqa: F401, F403
```

### Step 6：编写 `lift/lift_env_cfg.py`（基类）

```python
# ... license ...

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import (
    ArticulationCfg,
    AssetBaseCfg,
    RigidObjectCfg,
    DeformableObjectCfg,
)
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp


##
# Scene definition
##

@configclass
class BimanualLiftSceneCfg(InteractiveSceneCfg):
    """Scene with bimanual robot, table, and liftable object."""

    # robots: populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # target object: populated by agent env cfg
    object: RigidObjectCfg | DeformableObjectCfg = MISSING

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
        ),
    )

    # Ground plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # Lighting
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##

@configclass
class CommandsCfg:
    """Command: target object pose."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # set by concrete config
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.15, 0.35),
            pos_y=(-0.05, 0.05),
            pos_z=(0.15, 0.35),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    """Four action groups: left/right arm + left/right gripper."""

    left_arm_action: mdp.JointPositionActionCfg = MISSING
    right_arm_action: mdp.JointPositionActionCfg = MISSING
    left_gripper_action: mdp.BinaryJointPositionActionCfg = MISSING
    right_gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications."""

    @configclass
    class PolicyCfg(ObsGroup):
        left_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        f"openarm_left_joint{i}" for i in range(1, 8)
                    ],
                )
            },
        )
        right_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        f"openarm_right_joint{i}" for i in range(1, 8)
                    ],
                )
            },
        )
        left_finger_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["openarm_left_finger_joint.*"],
                )
            },
        )
        right_finger_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["openarm_right_finger_joint.*"],
                )
            },
        )
        left_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        f"openarm_left_joint{i}" for i in range(1, 8)
                    ],
                )
            },
        )
        right_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        f"openarm_right_joint{i}" for i in range(1, 8)
                    ],
                )
            },
        )
        left_finger_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["openarm_left_finger_joint.*"],
                )
            },
        )
        right_finger_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["openarm_right_finger_joint.*"],
                )
            },
        )
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_object_position = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "object_pose"}
        )
        object_orientation = ObsTerm(func=mdp.object_orientation_in_world)
        left_actions = ObsTerm(
            func=mdp.last_action, params={"action_name": "left_arm_action"}
        )
        right_actions = ObsTerm(
            func=mdp.last_action, params={"action_name": "right_arm_action"}
        )
        left_gripper_action_obs = ObsTerm(
            func=mdp.last_action, params={"action_name": "left_gripper_action"}
        )
        right_gripper_action_obs = ObsTerm(
            func=mdp.last_action, params={"action_name": "right_gripper_action"}
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Events: reset and randomize object position."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.15, 0.15), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for bimanual lift."""

    # -- Phase 1: Approach --
    left_reaching_object = RewTerm(
        func=mdp.body_reaching_object,
        weight=1.5,
        params={
            "std": 0.1,
            "robot_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "object_cfg": SceneEntityCfg("object"),
        },
    )
    right_reaching_object = RewTerm(
        func=mdp.body_reaching_object,
        weight=1.5,
        params={
            "std": 0.1,
            "robot_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "object_cfg": SceneEntityCfg("object"),
        },
    )

    # -- Phase 2: Grasp --
    grasp_both_ends = RewTerm(
        func=mdp.grasp_both_ends,
        weight=5.0,
        params={
            "distance_threshold": 0.05,
            "left_body_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "right_body_cfg": SceneEntityCfg("robot", body_names=MISSING),
        },
    )
    left_finger_closure = RewTerm(
        func=mdp.finger_closure_reward,
        weight=0.5,
        params={
            "distance_threshold": 0.05,
            "body_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "finger_cfg": SceneEntityCfg("robot", joint_names=["openarm_left_finger_joint.*"]),
        },
    )
    right_finger_closure = RewTerm(
        func=mdp.finger_closure_reward,
        weight=0.5,
        params={
            "distance_threshold": 0.05,
            "body_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "finger_cfg": SceneEntityCfg("robot", joint_names=["openarm_right_finger_joint.*"]),
        },
    )

    # -- Phase 3: Lift --
    lifting_object = RewTerm(
        func=mdp.object_is_lifted,
        params={"minimal_height": 0.06},
        weight=20.0,
    )

    # -- Phase 4: Goal tracking --
    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=16.0,
    )
    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=5.0,
    )

    # -- Levelness constraint (only meaningful when lifted) --
    levelness_constraint = RewTerm(
        func=mdp.levelness_constraint,
        weight=-2.0,
    )

    # -- Relative distance between hands --
    relative_distance = RewTerm(
        func=mdp.relative_distance_constraint,
        weight=0.5,
        params={
            "target_distance": 0.25,
            "std": 0.2,
            "left_body_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "right_body_cfg": SceneEntityCfg("robot", body_names=MISSING),
        },
    )

    # -- Regularization --
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    left_joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[f"openarm_left_joint{i}" for i in range(1, 8)],
            )
        },
    )
    right_joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[f"openarm_right_joint{i}" for i in range(1, 8)],
            )
        },
    )


@configclass
class TerminationsCfg:
    """Termination conditions."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_dropping = DoneTerm(
        func=mdp.object_dropped,
        params={"minimum_height": -0.05, "object_cfg": SceneEntityCfg("object")},
    )


@configclass
class CurriculumCfg:
    """Gradually increase smoothness penalties."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -5e-3, "num_steps": 10000},
    )
    left_joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "left_joint_vel", "weight": -1e-3, "num_steps": 10000},
    )
    right_joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "right_joint_vel", "weight": -1e-3, "num_steps": 10000},
    )


##
# Environment configuration
##

@configclass
class BimanualLiftEnvCfg(ManagerBasedRLEnvCfg):
    """Base configuration for bimanual lift."""

    scene: BimanualLiftSceneCfg = BimanualLiftSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 10.0
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
```

### Step 7：编写 `lift/config/joint_pos_env_cfg.py`（具体配置）

```python
# ... license ...

import math

from isaaclab.assets import RigidObjectCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, MassPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from .. import mdp
from ..lift_env_cfg import BimanualLiftEnvCfg

from source.openarm.openarm.tasks.manager_based.openarm_manipulation.assets.openarm_bimanual import (
    OPEN_ARM_HIGH_PD_CFG,
)


@configclass
class OpenArmBimanualLiftEnvCfg(BimanualLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # --- Robot ---
        self.scene.robot = OPEN_ARM_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={
                    "openarm_left_joint1": 0.0,
                    "openarm_left_joint2": 0.0,
                    "openarm_left_joint3": 0.0,
                    "openarm_left_joint4": 0.0,
                    "openarm_left_joint5": 0.0,
                    "openarm_left_joint6": 0.0,
                    "openarm_left_joint7": 0.0,
                    "openarm_right_joint1": 0.0,
                    "openarm_right_joint2": 0.0,
                    "openarm_right_joint3": 0.0,
                    "openarm_right_joint4": 0.0,
                    "openarm_right_joint5": 0.0,
                    "openarm_right_joint6": 0.0,
                    "openarm_right_joint7": 0.0,
                    "openarm_left_finger_joint.*": 0.044,  # open
                    "openarm_right_finger_joint.*": 0.044,  # open
                },
            ),
        )

        # --- Actions ---
        self.actions.left_arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[f"openarm_left_joint{i}" for i in range(1, 8)],
            scale=0.5,
            use_default_offset=True,
        )
        self.actions.right_arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[f"openarm_right_joint{i}" for i in range(1, 8)],
            scale=0.5,
            use_default_offset=True,
        )
        self.actions.left_gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["openarm_left_finger_joint.*"],
            open_command_expr={"openarm_left_finger_joint.*": 0.044},
            close_command_expr={"openarm_left_finger_joint.*": 0.0},
        )
        self.actions.right_gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["openarm_right_finger_joint.*"],
            open_command_expr={"openarm_right_finger_joint.*": 0.044},
            close_command_expr={"openarm_right_finger_joint.*": 0.0},
        )

        # --- Object (elongated tray-like) ---
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.35, 0, 0.04], rot=[1, 0, 0, 0]
            ),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(1.5, 0.3, 0.15),  # long in x, thin in y/z
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                mass_props=MassPropertiesCfg(mass=2.0),
            ),
        )

        # --- Command body reference ---
        self.commands.object_pose.body_name = "openarm_left_hand"

        # --- Fill in reward body names ---
        self.rewards.left_reaching_object.params["robot_cfg"].body_names = [
            "openarm_left_hand"
        ]
        self.rewards.right_reaching_object.params["robot_cfg"].body_names = [
            "openarm_right_hand"
        ]
        self.rewards.grasp_both_ends.params["left_body_cfg"].body_names = [
            "openarm_left_hand"
        ]
        self.rewards.grasp_both_ends.params["right_body_cfg"].body_names = [
            "openarm_right_hand"
        ]
        self.rewards.left_finger_closure.params["body_cfg"].body_names = [
            "openarm_left_hand"
        ]
        self.rewards.right_finger_closure.params["body_cfg"].body_names = [
            "openarm_right_hand"
        ]
        self.rewards.relative_distance.params["left_body_cfg"].body_names = [
            "openarm_left_hand"
        ]
        self.rewards.relative_distance.params["right_body_cfg"].body_names = [
            "openarm_right_hand"
        ]


@configclass
class OpenArmBimanualLiftEnvCfg_PLAY(OpenArmBimanualLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
```

### Step 8：编写 `lift/config/agents/rsl_rl_ppo_cfg.py`

```python
# ... license ...

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)
from isaaclab.utils import configclass


@configclass
class OpenArmBimanualLiftPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 50
    experiment_name = "openarm_bimanual_lift"
    run_name = ""
    resume = False
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
```

### Step 9：编写 `lift/config/agents/rl_games_ppo_cfg.yaml`

直接复用已有的双臂 reach 的 yaml 配置，修改 `name` 和 `max_epochs`：

```bash
cp bimanual/reach/config/agents/rl_games_ppo_cfg.yaml \
   bimanual/lift/config/agents/rl_games_ppo_cfg.yaml
```

然后将 yaml 中的 `name: openarm_bi_reach` 改为 `name: openarm_bimanual_lift`，`max_epochs: 3000`。

### Step 10：同理复制 skrl 配置

```bash
cp bimanual/reach/config/agents/skrl_ppo_cfg.yaml \
   bimanual/lift/config/agents/skrl_ppo_cfg.yaml
```

### Step 11：编写 `lift/config/__init__.py`（注册 Gym 环境）

```python
# ... license ...

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Lift-Tray-OpenArm-Bi-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:OpenArmBimanualLiftEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OpenArmBimanualLiftPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Lift-Tray-OpenArm-Bi-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:OpenArmBimanualLiftEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OpenArmBimanualLiftPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
```

### Step 12：创建空文件

```bash
touch bimanual/lift/__init__.py
touch bimanual/lift/config/agents/__init__.py
```

---

## 四、训练

### 4.1 启动训练（RSL-RL）

```bash
cd /home/jintao/isaac_ws/openarm_isaac_lab

python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Lift-Tray-OpenArm-Bi-v0 \
    --num_envs 4096 \
    --headless
```

### 4.2 验证训练效果

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Lift-Tray-OpenArm-Bi-Play-v0 \
    --num_envs 50
```

### 4.3 查看环境列表确认注册成功

```bash
python scripts/tools/list_envs.py
```

应该能看到 `Isaac-Lift-Tray-OpenArm-Bi-v0`。

---

## 五、调试与调参指南

### 5.1 分阶段检查

训练中通过 TensorBoard 观察各项奖励的曲线：

```bash
tensorboard --logdir logs/rsl_rl/openarm_bimanual_lift/
```

**阶段 1（前 ~500 iter）**：`left_reaching_object` 和 `right_reaching_object` 应快速上升，说明双臂学会了靠近物体。

**阶段 2（~500-1500 iter）**：`grasp_both_ends` 和 `finger_closure` 应上升，同时可能看到 `lifting_object` 偶尔触发。

**阶段 3（~1500-2500 iter）**：`lifting_object` 稳定触发，`object_goal_tracking` 开始增长。

**阶段 4（~2500+ iter）**：`levelness_constraint` 和 `relative_distance` 应同时优化。

如果某个阶段的奖励停滞不前，重点调该阶段的权重。

### 5.2 常见问题及调整

| 症状 | 可能原因 | 调整方向 |
|------|---------|---------|
| 双臂不靠近物体 | 接近奖励太弱 | 增大 `reaching_object` 权重到 2.0~3.0 |
| 靠近但不抓取 | 抓取奖励不足，或 `distance_threshold` 太严格 | 放宽 threshold 到 0.08m，增大 `grasp_both_ends` 到 10.0 |
| 抓起后立刻掉落 | 物体太重或 PD 刚度不够 | 降低物体 mass，或使用 `OPEN_ARM_CFG`（正常刚度 + 重力）而非 `OPEN_ARM_HIGH_PD_CFG`（无重力）|
| 托盘严重倾斜 | `levelness_constraint` 权重过低 | 增大到 -5.0 或 -10.0 |
| 双手撞在一起 | `relative_distance` 无效 | 加硬约束（额外终止条件 + 增大标准差惩罚）|
| 手指不闭合 | gripper action 探索不充分 | 在 event 中加入随机手指动作初始化 |

### 5.3 可能的增强方向

1. **两阶段课程学习**：第一阶段只优化接近+抓取（关闭目标跟踪），第二阶段再加入抬升+目标跟踪。
2. **力反馈观测**：如果 USD 模型支持 force/torque sensor，可以将手爪接触力加入观测，帮助判断是否真正抓住了物体。
3. **更真实的托盘**：用自定义 USD 或矩形 primitive 替换 DexCube，使其在视觉上更像托盘。
4. **不对称目标位置**：允许目标位置的 y 坐标在更大范围内采样，迫使双臂在不同位置协同。
5. **移动底座**：引入移动底座协同控制，让双臂系统可以走向物体。

---

## 六、文件清单总览

需要创建的新文件（共 12 个）：

```
bimanual/lift/
├── __init__.py                          # 空文件
├── lift_env_cfg.py                      # 基类（场景 + MDP 抽象配置）
├── mdp/
│   ├── __init__.py                      # re-export
│   ├── rewards.py                       # 9 个自定义奖励函数
│   ├── observations.py                  # 2 个自定义观测函数
│   └── terminations.py                  # 1 个自定义终止函数
└── config/
    ├── __init__.py                      # gym.register × 2
    ├── joint_pos_env_cfg.py             # 具体 robot + object 配置
    └── agents/
        ├── __init__.py                  # 空文件
        ├── rsl_rl_ppo_cfg.py            # PPO 超参数
        ├── rl_games_ppo_cfg.yaml        # 从 reach 复制并修改
        └── skrl_ppo_cfg.yaml            # 从 reach 复制
```

---

## 七、控制关节总结

| 部位 | 关节 | 数量 | 控制方式 |
|------|------|------|---------|
| 左臂 | `openarm_left_joint[1-7]` | **7** | Joint Position Control (scale=0.5) |
| 右臂 | `openarm_right_joint[1-7]` | **7** | Joint Position Control (scale=0.5) |
| 左手爪 | `openarm_left_finger_joint.*` (2 fingers) | **2** | Binary Position (open=0.044, close=0.0) |
| 右手爪 | `openarm_right_finger_joint.*` (2 fingers) | **2** | Binary Position (open=0.044, close=0.0) |
| **总计** | | **18 DOF, 20 维动作** | |
