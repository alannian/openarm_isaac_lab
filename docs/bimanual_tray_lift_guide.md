# 双臂协同托盘举升任务实现指南（最终版）

## 一、任务总览

### 1.1 目标

工作台上放置一个又长又重的托盘，单臂因抓取宽度不足或力量不够而无法平稳举起，
必须左右臂**同时抓住托盘两端**，并将其**平稳举升到目标高度**。

### 1.2 训练价值

| 训练价值 | 具体体现 |
|---------|---------|
| **对称协同** | 两臂必须分别抓住 +Y / -Y 两端，否则托盘倾斜脱落 |
| **姿态约束** | 举升过程中 roll / pitch 趋近于零（托盘保持水平） |
| **时序探索** | 先靠近 → 夹爪闭合 → 同步举升，三个阶段需要依次解锁 |

### 1.3 与现有任务的关系

| 现有任务 | 复用内容 | 新增内容 |
|---------|---------|---------|
| `bimanual/reach` | 双臂机器人模型、双臂动作结构 | — |
| `unimanual/lift` | 桌面场景、物体举升奖励模板 | 双臂协调 + 水平约束 |

---

## 二、整体设计

### 2.1 坐标约定

```
机器人双臂基座在原点：
  X 轴  → 朝前（托盘放在正前方 x ≈ 0.45 m）
  Y 轴  → 朝左（左臂在 +Y 侧，右臂在 -Y 侧）
  Z 轴  → 朝上

托盘长轴沿 Y 轴摆放（长 0.60 m，半长 0.30 m）：
  左端  = 托盘中心 + 0.30 * 托盘局部 Y 轴方向（+Y 端）
  右端  = 托盘中心 - 0.30 * 托盘局部 Y 轴方向（-Y 端）

左臂目标：靠近并夹住左端（+Y）
右臂目标：靠近并夹住右端（-Y）
```

### 2.2 动作空间（18 维）

```
左臂关节位置 × 7   （openarm_left_joint1~7，scale=0.5）
右臂关节位置 × 7   （openarm_right_joint1~7，scale=0.5）
左夹爪开合 × 1     （Binary: open=0.044 / close=0.0）
右夹爪开合 × 1     （同上）
左夹爪关节 × 1     ← 由 BinaryJointPositionAction 统一控制，含 2 根 finger joint
右夹爪关节 × 1     ← 同上
```

### 2.3 观测空间（约 62 维）

| 观测项 | 维度 | 说明 |
|--------|------|------|
| 左臂关节位置 | 7 | 加高斯噪声 ±0.01 |
| 右臂关节位置 | 7 | 同上 |
| 左夹爪关节位置 | 2 | |
| 右夹爪关节位置 | 2 | |
| 左臂关节速度 | 7 | 加高斯噪声 ±0.01 |
| 右臂关节速度 | 7 | 同上 |
| 左夹爪关节速度 | 2 | |
| 右夹爪关节速度 | 2 | |
| 托盘位置（机器人基座系）| 3 | |
| 托盘 roll / pitch | 2 | 感知倾斜状态 |
| 上一步左臂动作 | 7 | |
| 上一步右臂动作 | 7 | |
| 上一步夹爪动作 | 2 | left + right |
| **总计** | **59** | |

> **说明**：目标高度固定为 0.25 m，无需 Command 管理器。
> 若后期需要随机化目标高度，再引入 `UniformScalarCommandCfg` 并在观测中加入该项。

### 2.4 奖励函数总览（10 项）

训练分为四个自然阶段，奖励项逐步激活：

| 阶段 | 主导奖励项 | 目标 |
|------|-----------|------|
| Phase 1（0~500 iter）| `left/right_reach_tray` | 双臂靠近托盘两端 |
| Phase 2（500~1500 iter）| `grasp_both_ends`, `left/right_finger_closure` | 双手同时接触，夹爪闭合 |
| Phase 3（1500~2500 iter）| `tray_lifted`, `tray_goal_height` | 托盘离桌并到达目标高度 |
| Phase 4（2500+ iter）| `tray_tilt_penalty`, `grasp_symmetry` | 托盘水平，双臂对称 |

---

## 三、文件结构（需新建 12 个文件）

```
source/openarm/openarm/tasks/manager_based/openarm_manipulation/bimanual/
├── __init__.py                   ← 已存在，无需修改（自动发现子包）
├── reach/                        ← 已存在，参考用
└── lift/                         ← 【全部新建】
    ├── __init__.py
    ├── lift_env_cfg.py            ← 抽象基类（场景 + MDP）
    ├── mdp/
    │   ├── __init__.py
    │   ├── rewards.py             ← 10 个奖励函数
    │   ├── observations.py        ← 2 个观测函数
    │   └── terminations.py        ← 1 个终止函数
    └── config/
        ├── __init__.py            ← gym.register × 2
        ├── joint_pos_env_cfg.py   ← 绑定 OpenArm 双臂 + 托盘 + EE frame
        └── agents/
            ├── __init__.py
            ├── rsl_rl_ppo_cfg.py
            ├── rl_games_ppo_cfg.yaml
            └── skrl_ppo_cfg.yaml
```

---

## 四、Step-by-Step 实现

### Step 0：创建目录

```bash
cd /home/jintao/isaac_ws/openarm_isaac_lab/source/openarm/openarm/tasks/manager_based/openarm_manipulation/bimanual

mkdir -p lift/mdp
mkdir -p lift/config/agents
```

---

### Step 1：`lift/mdp/rewards.py`

这是整个任务的核心，包含 10 个奖励函数。

```python
# Copyright 2025 Enactic, Inc.
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

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ─────────────────────────────────────────────────────────────────────
# 内部工具：计算托盘两端在世界坐标系中的位置
# ─────────────────────────────────────────────────────────────────────

def _get_tray_ends(
    env: ManagerBasedRLEnv,
    tray_cfg: SceneEntityCfg,
    half_length: float = 0.30,
):
    """返回 (left_end, right_end)，均为 (N, 3) 世界坐标。

    托盘长轴沿局部 Y 轴，half_length 必须与 CuboidCfg size[1]/2 一致。
    left_end  = 托盘中心 + half_length * 旋转后的局部 +Y 方向
    right_end = 托盘中心 - half_length * 旋转后的局部 +Y 方向
    """
    tray: RigidObject = env.scene[tray_cfg.name]
    tray_pos = tray.data.root_pos_w          # (N, 3)
    tray_quat = tray.data.root_quat_w        # (N, 4) wxyz

    local_y = torch.zeros(tray_pos.shape[0], 3, device=tray_pos.device)
    local_y[:, 1] = 1.0
    world_y = quat_apply(tray_quat, local_y)  # (N, 3)

    left_end  = tray_pos + half_length * world_y
    right_end = tray_pos - half_length * world_y
    return left_end, right_end


# ─────────────────────────────────────────────────────────────────────
# 1. EE 靠近托盘对应端（Phase 1）
# ─────────────────────────────────────────────────────────────────────

def ee_reach_tray_end(
    env: ManagerBasedRLEnv,
    std: float,
    ee_frame_cfg: SceneEntityCfg,
    side: str,                               # "left" 或 "right"
    tray_cfg: SceneEntityCfg = SceneEntityCfg("tray"),
    half_length: float = 0.30,
) -> torch.Tensor:
    """tanh 核奖励：末端执行器靠近托盘对应端。

    side="left"  → 左臂靠近 +Y 端
    side="right" → 右臂靠近 -Y 端
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]   # (N, 3)

    left_end, right_end = _get_tray_ends(env, tray_cfg, half_length)
    target = left_end if side == "left" else right_end

    dist = torch.norm(ee_pos - target, dim=1)
    return 1.0 - torch.tanh(dist / std)


# ─────────────────────────────────────────────────────────────────────
# 2. 双手同时接近（Phase 2，一次性触发奖励）
# ─────────────────────────────────────────────────────────────────────

def grasp_both_ends(
    env: ManagerBasedRLEnv,
    distance_threshold: float,
    left_ee_cfg: SceneEntityCfg,
    right_ee_cfg: SceneEntityCfg,
    tray_cfg: SceneEntityCfg = SceneEntityCfg("tray"),
    half_length: float = 0.30,
) -> torch.Tensor:
    """二值奖励：左右手同时进入各自端点的 distance_threshold 范围内时给 1.0。"""
    left_ee: FrameTransformer = env.scene[left_ee_cfg.name]
    right_ee: FrameTransformer = env.scene[right_ee_cfg.name]
    left_pos  = left_ee.data.target_pos_w[..., 0, :]
    right_pos = right_ee.data.target_pos_w[..., 0, :]

    left_end, right_end = _get_tray_ends(env, tray_cfg, half_length)

    left_close  = torch.norm(left_pos  - left_end,  dim=1) < distance_threshold
    right_close = torch.norm(right_pos - right_end, dim=1) < distance_threshold
    return (left_close & right_close).float()


# ─────────────────────────────────────────────────────────────────────
# 3. 夹爪闭合奖励（Phase 2，各手独立）
# ─────────────────────────────────────────────────────────────────────

def finger_closure_reward(
    env: ManagerBasedRLEnv,
    distance_threshold: float,
    ee_frame_cfg: SceneEntityCfg,
    finger_cfg: SceneEntityCfg,
    side: str,
    tray_cfg: SceneEntityCfg = SceneEntityCfg("tray"),
    half_length: float = 0.30,
) -> torch.Tensor:
    """当 EE 靠近对应端时，奖励夹爪闭合；远离时不强求。

    逻辑：
        near & closed  → +1
        near & open    →  0
        far  & open    → +1（不干扰探索）
        far  & closed  →  0
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]
    left_end, right_end = _get_tray_ends(env, tray_cfg, half_length)
    target = left_end if side == "left" else right_end

    near = torch.norm(ee_pos - target, dim=1) < distance_threshold  # bool (N,)

    robot: RigidObject = env.scene[finger_cfg.name]
    finger_pos = robot.data.joint_pos[:, finger_cfg.joint_ids].mean(dim=1)
    closed = finger_pos < 0.005  # 关节位置接近 0 = 完全闭合

    return (near.float() * closed.float()) + ((1 - near.float()) * (1 - closed.float()))


# ─────────────────────────────────────────────────────────────────────
# 4. 托盘被举起（Phase 3，二值）
# ─────────────────────────────────────────────────────────────────────

def tray_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    tray_cfg: SceneEntityCfg = SceneEntityCfg("tray"),
) -> torch.Tensor:
    """托盘质心高于 minimal_height 时给 1.0。"""
    tray: RigidObject = env.scene[tray_cfg.name]
    return torch.where(tray.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


# ─────────────────────────────────────────────────────────────────────
# 5. 托盘到达目标高度（Phase 3，tanh）
# ─────────────────────────────────────────────────────────────────────

def tray_goal_height_tracking(
    env: ManagerBasedRLEnv,
    target_height: float,
    std: float,
    minimal_height: float = 0.04,
    tray_cfg: SceneEntityCfg = SceneEntityCfg("tray"),
) -> torch.Tensor:
    """托盘已被举起时，用 tanh 核奖励高度接近 target_height。"""
    tray: RigidObject = env.scene[tray_cfg.name]
    tray_z = tray.data.root_pos_w[:, 2]
    height_err = torch.abs(tray_z - target_height)
    is_lifted = (tray_z > minimal_height).float()
    return is_lifted * (1.0 - torch.tanh(height_err / std))


# ─────────────────────────────────────────────────────────────────────
# 6. 对称抓取惩罚（Phase 4）
# ─────────────────────────────────────────────────────────────────────

def grasp_symmetry_penalty(
    env: ManagerBasedRLEnv,
    left_ee_cfg: SceneEntityCfg,
    right_ee_cfg: SceneEntityCfg,
    tray_cfg: SceneEntityCfg = SceneEntityCfg("tray"),
) -> torch.Tensor:
    """惩罚左右臂到托盘中心的距离不对称。

    penalty = |dist(left_ee, tray_center) - dist(right_ee, tray_center)|
    权重取负，越不对称惩罚越大。
    """
    tray: RigidObject = env.scene[tray_cfg.name]
    left_ee: FrameTransformer = env.scene[left_ee_cfg.name]
    right_ee: FrameTransformer = env.scene[right_ee_cfg.name]

    tray_center = tray.data.root_pos_w
    left_pos  = left_ee.data.target_pos_w[..., 0, :]
    right_pos = right_ee.data.target_pos_w[..., 0, :]

    dist_left  = torch.norm(left_pos  - tray_center, dim=1)
    dist_right = torch.norm(right_pos - tray_center, dim=1)
    return torch.abs(dist_left - dist_right)


# ─────────────────────────────────────────────────────────────────────
# 7. 托盘水平姿态惩罚（Phase 4）
# ─────────────────────────────────────────────────────────────────────

def tray_tilt_penalty(
    env: ManagerBasedRLEnv,
    max_tilt_rad: float = 0.1,
    tray_cfg: SceneEntityCfg = SceneEntityCfg("tray"),
) -> torch.Tensor:
    """惩罚托盘倾斜，仅在托盘被举起后激活。

    计算物体局部 Z 轴与世界 Z 轴的夹角：
      obj_z_dot_world_z = 1 - 2*(qx² + qy²)
      tilt = 1 - obj_z_dot_world_z   →  0=水平, 最大=2（倒置）

    超过 max_tilt_rad 等效阈值的部分才惩罚，阈值内给容忍度。
    """
    tray: RigidObject = env.scene[tray_cfg.name]
    quat = tray.data.root_quat_w          # (N, 4) wxyz
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    # 物体局部 Z 在世界系的 z 分量 = 1 - 2*(x² + y²)
    obj_z_world_z = 1.0 - 2.0 * (x ** 2 + y ** 2)
    tilt = 1.0 - obj_z_world_z           # 0=水平

    excess = torch.clamp(tilt - max_tilt_rad, min=0.0)
    is_lifted = (tray.data.root_pos_w[:, 2] > 0.04).float()
    return is_lifted * excess             # 权重取负


# ─────────────────────────────────────────────────────────────────────
# 8. 双手间距约束（辅助，促进持续抓握）
# ─────────────────────────────────────────────────────────────────────

def hand_distance_reward(
    env: ManagerBasedRLEnv,
    target_distance: float,
    std: float,
    left_ee_cfg: SceneEntityCfg,
    right_ee_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """奖励双手保持约 target_distance 的间距（tanh 核）。"""
    left_ee: FrameTransformer = env.scene[left_ee_cfg.name]
    right_ee: FrameTransformer = env.scene[right_ee_cfg.name]
    left_pos  = left_ee.data.target_pos_w[..., 0, :]
    right_pos = right_ee.data.target_pos_w[..., 0, :]

    current_dist = torch.norm(left_pos - right_pos, dim=1)
    deviation = torch.abs(current_dist - target_distance) / target_distance
    return 1.0 - torch.tanh(deviation / std)
```

---

### Step 2：`lift/mdp/observations.py`

```python
# Copyright 2025 Enactic, Inc.
# ... (license header same as above) ...

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms, quat_to_euler_xyz

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def tray_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    tray_cfg: SceneEntityCfg = SceneEntityCfg("tray"),
) -> torch.Tensor:
    """托盘质心在机器人基座坐标系下的位置 (N, 3)。"""
    robot: RigidObject = env.scene[robot_cfg.name]
    tray: RigidObject = env.scene[tray_cfg.name]
    tray_pos_w = tray.data.root_pos_w[:, :3]
    tray_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, tray_pos_w
    )
    return tray_pos_b


def tray_roll_pitch(
    env: ManagerBasedRLEnv,
    tray_cfg: SceneEntityCfg = SceneEntityCfg("tray"),
) -> torch.Tensor:
    """托盘的 roll / pitch 欧拉角 (N, 2)，让策略感知倾斜状态。"""
    tray: RigidObject = env.scene[tray_cfg.name]
    euler = quat_to_euler_xyz(tray.data.root_quat_w)   # (N, 3) roll, pitch, yaw
    return euler[:, :2]
```

---

### Step 3：`lift/mdp/terminations.py`

```python
# Copyright 2025 Enactic, Inc.
# ... (license header) ...

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def tray_dropped(
    env: ManagerBasedRLEnv,
    minimum_height: float = -0.05,
    tray_cfg: SceneEntityCfg = SceneEntityCfg("tray"),
) -> torch.Tensor:
    """托盘坠落到桌面以下时终止 episode。"""
    tray: RigidObject = env.scene[tray_cfg.name]
    return tray.data.root_pos_w[:, 2] < minimum_height
```

---

### Step 4：`lift/mdp/__init__.py`

```python
# Copyright 2025 Enactic, Inc.
# ... (license header) ...

"""This sub-module contains the functions that are specific to the bimanual lift environments."""

from isaaclab.envs.mdp import *  # noqa: F401, F403

from .observations import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403
from .terminations import *  # noqa: F401, F403
```

---

### Step 5：`lift/lift_env_cfg.py`（抽象基类）

```python
# Copyright 2025 Enactic, Inc.
# ... (license header) ...

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg, GroundPlaneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp


##
# 场景
##

@configclass
class TrayLiftSceneCfg(InteractiveSceneCfg):
    """双臂托盘举升场景：机器人 + 桌子 + 托盘。
    robot / left_ee_frame / right_ee_frame / tray 由子类填充。
    """
    robot: ArticulationCfg = MISSING
    left_ee_frame: FrameTransformerCfg = MISSING
    right_ee_frame: FrameTransformerCfg = MISSING
    tray: RigidObjectCfg = MISSING

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
        ),
    )
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# 动作
##

@configclass
class ActionsCfg:
    left_arm_action: mdp.JointPositionActionCfg = MISSING
    right_arm_action: mdp.JointPositionActionCfg = MISSING
    left_gripper_action: mdp.BinaryJointPositionActionCfg = MISSING
    right_gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


##
# 观测
##

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # 左臂关节
        left_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg(
                "robot", joint_names=[f"openarm_left_joint{i}" for i in range(1, 8)]
            )},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        left_finger_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg(
                "robot", joint_names=["openarm_left_finger_joint.*"]
            )},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        # 右臂关节
        right_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg(
                "robot", joint_names=[f"openarm_right_joint{i}" for i in range(1, 8)]
            )},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        right_finger_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg(
                "robot", joint_names=["openarm_right_finger_joint.*"]
            )},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        # 关节速度
        left_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg(
                "robot", joint_names=[f"openarm_left_joint{i}" for i in range(1, 8)]
            )},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        left_finger_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg(
                "robot", joint_names=["openarm_left_finger_joint.*"]
            )},
        )
        right_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg(
                "robot", joint_names=[f"openarm_right_joint{i}" for i in range(1, 8)]
            )},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        right_finger_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg(
                "robot", joint_names=["openarm_right_finger_joint.*"]
            )},
        )
        # 托盘状态
        tray_position = ObsTerm(func=mdp.tray_position_in_robot_root_frame)
        tray_tilt = ObsTerm(func=mdp.tray_roll_pitch)
        # 上一步动作
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


##
# 事件（Reset）
##

@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_tray_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (0.0, 0.0),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("tray", body_names="Tray"),
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.8, 1.2), "velocity_range": (0.0, 0.0)},
    )


##
# 奖励
##

@configclass
class RewardsCfg:
    # Phase 1：靠近
    left_reach_tray = RewTerm(
        func=mdp.ee_reach_tray_end,
        weight=1.5,
        params={
            "std": 0.1,
            "ee_frame_cfg": SceneEntityCfg("left_ee_frame"),
            "side": "left",
        },
    )
    right_reach_tray = RewTerm(
        func=mdp.ee_reach_tray_end,
        weight=1.5,
        params={
            "std": 0.1,
            "ee_frame_cfg": SceneEntityCfg("right_ee_frame"),
            "side": "right",
        },
    )

    # Phase 2：双手同时接触
    grasp_both_ends = RewTerm(
        func=mdp.grasp_both_ends,
        weight=5.0,
        params={
            "distance_threshold": 0.06,
            "left_ee_cfg": SceneEntityCfg("left_ee_frame"),
            "right_ee_cfg": SceneEntityCfg("right_ee_frame"),
        },
    )
    left_finger_closure = RewTerm(
        func=mdp.finger_closure_reward,
        weight=0.5,
        params={
            "distance_threshold": 0.06,
            "ee_frame_cfg": SceneEntityCfg("left_ee_frame"),
            "finger_cfg": SceneEntityCfg("robot", joint_names=["openarm_left_finger_joint.*"]),
            "side": "left",
        },
    )
    right_finger_closure = RewTerm(
        func=mdp.finger_closure_reward,
        weight=0.5,
        params={
            "distance_threshold": 0.06,
            "ee_frame_cfg": SceneEntityCfg("right_ee_frame"),
            "finger_cfg": SceneEntityCfg("robot", joint_names=["openarm_right_finger_joint.*"]),
            "side": "right",
        },
    )

    # Phase 3：举起
    tray_lifted = RewTerm(
        func=mdp.tray_is_lifted,
        weight=20.0,
        params={"minimal_height": 0.06},
    )
    tray_goal_height = RewTerm(
        func=mdp.tray_goal_height_tracking,
        weight=16.0,
        params={"target_height": 0.25, "std": 0.1, "minimal_height": 0.04},
    )
    tray_goal_height_fine = RewTerm(
        func=mdp.tray_goal_height_tracking,
        weight=5.0,
        params={"target_height": 0.25, "std": 0.03, "minimal_height": 0.04},
    )

    # Phase 4：协同约束
    grasp_symmetry = RewTerm(
        func=mdp.grasp_symmetry_penalty,
        weight=-2.0,
        params={
            "left_ee_cfg": SceneEntityCfg("left_ee_frame"),
            "right_ee_cfg": SceneEntityCfg("right_ee_frame"),
        },
    )
    tray_tilt = RewTerm(
        func=mdp.tray_tilt_penalty,
        weight=-3.0,
        params={"max_tilt_rad": 0.1},
    )
    hand_distance = RewTerm(
        func=mdp.hand_distance_reward,
        weight=0.5,
        params={
            "target_distance": 0.60,     # 托盘长度，两手保持约 0.60 m
            "std": 0.2,
            "left_ee_cfg": SceneEntityCfg("left_ee_frame"),
            "right_ee_cfg": SceneEntityCfg("right_ee_frame"),
        },
    )

    # 平滑性惩罚
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    left_joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg(
            "robot", joint_names=[f"openarm_left_joint{i}" for i in range(1, 8)]
        )},
    )
    right_joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg(
            "robot", joint_names=[f"openarm_right_joint{i}" for i in range(1, 8)]
        )},
    )


##
# 终止条件
##

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    tray_dropped = DoneTerm(
        func=mdp.tray_dropped,
        params={"minimum_height": -0.05, "tray_cfg": SceneEntityCfg("tray")},
    )


##
# 课程学习
##

@configclass
class CurriculumCfg:
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -5e-3, "num_steps": 20000},
    )
    left_joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "left_joint_vel", "weight": -1e-3, "num_steps": 20000},
    )
    right_joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "right_joint_vel", "weight": -1e-3, "num_steps": 20000},
    )


##
# 顶层环境配置
##

@configclass
class BimanualTrayLiftEnvCfg(ManagerBasedRLEnvCfg):
    """双臂托盘举升基类配置。"""

    scene: TrayLiftSceneCfg = TrayLiftSceneCfg(num_envs=2048, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 10.0          # 比单臂任务更长
        self.sim.dt = 0.01                    # 100 Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
```

---

### Step 6：`lift/config/joint_pos_env_cfg.py`（绑定具体机器人）

```python
# Copyright 2025 Enactic, Inc.
# ... (license header) ...

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from .. import mdp
from ..lift_env_cfg import BimanualTrayLiftEnvCfg

from source.openarm.openarm.tasks.manager_based.openarm_manipulation.assets.openarm_bimanual import (
    OPEN_ARM_CFG,
)


@configclass
class OpenArmTrayLiftEnvCfg(BimanualTrayLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # ── 机器人（使用标准刚度 + 重力，与单臂 lift 保持一致） ──
        self.scene.robot = OPEN_ARM_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={
                    "openarm_left_joint1":  0.0,
                    "openarm_left_joint2":  0.0,
                    "openarm_left_joint3":  0.0,
                    "openarm_left_joint4":  0.0,
                    "openarm_left_joint5":  0.0,
                    "openarm_left_joint6":  0.0,
                    "openarm_left_joint7":  0.0,
                    "openarm_right_joint1": 0.0,
                    "openarm_right_joint2": 0.0,
                    "openarm_right_joint3": 0.0,
                    "openarm_right_joint4": 0.0,
                    "openarm_right_joint5": 0.0,
                    "openarm_right_joint6": 0.0,
                    "openarm_right_joint7": 0.0,
                    "openarm_left_finger_joint.*":  0.044,  # 初始张开
                    "openarm_right_finger_joint.*": 0.044,
                },
            ),
        )

        # ── 托盘（程序化长方体，长轴沿 Y 轴） ──
        # size = (x_width, y_length, z_height) = (0.12, 0.60, 0.03)
        # half_length = 0.30，与 rewards.py 中的 half_length 保持一致
        self.scene.tray = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Tray",
            spawn=sim_utils.CuboidCfg(
                size=(0.12, 0.60, 0.03),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_depenetration_velocity=1.0,
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.5),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.55, 0.35, 0.15), roughness=0.6
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.40, 0.0, 0.055),   # 桌面上，正前方
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )

        # ── 左臂 EE FrameTransformer ──
        # 注意：prim_path 填写机器人根 link，target_frames 填写实际 EE link
        # 请通过 Isaac Sim Stage 面板确认 "openarm_left_ee_tcp" 确实存在
        self.scene.left_ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/openarm_left_link0",
            debug_vis=True,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/openarm_left_ee_tcp",
                    name="left_ee_tcp",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
                ),
            ],
        )

        # ── 右臂 EE FrameTransformer ──
        self.scene.right_ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/openarm_right_link0",
            debug_vis=True,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/openarm_right_ee_tcp",
                    name="right_ee_tcp",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
                ),
            ],
        )

        # ── 动作 ──
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


@configclass
class OpenArmTrayLiftEnvCfg_PLAY(OpenArmTrayLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 3.5
        self.observations.policy.enable_corruption = False
```

---

### Step 7：`lift/config/__init__.py`（注册环境）

```python
# Copyright 2025 Enactic, Inc.
# ... (license header) ...

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Lift-Tray-OpenArm-Bi-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:OpenArmTrayLiftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OpenArmTrayLiftPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Lift-Tray-OpenArm-Bi-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:OpenArmTrayLiftEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OpenArmTrayLiftPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
```

---

### Step 8：`lift/config/agents/rsl_rl_ppo_cfg.py`

```python
# Copyright 2025 Enactic, Inc.
# ... (license header) ...

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)
from isaaclab.utils import configclass


@configclass
class OpenArmTrayLiftPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 48           # episode 更长，增大 rollout 步数
    max_iterations = 3000
    save_interval = 100
    experiment_name = "openarm_bi_tray_lift"
    run_name = ""
    resume = False
    empirical_normalization = True   # 对复杂任务有帮助

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
        entropy_coef=0.005,           # 适当增大熵系数，鼓励早期探索
        num_learning_epochs=8,
        num_mini_batches=8,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
```

---

### Step 9：复制 yaml 配置文件

```bash
# 以下命令在 bimanual/ 目录执行
cp reach/config/agents/rl_games_ppo_cfg.yaml lift/config/agents/rl_games_ppo_cfg.yaml
cp reach/config/agents/skrl_ppo_cfg.yaml     lift/config/agents/skrl_ppo_cfg.yaml
```

然后用文本编辑器打开 `lift/config/agents/rl_games_ppo_cfg.yaml`，
将 `name: openarm_bi_reach` 改为 `name: openarm_bi_tray_lift`，
`max_epochs` 改为 `3000`。

---

### Step 10：创建空的 `__init__.py` 文件

```bash
touch lift/__init__.py
touch lift/config/agents/__init__.py
```

> **关于 bimanual/__init__.py**：
> 当前 `bimanual/__init__.py` 只有 license header，没有显式 import。
> 上层 `tasks/__init__.py` 使用 `import_packages()` 自动发现子包，
> 所以只要 `lift/config/__init__.py` 中有 `gym.register()`，就会被自动找到，
> **无需修改 `bimanual/__init__.py`**。

---

## 五、关键注意事项

### 5.1 EE Link 名称确认

`openarm_left_ee_tcp` 和 `openarm_right_ee_tcp` 必须与 USD 文件中的实际 link 名称完全匹配。
在训练之前，用以下方式确认：

```bash
# 用训练脚本跑 1 步，在日志里会打印 body_names（需在 joint_pos_env_cfg.py 中临时加 print）
# 或直接启动训练，遇到 prim not found 错误时看错误信息里的实际 link 名称
/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Lift-Tray-OpenArm-Bi-v0 \
    --num_envs 1 --max_iterations 1 --headless 2>&1 | grep -i "body_name\|link\|tcp"
```

如果名称不对，修改 `joint_pos_env_cfg.py` 中两个 `FrameTransformerCfg` 的 `target_frames[0].prim_path`。

### 5.2 托盘尺寸与 half_length 的一致性

`CuboidCfg(size=(0.12, 0.60, 0.03))` 中 Y 轴长度为 0.60 m，所以 `half_length=0.30`。

若修改托盘尺寸，**必须同步修改** `rewards.py` 中所有调用 `_get_tray_ends()` 的 `half_length` 参数，
以及 `hand_distance_reward` 中的 `target_distance`。

### 5.3 夹爪接触力调整

如果出现"夹爪碰到托盘但无法夹住"的情况，需要调整 `openarm_bimanual.py` 中夹爪执行器的物理参数：

```python
# 提高夹爪刚度（在 openarm_bimanual.py 中）
"openarm_gripper": ImplicitActuatorCfg(
    ...
    stiffness=5e3,     # 原来 2e3，适当提高
    damping=1e2,
    effort_limit_sim=500.0,   # 原来 333.33，适当提高
),
```

或为托盘材质添加高摩擦系数（在 `CuboidCfg` 的 `visual_material` 中暂不支持物理摩擦，
需要在 physics material 属性中单独设置）。

---

## 六、训练与调试

### 6.1 验证注册

```bash
cd ~/isaac_ws/openarm_isaac_lab
/workspace/isaaclab/isaaclab.sh -p scripts/tools/list_envs.py --headless 2>&1 | grep -iE "tray|error"
# 期望输出：Isaac-Lift-Tray-OpenArm-Bi-v0 和 Isaac-Lift-Tray-OpenArm-Bi-Play-v0
```

> **注意**：必须使用 `/workspace/isaaclab/isaaclab.sh -p` 而非系统 `python`，且需要加 `--headless` 参数（无显示器时）。启动需等待 1~3 分钟。

### 6.2 启动训练

```bash
/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Lift-Tray-OpenArm-Bi-v0 \
    --num_envs 2048 \
    --headless
```

**建议先用小规模验证场景可以跑通：**

```bash
/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Lift-Tray-OpenArm-Bi-v0 \
    --num_envs 64 \
    --max_iterations 5 \
    --headless
```

### 6.3 监控训练曲线

```bash
tensorboard --logdir logs/rsl_rl/openarm_bi_tray_lift
```

**各阶段关键指标：**

| 阶段 | 迭代数 | 现象 | 验证指标 |
|------|--------|------|---------|
| Phase 1 | 0–500 | 双臂开始靠近托盘 | `left_reach_tray` + `right_reach_tray` > 1.5 |
| Phase 2 | 500–1500 | 夹爪闭合，双手同时接触 | `grasp_both_ends` > 0，`finger_closure` > 0 |
| Phase 3 | 1500–2500 | 托盘离桌 | `tray_lifted` > 0，`tray_goal_height` 上升 |
| Phase 4 | 2500+ | 稳定水平举升 | `tray_tilt` 减小，`grasp_symmetry` 减小 |

### 6.4 推理回放

```bash
/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Lift-Tray-OpenArm-Bi-Play-v0 \
    --num_envs 16 \
    --checkpoint logs/rsl_rl/openarm_bi_tray_lift/<timestamp>/model_XXXX.pt
```

---

## 七、调参指南

### 7.1 常见症状与解法

| 症状 | 可能原因 | 调整方法 |
|------|---------|---------|
| 双臂不靠近托盘 | 接近奖励权重不足 | `left/right_reach_tray` weight: 1.5 → 3.0 |
| 靠近但不抓取 | `distance_threshold` 太严 | `grasp_both_ends.distance_threshold`: 0.06 → 0.10 |
| 单臂完成后另一臂不动 | `grasp_both_ends` 的联合触发还不够强 | weight: 5.0 → 10.0 |
| 抓起后立刻掉落 | 夹爪力不够 | 参考 5.3 提高 `stiffness` 和 `effort_limit_sim` |
| 托盘抖动严重 | action 平滑性惩罚不足 | 提前触发 curriculum，或直接把 `action_rate` weight 调大 |
| 托盘严重倾斜 | `tray_tilt_penalty` 激活太晚 | 降低激活阈值 `0.04 → 0.02` |
| 策略陷入单臂局部最优 | reset 初始化过于保守 | 增大 `reset_robot_joints position_range` 到 (0.5, 1.5) |
| 训练收敛极慢 | 策略网络太复杂 | 先用 `[64, 64]` 验证任务可学，再扩到 `[256, 128, 64]` |

### 7.2 两阶段课程学习（可选）

如果三个阶段无法同时学习，可以分两步：

**阶段 A（先训 1000 iter）**：只保留接近 + 抓取奖励，关闭举升和高度奖励：

```python
# 在 lift_env_cfg.py 的 RewardsCfg 中，将举升项权重临时设为 0
tray_lifted = RewTerm(..., weight=0.0)
tray_goal_height = RewTerm(..., weight=0.0)
```

**阶段 B**：加载阶段 A 的 checkpoint，加回举升奖励，继续训练：

```bash
/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Lift-Tray-OpenArm-Bi-v0 \
    --num_envs 2048 \
    --headless \
    --load_run <阶段A的实验目录>
```

---

## 八、进阶扩展

| 方向 | 实现方法 |
|------|---------|
| **随机化目标高度** | 引入 `UniformScalarCommandCfg`，观测加入目标高度维度 |
| **托盘随机初始偏转** | `reset_tray_position` 的 `pose_range` 加 `yaw: (-0.3, 0.3)` |
| **Domain Randomization** | 对托盘质量 (0.8~2.0 kg)、摩擦系数进行随机化，提升 sim2real |
| **虚拟球不滚落** | 在托盘上方 5 cm 添加轻质小球 `RigidObjectCfg`，其坠落触发终止信号 |
| **力反馈观测** | USD 模型支持 contact sensor 时，将夹爪接触力加入观测（12 维），帮助判断是否真正夹住 |
| **移动基座协同** | 将双臂安装在带轮底座上，联合训练底座移动 + 双臂协同 |

---

## 九、文件清单总览

| 文件 | 作用 | 来源 |
|------|------|------|
| `lift/__init__.py` | 空文件（license） | 新建 |
| `lift/lift_env_cfg.py` | 抽象基类 | 新建 |
| `lift/mdp/__init__.py` | re-export mdp | 新建 |
| `lift/mdp/rewards.py` | 10 个奖励函数 | 新建（核心） |
| `lift/mdp/observations.py` | 2 个观测函数 | 新建 |
| `lift/mdp/terminations.py` | 1 个终止函数 | 新建 |
| `lift/config/__init__.py` | gym.register × 2 | 新建 |
| `lift/config/joint_pos_env_cfg.py` | OpenArm 具体配置 | 新建 |
| `lift/config/agents/__init__.py` | 空文件 | 新建 |
| `lift/config/agents/rsl_rl_ppo_cfg.py` | PPO 超参数 | 新建 |
| `lift/config/agents/rl_games_ppo_cfg.yaml` | rl_games 配置 | 从 reach 复制并修改 |
| `lift/config/agents/skrl_ppo_cfg.yaml` | skrl 配置 | 从 reach 复制 |
