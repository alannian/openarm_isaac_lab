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
    side: str,
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
# 2. 双手同时接近（Phase 2，联合触发奖励）
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

    robot = env.scene[finger_cfg.name]
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
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]  # noqa: F841

    # 物体局部 Z 在世界系的 z 分量 = 1 - 2*(x² + y²)
    obj_z_world_z = 1.0 - 2.0 * (x ** 2 + y ** 2)
    tilt = 1.0 - obj_z_world_z           # 0=水平

    excess = torch.clamp(tilt - max_tilt_rad, min=0.0)
    is_lifted = (tray.data.root_pos_w[:, 2] > 0.40).float()
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
