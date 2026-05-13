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


def _proximity_signal(distance: torch.Tensor, std: float) -> torch.Tensor:
    return torch.exp(-0.5 * (distance / max(std, 1e-6)) ** 2)


def _finger_grip_signal(
    finger_pos: torch.Tensor,
    target: float = 0.015,
    std: float = 0.007,
) -> torch.Tensor:
    return torch.exp(-0.5 * ((finger_pos - target) / max(std, 1e-6)) ** 2)


def _bimanual_grasp_signal(
    env: ManagerBasedRLEnv,
    grasp_distance_threshold: float,
    left_ee_cfg: SceneEntityCfg,
    right_ee_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    tray_cfg: SceneEntityCfg = SceneEntityCfg("tray"),
    half_length: float = 0.22,
    grip_std: float = 0.007,
) -> torch.Tensor:
    left_ee: FrameTransformer = env.scene[left_ee_cfg.name]
    right_ee: FrameTransformer = env.scene[right_ee_cfg.name]
    left_pos = left_ee.data.target_pos_w[..., 0, :]
    right_pos = right_ee.data.target_pos_w[..., 0, :]

    left_end, right_end = _get_tray_ends(env, tray_cfg, half_length)
    left_dist = torch.norm(left_pos - left_end, dim=1)
    right_dist = torch.norm(right_pos - right_end, dim=1)

    left_robot = env.scene[left_finger_cfg.name]
    right_robot = env.scene[right_finger_cfg.name]
    left_finger = left_robot.data.joint_pos[:, left_finger_cfg.joint_ids].mean(dim=1)
    right_finger = right_robot.data.joint_pos[:, right_finger_cfg.joint_ids].mean(dim=1)

    left_contact = _proximity_signal(left_dist, grasp_distance_threshold)
    right_contact = _proximity_signal(right_dist, grasp_distance_threshold)
    left_grip = _finger_grip_signal(left_finger, std=grip_std)
    right_grip = _finger_grip_signal(right_finger, std=grip_std)
    return left_contact * left_grip * right_contact * right_grip


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
    """连续奖励：左右手越同时接近各自端点，奖励越高。"""
    left_ee: FrameTransformer = env.scene[left_ee_cfg.name]
    right_ee: FrameTransformer = env.scene[right_ee_cfg.name]
    left_pos  = left_ee.data.target_pos_w[..., 0, :]
    right_pos = right_ee.data.target_pos_w[..., 0, :]

    left_end, right_end = _get_tray_ends(env, tray_cfg, half_length)

    left_dist = torch.norm(left_pos - left_end, dim=1)
    right_dist = torch.norm(right_pos - right_end, dim=1)
    return _proximity_signal(left_dist, distance_threshold) * _proximity_signal(
        right_dist, distance_threshold
    )


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

    dist = torch.norm(ee_pos - target, dim=1)
    approach = 1.0 - torch.tanh(dist / distance_threshold)

    robot = env.scene[finger_cfg.name]
    finger_pos = robot.data.joint_pos[:, finger_cfg.joint_ids].mean(dim=1)
    grip_signal = _finger_grip_signal(finger_pos, std=0.006)

    # 越靠近 + 越精确夹住托盘，奖励越高
    return approach * grip_signal


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
    distance_threshold: float | None = None,
    tray_cfg: SceneEntityCfg = SceneEntityCfg("tray"),
    half_length: float = 0.22,
) -> torch.Tensor:
    """奖励双手保持约 target_distance 的间距（tanh 核）。"""
    left_ee: FrameTransformer = env.scene[left_ee_cfg.name]
    right_ee: FrameTransformer = env.scene[right_ee_cfg.name]
    left_pos  = left_ee.data.target_pos_w[..., 0, :]
    right_pos = right_ee.data.target_pos_w[..., 0, :]

    current_dist = torch.norm(left_pos - right_pos, dim=1)
    deviation = torch.abs(current_dist - target_distance) / target_distance
    reward = 1.0 - torch.tanh(deviation / std)
    if distance_threshold is None:
        return reward

    left_end, right_end = _get_tray_ends(env, tray_cfg, half_length)
    left_near = _proximity_signal(torch.norm(left_pos - left_end, dim=1), distance_threshold)
    right_near = _proximity_signal(torch.norm(right_pos - right_end, dim=1), distance_threshold)
    return reward * left_near * right_near


# ─────────────────────────────────────────────────────────────────────
# 9. EE 高度对齐奖励（防止从下方推托盘）
# ─────────────────────────────────────────────────────────────────────

def ee_height_align_reward(

    env: ManagerBasedRLEnv,
    std: float,
    ee_frame_cfg: SceneEntityCfg,
    tray_cfg: SceneEntityCfg = SceneEntityCfg("tray"),
) -> torch.Tensor:
    """奖励 EE z 高度与托盘质心对齐，引导夹爪从侧面正确接近，防止从下方托推。

    当 EE 高度与托盘中心高度误差在 std 范围内时奖励最高；
    偏低（从下方推）和偏高（从上方压）均减少奖励。
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_z = ee_frame.data.target_pos_w[..., 0, 2]
    tray: RigidObject = env.scene[tray_cfg.name]
    tray_z = tray.data.root_pos_w[:, 2]
    z_err = torch.abs(ee_z - tray_z)
    return 1.0 - torch.tanh(z_err / std)


# ─────────────────────────────────────────────────────────────────────
# 10. EE 水平接近方向奖励（不依赖 EE 轴约定，纯位置几何）
# ─────────────────────────────────────────────────────────────────────

def ee_approach_direction_reward(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg,
    side: str,
    distance_std: float = 0.12,
    tray_cfg: SceneEntityCfg = SceneEntityCfg("tray"),
    half_length: float = 0.22,
) -> torch.Tensor:
    """奖励 EE 从托盘端面外侧沿托盘 Y 轴方向水平接近。

    正确的接近方向唯一确定：
      - 左臂：从 +Y 方向接近 +Y 端（displacement 朝 +Y）
      - 右臂：从 -Y 方向接近 -Y 端（displacement 朝 -Y）
      - 两臂接近方向都沿托盘长轴（Y 轴），不从 X 方向或 Z 方向接近

    两个正交约束的乘积：
      1. horizontal：Z 分量小 → 不从上下接近
      2. along_tray_y：位移方向与托盘 Y 轴对齐 → 从端面正外侧接近，不从托盘侧面(X)接近

    只有同时满足"水平 + 沿Y轴"，两个约束才都接近 1.0。
    这确保了左右手臂的接近方向完全镜像对称。
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]   # (N, 3)

    tray: RigidObject = env.scene[tray_cfg.name]
    tray_quat = tray.data.root_quat_w                # (N, 4) wxyz

    left_end, right_end = _get_tray_ends(env, tray_cfg, half_length)
    target = left_end if side == "left" else right_end

    displacement = ee_pos - target                                       # (N, 3)
    dist = torch.norm(displacement, dim=1, keepdim=True).clamp(min=1e-6)
    normalized = displacement / dist                                     # (N, 3) 单位向量
    proximity = _proximity_signal(dist.squeeze(1), distance_std)

    # 约束1：水平接近（Z 分量小）
    horizontal = 1.0 - torch.abs(normalized[:, 2])   # (N,)

    # 约束2：沿托盘 Y 轴方向接近（|dot(normalized, tray_Y_world)| 大）
    # 托盘局部 +Y 轴在世界系的方向
    local_y = torch.zeros(ee_pos.shape[0], 3, device=ee_pos.device)
    local_y[:, 1] = 1.0
    tray_y_world = quat_apply(tray_quat, local_y)    # (N, 3)
    along_tray_y = torch.abs((normalized * tray_y_world).sum(dim=1))    # (N,)

    return proximity * horizontal * along_tray_y


# ─────────────────────────────────────────────────────────────────────
# 11. 条件门控举升奖励（阶段B核心：只有夹住才能得举升奖励）
# ─────────────────────────────────────────────────────────────────────

def tray_is_lifted_grasped(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    grasp_distance_threshold: float,
    left_ee_cfg: SceneEntityCfg,
    right_ee_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    base_height: float = 0.375,
    tray_cfg: SceneEntityCfg = SceneEntityCfg("tray"),
    half_length: float = 0.22,
    grip_std: float = 0.007,
) -> torch.Tensor:
    """连续举升奖励：越稳地夹住托盘并把它从支架上抬起，奖励越高。"""
    tray: RigidObject = env.scene[tray_cfg.name]
    tray_z = tray.data.root_pos_w[:, 2]
    lift_span = max(minimal_height - base_height, 1e-6)
    lift_progress = torch.clamp((tray_z - base_height) / lift_span, min=0.0, max=1.0)
    grasp_signal = _bimanual_grasp_signal(
        env,
        grasp_distance_threshold,
        left_ee_cfg,
        right_ee_cfg,
        left_finger_cfg,
        right_finger_cfg,
        tray_cfg=tray_cfg,
        half_length=half_length,
        grip_std=grip_std,
    )
    return lift_progress * grasp_signal


def tray_goal_height_tracking_grasped(
    env: ManagerBasedRLEnv,
    target_height: float,
    std: float,
    grasp_distance_threshold: float,
    left_ee_cfg: SceneEntityCfg,
    right_ee_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    minimal_height: float = 0.40,
    base_height: float = 0.375,
    tray_cfg: SceneEntityCfg = SceneEntityCfg("tray"),
    half_length: float = 0.22,
    grip_std: float = 0.007,
) -> torch.Tensor:
    """门控高度追踪奖励：只有双手夹住时才奖励高度追踪。"""
    tray: RigidObject = env.scene[tray_cfg.name]
    tray_z = tray.data.root_pos_w[:, 2]
    height_err = torch.abs(tray_z - target_height)
    lift_span = max(minimal_height - base_height, 1e-6)
    lift_progress = torch.clamp((tray_z - base_height) / lift_span, min=0.0, max=1.0)
    grasp_signal = _bimanual_grasp_signal(
        env,
        grasp_distance_threshold,
        left_ee_cfg,
        right_ee_cfg,
        left_finger_cfg,
        right_finger_cfg,
        tray_cfg=tray_cfg,
        half_length=half_length,
        grip_std=grip_std,
    )
    return lift_progress * grasp_signal * (1.0 - torch.tanh(height_err / std))
