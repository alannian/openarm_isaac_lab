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
from isaaclab.assets import Articulation
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

    dist = torch.norm(ee_pos - target, dim=1)
    # 连续接近信号（tanh 核，EE 越靠近托盘端点奖励越高）
    approach = 1.0 - torch.tanh(dist / distance_threshold)

    robot = env.scene[finger_cfg.name]
    finger_pos = robot.data.joint_pos[:, finger_cfg.joint_ids].mean(dim=1)
    # 高斯夹取信号，峰值在 0.015m（托盘厚 0.03m → 每指阻挡位约 0.015m）
    # 物理含义：
    #   finger_pos ≈ 0.044  → 完全张开，奖励 ≈ 0
    #   finger_pos ≈ 0.015  → 托盘在手指之间，真实夹住，奖励 = 1.0
    #   finger_pos ≈ 0.000  → 空握（没夹住任何东西），奖励 ≈ 0.011 ≈ 0
    grip_signal = torch.exp(-0.5 * ((finger_pos - 0.015) / 0.005) ** 2)

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
) -> torch.Tensor:
    """奖励双手保持约 target_distance 的间距（tanh 核）。"""
    left_ee: FrameTransformer = env.scene[left_ee_cfg.name]
    right_ee: FrameTransformer = env.scene[right_ee_cfg.name]
    left_pos  = left_ee.data.target_pos_w[..., 0, :]
    right_pos = right_ee.data.target_pos_w[..., 0, :]

    current_dist = torch.norm(left_pos - right_pos, dim=1)
    deviation = torch.abs(current_dist - target_distance) / target_distance
    return 1.0 - torch.tanh(deviation / std)


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
# 10. 夹爪朝向对齐奖励（Stage A 核心：强迫夹爪以正确姿态接近托盘）
# ─────────────────────────────────────────────────────────────────────

def ee_grasp_orientation_reward(
    env: ManagerBasedRLEnv,
    ee_body_cfg: SceneEntityCfg,
    tray_cfg: SceneEntityCfg = SceneEntityCfg("tray"),
) -> torch.Tensor:
    """奖励夹爪以正确姿态接近托盘（水平插入，上下夹住）。

    托盘是水平放置的薄板（厚 3cm），正确夹取需要：
      - EE 局部 Z 轴（进入/接近方向）水平 → 从托盘端面侧向插入
      - EE 局部 X 或 Y 轴（手指开合方向）竖直 → 一指在托盘上方，一指在下方

    错误行为（之前的设计）：
      - EE Z 轴朝下 → 夹爪像抓柱子一样从上方插入，无法夹住水平薄板

    计算两个正交约束：
      1. horizontal_approach = 1 - |world_z_of_ee_Z_component|  （接近轴应水平，z分量应≈0）
      2. vertical_opening    = max(|world_z_of_ee_X_component|, |world_z_of_ee_Y_component|)
                                 （开合轴应竖直，X或Y中必须有一个z分量≈1）

    两者之积：仅当接近方向正确且开合方向正确时，奖励才接近 1.0。
    """
    robot: Articulation = env.scene[ee_body_cfg.name]
    ee_quat = robot.data.body_quat_w[:, ee_body_cfg.body_ids[0], :]   # (N, 4) wxyz

    # 把 EE 三个局部轴分别变换到世界系
    local_x = torch.zeros(ee_quat.shape[0], 3, device=ee_quat.device)
    local_x[:, 0] = 1.0
    local_y = torch.zeros_like(local_x); local_y[:, 1] = 1.0
    local_z = torch.zeros_like(local_x); local_z[:, 2] = 1.0

    world_x = quat_apply(ee_quat, local_x)   # (N, 3)
    world_y = quat_apply(ee_quat, local_y)
    world_z = quat_apply(ee_quat, local_z)

    # 约束1：进入方向（EE Z）必须水平 → 其世界系 z 分量应接近 0
    horizontal_approach = 1.0 - torch.abs(world_z[:, 2])   # (N,)

    # 约束2：开合方向（EE X 或 Y）必须竖直 → 其世界系 z 分量的绝对值应接近 1
    # 取 X、Y 中 z 分量较大的那个（不用事先知道哪个轴是开合轴）
    vertical_opening = torch.max(
        torch.abs(world_x[:, 2]),
        torch.abs(world_y[:, 2]),
    )   # (N,)

    # 两个约束同时满足才能得高分
    return horizontal_approach * vertical_opening


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
    tray_cfg: SceneEntityCfg = SceneEntityCfg("tray"),
    half_length: float = 0.22,
) -> torch.Tensor:
    """门控举升奖励：必须同时满足双手已夹住托盘 + 托盘高于阈值，才给 1.0。

    这是两阶段课程学习的关键——策略无法通过手臂推托盘来获得举升奖励，
    必须先学会夹住再举起。
    """
    tray: RigidObject = env.scene[tray_cfg.name]
    tray_z = tray.data.root_pos_w[:, 2]

    # 条件1：托盘被举离支架
    is_lifted = tray_z > minimal_height

    # 条件2：左手夹住（EE靠近 + 夹爪闭合）
    left_ee: FrameTransformer = env.scene[left_ee_cfg.name]
    left_pos = left_ee.data.target_pos_w[..., 0, :]
    left_end, right_end = _get_tray_ends(env, tray_cfg, half_length)
    left_near = torch.norm(left_pos - left_end, dim=1) < grasp_distance_threshold
    left_robot = env.scene[left_finger_cfg.name]
    left_finger = left_robot.data.joint_pos[:, left_finger_cfg.joint_ids].mean(dim=1)
    # 范围 (0.005, 0.030) 表示托盘在手指之间
    # 空握=0.0（不在范围内），真实夹住≈0.015（在范围内），张开=0.044（不在范围内）
    left_gripping = (left_finger > 0.005) & (left_finger < 0.030)

    # 条件3：右手夹住（EE靠近 + 夹爪实际夹住托盘）
    right_ee: FrameTransformer = env.scene[right_ee_cfg.name]
    right_pos = right_ee.data.target_pos_w[..., 0, :]
    right_near = torch.norm(right_pos - right_end, dim=1) < grasp_distance_threshold
    right_robot = env.scene[right_finger_cfg.name]
    right_finger = right_robot.data.joint_pos[:, right_finger_cfg.joint_ids].mean(dim=1)
    right_gripping = (right_finger > 0.005) & (right_finger < 0.030)

    grasped = left_near & left_gripping & right_near & right_gripping
    return (is_lifted & grasped).float()


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
    tray_cfg: SceneEntityCfg = SceneEntityCfg("tray"),
    half_length: float = 0.22,
) -> torch.Tensor:
    """门控高度追踪奖励：只有双手夹住时才奖励高度追踪。"""
    tray: RigidObject = env.scene[tray_cfg.name]
    tray_z = tray.data.root_pos_w[:, 2]
    height_err = torch.abs(tray_z - target_height)
    is_lifted = (tray_z > minimal_height).float()

    left_ee: FrameTransformer = env.scene[left_ee_cfg.name]
    left_pos = left_ee.data.target_pos_w[..., 0, :]
    left_end, right_end = _get_tray_ends(env, tray_cfg, half_length)
    left_near = torch.norm(left_pos - left_end, dim=1) < grasp_distance_threshold
    left_robot = env.scene[left_finger_cfg.name]
    left_finger = left_robot.data.joint_pos[:, left_finger_cfg.joint_ids].mean(dim=1)
    left_gripping = (left_finger > 0.005) & (left_finger < 0.030)

    right_ee: FrameTransformer = env.scene[right_ee_cfg.name]
    right_pos = right_ee.data.target_pos_w[..., 0, :]
    right_near = torch.norm(right_pos - right_end, dim=1) < grasp_distance_threshold
    right_robot = env.scene[right_finger_cfg.name]
    right_finger = right_robot.data.joint_pos[:, right_finger_cfg.joint_ids].mean(dim=1)
    right_gripping = (right_finger > 0.005) & (right_finger < 0.030)

    grasped = (left_near & left_gripping & right_near & right_gripping).float()
    return is_lifted * grasped * (1.0 - torch.tanh(height_err / std))
