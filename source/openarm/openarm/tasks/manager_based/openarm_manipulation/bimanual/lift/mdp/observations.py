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

"""双臂托盘举升任务的观测项（重构版）。

提供策略学习所需的几何线索：TCP 位置、到把手抓取点的相对向量、托盘位姿/水平度/
速度、手部"朝下"标量。所有 3D 量给在机器人 root 系下（对底座绝对位置不变），
姿态用轴向投影（避免欧拉角奇异）。
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import quat_apply, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# 几何默认值（与 rewards.py / tray.usda / lift_env_cfg.py 一致）
HALF_GRASP_Y = 0.22
GRASP_Z_OFFSET = 0.035


# ─────────────────────────────────────────────────────────────────────
# 内部工具
# ─────────────────────────────────────────────────────────────────────

def _to_root_frame(robot, point_w: torch.Tensor) -> torch.Tensor:
    point_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, point_w
    )
    return point_b


def _handle_target_w(
    env: ManagerBasedRLEnv,
    side: str,
    tray_cfg: SceneEntityCfg,
    half_grasp_y: float,
    grasp_z_offset: float,
) -> torch.Tensor:
    """某侧把手抓取点世界坐标 (N, 3)，定义与 rewards._handle_target_w 一致。"""
    tray: RigidObject = env.scene[tray_cfg.name]
    pos = tray.data.root_pos_w
    quat = tray.data.root_quat_w
    sign = 1.0 if side == "left" else -1.0
    local = torch.zeros_like(pos)
    local[:, 1] = sign * half_grasp_y
    local[:, 2] = grasp_z_offset
    return pos + quat_apply(quat, local)


# ─────────────────────────────────────────────────────────────────────
# 1. 托盘位姿（root 系）
# ─────────────────────────────────────────────────────────────────────

def tray_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    tray_cfg: SceneEntityCfg = SceneEntityCfg("tray"),
) -> torch.Tensor:
    """托盘质心位置 (N, 3)，root 系。"""
    robot: RigidObject = env.scene[robot_cfg.name]
    tray: RigidObject = env.scene[tray_cfg.name]
    return _to_root_frame(robot, tray.data.root_pos_w[:, :3])


def tray_orientation_features(
    env: ManagerBasedRLEnv,
    tray_cfg: SceneEntityCfg = SceneEntityCfg("tray"),
) -> torch.Tensor:
    """托盘姿态摘要 (N, 4) = [tilt_x, tilt_y, long_axis_x, long_axis_y]。

    - tilt_x/y：托盘局部 +Z 在世界系的 x/y 分量（水平时为 0），刻画倾斜方向与程度。
    - long_axis_x/y：托盘局部 +Y（长轴）在世界系的水平投影，刻画偏航。
    用轴向投影而非欧拉角，避免万向锁与不连续。
    """
    tray: RigidObject = env.scene[tray_cfg.name]
    quat = tray.data.root_quat_w
    local_y = torch.zeros(quat.shape[0], 3, device=quat.device)
    local_z = torch.zeros(quat.shape[0], 3, device=quat.device)
    local_y[:, 1] = 1.0
    local_z[:, 2] = 1.0
    world_y = quat_apply(quat, local_y)
    world_z = quat_apply(quat, local_z)
    return torch.stack([world_z[:, 0], world_z[:, 1], world_y[:, 0], world_y[:, 1]], dim=1)


def tray_linear_velocity(
    env: ManagerBasedRLEnv,
    tray_cfg: SceneEntityCfg = SceneEntityCfg("tray"),
) -> torch.Tensor:
    """托盘线速度 (N, 3)，世界系。"""
    tray: RigidObject = env.scene[tray_cfg.name]
    return tray.data.root_lin_vel_w


def tray_angular_velocity(
    env: ManagerBasedRLEnv,
    tray_cfg: SceneEntityCfg = SceneEntityCfg("tray"),
) -> torch.Tensor:
    """托盘角速度 (N, 3)，世界系（感知摇晃）。"""
    tray: RigidObject = env.scene[tray_cfg.name]
    return tray.data.root_ang_vel_w


# ─────────────────────────────────────────────────────────────────────
# 2. TCP 与把手抓取点的相对几何（root 系）
# ─────────────────────────────────────────────────────────────────────

def ee_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """TCP 在 root 系的位置 (N, 3)。"""
    robot: RigidObject = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return _to_root_frame(robot, ee_frame.data.target_pos_w[..., 0, :])


def ee_to_handle_vector_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg,
    side: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    tray_cfg: SceneEntityCfg = SceneEntityCfg("tray"),
    half_grasp_y: float = HALF_GRASP_Y,
    grasp_z_offset: float = GRASP_Z_OFFSET,
) -> torch.Tensor:
    """从 TCP 指向该侧把手抓取点的位移向量 (N, 3)，root 系。

    抓取点与 rewards 中完全一致，让策略直接感知"还差多少"。
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    target_w = _handle_target_w(env, side, tray_cfg, half_grasp_y, grasp_z_offset)
    ee_b = _to_root_frame(robot, ee_pos_w)
    target_b = _to_root_frame(robot, target_w)
    return target_b - ee_b


# ─────────────────────────────────────────────────────────────────────
# 3. 手部姿态摘要（标量，不依赖未确认的轴细节）
# ─────────────────────────────────────────────────────────────────────

def hand_down_alignment(
    env: ManagerBasedRLEnv,
    hand_cfg: SceneEntityCfg,
    axis: int = 2,
) -> torch.Tensor:
    """手部某局部轴与世界 -Z 的对齐量 (N, 1) ∈ [-1, 1]；朝下时 ≈ +1。"""
    robot: Articulation = env.scene[hand_cfg.name]
    quat = robot.data.body_quat_w[:, hand_cfg.body_ids[0]]
    local = torch.zeros(quat.shape[0], 3, device=quat.device)
    local[:, axis] = 1.0
    world_axis = quat_apply(quat, local)
    return (-world_axis[:, 2:3]).clone()  # (N, 1)


def hand_span_alignment(
    env: ManagerBasedRLEnv,
    hand_cfg: SceneEntityCfg,
    tray_cfg: SceneEntityCfg = SceneEntityCfg("tray"),
    span_axis: int = 1,
    handle_long_axis: int = 0,
) -> torch.Tensor:
    """夹爪开合轴（hand 局部 ±Y）与把手长轴（tray 局部 ±X）垂直的程度 (N, 1) ∈ [0, 1]。

    越接近 1 表示越正交（越正确：夹爪正好横跨把手细杆闭合）。
    """
    robot: Articulation = env.scene[hand_cfg.name]
    hand_quat = robot.data.body_quat_w[:, hand_cfg.body_ids[0]]
    tray: RigidObject = env.scene[tray_cfg.name]
    tray_quat = tray.data.root_quat_w

    local_hand = torch.zeros(hand_quat.shape[0], 3, device=hand_quat.device)
    local_hand[:, span_axis] = 1.0
    hand_span_world = quat_apply(hand_quat, local_hand)

    local_tray = torch.zeros(tray_quat.shape[0], 3, device=tray_quat.device)
    local_tray[:, handle_long_axis] = 1.0
    handle_long_world = quat_apply(tray_quat, local_tray)

    dot = (hand_span_world * handle_long_world).sum(dim=1, keepdim=True).abs()
    return (1.0 - dot).clamp(min=0.0)
