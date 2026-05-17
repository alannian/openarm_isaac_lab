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

"""双臂托盘举升任务的观测项。

整体设计思想：
- 提供策略学习所需的"几何线索"：TCP 位置、托盘端目标、相对向量、手部"朝下"对齐量。
- 不堆砌冗余角度，避免观测维度过大反而稀释信号。
- 所有 3D 量都给在机器人 root 坐标系下，姿态相关给标量对齐分（不依赖未确认的轴约定）。
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


# ─────────────────────────────────────────────────────────────────────
# 内部工具
# ─────────────────────────────────────────────────────────────────────

def _tray_ends_world(
    env: ManagerBasedRLEnv,
    tray_cfg: SceneEntityCfg,
    half_length: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 (left_end_w, right_end_w)，均为 (N, 3)。

    约定托盘长轴沿其局部 +Y。
        left_end  = root + half_length * R(+Y)
        right_end = root - half_length * R(+Y)
    """
    tray: RigidObject = env.scene[tray_cfg.name]
    tray_pos = tray.data.root_pos_w
    tray_quat = tray.data.root_quat_w
    local_y = torch.zeros_like(tray_pos)
    local_y[:, 1] = 1.0
    world_y = quat_apply(tray_quat, local_y)
    return tray_pos + half_length * world_y, tray_pos - half_length * world_y


def _to_root_frame(robot, point_w: torch.Tensor) -> torch.Tensor:
    """把世界坐标的点投影到 robot root 坐标系。"""
    point_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, point_w
    )
    return point_b


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
    """托盘姿态摘要 (N, 4)：

        [tilt_x, tilt_y, long_axis_x, long_axis_y]

    - tilt_x / tilt_y：托盘局部 +Z 在世界系中的 x/y 分量，刻画倾斜方向与程度（水平时全为 0）。
    - long_axis_x / long_axis_y：托盘局部 +Y（长轴）在世界系下的水平投影（不取 z，避免冗余）。

    选用"轴向投影"而不是欧拉角，避免万向锁与奇异点带来的不连续。
    """
    tray: RigidObject = env.scene[tray_cfg.name]
    quat = tray.data.root_quat_w
    local_y = torch.zeros(quat.shape[0], 3, device=quat.device)
    local_z = torch.zeros(quat.shape[0], 3, device=quat.device)
    local_y[:, 1] = 1.0
    local_z[:, 2] = 1.0
    world_y = quat_apply(quat, local_y)  # 长轴
    world_z = quat_apply(quat, local_z)  # 法向
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
    """托盘角速度 (N, 3)，世界系。让策略感知摇晃。"""
    tray: RigidObject = env.scene[tray_cfg.name]
    return tray.data.root_ang_vel_w


# ─────────────────────────────────────────────────────────────────────
# 2. TCP 与抓取目标的相对几何（root 系）
# ─────────────────────────────────────────────────────────────────────

def ee_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """TCP 在 root 系下的位置 (N, 3)。"""
    robot: RigidObject = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return _to_root_frame(robot, ee_frame.data.target_pos_w[..., 0, :])


def ee_to_tray_end_vector_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg,
    side: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    tray_cfg: SceneEntityCfg = SceneEntityCfg("tray"),
    half_length: float = 0.25,
    grasp_z_offset: float = 0.02,
) -> torch.Tensor:
    """从 TCP 指向托盘端"抓取点"的位移向量 (N, 3)，root 系。

    抓取点定义：托盘端中心 + (0, 0, grasp_z_offset)，即托盘端正上方 2cm 处，
    与奖励里使用的目标点保持一致，让策略能直接感知"还差多少"。
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    left_end, right_end = _tray_ends_world(env, tray_cfg, half_length)
    target = left_end if side == "left" else right_end
    target = target.clone()
    target[:, 2] += grasp_z_offset

    ee_b = _to_root_frame(robot, ee_pos_w)
    target_b = _to_root_frame(robot, target)
    return target_b - ee_b


# ─────────────────────────────────────────────────────────────────────
# 3. 手部姿态摘要（朝下程度，不依赖具体轴约定）
# ─────────────────────────────────────────────────────────────────────

def hand_down_alignment(
    env: ManagerBasedRLEnv,
    hand_cfg: SceneEntityCfg,
    axis: int = 2,
) -> torch.Tensor:
    """手部某局部轴与世界 -Z 的对齐量 (N, 1) ∈ [-1, 1]。

    上手类机器人惯例：hand 局部 +Z 指向夹爪前向（手腕向指尖）。
    手"朝下"时，hand_z_world · (-world_z) ≈ +1。
    用作 top-down 抓取的姿态线索。
    """
    robot: Articulation = env.scene[hand_cfg.name]
    quat = robot.data.body_quat_w[:, hand_cfg.body_ids[0]]
    local = torch.zeros(quat.shape[0], 3, device=quat.device)
    local[:, axis] = 1.0
    world_axis = quat_apply(quat, local)
    return (-world_axis[:, 2:3]).clone()  # (N, 1)


def hand_yaw_alignment(
    env: ManagerBasedRLEnv,
    hand_cfg: SceneEntityCfg,
    tray_cfg: SceneEntityCfg = SceneEntityCfg("tray"),
    span_axis: int = 1,
) -> torch.Tensor:
    """手部"夹爪展开轴"与托盘长轴垂直的程度 (N, 1) ∈ [0, 1]。

    parallel jaw 夹爪通常沿 hand 局部 ±Y 张开；为了从上方夹住 bar，
    展开方向应在水平面内、且与 bar 长轴垂直 → hand_y_world · tray_y_world ≈ 0。

    返回 |1 - |dot||，越接近 1 表示越正交（也就是越正确）。
    """
    robot: Articulation = env.scene[hand_cfg.name]
    hand_quat = robot.data.body_quat_w[:, hand_cfg.body_ids[0]]
    tray: RigidObject = env.scene[tray_cfg.name]
    tray_quat = tray.data.root_quat_w

    local_hand = torch.zeros(hand_quat.shape[0], 3, device=hand_quat.device)
    local_hand[:, span_axis] = 1.0
    hand_span_world = quat_apply(hand_quat, local_hand)

    local_tray_y = torch.zeros(tray_quat.shape[0], 3, device=tray_quat.device)
    local_tray_y[:, 1] = 1.0
    tray_long_world = quat_apply(tray_quat, local_tray_y)

    dot = (hand_span_world * tray_long_world).sum(dim=1, keepdim=True).abs()
    return (1.0 - dot).clamp(min=0.0)
