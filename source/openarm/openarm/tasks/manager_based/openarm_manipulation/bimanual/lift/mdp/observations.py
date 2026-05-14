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
from isaaclab.utils.math import quat_apply, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_tray_ends(
    env: ManagerBasedRLEnv,
    tray_cfg: SceneEntityCfg,
    half_length: float = 0.22,
) -> tuple[torch.Tensor, torch.Tensor]:
    tray: RigidObject = env.scene[tray_cfg.name]
    tray_pos = tray.data.root_pos_w
    tray_quat = tray.data.root_quat_w

    local_y = torch.zeros(tray_pos.shape[0], 3, device=tray_pos.device)
    local_y[:, 1] = 1.0
    world_y = quat_apply(tray_quat, local_y)
    return tray_pos + half_length * world_y, tray_pos - half_length * world_y


def _tray_axes_world(tray_quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    local_x = torch.zeros(tray_quat.shape[0], 3, device=tray_quat.device)
    local_z = torch.zeros(tray_quat.shape[0], 3, device=tray_quat.device)
    local_x[:, 0] = 1.0
    local_z[:, 2] = 1.0
    return quat_apply(tray_quat, local_x), quat_apply(tray_quat, local_z)


def _body_axis_world(body_quat: torch.Tensor, axis_index: int) -> torch.Tensor:
    axis = torch.zeros(body_quat.shape[0], 3, device=body_quat.device)
    axis[:, axis_index] = 1.0
    return quat_apply(body_quat, axis)


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
    quat = tray.data.root_quat_w  # (N, 4) wxyz
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    # roll (x-axis rotation)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    # pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    sinp = torch.clamp(sinp, -1.0, 1.0)
    pitch = torch.asin(sinp)
    return torch.stack([roll, pitch], dim=1)


def ee_to_tray_end_vector_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg,
    side: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    tray_cfg: SceneEntityCfg = SceneEntityCfg("tray"),
    half_length: float = 0.22,
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    left_end, right_end = _get_tray_ends(env, tray_cfg, half_length)
    target_pos_w = left_end if side == "left" else right_end

    ee_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, ee_pos_w
    )
    target_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, target_pos_w
    )
    return target_pos_b - ee_pos_b


def hand_grasp_pose_metrics(
    env: ManagerBasedRLEnv,
    hand_cfg: SceneEntityCfg,
    tray_cfg: SceneEntityCfg = SceneEntityCfg("tray"),
) -> torch.Tensor:
    robot: RigidObject = env.scene[hand_cfg.name]
    tray: RigidObject = env.scene[tray_cfg.name]

    hand_quat = robot.data.body_quat_w[:, hand_cfg.body_ids[0]]
    tray_x_world, tray_z_world = _tray_axes_world(tray.data.root_quat_w)
    hand_close_world = _body_axis_world(hand_quat, axis_index=1)
    hand_forward_world = _body_axis_world(hand_quat, axis_index=2)

    close_align = torch.abs((hand_close_world * tray_z_world).sum(dim=1))
    forward_align = torch.abs((hand_forward_world * tray_x_world).sum(dim=1))
    return torch.stack([close_align, forward_align], dim=1)
