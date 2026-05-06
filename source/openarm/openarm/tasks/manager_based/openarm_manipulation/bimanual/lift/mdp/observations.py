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
from isaaclab.utils.math import subtract_frame_transforms

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
    quat = tray.data.root_quat_w  # (N, 4) wxyz
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    # roll (x-axis rotation)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    # pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    sinp = torch.clamp(sinp, -1.0, 1.0)
    pitch = torch.asin(sinp)
    return torch.stack([roll, pitch], dim=1)
