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

"""双臂托盘举升任务的奖励项（从头重新设计）。

核心理念：
1. **加性优先，乘性最少**。多个奖励项相乘会让任意一项为 0 时整体梯度消失。
   只在"是否已完成抓取"这种二段式判断里使用最小的门控。
2. **每个 reward 单一职责**，互不重叠。这样课程学习里调权重不会出现耦合反弹。
3. **奖励范围 ≈ [0, 1] 单位化**，权重直接体现"该项重要性"。
4. **几何统一**：所有 reward 共用一组目标点定义函数，避免不同地方对 half_length / grasp_offset 的语义漂移。

任务流程被分解为：
    A. 接近：把 TCP 移到托盘端正上方
    B. 下沉与对齐：让 TCP 处于 grasp 高度、手朝下、夹爪展开轴与托盘长轴垂直
    C. 闭合：在 grasp 半径内闭合夹爪
    D. 举升：双手都已闭合并处于 grasp 半径时，奖励托盘升高直到目标高度
    E. 平稳：举起后惩罚倾斜与摆动
"""

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
# 内部工具
# ─────────────────────────────────────────────────────────────────────

def _tray_ends_world(
    env: ManagerBasedRLEnv,
    tray_cfg: SceneEntityCfg,
    half_length: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """托盘两端（沿托盘局部 +Y）的世界坐标 (N, 3) × 2。"""
    tray: RigidObject = env.scene[tray_cfg.name]
    tray_pos = tray.data.root_pos_w
    tray_quat = tray.data.root_quat_w
    local_y = torch.zeros_like(tray_pos)
    local_y[:, 1] = 1.0
    world_y = quat_apply(tray_quat, local_y)
    return tray_pos + half_length * world_y, tray_pos - half_length * world_y


def _grasp_target(
    env: ManagerBasedRLEnv,
    tray_cfg: SceneEntityCfg,
    side: str,
    half_length: float,
    z_offset: float,
) -> torch.Tensor:
    """抓取目标点 (N, 3)，世界系：托盘端正上方 z_offset。"""
    left, right = _tray_ends_world(env, tray_cfg, half_length)
    target = left if side == "left" else right
    target = target.clone()
    target[:, 2] = target[:, 2] + z_offset
    return target


def _ee_pos(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg) -> torch.Tensor:
    """TCP 世界坐标 (N, 3)。"""
    ee: FrameTransformer = env.scene[ee_frame_cfg.name]
    return ee.data.target_pos_w[..., 0, :]


def _finger_pos(env: ManagerBasedRLEnv, finger_cfg: SceneEntityCfg) -> torch.Tensor:
    """夹爪平均关节位置 (N,)；越小越闭合（0 = 完全闭合, 0.044 = 完全张开）。"""
    robot = env.scene[finger_cfg.name]
    return robot.data.joint_pos[:, finger_cfg.joint_ids].mean(dim=1)


def _hand_axis_world(env: ManagerBasedRLEnv, hand_cfg: SceneEntityCfg, axis: int) -> torch.Tensor:
    """手部局部某轴在世界系下的方向向量 (N, 3)。"""
    robot = env.scene[hand_cfg.name]
    quat = robot.data.body_quat_w[:, hand_cfg.body_ids[0]]
    local = torch.zeros(quat.shape[0], 3, device=quat.device)
    local[:, axis] = 1.0
    return quat_apply(quat, local)


def _is_grasped_per_env(
    env: ManagerBasedRLEnv,
    left_ee_cfg: SceneEntityCfg,
    right_ee_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    tray_cfg: SceneEntityCfg,
    half_length: float,
    grasp_radius: float,
    finger_closed_thresh: float,
    grasp_z_offset: float,
) -> torch.Tensor:
    """门控掩码 (N,) ∈ [0, 1]：双手都进入 grasp 半径且都闭合到一定程度时为 1。

    使用连续 sigmoid-like 内核，避免硬阈值带来的梯度断裂。
    """
    left_pos = _ee_pos(env, left_ee_cfg)
    right_pos = _ee_pos(env, right_ee_cfg)
    left_tgt = _grasp_target(env, tray_cfg, "left", half_length, grasp_z_offset)
    right_tgt = _grasp_target(env, tray_cfg, "right", half_length, grasp_z_offset)

    left_d = torch.norm(left_pos - left_tgt, dim=1)
    right_d = torch.norm(right_pos - right_tgt, dim=1)
    left_close = _finger_pos(env, left_finger_cfg)
    right_close = _finger_pos(env, right_finger_cfg)

    # 连续指示：r→0 越小，c→0 越小（更闭合）越好
    pos_kernel = lambda d, r: 1.0 / (1.0 + (d / r) ** 4)
    close_kernel = lambda c, t: 1.0 / (1.0 + ((c - t).clamp(min=0.0) / 0.012) ** 4)

    left_grip = pos_kernel(left_d, grasp_radius) * close_kernel(left_close, finger_closed_thresh)
    right_grip = pos_kernel(right_d, grasp_radius) * close_kernel(right_close, finger_closed_thresh)
    return left_grip * right_grip


# ─────────────────────────────────────────────────────────────────────
# A. 接近：TCP → 托盘端上方
# ─────────────────────────────────────────────────────────────────────

def reach_grasp_target(
    env: ManagerBasedRLEnv,
    std: float,
    ee_frame_cfg: SceneEntityCfg,
    side: str,
    tray_cfg: SceneEntityCfg = SceneEntityCfg("tray"),
    half_length: float = 0.25,
    grasp_z_offset: float = 0.02,
) -> torch.Tensor:
    """tanh 内核：TCP 越接近 grasp 目标点，奖励越高（∈ [0, 1]）。

    grasp 目标点 = 托盘端 + (0, 0, grasp_z_offset)，是手腕实际应到达的位置。
    使用 1 - tanh 让梯度在远距离时仍然存在，避免 reach-failure。
    """
    ee_pos = _ee_pos(env, ee_frame_cfg)
    target = _grasp_target(env, tray_cfg, side, half_length, grasp_z_offset)
    dist = torch.norm(ee_pos - target, dim=1)
    return 1.0 - torch.tanh(dist / std)


def reach_grasp_target_fine(
    env: ManagerBasedRLEnv,
    std: float,
    ee_frame_cfg: SceneEntityCfg,
    side: str,
    tray_cfg: SceneEntityCfg = SceneEntityCfg("tray"),
    half_length: float = 0.25,
    grasp_z_offset: float = 0.02,
) -> torch.Tensor:
    """同上但 std 更小，提供"最后一公里"的密集梯度，权重比 coarse 项小。"""
    return reach_grasp_target(env, std, ee_frame_cfg, side, tray_cfg, half_length, grasp_z_offset)


# ─────────────────────────────────────────────────────────────────────
# B. 接近偏置：必须从上方接近（EE 不得低于托盘）
# ─────────────────────────────────────────────────────────────────────

def ee_above_tray_penalty(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg,
    tray_cfg: SceneEntityCfg = SceneEntityCfg("tray"),
    margin: float = 0.0,
) -> torch.Tensor:
    """惩罚 EE z 低于 (tray_z + margin) 的部分，平方项。

    设计动机：top-down 抓取必须从托盘上方下沉，禁止 EE 从下方掏起或从侧面横插。
    返回非负值，使用时权重应取负。
    """
    ee_z = _ee_pos(env, ee_frame_cfg)[:, 2]
    tray: RigidObject = env.scene[tray_cfg.name]
    tray_z = tray.data.root_pos_w[:, 2]
    deficit = torch.clamp((tray_z + margin) - ee_z, min=0.0)
    return deficit ** 2


# ─────────────────────────────────────────────────────────────────────
# C. 手部朝向：开口轴向下 + 张开方向与托盘长轴垂直
# ─────────────────────────────────────────────────────────────────────

def hand_pointing_down(
    env: ManagerBasedRLEnv,
    hand_cfg: SceneEntityCfg,
    forward_axis: int = 2,
) -> torch.Tensor:
    """奖励手部 forward 轴指向世界 -Z 方向 (∈ [0, 1])。

    上手类机器人惯例：hand 局部 +Z 指向手腕->指尖方向。
    `score = clamp(-axis_world_z, 0, 1)`：朝下时为 1，水平时为 0，朝上时为 0（被截断）。
    """
    axis_world = _hand_axis_world(env, hand_cfg, forward_axis)
    return (-axis_world[:, 2]).clamp(min=0.0, max=1.0)


def gripper_yaw_align(
    env: ManagerBasedRLEnv,
    hand_cfg: SceneEntityCfg,
    tray_cfg: SceneEntityCfg = SceneEntityCfg("tray"),
    span_axis: int = 1,
) -> torch.Tensor:
    """奖励夹爪展开轴（hand 局部 ±Y）与托盘长轴（tray 局部 ±Y）垂直 (∈ [0, 1])。

    parallel jaw 夹爪要从上方夹住 bar，两指方向必须横跨 bar 的长轴 →
    span_world ⟂ bar_long_axis_world → |span · long| 接近 0 → 奖励 1 - |·|。
    """
    span_world = _hand_axis_world(env, hand_cfg, span_axis)
    tray: RigidObject = env.scene[tray_cfg.name]
    tray_quat = tray.data.root_quat_w
    local_y = torch.zeros(tray_quat.shape[0], 3, device=tray_quat.device)
    local_y[:, 1] = 1.0
    long_world = quat_apply(tray_quat, local_y)
    dot = (span_world * long_world).sum(dim=1).abs()
    return (1.0 - dot).clamp(min=0.0, max=1.0)


# ─────────────────────────────────────────────────────────────────────
# D. 夹爪闭合：仅当 TCP 接近 grasp 点时才奖励
# ─────────────────────────────────────────────────────────────────────

def gripper_close_when_near(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg,
    finger_cfg: SceneEntityCfg,
    side: str,
    grasp_radius: float = 0.06,
    target_finger_pos: float = 0.012,
    tray_cfg: SceneEntityCfg = SceneEntityCfg("tray"),
    half_length: float = 0.25,
    grasp_z_offset: float = 0.02,
) -> torch.Tensor:
    """在 grasp 半径内时奖励 finger 闭合到 target_finger_pos 附近。

    near_kernel = 1 / (1 + (d / r)^4)，r 外快速衰减到 ~0
    close_kernel = 1 / (1 + ((f - target).clamp_min(0) / 0.012)^4)
        ：只惩罚"开"，不惩罚"过分闭合"（接触托盘后 f 会被卡在 ~target）

    远离 grasp 点时该项接近 0 → 策略不会有"全程闭合"的退化解。
    """
    ee_pos = _ee_pos(env, ee_frame_cfg)
    target = _grasp_target(env, tray_cfg, side, half_length, grasp_z_offset)
    dist = torch.norm(ee_pos - target, dim=1)
    near = 1.0 / (1.0 + (dist / grasp_radius) ** 4)

    f = _finger_pos(env, finger_cfg)
    close = 1.0 / (1.0 + ((f - target_finger_pos).clamp(min=0.0) / 0.012) ** 4)
    return near * close


# ─────────────────────────────────────────────────────────────────────
# E. 举升：双手都已抓住时奖励托盘升高 → 目标高度
# ─────────────────────────────────────────────────────────────────────

def lift_progress_when_grasped(
    env: ManagerBasedRLEnv,
    base_height: float,
    target_height: float,
    left_ee_cfg: SceneEntityCfg,
    right_ee_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    tray_cfg: SceneEntityCfg = SceneEntityCfg("tray"),
    half_length: float = 0.25,
    grasp_radius: float = 0.08,
    grasp_z_offset: float = 0.02,
    finger_closed_thresh: float = 0.025,
) -> torch.Tensor:
    """线性奖励：托盘 z 从 base_height → target_height，∈ [0, 1]，乘以"已抓"门控掩码。

    门控的存在确保策略不会把托盘"撞飞"或"撩起来"——必须真的夹住才算分。
    """
    tray: RigidObject = env.scene[tray_cfg.name]
    tray_z = tray.data.root_pos_w[:, 2]
    span = max(target_height - base_height, 1e-6)
    progress = ((tray_z - base_height) / span).clamp(min=0.0, max=1.0)

    grasped = _is_grasped_per_env(
        env, left_ee_cfg, right_ee_cfg, left_finger_cfg, right_finger_cfg,
        tray_cfg, half_length, grasp_radius, finger_closed_thresh, grasp_z_offset,
    )
    return progress * grasped


def goal_height_tracking_when_grasped(
    env: ManagerBasedRLEnv,
    target_height: float,
    std: float,
    left_ee_cfg: SceneEntityCfg,
    right_ee_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    tray_cfg: SceneEntityCfg = SceneEntityCfg("tray"),
    half_length: float = 0.25,
    grasp_radius: float = 0.08,
    grasp_z_offset: float = 0.02,
    finger_closed_thresh: float = 0.025,
) -> torch.Tensor:
    """tanh 内核：托盘 z 接近 target_height 时奖励 1，远离衰减。

    与 lift_progress 互补：progress 提供"往上走"的单调梯度，goal_height
    提供"停在目标"的稳态信号；两者都受 grasp 门控约束。
    """
    tray: RigidObject = env.scene[tray_cfg.name]
    tray_z = tray.data.root_pos_w[:, 2]
    err = (tray_z - target_height).abs()
    bell = 1.0 - torch.tanh(err / std)

    grasped = _is_grasped_per_env(
        env, left_ee_cfg, right_ee_cfg, left_finger_cfg, right_finger_cfg,
        tray_cfg, half_length, grasp_radius, finger_closed_thresh, grasp_z_offset,
    )
    return bell * grasped


# ─────────────────────────────────────────────────────────────────────
# F. 平稳性：举升后惩罚倾斜与摆动
# ─────────────────────────────────────────────────────────────────────

def tray_tilt_when_lifted(
    env: ManagerBasedRLEnv,
    lift_threshold: float,
    tray_cfg: SceneEntityCfg = SceneEntityCfg("tray"),
) -> torch.Tensor:
    """惩罚托盘倾斜：tilt = 1 - (tray.local +Z) · world +Z，仅当托盘 z > lift_threshold 时激活。

    返回非负，使用时权重取负。
    """
    tray: RigidObject = env.scene[tray_cfg.name]
    quat = tray.data.root_quat_w
    x = quat[:, 1]
    y = quat[:, 2]
    # local +Z 的世界 z 分量：1 - 2(x² + y²)
    obj_z_dot_world_z = 1.0 - 2.0 * (x * x + y * y)
    tilt = (1.0 - obj_z_dot_world_z).clamp(min=0.0)
    is_lifted = (tray.data.root_pos_w[:, 2] > lift_threshold).float()
    return is_lifted * tilt


def tray_angular_speed_when_lifted(
    env: ManagerBasedRLEnv,
    lift_threshold: float,
    tray_cfg: SceneEntityCfg = SceneEntityCfg("tray"),
) -> torch.Tensor:
    """惩罚托盘角速度模 (rad/s)：举升后激活。让动作"稳"。"""
    tray: RigidObject = env.scene[tray_cfg.name]
    ang_speed = torch.norm(tray.data.root_ang_vel_w, dim=1)
    is_lifted = (tray.data.root_pos_w[:, 2] > lift_threshold).float()
    return is_lifted * ang_speed


# ─────────────────────────────────────────────────────────────────────
# G. 双手间距：保持约托盘长度
# ─────────────────────────────────────────────────────────────────────

def hand_spacing(
    env: ManagerBasedRLEnv,
    target_distance: float,
    std: float,
    left_ee_cfg: SceneEntityCfg,
    right_ee_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """tanh 内核：双手间距越接近 target_distance（默认 = 2 × half_length）越好。

    单独存在，不与抓取门控耦合，使策略在 reach 阶段也有"两手分开"的弱信号。
    """
    left_pos = _ee_pos(env, left_ee_cfg)
    right_pos = _ee_pos(env, right_ee_cfg)
    d = torch.norm(left_pos - right_pos, dim=1)
    err = (d - target_distance).abs()
    return 1.0 - torch.tanh(err / std)


# ─────────────────────────────────────────────────────────────────────
# H. 选择性 action-rate（只惩罚双臂关节维度，跳过二值夹爪）
# ─────────────────────────────────────────────────────────────────────

def action_rate_l2_arm_only(
    env: ManagerBasedRLEnv,
    arm_action_names: tuple[str, ...] = ("left_arm_action", "right_arm_action"),
) -> torch.Tensor:
    """对相邻两步 raw action 的 L2 变化做平滑性惩罚，但仅覆盖给定的若干臂关节动作项，
    跳过 ``BinaryJointPositionAction`` 等"sign-only"动作 —— 后者的幅值对环境没有梯度，
    若纳入该惩罚会导致幅值无界漂移并把 critic 训飞（value_loss → inf）。

    实现方式：在拼接后的 ``action_manager.action`` 上取这些臂动作项对应的切片。
    """
    am = env.action_manager
    selected = set(arm_action_names)
    indices: list[int] = []
    cursor = 0
    for name in am.active_terms:
        dim = am.get_term(name).action_dim
        if name in selected:
            indices.extend(range(cursor, cursor + dim))
        cursor += dim
    if not indices:
        return torch.zeros(env.num_envs, device=env.device)
    idx = torch.as_tensor(indices, device=am.action.device, dtype=torch.long)
    diff = am.action.index_select(1, idx) - am.prev_action.index_select(1, idx)
    return torch.sum(torch.square(diff), dim=1)
