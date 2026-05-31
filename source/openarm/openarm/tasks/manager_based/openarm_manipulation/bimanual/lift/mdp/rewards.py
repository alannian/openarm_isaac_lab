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

"""双臂托盘举升任务的奖励项（彻底重构版）。

核心设计原则（针对旧版"练不出来"的根因）：

1. **举升奖励永不门控**。旧版把举升奖励乘以"双手都进半径且都闭合"的硬门控，
   随机策略几乎采不到 → 永远没有举升梯度 → 收敛到"在托盘上方悬停"的局部最优。
   新版照搬仓库里**能跑通的单臂 lift** 的配方：
     - `reach`（密集，全程有梯度，把手 → 把手抓取点）
     - `tray_lift_height`（**密集、不门控**，托盘升高即线性给分，是主力信号）
     - `tray_is_lifted`（越过最小高度的台阶奖励）
     - `tray_goal_height`（粗 + 精，向目标高度收敛，仅用"已离台"软门控）
   夹爪是"为了把托盘举起来而自然学会闭合"的，不需要任何硬门控。

2. **不再有"必须在托盘上方"的反向惩罚**——它和"下探去抓"直接对抗，是旧版把
   手钉在空中的元凶之一。

3. **平稳 / 对称 / 平滑** 这类约束初始给很小的权重，靠课程学习在策略学会
   "抓 + 抬"之后再调大，避免训练早期梯度冲突。

4. 每个 reward 单一职责、范围约 [0, 1]，权重直接体现重要性；几何统一由
   `_handle_target_w` 定义，避免不同地方语义漂移。
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ─────────────────────────────────────────────────────────────────────
# 几何默认值（与 tray.usda / lift_env_cfg.py 保持一致）
#   托盘局部系：原点在板中心，长轴 +Y，把手沿 X，把手中心在 (0, ±0.22, +0.035)
# ─────────────────────────────────────────────────────────────────────
HALF_GRASP_Y = 0.22       # 把手中心距托盘中心的 |Y| (m)
GRASP_Z_OFFSET = 0.035    # 把手中心相对托盘中心的高度 (m)


# ─────────────────────────────────────────────────────────────────────
# 内部工具
# ─────────────────────────────────────────────────────────────────────

def _tray(env: ManagerBasedRLEnv, tray_cfg: SceneEntityCfg) -> RigidObject:
    return env.scene[tray_cfg.name]


def _handle_target_w(
    env: ManagerBasedRLEnv,
    side: str,
    tray_cfg: SceneEntityCfg,
    half_grasp_y: float,
    grasp_z_offset: float,
) -> torch.Tensor:
    """某侧把手抓取点的世界坐标 (N, 3)。

    left  → 托盘局部 +Y 端；right → -Y 端；都在板面上方 grasp_z_offset 处。
    通过托盘四元数旋转局部偏移，确保托盘平移/偏航后抓取点依然正确。
    """
    tray = _tray(env, tray_cfg)
    pos = tray.data.root_pos_w
    quat = tray.data.root_quat_w
    sign = 1.0 if side == "left" else -1.0
    local = torch.zeros_like(pos)
    local[:, 1] = sign * half_grasp_y
    local[:, 2] = grasp_z_offset
    return pos + quat_apply(quat, local)


def _ee_pos_w(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg) -> torch.Tensor:
    ee: FrameTransformer = env.scene[ee_frame_cfg.name]
    return ee.data.target_pos_w[..., 0, :]


def _finger_open(env: ManagerBasedRLEnv, finger_cfg: SceneEntityCfg) -> torch.Tensor:
    """夹爪平均开度 (N,)：0 = 完全闭合，0.044 = 完全张开。"""
    robot: Articulation = env.scene[finger_cfg.name]
    return robot.data.joint_pos[:, finger_cfg.joint_ids].mean(dim=1)


def _hand_axis_world(env: ManagerBasedRLEnv, hand_cfg: SceneEntityCfg, axis: int) -> torch.Tensor:
    robot: Articulation = env.scene[hand_cfg.name]
    quat = robot.data.body_quat_w[:, hand_cfg.body_ids[0]]
    local = torch.zeros(quat.shape[0], 3, device=quat.device)
    local[:, axis] = 1.0
    return quat_apply(quat, local)


def _tray_axis_world(env: ManagerBasedRLEnv, tray_cfg: SceneEntityCfg, axis: int) -> torch.Tensor:
    tray = _tray(env, tray_cfg)
    quat = tray.data.root_quat_w
    local = torch.zeros(quat.shape[0], 3, device=quat.device)
    local[:, axis] = 1.0
    return quat_apply(quat, local)


# ─────────────────────────────────────────────────────────────────────
# A. 接近：每只手 → 自己那侧的把手抓取点（密集，全程有梯度）
# ─────────────────────────────────────────────────────────────────────

def reach_handle(
    env: ManagerBasedRLEnv,
    std: float,
    ee_frame_cfg: SceneEntityCfg,
    side: str,
    tray_cfg: SceneEntityCfg = SceneEntityCfg("tray"),
    half_grasp_y: float = HALF_GRASP_Y,
    grasp_z_offset: float = GRASP_Z_OFFSET,
) -> torch.Tensor:
    """tanh 内核：TCP 越接近把手抓取点，奖励越高 (∈ [0, 1])。

    std 大 → 远处也有梯度（coarse）；std 小 → 最后一公里更密集（fine）。
    """
    ee = _ee_pos_w(env, ee_frame_cfg)
    tgt = _handle_target_w(env, side, tray_cfg, half_grasp_y, grasp_z_offset)
    dist = torch.norm(ee - tgt, dim=1)
    return 1.0 - torch.tanh(dist / std)


# ─────────────────────────────────────────────────────────────────────
# B. 举升（核心）：不门控的密集高度信号 + 越台阶奖励 + 目标高度跟踪
# ─────────────────────────────────────────────────────────────────────

def tray_lift_height(
    env: ManagerBasedRLEnv,
    base_height: float,
    target_height: float,
    tray_cfg: SceneEntityCfg = SceneEntityCfg("tray"),
) -> torch.Tensor:
    """**不门控**的线性高度进度 ∈ [0, 1]：托盘 z 从 base → target。

    这是逃出"悬停局部最优"的关键——托盘只要被抬高一点点就立刻有正比例奖励，
    策略一旦偶然抓住并抬起就能得到密集正反馈，从而强化"抓 + 抬"。
    """
    tray = _tray(env, tray_cfg)
    z = tray.data.root_pos_w[:, 2]
    span = max(target_height - base_height, 1e-6)
    return ((z - base_height) / span).clamp(min=0.0, max=1.0)


def tray_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    tray_cfg: SceneEntityCfg = SceneEntityCfg("tray"),
) -> torch.Tensor:
    """托盘质心越过 minimal_height 时给 1，否则 0（**不门控**）。"""
    tray = _tray(env, tray_cfg)
    return (tray.data.root_pos_w[:, 2] > minimal_height).float()


def tray_goal_height(
    env: ManagerBasedRLEnv,
    target_height: float,
    std: float,
    minimal_height: float,
    tray_cfg: SceneEntityCfg = SceneEntityCfg("tray"),
) -> torch.Tensor:
    """tanh 内核：托盘 z 接近 target_height 时奖励 1，仅用"已离台"软门控。"""
    tray = _tray(env, tray_cfg)
    z = tray.data.root_pos_w[:, 2]
    is_lifted = (z > minimal_height).float()
    return is_lifted * (1.0 - torch.tanh((z - target_height).abs() / std))


# ─────────────────────────────────────────────────────────────────────
# C. 平稳 / 对称（初始弱，课程后期加大）
# ─────────────────────────────────────────────────────────────────────

def tray_flat(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    tray_cfg: SceneEntityCfg = SceneEntityCfg("tray"),
) -> torch.Tensor:
    """托盘水平度奖励 (∈ [0, 1])：托盘局部 +Z 与世界 +Z 的点积，举起后才计分。

    completely flat → 1；倾斜越大越小。鼓励"平稳举起、保持水平"。
    """
    tray = _tray(env, tray_cfg)
    quat = tray.data.root_quat_w
    x = quat[:, 1]
    y = quat[:, 2]
    obj_z_dot_world_z = 1.0 - 2.0 * (x * x + y * y)
    is_lifted = (tray.data.root_pos_w[:, 2] > minimal_height).float()
    return is_lifted * obj_z_dot_world_z.clamp(min=0.0, max=1.0)


def ee_height_symmetry(
    env: ManagerBasedRLEnv,
    std: float,
    left_ee_cfg: SceneEntityCfg,
    right_ee_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """双手 TCP 等高奖励 (∈ [0, 1])：两手高度差越小越好。

    对称同步施力 → 托盘不偏不倒。全程激活（接近阶段也促使两手同高摆位）。
    """
    lz = _ee_pos_w(env, left_ee_cfg)[:, 2]
    rz = _ee_pos_w(env, right_ee_cfg)[:, 2]
    return 1.0 - torch.tanh((lz - rz).abs() / std)


# ─────────────────────────────────────────────────────────────────────
# D. 抓握辅助（**加分项，绝不门控**）：靠近把手时奖励闭合夹爪
# ─────────────────────────────────────────────────────────────────────

def grasp_bonus(
    env: ManagerBasedRLEnv,
    left_ee_cfg: SceneEntityCfg,
    right_ee_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    tray_cfg: SceneEntityCfg = SceneEntityCfg("tray"),
    half_grasp_y: float = HALF_GRASP_Y,
    grasp_z_offset: float = GRASP_Z_OFFSET,
    grasp_radius: float = 0.08,
) -> torch.Tensor:
    """软加分 (∈ [0, 1])：每只手"靠近把手 × 已闭合"的乘积，两侧求平均。

    near = 1/(1+(d/r)^2) 在抓取半径外快速衰减；closed = 1 - tanh(open/0.015)。
    远离把手时几乎为 0 → 不会鼓励"全程闭合"的退化解；它只是加速"在正确位置
    闭合夹爪"的发现，**不作为举升奖励的门控**。
    """
    def side_term(ee_cfg: SceneEntityCfg, finger_cfg: SceneEntityCfg, side: str) -> torch.Tensor:
        d = torch.norm(
            _ee_pos_w(env, ee_cfg) - _handle_target_w(env, side, tray_cfg, half_grasp_y, grasp_z_offset),
            dim=1,
        )
        near = 1.0 / (1.0 + (d / grasp_radius) ** 2)
        closed = 1.0 - torch.tanh(_finger_open(env, finger_cfg) / 0.015)
        return near * closed

    left = side_term(left_ee_cfg, left_finger_cfg, "left")
    right = side_term(right_ee_cfg, right_finger_cfg, "right")
    return 0.5 * (left + right)


# ─────────────────────────────────────────────────────────────────────
# E. 姿态偏置（弱）：手朝下 + 夹爪开合轴横跨把手长轴
# ─────────────────────────────────────────────────────────────────────

def hand_pointing_down(
    env: ManagerBasedRLEnv,
    hand_cfg: SceneEntityCfg,
    forward_axis: int = 2,
) -> torch.Tensor:
    """手部 forward 轴（hand 局部 +Z）指向世界 -Z 的程度 (∈ [0, 1])。

    偏置"从上方下压抓把手"的姿态；弱权重，避免压制举升主信号。
    """
    axis_w = _hand_axis_world(env, hand_cfg, forward_axis)
    return (-axis_w[:, 2]).clamp(min=0.0, max=1.0)


def gripper_span_align(
    env: ManagerBasedRLEnv,
    hand_cfg: SceneEntityCfg,
    tray_cfg: SceneEntityCfg = SceneEntityCfg("tray"),
    span_axis: int = 1,
    handle_long_axis: int = 0,
) -> torch.Tensor:
    """夹爪开合轴（hand 局部 ±Y）与把手长轴（tray 局部 ±X）垂直的程度 (∈ [0, 1])。

    把手是沿 X 的细横杆，平行夹爪要横跨它闭合 → span ⟂ handle_long →
    |span · handle_long| ≈ 0 → 奖励 1 - |·|。
    """
    span_w = _hand_axis_world(env, hand_cfg, span_axis)
    long_w = _tray_axis_world(env, tray_cfg, handle_long_axis)
    dot = (span_w * long_w).sum(dim=1).abs()
    return (1.0 - dot).clamp(min=0.0, max=1.0)


# ─────────────────────────────────────────────────────────────────────
# F. 选择性 action-rate：只惩罚双臂关节维度，跳过二值夹爪
# ─────────────────────────────────────────────────────────────────────

def action_rate_l2_arm_only(
    env: ManagerBasedRLEnv,
    arm_action_names: tuple[str, ...] = ("left_arm_action", "right_arm_action"),
) -> torch.Tensor:
    """对相邻两步 raw action 的 L2 变化做平滑性惩罚，但仅覆盖给定的臂关节动作项，
    跳过 ``BinaryJointPositionAction`` —— 后者 raw 幅值对环境无梯度，纳入会无界漂移
    并把 critic 训飞（value_loss → inf）。
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
