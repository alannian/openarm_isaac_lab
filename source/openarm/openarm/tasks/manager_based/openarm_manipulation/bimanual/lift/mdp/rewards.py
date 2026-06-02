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

"""双臂托盘举升任务的奖励项（侧面抓取 + 分阶段课程 + 抓握门控版）。

为什么这样设计（针对前两版的两个失败）：

* v1：举升奖励**硬门控** → 随机策略采不到 → 没有举升梯度 → 收敛到"悬停不抓"。
* v2：举升奖励**完全不门控** → 策略发现"把托盘推倒/顶竖起来"也能让质心升高
      从而骗到举升分（截图里托盘被怼成竖直）→ 根本不去抓。

正确做法 = 用户的思路：**先夹住、再抬起**，分阶段：

阶段 1（先夹住）——主力信号是 ``reach`` + ``grasp_hold``：
    把两只手带到托盘两端的抓取点并闭合夹爪。**举升奖励权重此时为 0**
    （由 lift_env_cfg 的课程控制），策略只学"双手稳稳夹住托盘"。

阶段 2（再抬起）——课程在 N 步后开启举升奖励，且举升奖励 **乘以抓握门控**
    ``_grasp_gate``（两只手都"靠近抓取点 × 已闭合"才接近 1）：
        - 真夹住才有举升梯度 → 学会协同抬升；
        - 不夹只推 → 门控 ≈ 0 → 拿不到举升分 → "推倒作弊"被彻底堵死。

托盘几何（配合新的扁平板 tray.usda）：
    托盘是一块 0.36×0.50×0.03 的扁平板，两端伸出中央支架之外悬空。夹爪从侧面
    把一指探到板下、一指压在板上夹住板的两端 → 抬升时下指**托着**板（形封闭），
    比靠摩擦夹横杆稳得多。抓取点在托盘局部 (0, ±HALF_GRASP_Y, 0)。
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
#   托盘局部系：原点在板中心，长轴 +Y。抓取点在两端 (0, ±0.23, 0)，
#   即板中厚处、距中心 0.23 m（落在悬空的两端，夹爪可从侧面套住板）。
# ─────────────────────────────────────────────────────────────────────
HALF_GRASP_Y = 0.23       # 抓取点距托盘中心的 |Y| (m)
GRASP_Z_OFFSET = 0.0      # 抓取点相对托盘中心的高度 (m)（板中厚处）
FINGER_OPEN_MAX = 0.044   # 夹爪单指完全张开值；0 = 完全闭合


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
    """某侧抓取点的世界坐标 (N, 3)。

    left → 托盘局部 +Y 端；right → -Y 端。通过托盘四元数旋转局部偏移，
    确保托盘平移/偏航后抓取点依然贴在板上。
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


def _closing_frac(env: ManagerBasedRLEnv, finger_cfg: SceneEntityCfg) -> torch.Tensor:
    """闭合比例 (N,) ∈ [0, 1]：完全张开 → 0，完全闭合 → 1。

    仅用于 ``grasp_attempt`` 的**弱引导**（靠近时鼓励尝试闭合），不用于门控——
    因为它分不清"夹住了"还是"空中闭合"。真实抓握判据见 ``_capture_band``。
    """
    open_val = _finger_open(env, finger_cfg)
    return ((FINGER_OPEN_MAX - open_val) / FINGER_OPEN_MAX).clamp(min=0.0, max=1.0)


# 板半厚（夹住 30 mm 板时单指停留的开度）；带通中心。与 tray.usda 的 0.03 厚一致。
_BOARD_HALF = 0.015


def _capture_band(env: ManagerBasedRLEnv, finger_cfg: SceneEntityCfg) -> torch.Tensor:
    """**真实抓握判据** (N,) ∈ [0, 1]：手指开度被"卡"在板半厚附近 → 1。

    关键物理事实：二值夹爪的目标只有 0（闭）或 0.044（开）两种。
      - 空中闭合 → 手指一路驱动到 ~0 → 带通 ≈ 0；
      - 完全张开 → ~0.044 → 带通 ≈ 0；
      - **只有**当一块板被卡在两指之间时，手指才能稳定停在 ~0.015（既到不了 0，
        也不在 0.044）→ 带通 ≈ 1。
    因此"带通持续 ≈ 1" 等价于"板确实被夹在两指之间"，**无法靠空中闭合或从下方
    顶托伪造**（那两种情况手指都会到 ~0）。带通在 ~[0.010, 0.026] 为平台，容忍
    夹取略偏心。
    """
    open_val = _finger_open(env, finger_cfg)
    rise = ((open_val - 0.004) / 0.006).clamp(min=0.0, max=1.0)   # 低于 ~0.004 → 0
    fall = ((0.032 - open_val) / 0.006).clamp(min=0.0, max=1.0)   # 高于 ~0.032 → 0
    return rise * fall


def _capture_quality(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg,
    finger_cfg: SceneEntityCfg,
    side: str,
    tray_cfg: SceneEntityCfg,
    half_grasp_y: float,
    grasp_z_offset: float,
    grasp_radius: float,
) -> torch.Tensor:
    """单手真实抓握质量 (N,) ∈ [0, 1] = near(TCP→该侧抓取点) × capture_band。

    near 保证"夹在正确的那一端"，capture_band 保证"板真的在两指之间"。两者缺一
    不可：停在附近但没夹住 → capture_band≈0；夹住了别处 → near≈0。
    """
    near = _near(env, ee_cfg, side, tray_cfg, half_grasp_y, grasp_z_offset, grasp_radius)
    return near * _capture_band(env, finger_cfg)


def _near(env: ManagerBasedRLEnv, ee_cfg: SceneEntityCfg, side: str,
          tray_cfg: SceneEntityCfg, half_grasp_y: float, grasp_z_offset: float,
          grasp_radius: float) -> torch.Tensor:
    """TCP 落在该侧抓取点附近的程度 (N,) ∈ (0, 1]：1/(1+(d/r)^2)，半径外快速衰减。"""
    d = torch.norm(
        _ee_pos_w(env, ee_cfg) - _handle_target_w(env, side, tray_cfg, half_grasp_y, grasp_z_offset),
        dim=1,
    )
    return 1.0 / (1.0 + (d / grasp_radius) ** 2)


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
# 抓握门控（关键）：两只手都"夹在正确端 × 板真的在两指间" → 接近 1，否则 ≈ 0
# ─────────────────────────────────────────────────────────────────────

def _grasp_gate(
    env: ManagerBasedRLEnv,
    left_ee_cfg: SceneEntityCfg,
    right_ee_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    tray_cfg: SceneEntityCfg,
    half_grasp_y: float,
    grasp_z_offset: float,
    grasp_radius: float,
) -> torch.Tensor:
    """双手**真实抓握质量**的乘积 (N,) ∈ [0, 1]。

    单手质量 = near × capture_band（见 ``_capture_quality``）。两手相乘 → **必须
    左右都真的把板夹在两指之间**门控才接近 1。用作举升奖励的乘子，杜绝两类作弊：
      - 用手臂从下方顶托抬板（手指空中→0，capture_band≈0 → 门控 0）；
      - 停在距离合格处但没夹住（capture_band≈0 → 门控 0）。
    """
    lq = _capture_quality(env, left_ee_cfg, left_finger_cfg, "left",
                          tray_cfg, half_grasp_y, grasp_z_offset, grasp_radius)
    rq = _capture_quality(env, right_ee_cfg, right_finger_cfg, "right",
                          tray_cfg, half_grasp_y, grasp_z_offset, grasp_radius)
    return lq * rq


# ─────────────────────────────────────────────────────────────────────
# A. 接近：每只手 → 自己那侧的抓取点（密集，全程有梯度）
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
    """tanh 内核：TCP 越接近该侧抓取点，奖励越高 (∈ [0, 1])。

    std 大 → 远处也有梯度（coarse）；std 小 → 最后一公里更密集（fine）。
    """
    ee = _ee_pos_w(env, ee_frame_cfg)
    tgt = _handle_target_w(env, side, tray_cfg, half_grasp_y, grasp_z_offset)
    dist = torch.norm(ee - tgt, dim=1)
    return 1.0 - torch.tanh(dist / std)


# ─────────────────────────────────────────────────────────────────────
# B. 抓握（阶段 1 主力）：grasp_hold = 真实抓握；grasp_attempt = 弱引导闭合
# ─────────────────────────────────────────────────────────────────────

def grasp_hold(
    env: ManagerBasedRLEnv,
    left_ee_cfg: SceneEntityCfg,
    right_ee_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    tray_cfg: SceneEntityCfg = SceneEntityCfg("tray"),
    half_grasp_y: float = HALF_GRASP_Y,
    grasp_z_offset: float = GRASP_Z_OFFSET,
    grasp_radius: float = 0.09,
) -> torch.Tensor:
    """双手**真实抓握质量**的均值 (∈ [0, 1])：0.5·(left_q + right_q)，
    单手 q = near × capture_band（板真的在两指之间）。

    用均值（而非乘积）让"单手先夹上"也能拿到一半分 → 平滑引导。它是阶段 1 的
    主力信号，权重明显高于 reach，使"在正确端把板夹进两指"成为该阶段最优策略。
    **只奖励真实夹住**：空中闭合 / 从下顶托 / 停在附近没夹住，capture_band 都≈0。
    """
    lq = _capture_quality(env, left_ee_cfg, left_finger_cfg, "left",
                          tray_cfg, half_grasp_y, grasp_z_offset, grasp_radius)
    rq = _capture_quality(env, right_ee_cfg, right_finger_cfg, "right",
                          tray_cfg, half_grasp_y, grasp_z_offset, grasp_radius)
    return 0.5 * (lq + rq)


def grasp_attempt(
    env: ManagerBasedRLEnv,
    left_ee_cfg: SceneEntityCfg,
    right_ee_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    tray_cfg: SceneEntityCfg = SceneEntityCfg("tray"),
    half_grasp_y: float = HALF_GRASP_Y,
    grasp_z_offset: float = GRASP_Z_OFFSET,
    grasp_radius: float = 0.09,
) -> torch.Tensor:
    """**弱引导** (∈ [0, 1])：靠近抓取点时鼓励"尝试闭合"= 0.5·Σ near × closing。

    作用是给"在板边把夹爪合上"提供一点密集梯度，加速发现真实抓握（否则
    capture_band 在抓住前一直是 0，纯靠探索较慢）。权重必须**远小于** grasp_hold，
    且它**不进举升门控**——所以即便策略"在附近空中闭合"也只能拿到这一点点分，
    拿不到举升分；真实夹住才同时拿到 grasp_hold + 举升。
    """
    def side(ee_cfg, fg_cfg, s):
        near = _near(env, ee_cfg, s, tray_cfg, half_grasp_y, grasp_z_offset, grasp_radius)
        return near * _closing_frac(env, fg_cfg)
    return 0.5 * (side(left_ee_cfg, left_finger_cfg, "left")
                  + side(right_ee_cfg, right_finger_cfg, "right"))


# ─────────────────────────────────────────────────────────────────────
# C. 举升（阶段 2，**乘抓握门控**）：密集高度 + 越台阶 + 目标高度跟踪
# ─────────────────────────────────────────────────────────────────────

def tray_lift_height(
    env: ManagerBasedRLEnv,
    base_height: float,
    target_height: float,
    left_ee_cfg: SceneEntityCfg,
    right_ee_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    tray_cfg: SceneEntityCfg = SceneEntityCfg("tray"),
    half_grasp_y: float = HALF_GRASP_Y,
    grasp_z_offset: float = GRASP_Z_OFFSET,
    grasp_radius: float = 0.09,
) -> torch.Tensor:
    """线性高度进度 × 抓握门控 ∈ [0, 1]：托盘 z 从 base→target，且必须真夹住。

    门控杜绝了"推倒托盘骗高度"的退化解——不夹住时门控 ≈ 0，推得再高也无分。
    """
    tray = _tray(env, tray_cfg)
    z = tray.data.root_pos_w[:, 2]
    span = max(target_height - base_height, 1e-6)
    progress = ((z - base_height) / span).clamp(min=0.0, max=1.0)
    gate = _grasp_gate(env, left_ee_cfg, right_ee_cfg, left_finger_cfg, right_finger_cfg,
                       tray_cfg, half_grasp_y, grasp_z_offset, grasp_radius)
    return progress * gate


def tray_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    left_ee_cfg: SceneEntityCfg,
    right_ee_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    tray_cfg: SceneEntityCfg = SceneEntityCfg("tray"),
    half_grasp_y: float = HALF_GRASP_Y,
    grasp_z_offset: float = GRASP_Z_OFFSET,
    grasp_radius: float = 0.09,
) -> torch.Tensor:
    """托盘越过 minimal_height 的台阶奖励 × 抓握门控 (∈ [0, 1])。"""
    tray = _tray(env, tray_cfg)
    lifted = (tray.data.root_pos_w[:, 2] > minimal_height).float()
    gate = _grasp_gate(env, left_ee_cfg, right_ee_cfg, left_finger_cfg, right_finger_cfg,
                       tray_cfg, half_grasp_y, grasp_z_offset, grasp_radius)
    return lifted * gate


def tray_goal_height(
    env: ManagerBasedRLEnv,
    target_height: float,
    std: float,
    minimal_height: float,
    left_ee_cfg: SceneEntityCfg,
    right_ee_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    tray_cfg: SceneEntityCfg = SceneEntityCfg("tray"),
    half_grasp_y: float = HALF_GRASP_Y,
    grasp_z_offset: float = GRASP_Z_OFFSET,
    grasp_radius: float = 0.09,
) -> torch.Tensor:
    """tanh 内核：托盘 z 接近 target_height 时奖励 1，× 已离台 × 抓握门控。"""
    tray = _tray(env, tray_cfg)
    z = tray.data.root_pos_w[:, 2]
    is_lifted = (z > minimal_height).float()
    track = 1.0 - torch.tanh((z - target_height).abs() / std)
    gate = _grasp_gate(env, left_ee_cfg, right_ee_cfg, left_finger_cfg, right_finger_cfg,
                       tray_cfg, half_grasp_y, grasp_z_offset, grasp_radius)
    return is_lifted * track * gate


# ─────────────────────────────────────────────────────────────────────
# D. 平稳 / 对称（初始弱，课程后期调大）
# ─────────────────────────────────────────────────────────────────────

def tray_flat(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    tray_cfg: SceneEntityCfg = SceneEntityCfg("tray"),
) -> torch.Tensor:
    """托盘水平度奖励 (∈ [0, 1])：托盘局部 +Z 与世界 +Z 的点积，举起后才计分。"""
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
    """双手 TCP 等高奖励 (∈ [0, 1])：两手高度差越小越好 → 对称同步施力、托盘不偏。"""
    lz = _ee_pos_w(env, left_ee_cfg)[:, 2]
    rz = _ee_pos_w(env, right_ee_cfg)[:, 2]
    return 1.0 - torch.tanh((lz - rz).abs() / std)


def tray_ang_vel_penalty(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    tray_cfg: SceneEntityCfg = SceneEntityCfg("tray"),
) -> torch.Tensor:
    """托盘角速度 L2（仅举起后计），配合 tray_flat 抑制摇摆。"""
    tray = _tray(env, tray_cfg)
    lifted = (tray.data.root_pos_w[:, 2] > minimal_height).float()
    ang = tray.data.root_ang_vel_w
    return lifted * torch.sum(torch.square(ang), dim=1)


# ─────────────────────────────────────────────────────────────────────
# E. 选择性 action-rate：只惩罚双臂关节维度，跳过二值夹爪
# ─────────────────────────────────────────────────────────────────────

def action_rate_l2_arm_only(
    env: ManagerBasedRLEnv,
    arm_action_names: tuple[str, ...] = ("left_arm_action", "right_arm_action"),
) -> torch.Tensor:
    """对相邻两步 raw action 的 L2 变化做平滑性惩罚，仅覆盖臂关节动作项，
    跳过 ``BinaryJointPositionAction``（其 raw 幅值对环境无梯度，纳入会把 critic 训飞）。
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
    return torch.sum(torch.square(am.action[:, idx] - am.prev_action[:, idx]), dim=1)
