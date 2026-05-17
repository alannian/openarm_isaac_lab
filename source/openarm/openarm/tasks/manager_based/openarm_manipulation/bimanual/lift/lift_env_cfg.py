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

"""双臂托盘举升任务的顶层配置（从头重新设计）。

任务定义：
    两个 7 自由度 OpenArm 机械臂从上方分别抓住一根长杆形托盘的两端，
    将其平稳举起到目标高度并保持水平。

几何约定（全部以 robot root 系描述，root 在两臂中线、地面处）：
    - 托盘 (tray):     bar 形, size = (0.04, 0.50, 0.025) m, mass = 0.4 kg
                       长轴沿 +Y, 初始 pos = (0.40, 0.0, 0.245)
    - 支架 (stand):    短粗立柱, size = (0.08, 0.08, 0.22) m
                       置于 (0.40, 0.0, 0.11)，仅在托盘中部下方
                       → 两端 ±0.25 m 处腾空，方便从上方抓取
    - target_height:   0.55 m  （比初始高 0.30 m）
    - grasp 半径:      0.08 m
    - half_length:     0.25 m

整体奖励权重（加性）：
    reach (coarse / fine) ........ 2.0 / 1.0  × 2 sides = 6.0
    ee_above_tray (penalty) ...... -2.0       × 2 sides = -4.0
    hand_pointing_down ........... 1.5        × 2 sides = 3.0
    gripper_yaw_align ............ 1.0        × 2 sides = 2.0
    gripper_close_when_near ...... 4.0        × 2 sides = 8.0
    hand_spacing ................. 0.5
    lift_progress (curriculum) ... 0 → 8.0
    goal_height_tracking (curr.).. 0 → 10.0
    tray_tilt_when_lifted ........ -2.0
    tray_ang_speed_when_lifted ... -0.05
    action_rate .................. -1e-3
    joint_vel .................... -5e-4
"""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp


# ─────────────────────────────────────────────────────────────────────
# 常量（与 mdp 内部默认值保持一致）
# ─────────────────────────────────────────────────────────────────────
HALF_LENGTH = 0.25           # 托盘半长 (m)；与托盘 size_y=0.50 一致
GRASP_Z_OFFSET = 0.02        # 抓取点距托盘中心向上的 offset (m)
GRASP_RADIUS = 0.08          # 进入抓取的 TCP 半径 (m)
TRAY_BASE_HEIGHT = 0.2325    # 托盘初始 z（stand top 0.22 + 半厚 0.0125）
LIFT_THRESHOLD = 0.30        # 视为"已举起"的高度
TARGET_HEIGHT = 0.55         # 最终目标高度（比初始高 ~0.32 m）
HAND_SPACING_TARGET = 0.50   # 双手期望间距 = 2 × HALF_LENGTH

_LEFT_HAND_BODY = SceneEntityCfg("robot", body_names=["openarm_left_hand"])
_RIGHT_HAND_BODY = SceneEntityCfg("robot", body_names=["openarm_right_hand"])
_LEFT_FINGER_CFG = SceneEntityCfg("robot", joint_names=["openarm_left_finger_joint.*"])
_RIGHT_FINGER_CFG = SceneEntityCfg("robot", joint_names=["openarm_right_finger_joint.*"])


# ─────────────────────────────────────────────────────────────────────
# 1. 场景
# ─────────────────────────────────────────────────────────────────────

@configclass
class TrayLiftSceneCfg(InteractiveSceneCfg):
    """场景：双臂机器人 + 细支架 + 长杆托盘。

    机器人 / 末端坐标系 / 托盘 由子类填充。
    """
    robot: ArticulationCfg = MISSING
    left_ee_frame: FrameTransformerCfg = MISSING
    right_ee_frame: FrameTransformerCfg = MISSING
    tray: RigidObjectCfg = MISSING

    # 中央细支架（窄于托盘，使两端 ±0.25 m 腾空可抓取）
    stand = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Stand",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.40, 0.0, 0.11]),
        spawn=sim_utils.CuboidCfg(
            size=(0.08, 0.08, 0.22),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.20, 0.20, 0.20), roughness=0.8
            ),
        ),
    )
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 0]),
        spawn=GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


# ─────────────────────────────────────────────────────────────────────
# 2. 动作：双臂 + 双夹爪
# ─────────────────────────────────────────────────────────────────────

@configclass
class ActionsCfg:
    """动作维度 = 7 + 7 + 1 + 1 = 16
    （二值夹爪每边 1 维 sign(a)）。"""
    left_arm_action: mdp.JointPositionActionCfg = MISSING
    right_arm_action: mdp.JointPositionActionCfg = MISSING
    left_gripper_action: mdp.BinaryJointPositionActionCfg = MISSING
    right_gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


# ─────────────────────────────────────────────────────────────────────
# 3. 观测
# ─────────────────────────────────────────────────────────────────────

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """精简观测，聚焦"做这个任务必须看到的几何线索"。"""

        # ── 本体感知 ───────────────────────────────────────────────
        left_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg(
                "robot", joint_names=[f"openarm_left_joint{i}" for i in range(1, 8)]
            )},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        right_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg(
                "robot", joint_names=[f"openarm_right_joint{i}" for i in range(1, 8)]
            )},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        left_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg(
                "robot", joint_names=[f"openarm_left_joint{i}" for i in range(1, 8)]
            )},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        right_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg(
                "robot", joint_names=[f"openarm_right_joint{i}" for i in range(1, 8)]
            )},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        left_finger_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": _LEFT_FINGER_CFG},
            noise=Unoise(n_min=-0.005, n_max=0.005),
        )
        right_finger_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": _RIGHT_FINGER_CFG},
            noise=Unoise(n_min=-0.005, n_max=0.005),
        )

        # ── 抓取几何（root 系） ────────────────────────────────────
        left_ee_pos = ObsTerm(
            func=mdp.ee_position_in_robot_root_frame,
            params={"ee_frame_cfg": SceneEntityCfg("left_ee_frame")},
        )
        right_ee_pos = ObsTerm(
            func=mdp.ee_position_in_robot_root_frame,
            params={"ee_frame_cfg": SceneEntityCfg("right_ee_frame")},
        )
        left_to_target = ObsTerm(
            func=mdp.ee_to_tray_end_vector_in_robot_root_frame,
            params={
                "ee_frame_cfg": SceneEntityCfg("left_ee_frame"),
                "side": "left",
                "half_length": HALF_LENGTH,
                "grasp_z_offset": GRASP_Z_OFFSET,
            },
        )
        right_to_target = ObsTerm(
            func=mdp.ee_to_tray_end_vector_in_robot_root_frame,
            params={
                "ee_frame_cfg": SceneEntityCfg("right_ee_frame"),
                "side": "right",
                "half_length": HALF_LENGTH,
                "grasp_z_offset": GRASP_Z_OFFSET,
            },
        )

        # ── 托盘状态 ──────────────────────────────────────────────
        tray_position = ObsTerm(func=mdp.tray_position_in_robot_root_frame)
        tray_orientation = ObsTerm(func=mdp.tray_orientation_features)
        tray_lin_vel = ObsTerm(func=mdp.tray_linear_velocity)
        tray_ang_vel = ObsTerm(func=mdp.tray_angular_velocity)

        # ── 手部姿态（朝向标量） ──────────────────────────────────
        left_hand_down = ObsTerm(
            func=mdp.hand_down_alignment,
            params={"hand_cfg": _LEFT_HAND_BODY},
        )
        right_hand_down = ObsTerm(
            func=mdp.hand_down_alignment,
            params={"hand_cfg": _RIGHT_HAND_BODY},
        )
        left_hand_yaw = ObsTerm(
            func=mdp.hand_yaw_alignment,
            params={"hand_cfg": _LEFT_HAND_BODY},
        )
        right_hand_yaw = ObsTerm(
            func=mdp.hand_yaw_alignment,
            params={"hand_cfg": _RIGHT_HAND_BODY},
        )

        # ── 历史动作 ──────────────────────────────────────────────
        left_actions = ObsTerm(func=mdp.last_action, params={"action_name": "left_arm_action"})
        right_actions = ObsTerm(func=mdp.last_action, params={"action_name": "right_arm_action"})
        left_grip_action = ObsTerm(func=mdp.last_action, params={"action_name": "left_gripper_action"})
        right_grip_action = ObsTerm(func=mdp.last_action, params={"action_name": "right_gripper_action"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ─────────────────────────────────────────────────────────────────────
# 4. 事件（reset & domain randomization）
# ─────────────────────────────────────────────────────────────────────

@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # 托盘小幅平面随机化，强化策略的泛化能力
    reset_tray_pose = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.02, 0.02),
                "y": (-0.02, 0.02),
                "z": (0.0, 0.0),
                "yaw": (-0.05, 0.05),   # ±3° 微旋
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("tray"),
        },
    )

    # 机器人关节微抖动，避免每个 episode 完全确定性的初态
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (-0.05, 0.05), "velocity_range": (0.0, 0.0)},
    )


# ─────────────────────────────────────────────────────────────────────
# 5. 奖励
# ─────────────────────────────────────────────────────────────────────

@configclass
class RewardsCfg:
    """全加性奖励。

    阶段 A（reach / orient / close）的奖励全程激活，提供从零开始的密集梯度。
    阶段 B（lift / goal）通过课程从 0 线性激活，避免训练初期"夹未稳就被举升信号
    带跑"的副作用。
    """

    # ── A1. 接近：粗 + 精 ───────────────────────────────────────
    left_reach_coarse = RewTerm(
        func=mdp.reach_grasp_target,
        weight=2.0,
        params={
            "std": 0.15,
            "ee_frame_cfg": SceneEntityCfg("left_ee_frame"),
            "side": "left",
            "half_length": HALF_LENGTH,
            "grasp_z_offset": GRASP_Z_OFFSET,
        },
    )
    right_reach_coarse = RewTerm(
        func=mdp.reach_grasp_target,
        weight=2.0,
        params={
            "std": 0.15,
            "ee_frame_cfg": SceneEntityCfg("right_ee_frame"),
            "side": "right",
            "half_length": HALF_LENGTH,
            "grasp_z_offset": GRASP_Z_OFFSET,
        },
    )
    left_reach_fine = RewTerm(
        func=mdp.reach_grasp_target_fine,
        weight=1.0,
        params={
            "std": 0.04,
            "ee_frame_cfg": SceneEntityCfg("left_ee_frame"),
            "side": "left",
            "half_length": HALF_LENGTH,
            "grasp_z_offset": GRASP_Z_OFFSET,
        },
    )
    right_reach_fine = RewTerm(
        func=mdp.reach_grasp_target_fine,
        weight=1.0,
        params={
            "std": 0.04,
            "ee_frame_cfg": SceneEntityCfg("right_ee_frame"),
            "side": "right",
            "half_length": HALF_LENGTH,
            "grasp_z_offset": GRASP_Z_OFFSET,
        },
    )

    # ── A2. 从上方接近的几何偏置 ──────────────────────────────
    left_above_tray = RewTerm(
        func=mdp.ee_above_tray_penalty,
        weight=-2.0,
        params={
            "ee_frame_cfg": SceneEntityCfg("left_ee_frame"),
            "margin": 0.0,
        },
    )
    right_above_tray = RewTerm(
        func=mdp.ee_above_tray_penalty,
        weight=-2.0,
        params={
            "ee_frame_cfg": SceneEntityCfg("right_ee_frame"),
            "margin": 0.0,
        },
    )

    # ── A3. 手部朝下 + yaw 与托盘长轴垂直 ─────────────────────
    left_hand_down = RewTerm(
        func=mdp.hand_pointing_down,
        weight=1.5,
        params={"hand_cfg": _LEFT_HAND_BODY},
    )
    right_hand_down = RewTerm(
        func=mdp.hand_pointing_down,
        weight=1.5,
        params={"hand_cfg": _RIGHT_HAND_BODY},
    )
    left_hand_yaw = RewTerm(
        func=mdp.gripper_yaw_align,
        weight=1.0,
        params={"hand_cfg": _LEFT_HAND_BODY},
    )
    right_hand_yaw = RewTerm(
        func=mdp.gripper_yaw_align,
        weight=1.0,
        params={"hand_cfg": _RIGHT_HAND_BODY},
    )

    # ── A4. 进入抓取半径后才奖励夹爪闭合 ──────────────────────
    left_grip_close = RewTerm(
        func=mdp.gripper_close_when_near,
        weight=4.0,
        params={
            "ee_frame_cfg": SceneEntityCfg("left_ee_frame"),
            "finger_cfg": _LEFT_FINGER_CFG,
            "side": "left",
            "grasp_radius": GRASP_RADIUS,
            "target_finger_pos": 0.012,
            "half_length": HALF_LENGTH,
            "grasp_z_offset": GRASP_Z_OFFSET,
        },
    )
    right_grip_close = RewTerm(
        func=mdp.gripper_close_when_near,
        weight=4.0,
        params={
            "ee_frame_cfg": SceneEntityCfg("right_ee_frame"),
            "finger_cfg": _RIGHT_FINGER_CFG,
            "side": "right",
            "grasp_radius": GRASP_RADIUS,
            "target_finger_pos": 0.012,
            "half_length": HALF_LENGTH,
            "grasp_z_offset": GRASP_Z_OFFSET,
        },
    )

    # ── A5. 双手保持托盘长度间距（弱信号） ─────────────────────
    hand_spacing = RewTerm(
        func=mdp.hand_spacing,
        weight=0.5,
        params={
            "target_distance": HAND_SPACING_TARGET,
            "std": 0.15,
            "left_ee_cfg": SceneEntityCfg("left_ee_frame"),
            "right_ee_cfg": SceneEntityCfg("right_ee_frame"),
        },
    )

    # ── B1. 抓住后的举升进度（课程） ───────────────────────────
    lift_progress = RewTerm(
        func=mdp.lift_progress_when_grasped,
        weight=0.0,
        params={
            "base_height": TRAY_BASE_HEIGHT,
            "target_height": TARGET_HEIGHT,
            "left_ee_cfg": SceneEntityCfg("left_ee_frame"),
            "right_ee_cfg": SceneEntityCfg("right_ee_frame"),
            "left_finger_cfg": _LEFT_FINGER_CFG,
            "right_finger_cfg": _RIGHT_FINGER_CFG,
            "half_length": HALF_LENGTH,
            "grasp_radius": GRASP_RADIUS,
            "grasp_z_offset": GRASP_Z_OFFSET,
            "finger_closed_thresh": 0.025,
        },
    )
    goal_height = RewTerm(
        func=mdp.goal_height_tracking_when_grasped,
        weight=0.0,
        params={
            "target_height": TARGET_HEIGHT,
            "std": 0.08,
            "left_ee_cfg": SceneEntityCfg("left_ee_frame"),
            "right_ee_cfg": SceneEntityCfg("right_ee_frame"),
            "left_finger_cfg": _LEFT_FINGER_CFG,
            "right_finger_cfg": _RIGHT_FINGER_CFG,
            "half_length": HALF_LENGTH,
            "grasp_radius": GRASP_RADIUS,
            "grasp_z_offset": GRASP_Z_OFFSET,
            "finger_closed_thresh": 0.025,
        },
    )

    # ── B2. 平稳性：举升后惩罚倾斜 / 摆动 ─────────────────────
    tray_tilt = RewTerm(
        func=mdp.tray_tilt_when_lifted,
        weight=-2.0,
        params={"lift_threshold": LIFT_THRESHOLD},
    )
    tray_ang_speed = RewTerm(
        func=mdp.tray_angular_speed_when_lifted,
        weight=-0.05,
        params={"lift_threshold": LIFT_THRESHOLD},
    )

    # ── A6. 控制平滑性 ────────────────────────────────────────
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-3)
    left_joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-5e-4,
        params={"asset_cfg": SceneEntityCfg(
            "robot", joint_names=[f"openarm_left_joint{i}" for i in range(1, 8)]
        )},
    )
    right_joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-5e-4,
        params={"asset_cfg": SceneEntityCfg(
            "robot", joint_names=[f"openarm_right_joint{i}" for i in range(1, 8)]
        )},
    )


# ─────────────────────────────────────────────────────────────────────
# 6. 终止条件
# ─────────────────────────────────────────────────────────────────────

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    tray_dropped = DoneTerm(
        func=mdp.tray_dropped,
        params={"minimum_height": 0.18, "tray_cfg": SceneEntityCfg("tray")},
    )


# ─────────────────────────────────────────────────────────────────────
# 7. 课程
# ─────────────────────────────────────────────────────────────────────

@configclass
class CurriculumCfg:
    """阶段切换：

    - 前 ~10000 步策略学接近 + 闭合（lift / goal 权重为 0）
    - 10000 - 25000 步线性激活 lift / goal（≈ 235 iters，比旧版 60000 步快得多）
    - 之后维持终值
    """
    # 注意：`num_steps` 单位是"全局并行环境步" = num_envs × policy_step。
    # 单卡 4096 env：200k 步 ≈ 49 iter（每 iter 64 step × 4096 env = 262k）
    # 双卡 4096×2：200k 步 ≈ 24 iter（每 iter 524k）
    # 也就是 lift / goal 在前 ~25-50 iter 内线性激活到位，之后维持。
    activate_lift_progress = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "lift_progress", "weight": 8.0, "num_steps": 200_000},
    )
    activate_goal_height = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "goal_height", "weight": 10.0, "num_steps": 200_000},
    )


# ─────────────────────────────────────────────────────────────────────
# 8. 顶层 RL 环境配置
# ─────────────────────────────────────────────────────────────────────

@configclass
class BimanualTrayLiftEnvCfg(ManagerBasedRLEnvCfg):
    """双臂托盘举升基类配置。"""

    scene: TrayLiftSceneCfg = TrayLiftSceneCfg(num_envs=4096, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 8.0
        self.sim.dt = 0.01                       # 100 Hz physics
        self.sim.render_interval = self.decimation

        # PhysX：抓取细瘦物体需要足够的接触迭代
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 64 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
