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

"""双臂托盘举升任务顶层配置（彻底重构版）。

任务：两个 7-DoF OpenArm 机械臂从上方分别抓住一块"托盘"两端的把手，协同将其
从中央支架上平稳举起到目标高度并保持水平。

几何约定（机器人 root 系；机器人朝 +X，左臂在 +Y、右臂在 -Y）：
    - 托盘 (tray):  扁平板 0.30(X)×0.50(Y)×0.025(Z) + 两端把手横杆，单刚体，
                    见 ``usds/tray/tray.usda``。把手抓取点在托盘局部
                    (0, ±0.22, +0.035)。初始置于中央支架上。
    - 支架 (stand): 中央细立柱 0.10×0.12×0.20，仅托住托盘中部，两端把手腾空可抓。
    - 静置高度 TRAY_BASE_HEIGHT = 0.2125（支架顶 0.20 + 板半厚 0.0125）
    - 目标高度 TARGET_HEIGHT    = 0.46（≈ 抬升 25 cm）

奖励哲学（修复旧版"练不出来"的根因，详见 mdp/rewards.py）：
    - 接近、举升、目标高度等主信号**全程激活、绝不门控**；
    - 去掉"必须在托盘上方"的反向惩罚；
    - 平稳/对称/平滑等约束初始权重很小，靠课程在学会"抓+抬"之后再调大。
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
# 几何常量（必须与 usds/tray/tray.usda 与 config/joint_pos_env_cfg.py 一致）
# ─────────────────────────────────────────────────────────────────────
HALF_GRASP_Y = 0.22          # 把手抓取点距托盘中心的 |Y| (m)
GRASP_Z_OFFSET = 0.035       # 把手抓取点相对托盘中心的高度 (m)
GRASP_RADIUS = 0.08          # 抓握辅助/加分项的 TCP 半径 (m)

STAND_TOP = 0.20             # 支架顶面高度 (m)
DECK_HALF_THICKNESS = 0.0125 # 托盘板半厚 (m)
TRAY_BASE_HEIGHT = STAND_TOP + DECK_HALF_THICKNESS  # 0.2125：托盘静置质心高度
MINIMAL_LIFT_HEIGHT = TRAY_BASE_HEIGHT + 0.04       # 0.2525：视为"已离台"
TARGET_HEIGHT = 0.46         # 目标举升高度（≈ +25 cm）

_LEFT_HAND_BODY = SceneEntityCfg("robot", body_names=["openarm_left_hand"])
_RIGHT_HAND_BODY = SceneEntityCfg("robot", body_names=["openarm_right_hand"])
_LEFT_FINGER_CFG = SceneEntityCfg("robot", joint_names=["openarm_left_finger_joint.*"])
_RIGHT_FINGER_CFG = SceneEntityCfg("robot", joint_names=["openarm_right_finger_joint.*"])
_LEFT_EE = SceneEntityCfg("left_ee_frame")
_RIGHT_EE = SceneEntityCfg("right_ee_frame")


# ─────────────────────────────────────────────────────────────────────
# 1. 场景
# ─────────────────────────────────────────────────────────────────────

@configclass
class TrayLiftSceneCfg(InteractiveSceneCfg):
    """场景：双臂机器人 + 中央细支架 + 带把手的托盘（机器人/EE/托盘由子类填充）。"""

    robot: ArticulationCfg = MISSING
    left_ee_frame: FrameTransformerCfg = MISSING
    right_ee_frame: FrameTransformerCfg = MISSING
    tray: RigidObjectCfg = MISSING

    # 中央支架：托住托盘中部 0.12×0.24 区域（使板静置稳定、不易侧翻），
    # 两端把手 (±0.22 m) 仍在支架外侧腾空，可从上方无碰撞抓取。
    stand = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Stand",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.40, 0.0, STAND_TOP / 2.0]),
        spawn=sim_utils.CuboidCfg(
            size=(0.12, 0.24, STAND_TOP),
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
    """动作维度 = 7 + 7 + 1 + 1 = 16（二值夹爪每边 1 维 sign(a)）。"""
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
        """聚焦做这个任务必须看到的几何线索。"""

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
            params={"ee_frame_cfg": _LEFT_EE},
        )
        right_ee_pos = ObsTerm(
            func=mdp.ee_position_in_robot_root_frame,
            params={"ee_frame_cfg": _RIGHT_EE},
        )
        left_to_handle = ObsTerm(
            func=mdp.ee_to_handle_vector_in_robot_root_frame,
            params={
                "ee_frame_cfg": _LEFT_EE,
                "side": "left",
                "half_grasp_y": HALF_GRASP_Y,
                "grasp_z_offset": GRASP_Z_OFFSET,
            },
        )
        right_to_handle = ObsTerm(
            func=mdp.ee_to_handle_vector_in_robot_root_frame,
            params={
                "ee_frame_cfg": _RIGHT_EE,
                "side": "right",
                "half_grasp_y": HALF_GRASP_Y,
                "grasp_z_offset": GRASP_Z_OFFSET,
            },
        )

        # ── 托盘状态 ──────────────────────────────────────────────
        tray_position = ObsTerm(func=mdp.tray_position_in_robot_root_frame)
        tray_orientation = ObsTerm(func=mdp.tray_orientation_features)
        tray_lin_vel = ObsTerm(func=mdp.tray_linear_velocity)
        tray_ang_vel = ObsTerm(func=mdp.tray_angular_velocity)

        # ── 手部姿态（标量） ──────────────────────────────────────
        left_hand_down = ObsTerm(func=mdp.hand_down_alignment, params={"hand_cfg": _LEFT_HAND_BODY})
        right_hand_down = ObsTerm(func=mdp.hand_down_alignment, params={"hand_cfg": _RIGHT_HAND_BODY})
        left_hand_span = ObsTerm(func=mdp.hand_span_alignment, params={"hand_cfg": _LEFT_HAND_BODY})
        right_hand_span = ObsTerm(func=mdp.hand_span_alignment, params={"hand_cfg": _RIGHT_HAND_BODY})

        # ── 历史动作（仅两臂；二值夹爪 raw 幅值无界，不入观测） ────
        left_actions = ObsTerm(func=mdp.last_action, params={"action_name": "left_arm_action"})
        right_actions = ObsTerm(func=mdp.last_action, params={"action_name": "right_arm_action"})

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

    reset_tray_pose = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.03, 0.03),
                "y": (-0.03, 0.03),
                "z": (0.0, 0.0),
                "yaw": (-0.05, 0.05),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("tray"),
        },
    )

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
    """全加性奖励。主信号（reach / lift / goal）全程激活、不门控；平稳/对称/平滑
    初始弱，靠课程后期调大。"""

    # ── A. 接近：每只手 → 自己那侧的把手（粗 + 精） ───────────────
    left_reach_coarse = RewTerm(
        func=mdp.reach_handle, weight=2.0,
        params={"std": 0.10, "ee_frame_cfg": _LEFT_EE, "side": "left",
                "half_grasp_y": HALF_GRASP_Y, "grasp_z_offset": GRASP_Z_OFFSET},
    )
    right_reach_coarse = RewTerm(
        func=mdp.reach_handle, weight=2.0,
        params={"std": 0.10, "ee_frame_cfg": _RIGHT_EE, "side": "right",
                "half_grasp_y": HALF_GRASP_Y, "grasp_z_offset": GRASP_Z_OFFSET},
    )
    left_reach_fine = RewTerm(
        func=mdp.reach_handle, weight=1.0,
        params={"std": 0.03, "ee_frame_cfg": _LEFT_EE, "side": "left",
                "half_grasp_y": HALF_GRASP_Y, "grasp_z_offset": GRASP_Z_OFFSET},
    )
    right_reach_fine = RewTerm(
        func=mdp.reach_handle, weight=1.0,
        params={"std": 0.03, "ee_frame_cfg": _RIGHT_EE, "side": "right",
                "half_grasp_y": HALF_GRASP_Y, "grasp_z_offset": GRASP_Z_OFFSET},
    )

    # ── B. 姿态偏置（弱）：手朝下 + 夹爪开合轴横跨把手 ─────────────
    left_hand_down = RewTerm(func=mdp.hand_pointing_down, weight=0.5, params={"hand_cfg": _LEFT_HAND_BODY})
    right_hand_down = RewTerm(func=mdp.hand_pointing_down, weight=0.5, params={"hand_cfg": _RIGHT_HAND_BODY})
    left_span_align = RewTerm(func=mdp.gripper_span_align, weight=0.5, params={"hand_cfg": _LEFT_HAND_BODY})
    right_span_align = RewTerm(func=mdp.gripper_span_align, weight=0.5, params={"hand_cfg": _RIGHT_HAND_BODY})

    # ── C. 抓握加分（软、不门控） ─────────────────────────────────
    grasp_bonus = RewTerm(
        func=mdp.grasp_bonus, weight=2.0,
        params={
            "left_ee_cfg": _LEFT_EE, "right_ee_cfg": _RIGHT_EE,
            "left_finger_cfg": _LEFT_FINGER_CFG, "right_finger_cfg": _RIGHT_FINGER_CFG,
            "half_grasp_y": HALF_GRASP_Y, "grasp_z_offset": GRASP_Z_OFFSET,
            "grasp_radius": GRASP_RADIUS,
        },
    )

    # ── D. 举升（核心，全程激活、不门控） ─────────────────────────
    lift_height = RewTerm(
        func=mdp.tray_lift_height, weight=8.0,
        params={"base_height": TRAY_BASE_HEIGHT, "target_height": TARGET_HEIGHT},
    )
    lifted = RewTerm(
        func=mdp.tray_is_lifted, weight=4.0,
        params={"minimal_height": MINIMAL_LIFT_HEIGHT},
    )
    goal_height_coarse = RewTerm(
        func=mdp.tray_goal_height, weight=8.0,
        params={"target_height": TARGET_HEIGHT, "std": 0.10, "minimal_height": MINIMAL_LIFT_HEIGHT},
    )
    goal_height_fine = RewTerm(
        func=mdp.tray_goal_height, weight=4.0,
        params={"target_height": TARGET_HEIGHT, "std": 0.03, "minimal_height": MINIMAL_LIFT_HEIGHT},
    )

    # ── E. 平稳 / 对称（初始弱，课程后期调大） ─────────────────────
    tray_flat = RewTerm(
        func=mdp.tray_flat, weight=0.5,
        params={"minimal_height": MINIMAL_LIFT_HEIGHT},
    )
    ee_symmetry = RewTerm(
        func=mdp.ee_height_symmetry, weight=0.5,
        params={"std": 0.05, "left_ee_cfg": _LEFT_EE, "right_ee_cfg": _RIGHT_EE},
    )

    # ── F. 平滑性（初始弱，课程后期调大） ─────────────────────────
    action_rate = RewTerm(
        func=mdp.action_rate_l2_arm_only, weight=-1e-4,
        params={"arm_action_names": ("left_arm_action", "right_arm_action")},
    )
    left_joint_vel = RewTerm(
        func=mdp.joint_vel_l2, weight=-5e-4,
        params={"asset_cfg": SceneEntityCfg(
            "robot", joint_names=[f"openarm_left_joint{i}" for i in range(1, 8)]
        )},
    )
    right_joint_vel = RewTerm(
        func=mdp.joint_vel_l2, weight=-5e-4,
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
        params={"minimum_height": 0.12, "tray_cfg": SceneEntityCfg("tray")},
    )


# ─────────────────────────────────────────────────────────────────────
# 7. 课程：学会"抓 + 抬"之后，再调大 平稳/对称/平滑 的权重
# ─────────────────────────────────────────────────────────────────────

@configclass
class CurriculumCfg:
    """``modify_reward_weight`` 在 ``common_step_counter > num_steps`` 时把权重切到目标值
    （阶跃，非线性）。num_steps 单位是策略步；按 num_steps_per_env=64 估算，
    8000 步 ≈ 125 次迭代——足够策略先学会抓握与举升，再开始强调平稳与平滑。

    注意：lift / reach / goal 等主信号**不在课程里**，从第 0 步就是满权重——这是
    与旧版（把举升权重从 0 慢慢加，结果一直没激活）最关键的区别。
    """
    bump_tray_flat = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "tray_flat", "weight": 3.0, "num_steps": 8000},
    )
    bump_ee_symmetry = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "ee_symmetry", "weight": 1.5, "num_steps": 8000},
    )
    bump_action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -5e-3, "num_steps": 8000},
    )
    bump_left_joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "left_joint_vel", "weight": -1e-3, "num_steps": 8000},
    )
    bump_right_joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "right_joint_vel", "weight": -1e-3, "num_steps": 8000},
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

        # PhysX：抓取细把手 + 双夹爪接触需要足够的接触迭代/容量
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 64 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
