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

"""双臂托盘举升任务顶层配置（自然初态 + 四阶段课程）。

任务：两个 7-DoF OpenArm 机械臂从**自然下垂**初态出发，先到达托盘两端抓取点，
再竖直骑夹板端、协同举升并保持托盘平稳。

四阶段课程（详见 CurriculumCfg）：
    1. 到达（reach）—— 双臂从下垂姿态伸到抓取点附近；
    2. 夹住（grasp）—— grasp_hold / grasp_attempt；
    3. 抬起（lift）—— 门控举升奖励；
    4. 平稳（stability）—— 托盘水平 + 低速 + 对称/平滑。
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
HALF_GRASP_Y = 0.23          # 侧面抓取点距托盘中心的 |Y| (m)（落在悬空的两端）
GRASP_Z_OFFSET = 0.0         # 抓取点相对托盘中心的高度 (m)（板中厚处）
GRASP_RADIUS = 0.09          # 抓握门控/grasp_hold 的 TCP 半径 (m)

STAND_TOP = 0.33             # 支架顶面高度 (m)（inspect 实测两臂自然落在 z≈0.31~0.40，
                             #   取 0.33 让左右臂都够得到、不必拼命对抗重力上举）
DECK_HALF_THICKNESS = 0.015  # 托盘板半厚 (m)
TRAY_BASE_HEIGHT = STAND_TOP + DECK_HALF_THICKNESS  # 0.345：托盘静置质心高度
MINIMAL_LIFT_HEIGHT = TRAY_BASE_HEIGHT + 0.04       # 0.385：视为"已离台"
TARGET_HEIGHT = TRAY_BASE_HEIGHT + 0.15             # 0.495：目标举升高度（≈ +15 cm）

_LEFT_HAND_BODY = SceneEntityCfg("robot", body_names=["openarm_left_hand"])
_RIGHT_HAND_BODY = SceneEntityCfg("robot", body_names=["openarm_right_hand"])
_LEFT_FINGER_CFG = SceneEntityCfg("robot", joint_names=["openarm_left_finger_joint.*"])
_RIGHT_FINGER_CFG = SceneEntityCfg("robot", joint_names=["openarm_right_finger_joint.*"])
_LEFT_EE = SceneEntityCfg("left_ee_frame")
_RIGHT_EE = SceneEntityCfg("right_ee_frame")

# 抓握门控/抓握保持共用的几何参数（两手 EE + 两手夹爪 + 抓取点半径）
_GRASP_CFGS = {
    "left_ee_cfg": _LEFT_EE,
    "right_ee_cfg": _RIGHT_EE,
    "left_finger_cfg": _LEFT_FINGER_CFG,
    "right_finger_cfg": _RIGHT_FINGER_CFG,
    "half_grasp_y": HALF_GRASP_Y,
    "grasp_z_offset": GRASP_Z_OFFSET,
    "grasp_radius": GRASP_RADIUS,
}


# ─────────────────────────────────────────────────────────────────────
# 1. 场景
# ─────────────────────────────────────────────────────────────────────

@configclass
class TrayLiftSceneCfg(InteractiveSceneCfg):
    """场景：双臂机器人 + 中央支架 + 扁平托盘（机器人/EE/资产由子类填充）。"""

    robot: ArticulationCfg = MISSING
    left_ee_frame: FrameTransformerCfg = MISSING
    right_ee_frame: FrameTransformerCfg = MISSING
    tray: RigidObjectCfg = MISSING

    # 中央支架：托住托盘中部 0.12×0.24 区域（使板静置稳定、不易侧翻），
    # 托盘两端 (±0.23 m) 在支架外侧腾空 → 夹爪可把一指探到板下夹住板端。
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
        # 自然下垂初态，±0.05 rad 抖动可接受（手不在托盘旁，不会 spawn 穿模）。
        params={"position_range": (-0.05, 0.05), "velocity_range": (0.0, 0.0)},
    )


# ─────────────────────────────────────────────────────────────────────
# 5. 奖励
# ─────────────────────────────────────────────────────────────────────

@configclass
class RewardsCfg:
    """四阶段加性奖励（权重由课程控制，见 CurriculumCfg）。"""

    # ── A. 接近（阶段 1 唯一主力） ─────────────────────────────────
    left_reach_coarse = RewTerm(
        func=mdp.reach_handle, weight=2.0,
        params={"std": 0.15, "ee_frame_cfg": _LEFT_EE, "side": "left",
                "half_grasp_y": HALF_GRASP_Y, "grasp_z_offset": GRASP_Z_OFFSET},
    )
    right_reach_coarse = RewTerm(
        func=mdp.reach_handle, weight=2.0,
        params={"std": 0.15, "ee_frame_cfg": _RIGHT_EE, "side": "right",
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

    # ── B. 抓握（阶段 2 开启，初始权重 0） ────────────────────────
    grasp_hold = RewTerm(
        func=mdp.grasp_hold, weight=0.0,
        params=dict(_GRASP_CFGS),
    )
    grasp_attempt = RewTerm(
        func=mdp.grasp_attempt, weight=0.0,
        params=dict(_GRASP_CFGS),
    )

    # ── C. 举升（阶段 2，权重初始 0 → 课程开启；项内乘抓握门控） ────
    lift_height = RewTerm(
        func=mdp.tray_lift_height, weight=0.0,
        params={**_GRASP_CFGS, "base_height": TRAY_BASE_HEIGHT, "target_height": TARGET_HEIGHT},
    )
    lifted = RewTerm(
        func=mdp.tray_is_lifted, weight=0.0,
        params={**_GRASP_CFGS, "minimal_height": MINIMAL_LIFT_HEIGHT},
    )
    goal_height_coarse = RewTerm(
        func=mdp.tray_goal_height, weight=0.0,
        params={**_GRASP_CFGS, "target_height": TARGET_HEIGHT, "std": 0.10,
                "minimal_height": MINIMAL_LIFT_HEIGHT},
    )
    goal_height_fine = RewTerm(
        func=mdp.tray_goal_height, weight=0.0,
        params={**_GRASP_CFGS, "target_height": TARGET_HEIGHT, "std": 0.03,
                "minimal_height": MINIMAL_LIFT_HEIGHT},
    )

    # ── D. 平稳 / 对称（阶段 4 调大） ─────────────────────────────
    tray_flat = RewTerm(
        func=mdp.tray_flat, weight=0.0,
        params={"minimal_height": MINIMAL_LIFT_HEIGHT},
    )
    ee_symmetry = RewTerm(
        func=mdp.ee_height_symmetry, weight=0.3,
        params={"std": 0.05, "left_ee_cfg": _LEFT_EE, "right_ee_cfg": _RIGHT_EE},
    )
    tray_ang_vel = RewTerm(
        func=mdp.tray_ang_vel_penalty, weight=0.0,
        params={"minimal_height": MINIMAL_LIFT_HEIGHT},
    )

    # ── E. 平滑性（初始弱，课程阶段 3 调大） ──────────────────────
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
        params={"minimum_height": 0.22, "tray_cfg": SceneEntityCfg("tray")},
    )


# ─────────────────────────────────────────────────────────────────────
# 7. 课程：学会"抓 + 抬"之后，再调大 平稳/对称/平滑 的权重
# ─────────────────────────────────────────────────────────────────────

@configclass
class CurriculumCfg:
    """四阶段课程（自然初态 → 到达 → 夹住 → 抬起 → 平稳）。

    num_steps 为策略步；num_steps_per_env=64 → 步数/64 ≈ iteration。

      阶段 1（0  ~ 15000 步, ≈0~234 iter）：**仅 reach**（grasp/lift 权重=0），
            从自然下垂学会伸到抓取点。
      阶段 2（>15000 步, ≈234 iter 起）：开启 grasp_hold + grasp_attempt。
      阶段 3（>35000 步, ≈547 iter 起）：开启门控举升 4 项。
      阶段 4（>55000 步, ≈859 iter 起）：调大 平稳/对称/平滑。
    """
    # ── 阶段 2：开启抓握 ───────────────────────────────────────────
    enable_grasp_hold = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "grasp_hold", "weight": 6.0, "num_steps": 15000},
    )
    enable_grasp_attempt = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "grasp_attempt", "weight": 1.5, "num_steps": 15000},
    )

    # ── 阶段 3：开启（已门控的）举升奖励 ──────────────────────────
    enable_lift_height = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "lift_height", "weight": 8.0, "num_steps": 35000},
    )
    enable_lifted = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "lifted", "weight": 4.0, "num_steps": 35000},
    )
    enable_goal_coarse = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "goal_height_coarse", "weight": 8.0, "num_steps": 35000},
    )
    enable_goal_fine = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "goal_height_fine", "weight": 4.0, "num_steps": 35000},
    )

    # ── 阶段 4：平稳（水平 + 对称 + 平滑） ───────────────────────
    bump_tray_flat = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "tray_flat", "weight": 5.0, "num_steps": 55000},
    )
    bump_tray_ang_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "tray_ang_vel", "weight": -2e-3, "num_steps": 55000},
    )
    bump_ee_symmetry = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "ee_symmetry", "weight": 2.0, "num_steps": 55000},
    )
    bump_action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -5e-3, "num_steps": 55000},
    )
    bump_left_joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "left_joint_vel", "weight": -1e-3, "num_steps": 55000},
    )
    bump_right_joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "right_joint_vel", "weight": -1e-3, "num_steps": 55000},
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
        self.episode_length_s = 10.0
        self.sim.dt = 0.01                       # 100 Hz physics
        self.sim.render_interval = self.decimation

        # PhysX：侧面夹板 + 双夹爪接触需要足够的接触迭代/容量
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 64 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
