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


##
# 场景
##

@configclass
class TrayLiftSceneCfg(InteractiveSceneCfg):
    """双臂托盘举升场景：机器人 + 桌子 + 托盘。
    robot / left_ee_frame / right_ee_frame / tray 由子类填充。
    """
    robot: ArticulationCfg = MISSING
    left_ee_frame: FrameTransformerCfg = MISSING
    right_ee_frame: FrameTransformerCfg = MISSING
    tray: RigidObjectCfg = MISSING

    # 小支架：比托盘（12×60cm）更窄，仅用于托高托盘，不遮挡机械臂
    # 支架顶面 z = 0.18 + 0.18 = 0.36m
    # 支架 y 方向仅 20cm，托盘两端（y=±0.22m）完全暴露供手臂抓握
    # x=0.28m：远离机器人基座，手臂需要向前伸展才能到达，自然从侧方接近托盘端部
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Stand",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.28, 0.0, 0.18]),
        spawn=sim_utils.CuboidCfg(
            size=(0.10, 0.20, 0.36),
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


##
# 动作
##

@configclass
class ActionsCfg:
    """双臂 + 双夹爪动作规格。"""
    left_arm_action: mdp.JointPositionActionCfg = MISSING
    right_arm_action: mdp.JointPositionActionCfg = MISSING
    left_gripper_action: mdp.BinaryJointPositionActionCfg = MISSING
    right_gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


##
# 观测
##

@configclass
class ObservationsCfg:
    """观测规格。"""

    @configclass
    class PolicyCfg(ObsGroup):
        """策略观测组。"""
        # 左臂关节
        left_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg(
                "robot", joint_names=[f"openarm_left_joint{i}" for i in range(1, 8)]
            )},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        left_finger_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg(
                "robot", joint_names=["openarm_left_finger_joint.*"]
            )},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        # 右臂关节
        right_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg(
                "robot", joint_names=[f"openarm_right_joint{i}" for i in range(1, 8)]
            )},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        right_finger_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg(
                "robot", joint_names=["openarm_right_finger_joint.*"]
            )},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        # 关节速度
        left_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg(
                "robot", joint_names=[f"openarm_left_joint{i}" for i in range(1, 8)]
            )},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        left_finger_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg(
                "robot", joint_names=["openarm_left_finger_joint.*"]
            )},
        )
        right_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg(
                "robot", joint_names=[f"openarm_right_joint{i}" for i in range(1, 8)]
            )},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        right_finger_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg(
                "robot", joint_names=["openarm_right_finger_joint.*"]
            )},
        )
        # 托盘状态
        tray_position = ObsTerm(func=mdp.tray_position_in_robot_root_frame)
        tray_tilt = ObsTerm(func=mdp.tray_roll_pitch)
        # 上一步动作
        left_actions = ObsTerm(
            func=mdp.last_action, params={"action_name": "left_arm_action"}
        )
        right_actions = ObsTerm(
            func=mdp.last_action, params={"action_name": "right_arm_action"}
        )
        left_gripper_action_obs = ObsTerm(
            func=mdp.last_action, params={"action_name": "left_gripper_action"}
        )
        right_gripper_action_obs = ObsTerm(
            func=mdp.last_action, params={"action_name": "right_gripper_action"}
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


##
# 事件（Reset）
##

@configclass
class EventCfg:
    """重置事件配置。"""
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_tray_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (0.0, 0.0),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("tray", body_names="Tray"),
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (-0.2, 0.2), "velocity_range": (0.0, 0.0)},
    )


##
# 奖励
##

@configclass
class RewardsCfg:
    """奖励项配置。"""
    # Phase 1：靠近（half_length=0.22：抓握点在托盘两端往内 8cm，在确认工作空间 y=±0.22m 内）
    left_reach_tray = RewTerm(
        func=mdp.ee_reach_tray_end,
        weight=1.5,
        params={
            "std": 0.1,
            "ee_frame_cfg": SceneEntityCfg("left_ee_frame"),
            "side": "left",
            "half_length": 0.22,
        },
    )
    right_reach_tray = RewTerm(
        func=mdp.ee_reach_tray_end,
        weight=1.5,
        params={
            "std": 0.1,
            "ee_frame_cfg": SceneEntityCfg("right_ee_frame"),
            "side": "right",
            "half_length": 0.22,
        },
    )

    # Phase 1 辅助：EE 高度对齐（引导夹爪从侧面正确接近，防止手臂从下方托推托盘）
    left_ee_height = RewTerm(
        func=mdp.ee_height_align_reward,
        weight=2.0,
        params={
            "std": 0.05,
            "ee_frame_cfg": SceneEntityCfg("left_ee_frame"),
        },
    )
    right_ee_height = RewTerm(
        func=mdp.ee_height_align_reward,
        weight=2.0,
        params={
            "std": 0.05,
            "ee_frame_cfg": SceneEntityCfg("right_ee_frame"),
        },
    )

    # Phase 2：双手同时接触
    grasp_both_ends = RewTerm(
        func=mdp.grasp_both_ends,
        weight=5.0,
        params={
            "distance_threshold": 0.06,
            "left_ee_cfg": SceneEntityCfg("left_ee_frame"),
            "right_ee_cfg": SceneEntityCfg("right_ee_frame"),
            "half_length": 0.22,
        },
    )
    left_finger_closure = RewTerm(
        func=mdp.finger_closure_reward,
        weight=0.5,
        params={
            "distance_threshold": 0.06,
            "ee_frame_cfg": SceneEntityCfg("left_ee_frame"),
            "finger_cfg": SceneEntityCfg("robot", joint_names=["openarm_left_finger_joint.*"]),
            "side": "left",
            "half_length": 0.22,
        },
    )
    right_finger_closure = RewTerm(
        func=mdp.finger_closure_reward,
        weight=0.5,
        params={
            "distance_threshold": 0.06,
            "ee_frame_cfg": SceneEntityCfg("right_ee_frame"),
            "finger_cfg": SceneEntityCfg("robot", joint_names=["openarm_right_finger_joint.*"]),
            "side": "right",
            "half_length": 0.22,
        },
    )

    # Phase 3：举起（托盘初始在 z=0.375，举离支架顶面 z>0.40 才算起，目标举到 z=0.52）
    tray_lifted = RewTerm(
        func=mdp.tray_is_lifted,
        weight=20.0,
        params={"minimal_height": 0.40},
    )
    tray_goal_height = RewTerm(
        func=mdp.tray_goal_height_tracking,
        weight=16.0,
        params={"target_height": 0.52, "std": 0.1, "minimal_height": 0.40},
    )
    tray_goal_height_fine = RewTerm(
        func=mdp.tray_goal_height_tracking,
        weight=5.0,
        params={"target_height": 0.52, "std": 0.03, "minimal_height": 0.40},
    )

    # Phase 4：协同约束
    grasp_symmetry = RewTerm(
        func=mdp.grasp_symmetry_penalty,
        weight=-2.0,
        params={
            "left_ee_cfg": SceneEntityCfg("left_ee_frame"),
            "right_ee_cfg": SceneEntityCfg("right_ee_frame"),
        },
    )
    tray_tilt = RewTerm(
        func=mdp.tray_tilt_penalty,
        weight=-3.0,
        params={"max_tilt_rad": 0.1},
    )
    hand_distance = RewTerm(
        func=mdp.hand_distance_reward,
        weight=0.5,
        params={
            "target_distance": 0.44,  # 2 × half_length = 2 × 0.22
            "std": 0.2,
            "left_ee_cfg": SceneEntityCfg("left_ee_frame"),
            "right_ee_cfg": SceneEntityCfg("right_ee_frame"),
        },
    )

    # 平滑性惩罚
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    left_joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg(
            "robot", joint_names=[f"openarm_left_joint{i}" for i in range(1, 8)]
        )},
    )
    right_joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg(
            "robot", joint_names=[f"openarm_right_joint{i}" for i in range(1, 8)]
        )},
    )


##
# 终止条件
##

@configclass
class TerminationsCfg:
    """终止条件配置。"""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    tray_dropped = DoneTerm(
        func=mdp.tray_dropped,
        params={"minimum_height": 0.33, "tray_cfg": SceneEntityCfg("tray")},
    )


##
# 课程学习
##

@configclass
class CurriculumCfg:
    """课程学习配置：逐步提高平滑性惩罚权重。"""
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -5e-3, "num_steps": 20000},
    )
    left_joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "left_joint_vel", "weight": -1e-3, "num_steps": 20000},
    )
    right_joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "right_joint_vel", "weight": -1e-3, "num_steps": 20000},
    )


##
# 顶层环境配置
##

@configclass
class BimanualTrayLiftEnvCfg(ManagerBasedRLEnvCfg):
    """双臂托盘举升基类配置。"""

    scene: TrayLiftSceneCfg = TrayLiftSceneCfg(num_envs=2048, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 10.0
        self.sim.dt = 0.01                    # 100 Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
