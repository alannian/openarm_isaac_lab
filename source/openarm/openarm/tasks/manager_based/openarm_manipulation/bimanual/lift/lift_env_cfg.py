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
        left_grasp_vector = ObsTerm(
            func=mdp.ee_to_tray_end_vector_in_robot_root_frame,
            params={
                "ee_frame_cfg": SceneEntityCfg("left_ee_frame"),
                "side": "left",
                "half_length": 0.22,
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        right_grasp_vector = ObsTerm(
            func=mdp.ee_to_tray_end_vector_in_robot_root_frame,
            params={
                "ee_frame_cfg": SceneEntityCfg("right_ee_frame"),
                "side": "right",
                "half_length": 0.22,
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
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

# 公共参数：抓握判定用
_GRASP_DIST = 0.10   # 放宽抓握判定半径，避免策略卡在“接近但不触发抓取”
_HALF_LEN   = 0.22   # 托盘抓握点偏移量（托盘半长 0.30m，抓握点向内 8cm）
_TRAY_BASE_HEIGHT = 0.375
_LEFT_FINGER_CFG  = SceneEntityCfg("robot", joint_names=["openarm_left_finger_joint.*"])
_RIGHT_FINGER_CFG = SceneEntityCfg("robot", joint_names=["openarm_right_finger_joint.*"])

@configclass
class RewardsCfg:
    """奖励项配置：两阶段课程学习设计。

    阶段A（0 ~ ~900 iter）：
        - 举升相关奖励权重初始为 0，策略专注学会双臂靠近 + 夹爪夹住
        - 接近奖励权重大幅提升，确保探索效率

    阶段B（~900 iter 起，由课程自动线性激活）：
        - 举升奖励通过 tray_is_lifted_grasped 和 tray_goal_height_tracking_grasped
          施加门控：必须双手夹住托盘才能获得举升奖励，彻底堵死"推托盘"捷径
        - 课程学习在 60000 步内将举升奖励从 0 线性增长到目标权重
    """
    # ── 阶段A：接近与抓取（全程有效，权重较大） ──────────────────────

    # Phase 1：靠近（half_length=0.22，抓握点在托盘两端往内 8cm）
    left_reach_tray = RewTerm(
        func=mdp.ee_reach_tray_end,
        weight=2.0,   # 从 3.0 降到 2.0，给高度对齐腾出奖励空间
        params={
            "std": 0.1,
            "ee_frame_cfg": SceneEntityCfg("left_ee_frame"),
            "side": "left",
            "half_length": _HALF_LEN,
        },
    )
    right_reach_tray = RewTerm(
        func=mdp.ee_reach_tray_end,
        weight=2.0,
        params={
            "std": 0.1,
            "ee_frame_cfg": SceneEntityCfg("right_ee_frame"),
            "side": "right",
            "half_length": _HALF_LEN,
        },
    )

    # Phase 1 辅助：EE 高度对齐
    # 关键几何约束：EE 中心必须在 tray_z ± 一定范围内，才能保证一根手指在托盘上面、一根在下面
    # std=0.015 时在 z_err=3cm 处奖励仅 0.04（梯度消失），策略无法纠正高度
    # std=0.04 时在 z_err=3cm 处奖励为 0.38，梯度有意义，训练可以收敛
    left_ee_height = RewTerm(
        func=mdp.ee_height_align_reward,
        weight=4.0,   # 提升权重：高度对齐是能否抓住的关键前提
        params={
            "std": 0.04,  # 从 0.015 改为 0.04：给早期训练足够的梯度覆盖范围
            "ee_frame_cfg": SceneEntityCfg("left_ee_frame"),
        },
    )
    right_ee_height = RewTerm(
        func=mdp.ee_height_align_reward,
        weight=4.0,
        params={
            "std": 0.04,
            "ee_frame_cfg": SceneEntityCfg("right_ee_frame"),
        },
    )

    # Phase 2：双手同时接触（权重提升，强化联合抓取）
    grasp_both_ends = RewTerm(
        func=mdp.grasp_both_ends,
        weight=10.0,  # 从 5.0 提升至 10.0，使联合抓取成为最强奖励信号
        params={
            "distance_threshold": _GRASP_DIST,
            "left_ee_cfg": SceneEntityCfg("left_ee_frame"),
            "right_ee_cfg": SceneEntityCfg("right_ee_frame"),
            "half_length": _HALF_LEN,
        },
    )
    left_finger_closure = RewTerm(
        func=mdp.finger_closure_reward,
        weight=4.0,   # 从 2.0 提升至 4.0：夹取奖励需与 grasp_both_ends 同量级才足以打破"悬停"局部最优
        params={
            "distance_threshold": _GRASP_DIST,
            "ee_frame_cfg": SceneEntityCfg("left_ee_frame"),
            "finger_cfg": _LEFT_FINGER_CFG,
            "side": "left",
            "half_length": _HALF_LEN,
        },
    )
    right_finger_closure = RewTerm(
        func=mdp.finger_closure_reward,
        weight=4.0,
        params={
            "distance_threshold": _GRASP_DIST,
            "ee_frame_cfg": SceneEntityCfg("right_ee_frame"),
            "finger_cfg": _RIGHT_FINGER_CFG,
            "side": "right",
            "half_length": _HALF_LEN,
        },
    )

    # Phase 2 辅助：EE 水平接近方向（奖励从侧面进入，不从上/下插入）
    # 计算 EE 到目标点的位移向量，Z 分量越小 → 接近方向越水平 → 奖励越高
    # 不依赖 EE 轴约定，避免之前因轴方向猜测错误导致的怪异姿态
    left_grasp_orientation = RewTerm(
        func=mdp.ee_approach_direction_reward,
        weight=3.0,   # 从 2.0 提升到 3.0：接近方向是确保托盘在手指之间的第二关键约束
        params={
            "ee_frame_cfg": SceneEntityCfg("left_ee_frame"),
            "side": "left",
            "distance_std": 0.12,
            "half_length": _HALF_LEN,
        },
    )
    right_grasp_orientation = RewTerm(
        func=mdp.ee_approach_direction_reward,
        weight=3.0,
        params={
            "ee_frame_cfg": SceneEntityCfg("right_ee_frame"),
            "side": "right",
            "distance_std": 0.12,
            "half_length": _HALF_LEN,
        },
    )

    # ── 阶段B：举升（初始权重=0，由课程学习线性激活） ─────────────────
    # 关键设计：使用门控版本，必须双手夹住才能获得举升奖励

    tray_lifted = RewTerm(
        func=mdp.tray_is_lifted_grasped,
        weight=0.0,   # 初始为 0，由课程学习激活到 20.0
        params={
            "minimal_height": 0.40,
            "grasp_distance_threshold": _GRASP_DIST,
            "left_ee_cfg": SceneEntityCfg("left_ee_frame"),
            "right_ee_cfg": SceneEntityCfg("right_ee_frame"),
            "left_finger_cfg": _LEFT_FINGER_CFG,
            "right_finger_cfg": _RIGHT_FINGER_CFG,
            "base_height": _TRAY_BASE_HEIGHT,
            "half_length": _HALF_LEN,
        },
    )
    tray_goal_height = RewTerm(
        func=mdp.tray_goal_height_tracking_grasped,
        weight=0.0,   # 初始为 0，由课程学习激活到 16.0
        params={
            "target_height": 0.52,
            "std": 0.1,
            "minimal_height": 0.40,
            "grasp_distance_threshold": _GRASP_DIST,
            "left_ee_cfg": SceneEntityCfg("left_ee_frame"),
            "right_ee_cfg": SceneEntityCfg("right_ee_frame"),
            "left_finger_cfg": _LEFT_FINGER_CFG,
            "right_finger_cfg": _RIGHT_FINGER_CFG,
            "base_height": _TRAY_BASE_HEIGHT,
            "half_length": _HALF_LEN,
        },
    )
    tray_goal_height_fine = RewTerm(
        func=mdp.tray_goal_height_tracking_grasped,
        weight=0.0,   # 初始为 0，由课程学习激活到 5.0
        params={
            "target_height": 0.52,
            "std": 0.03,
            "minimal_height": 0.40,
            "grasp_distance_threshold": _GRASP_DIST,
            "left_ee_cfg": SceneEntityCfg("left_ee_frame"),
            "right_ee_cfg": SceneEntityCfg("right_ee_frame"),
            "left_finger_cfg": _LEFT_FINGER_CFG,
            "right_finger_cfg": _RIGHT_FINGER_CFG,
            "base_height": _TRAY_BASE_HEIGHT,
            "half_length": _HALF_LEN,
        },
    )

    # ── 阶段B：协同约束（初始权重=0，由课程学习激活） ────────────────
    grasp_symmetry = RewTerm(
        func=mdp.grasp_symmetry_penalty,
        weight=0.0,   # 初始为 0，由课程学习激活到 -2.0
        params={
            "left_ee_cfg": SceneEntityCfg("left_ee_frame"),
            "right_ee_cfg": SceneEntityCfg("right_ee_frame"),
        },
    )
    tray_tilt = RewTerm(
        func=mdp.tray_tilt_penalty,
        weight=0.0,   # 初始为 0，由课程学习激活到 -3.0
        params={"max_tilt_rad": 0.1},
    )
    hand_distance = RewTerm(
        func=mdp.hand_distance_reward,
        weight=0.3,
        params={
            "target_distance": 0.44,  # 2 × half_length = 2 × 0.22
            "std": 0.2,
            "left_ee_cfg": SceneEntityCfg("left_ee_frame"),
            "right_ee_cfg": SceneEntityCfg("right_ee_frame"),
            "distance_threshold": _GRASP_DIST,
            "half_length": _HALF_LEN,
        },
    )

    # ── 平滑性惩罚（全程，初始很小，课程学习逐步增大） ──────────────
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
    """终止条件配置。

    注意：阶段A（学夹取）期间 tray_dropped 的阈值设得更低（0.30），
    避免因托盘轻微晃动就终止 episode，给策略更多时间学习夹取行为。
    """
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    tray_dropped = DoneTerm(
        func=mdp.tray_dropped,
        params={"minimum_height": 0.30, "tray_cfg": SceneEntityCfg("tray")},
    )


##
# 课程学习
##

@configclass
class CurriculumCfg:
    """两阶段课程学习配置。

    阶段A → 阶段B 切换：
        - tray_lifted / tray_goal_height / tray_goal_height_fine 从 0 线性增长
        - 增长在 60000 步（约 940 iter × 64 steps）内完成
        - 同期 grasp_symmetry / tray_tilt 也从 0 激活，确保举升质量

    平滑性惩罚：在 20000 步内从初始值增大，与之前保持一致。
    """
    # ── 阶段B 激活：举升奖励从 0 线性增长 ────────────────────────────
    activate_tray_lifted = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "tray_lifted", "weight": 20.0, "num_steps": 60000},
    )
    activate_tray_goal_height = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "tray_goal_height", "weight": 16.0, "num_steps": 60000},
    )
    activate_tray_goal_height_fine = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "tray_goal_height_fine", "weight": 5.0, "num_steps": 60000},
    )
    activate_grasp_symmetry = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "grasp_symmetry", "weight": -2.0, "num_steps": 60000},
    )
    activate_tray_tilt = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "tray_tilt", "weight": -3.0, "num_steps": 60000},
    )

    # ── 平滑性惩罚逐步增大 ───────────────────────────────────────────
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

    scene: TrayLiftSceneCfg = TrayLiftSceneCfg(num_envs=3072, env_spacing=3.0)
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
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 32 * 1024   # 8192 envs 需要 ~16400，翻倍留余量
        self.sim.physx.friction_correlation_distance = 0.00625

