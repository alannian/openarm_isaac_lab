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

"""双臂托盘举升任务（关节位置动作空间）具体环境配置。

设计要点：
1. **完全自然下垂初态**：全部臂关节 = 0（USD 默认休息位形），夹爪张开；
2. **托盘**：扁平板 USD，置于中央支架上。
3. **TCP** 直接跟踪 USD 内 ``openarm_*_ee_tcp`` link。
4. **二值夹爪**：open=0.044, close=0.0。
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils import configclass

from .. import mdp
from ..lift_env_cfg import (
    BimanualTrayLiftEnvCfg,
    TRAY_BASE_HEIGHT,
)

from source.openarm.openarm.tasks.manager_based.openarm_manipulation import (
    OPENARM_ROOT_DIR,
)
from source.openarm.openarm.tasks.manager_based.openarm_manipulation.assets.openarm_bimanual import (
    OPEN_ARM_CFG,
)


# ─────────────────────────────────────────────────────────────────────
# TCP：直接使用 USD 里的 ee_tcp link（与 solve_ready_pose.py / 单臂任务一致）。
# 不要用 hand + 固定 offset——offset 与 ee_tcp body 往往差几厘米，会导致
# "inspect 看着夹对了、reach/grasp 奖励却在追另一个点"。
# ─────────────────────────────────────────────────────────────────────


@configclass
class OpenArmTrayLiftEnvCfg(BimanualTrayLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # ─────────── 机器人 ───────────
        # 完全自然下垂初态：全部关节角 = 0（USD 默认休息位形），仅夹爪张开。
        # 不预置任何弯肘/外展——策略从体侧垂臂开始，由 reach 奖励学会伸到托盘。
        hang_joint_pos = {
            "openarm_left_joint1": 0.0,
            "openarm_left_joint2": 0.0,
            "openarm_left_joint3": 0.0,
            "openarm_left_joint4": 0.0,
            "openarm_left_joint5": 0.0,
            "openarm_left_joint6": 0.0,
            "openarm_left_joint7": 0.0,
            "openarm_right_joint1": 0.0,
            "openarm_right_joint2": 0.0,
            "openarm_right_joint3": 0.0,
            "openarm_right_joint4": 0.0,
            "openarm_right_joint5": 0.0,
            "openarm_right_joint6": 0.0,
            "openarm_right_joint7": 0.0,
            "openarm_left_finger_joint.*": 0.044,
            "openarm_right_finger_joint.*": 0.044,
        }
        self.scene.robot = OPEN_ARM_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(joint_pos=hang_joint_pos),
        )
        # 抬高双臂 PD 刚度：默认 stiffness=80 太软，机械臂伸到托盘高度 (~0.42m) 时
        # 在重力下会下垂 ~12cm（inspect 实测静置 TCP 掉到 z≈0.24–0.31）。提高刚度让
        # 机械臂能稳稳把手停在托盘侧面、并在举升时托住托盘。阻尼同步加大以防振荡。
        self.scene.robot.actuators["openarm_arm"].stiffness = 200.0
        self.scene.robot.actuators["openarm_arm"].damping = 24.0

        # ─────────── 托盘（扁平板，单刚体 USD） ───────────
        # 几何在 usds/tray/tray.usda 内定义；这里设质量 / 求解器迭代 / 初始位姿。
        # 初始质心 z = 支架顶 0.33 + 板半厚 0.015 = TRAY_BASE_HEIGHT(0.345)，
        # 接近双手自然伸展高度（inspect 实测 TCP z≈0.434），便于侧面水平夹取。
        self.scene.tray = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Tray",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{OPENARM_ROOT_DIR}/usds/tray/tray.usda",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_depenetration_velocity=1.0,
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_linear_velocity=4.0,
                    max_angular_velocity=10.0,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.40, 0.0, TRAY_BASE_HEIGHT),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )

        # ─────────── 末端 TCP FrameTransformer ───────────
        # 与 unimanual lift / solve_ready_pose 一致：跟踪 USD 内 ee_tcp link。
        self.scene.left_ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/openarm_body_link",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/openarm_left_ee_tcp",
                    name="left_ee_tcp",
                ),
            ],
        )
        self.scene.right_ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/openarm_body_link",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/openarm_right_ee_tcp",
                    name="right_ee_tcp",
                ),
            ],
        )

        # ─────────── 动作 ───────────
        # 关节增量控制，scale=0.5；use_default_offset=True → 网络绕初始姿态振荡。
        self.actions.left_arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[f"openarm_left_joint{i}" for i in range(1, 8)],
            scale=0.5,
            use_default_offset=True,
        )
        self.actions.right_arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[f"openarm_right_joint{i}" for i in range(1, 8)],
            scale=0.5,
            use_default_offset=True,
        )
        self.actions.left_gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["openarm_left_finger_joint.*"],
            open_command_expr={"openarm_left_finger_joint.*": 0.044},
            close_command_expr={"openarm_left_finger_joint.*": 0.0},
        )
        self.actions.right_gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["openarm_right_finger_joint.*"],
            open_command_expr={"openarm_right_finger_joint.*": 0.044},
            close_command_expr={"openarm_right_finger_joint.*": 0.0},
        )


@configclass
class OpenArmTrayLiftEnvCfg_PLAY(OpenArmTrayLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # 评估/可视化：减少并行，关闭观测噪声
        self.scene.num_envs = 16
        self.scene.env_spacing = 3.5
        self.observations.policy.enable_corruption = False
