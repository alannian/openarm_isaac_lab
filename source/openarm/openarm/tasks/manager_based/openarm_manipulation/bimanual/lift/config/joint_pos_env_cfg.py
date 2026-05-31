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
1. **托盘资产**改为真正的托盘：扁平板 + 两端把手横杆（单刚体），从
   ``usds/tray/tray.usda`` 加载。把手 24 mm 的薄边正好落入夹爪 88 mm 开口，
   从上方下压即可套住、闭合即抓牢，抬升稳定。
2. **末端 TCP 坐标系**用 FrameTransformer + OffsetCfg(pos=(0, 0, 0.105))，把
   ``openarm_left/right_hand`` body 的位姿平移到两指中心。
3. **初始关节姿态**让肩、肘略弯、腕朝下，使两个 TCP 从初态就大致悬在两端把手
   正上方，缩短 reach 阶段需要探索的距离（关节角均在 USD 硬限位内）。
4. **二值夹爪**：open=0.044, close=0.0；策略只需在合适时机切换。
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from .. import mdp
from ..lift_env_cfg import BimanualTrayLiftEnvCfg, TRAY_BASE_HEIGHT

from source.openarm.openarm.tasks.manager_based.openarm_manipulation import (
    OPENARM_ROOT_DIR,
)
from source.openarm.openarm.tasks.manager_based.openarm_manipulation.assets.openarm_bimanual import (
    OPEN_ARM_CFG,
)


# ─────────────────────────────────────────────────────────────────────
# TCP 偏移：从 openarm_*_hand body 原点到两指中心（hand 局部 +Z 指向指尖）
# ─────────────────────────────────────────────────────────────────────
TCP_OFFSET = (0.0, 0.0, 0.105)


@configclass
class OpenArmTrayLiftEnvCfg(BimanualTrayLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # ─────────── 机器人 ───────────
        # 初始关节略弯、腕朝下，让两 TCP 自然落到两端把手正上方，缩短 reach 阶段。
        # OpenArm USD 硬限位（来自 articulation 校验）：左右臂部分镜像，
        #   left_joint2  ∈ [-3.316,  0.175]   right_joint2 ∈ [-0.175, 3.316]
        #   joint4       ∈ [ 0.000,  2.443]   joint6       ∈ [-0.785, 0.785]
        # 故 joint2 左右取相反符号、joint4/joint6 左右同号，形成镜像姿态。
        ready_joint_pos = {
            "openarm_left_joint1": 0.0,
            "openarm_left_joint2": -0.40,
            "openarm_left_joint3": 0.0,
            "openarm_left_joint4": 1.20,
            "openarm_left_joint5": 0.0,
            "openarm_left_joint6": 0.60,
            "openarm_left_joint7": 0.0,
            "openarm_right_joint1": 0.0,
            "openarm_right_joint2": 0.40,
            "openarm_right_joint3": 0.0,
            "openarm_right_joint4": 1.20,
            "openarm_right_joint5": 0.0,
            "openarm_right_joint6": 0.60,
            "openarm_right_joint7": 0.0,
            "openarm_left_finger_joint.*": 0.044,
            "openarm_right_finger_joint.*": 0.044,
        }
        self.scene.robot = OPEN_ARM_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(joint_pos=ready_joint_pos),
        )

        # ─────────── 托盘（带把手的真正托盘，单刚体 USD） ───────────
        # 几何在 usds/tray/tray.usda 内定义；这里设质量 / 求解器迭代 / 初始位姿。
        # 初始质心 z = 支架顶 0.20 + 板半厚 0.0125 = TRAY_BASE_HEIGHT(0.2125)。
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
                mass_props=sim_utils.MassPropertiesCfg(mass=0.6),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.40, 0.0, TRAY_BASE_HEIGHT),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )

        # ─────────── 末端 TCP FrameTransformer ───────────
        self.scene.left_ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/openarm_left_hand",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/openarm_left_hand",
                    name="left_ee_tcp",
                    offset=OffsetCfg(pos=TCP_OFFSET),
                ),
            ],
        )
        self.scene.right_ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/openarm_right_hand",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/openarm_right_hand",
                    name="right_ee_tcp",
                    offset=OffsetCfg(pos=TCP_OFFSET),
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
