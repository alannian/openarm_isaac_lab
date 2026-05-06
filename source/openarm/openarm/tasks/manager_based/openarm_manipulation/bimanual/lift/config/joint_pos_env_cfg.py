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

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from .. import mdp
from ..lift_env_cfg import BimanualTrayLiftEnvCfg

from source.openarm.openarm.tasks.manager_based.openarm_manipulation.assets.openarm_bimanual import (
    OPEN_ARM_CFG,
)


@configclass
class OpenArmTrayLiftEnvCfg(BimanualTrayLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ── 机器人（使用标准刚度 + 重力开启，适合 lift 任务） ──
        self.scene.robot = OPEN_ARM_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={
                    "openarm_left_joint1":  0.0,
                    "openarm_left_joint2":  0.0,
                    "openarm_left_joint3":  0.0,
                    "openarm_left_joint4":  0.0,
                    "openarm_left_joint5":  0.0,
                    "openarm_left_joint6":  0.0,
                    "openarm_left_joint7":  0.0,
                    "openarm_right_joint1": 0.0,
                    "openarm_right_joint2": 0.0,
                    "openarm_right_joint3": 0.0,
                    "openarm_right_joint4": 0.0,
                    "openarm_right_joint5": 0.0,
                    "openarm_right_joint6": 0.0,
                    "openarm_right_joint7": 0.0,
                    "openarm_left_finger_joint.*":  0.044,  # 初始张开
                    "openarm_right_finger_joint.*": 0.044,
                },
            ),
        )

        # ── 托盘（程序化长方体，长轴沿 Y 轴） ──
        # size = (x_width, y_length, z_height) = (0.12, 0.60, 0.03)
        # half_length = 0.60/2 = 0.30，与 rewards.py 中的 half_length 保持一致
        self.scene.tray = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Tray",
            spawn=sim_utils.CuboidCfg(
                size=(0.12, 0.60, 0.03),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_depenetration_velocity=1.0,
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.5),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.55, 0.35, 0.15), roughness=0.6
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.28, 0.0, 0.375),   # x=0.28m: 远离基座，手臂向前伸展从侧接近托盘端
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )

        # ── 左臂 EE FrameTransformer ──
        # prim_path 使用已确认存在的 openarm_left_hand body（来自 reach 任务验证）
        self.scene.left_ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/openarm_left_hand",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/openarm_left_hand",
                    name="left_ee_tcp",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
                ),
            ],
        )

        # ── 右臂 EE FrameTransformer ──
        self.scene.right_ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/openarm_right_hand",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/openarm_right_hand",
                    name="right_ee_tcp",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
                ),
            ],
        )

        # ── 动作 ──
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
        self.scene.num_envs = 16
        self.scene.env_spacing = 3.5
        self.observations.policy.enable_corruption = False
