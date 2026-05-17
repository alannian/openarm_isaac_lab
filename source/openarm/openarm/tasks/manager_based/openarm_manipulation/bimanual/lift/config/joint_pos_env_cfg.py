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
1. **托盘几何**改为细长 bar：4 × 50 × 2.5 cm，质量 0.4 kg。
   - 2.5 cm 厚度远小于夹爪 88 mm 开口，从上方下沉时手指自然包住截面。
   - 50 cm 长度让两端 ±0.25 m 处腾空，避免与中央支架接触。
2. **末端 TCP 坐标系**使用 FrameTransformer + OffsetCfg(pos=(0, 0, 0.105))，
   把 `openarm_left/right_hand` body 的位姿平移到指尖中心。
   - 105 mm 经验值来自 unimanual cabinet 中 finger frame 偏移 (0, 0, 0.075)
     再加上 hand → finger base 的约 0.03 m。两指间为典型 Franka-style gripper。
3. **初始关节姿态**让肩、肘略弯，让两个 EE 从初态就大致悬于托盘端上方，
   减少 reach-only 阶段需要探索的距离。
4. **二值夹爪**保留：open=0.044, close=0.0；策略只需在合适时机切换。
"""

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


# ─────────────────────────────────────────────────────────────────────
# TCP 偏移：从 openarm_*_hand body 原点到两指间中心
# ─────────────────────────────────────────────────────────────────────
# 上手类夹爪：hand 局部 +Z 指向指尖方向。
# unimanual cabinet 里 finger body 到指尖中心是 (0, 0, 0.075)；
# 同 USD 里 hand body 位于 finger base 后方约 0.030 m → 总 ~0.105 m。
TCP_OFFSET = (0.0, 0.0, 0.105)


@configclass
class OpenArmTrayLiftEnvCfg(BimanualTrayLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # ─────────── 机器人 ───────────
        # 初始关节略弯，让两 EE 自然落到托盘端附近正上方，缩短 reach 阶段。
        # 双臂使用同一组角度（OpenArm 双臂沿 Y 镜像对称安装时，相同关节角会自然
        # 把双手摆成镜像姿态；若用户机器人不是这样，可在此处把右臂部分关节取负）。
        ready_joint_pos = {
            "openarm_left_joint1": 0.0,
            "openarm_left_joint2": 0.40,    # 抬肩朝前
            "openarm_left_joint3": 0.0,
            "openarm_left_joint4": -1.20,   # 肘 ~70°
            "openarm_left_joint5": 0.0,
            "openarm_left_joint6": 1.20,    # 腕朝下
            "openarm_left_joint7": 0.0,
            "openarm_right_joint1": 0.0,
            "openarm_right_joint2": 0.40,
            "openarm_right_joint3": 0.0,
            "openarm_right_joint4": -1.20,
            "openarm_right_joint5": 0.0,
            "openarm_right_joint6": 1.20,
            "openarm_right_joint7": 0.0,
            "openarm_left_finger_joint.*": 0.044,   # 初始张开
            "openarm_right_finger_joint.*": 0.044,
        }
        self.scene.robot = OPEN_ARM_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(joint_pos=ready_joint_pos),
        )

        # ─────────── 托盘（长杆 bar） ───────────
        # size = (x=0.04, y=0.50, z=0.025)，质量 0.4 kg
        # init z = stand_top (0.22) + half_thickness (0.0125) = 0.2325
        self.scene.tray = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Tray",
            spawn=sim_utils.CuboidCfg(
                size=(0.04, 0.50, 0.025),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_depenetration_velocity=1.0,
                    solver_position_iteration_count=32,   # 细瘦物体接触
                    solver_velocity_iteration_count=2,
                    max_linear_velocity=4.0,
                    max_angular_velocity=10.0,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.4),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.005,
                    rest_offset=0.0,
                ),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0,
                    dynamic_friction=0.9,
                    restitution=0.0,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.55, 0.35, 0.15), roughness=0.7
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.40, 0.0, 0.2325),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )

        # ─────────── 末端 TCP FrameTransformer ───────────
        # 把目标 frame 沿 hand 局部 +Z 平移 ~10.5 cm，落到两指中央。
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
        # 关节增量 (delta) 控制，scale=0.5 → 一步内最多 ±0.5 rad。
        # use_default_offset=True：神经网络输出绕初始 joint_pos 振荡。
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
