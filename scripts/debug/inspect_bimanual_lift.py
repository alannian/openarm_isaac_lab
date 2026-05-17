# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""双臂托盘举升任务的训练前 sanity check 脚本。

无需 ckpt，直接以"零动作"启动 Isaac Sim，在 GUI 中：
1. 把两个 EE TCP 的坐标轴可视化（red=X / green=Y / blue=Z）。
2. 静态保持初始关节姿态约 30 s，方便你转视角观察。
3. 打印 EE / 托盘的世界坐标与手部三轴方向，定量验证 TCP_OFFSET、forward_axis、span_axis、初始姿态。

用法：
    python scripts/debug/inspect_bimanual_lift.py

按 Ctrl-C 退出。
"""

import argparse

from isaaclab.app import AppLauncher

# CLI
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=3000, help="保持渲染的步数（每步 ~16ms）")
parser.add_argument("--print_every", type=int, default=120)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
# 强制带 GUI
args.headless = False
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ── 必须在 AppLauncher 之后再 import ────────────────────────────
import torch
import gymnasium as gym

import openarm.tasks  # noqa: F401  触发任务注册
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def main():
    task_id = "Isaac-Lift-Tray-OpenArm-Bi-Play-v0"

    # 关键：通过 parse_env_cfg 拿到 cfg 实例，再传给 gym.make
    env_cfg = parse_env_cfg(task_id, device="cuda:0", num_envs=args.num_envs)

    # 打开 TCP frame 可视化（即使 cfg 里关着也强制开）
    env_cfg.scene.left_ee_frame.debug_vis = True
    env_cfg.scene.right_ee_frame.debug_vis = True

    env = gym.make(task_id, cfg=env_cfg).unwrapped
    obs, _ = env.reset()

    action_dim = env.action_manager.total_action_dim
    zero_action = torch.zeros((args.num_envs, action_dim), device=env.device)

    print("\n" + "=" * 78)
    print(" SANITY CHECK — 双臂托盘举升")
    print("=" * 78)

    for step in range(args.steps):
        env.step(zero_action)

        if step % args.print_every == 0:
            scene = env.scene
            left_ee = scene["left_ee_frame"].data.target_pos_w[0, 0]
            right_ee = scene["right_ee_frame"].data.target_pos_w[0, 0]
            tray_pos = scene["tray"].data.root_pos_w[0]
            tray_quat = scene["tray"].data.root_quat_w[0]

            # 抓取目标点（root frame 原点假设在 (0,0,0)，世界系下即 tray ± half_length 上 0.02）
            from isaaclab.utils.math import quat_apply
            local_y = torch.tensor([0.0, 1.0, 0.0], device=tray_pos.device)
            tray_long = quat_apply(tray_quat.unsqueeze(0), local_y.unsqueeze(0))[0]
            half_len = 0.25
            left_tgt = tray_pos + half_len * tray_long + torch.tensor([0, 0, 0.02], device=tray_pos.device)
            right_tgt = tray_pos - half_len * tray_long + torch.tensor([0, 0, 0.02], device=tray_pos.device)

            # 手部三轴
            robot = scene["robot"]
            l_hand_id = robot.find_bodies("openarm_left_hand")[0][0]
            r_hand_id = robot.find_bodies("openarm_right_hand")[0][0]
            l_quat = robot.data.body_quat_w[0, l_hand_id]
            r_quat = robot.data.body_quat_w[0, r_hand_id]
            e = torch.eye(3, device=l_quat.device)
            l_axes = quat_apply(l_quat.unsqueeze(0).expand(3, -1), e)
            r_axes = quat_apply(r_quat.unsqueeze(0).expand(3, -1), e)

            print(f"\n--- step {step} ---")
            print(f"tray pos       : {tray_pos.cpu().tolist()}")
            print(f"tray long axis : {tray_long.cpu().tolist()}  (期望 ≈ (0,±1,0))")
            print(f"left  EE pos   : {left_ee.cpu().tolist()}")
            print(f"left  target   : {left_tgt.cpu().tolist()}")
            print(f"left  dist     : {torch.norm(left_ee - left_tgt):.3f}  (期望 < 0.20)")
            print(f"right EE pos   : {right_ee.cpu().tolist()}")
            print(f"right target   : {right_tgt.cpu().tolist()}")
            print(f"right dist     : {torch.norm(right_ee - right_tgt):.3f}  (期望 < 0.20)")
            print(f"left  hand axes (world):")
            for i, name in enumerate(["+X", "+Y", "+Z"]):
                v = l_axes[i].cpu().tolist()
                print(f"   local {name} -> world {[f'{x:+.2f}' for x in v]}"
                      f"  {'<-- 朝下 (forward_axis=' + str(i) + ')' if v[2] < -0.85 else ''}")
            print(f"right hand axes (world):")
            for i, name in enumerate(["+X", "+Y", "+Z"]):
                v = r_axes[i].cpu().tolist()
                print(f"   local {name} -> world {[f'{x:+.2f}' for x in v]}"
                      f"  {'<-- 朝下 (forward_axis=' + str(i) + ')' if v[2] < -0.85 else ''}")

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
