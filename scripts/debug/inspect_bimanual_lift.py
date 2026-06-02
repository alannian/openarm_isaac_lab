# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""双臂托盘举升任务的训练前 sanity check 脚本。

无需 ckpt，直接以"零动作"启动 Isaac Sim，在 GUI 中：
1. 把两个 EE TCP 的坐标轴可视化（red=X / green=Y / blue=Z）。
2. 静态保持初始关节姿态约 30 s，方便你转视角观察。
3. 打印 EE / 托盘的世界坐标、开合轴、手指-托盘最近距离（穿模检测）、
   capture_band / grasp_hold（spawn 时应≈0，避免假抓握奖励）。

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

    # 打开 TCP frame 可视化，并把坐标轴缩小到 4 cm（默认是米级巨箭头，会挡住夹爪/托盘）。
    from isaaclab.markers.config import FRAME_MARKER_CFG
    for tag, frame in (("L", env_cfg.scene.left_ee_frame), ("R", env_cfg.scene.right_ee_frame)):
        marker = FRAME_MARKER_CFG.copy()
        marker.markers["frame"].scale = (0.04, 0.04, 0.04)
        marker.prim_path = f"/Visuals/EEFrame{tag}"
        frame.visualizer_cfg = marker
        frame.debug_vis = True

    env = gym.make(task_id, cfg=env_cfg).unwrapped
    obs, _ = env.reset()

    action_dim = env.action_manager.total_action_dim
    zero_action = torch.zeros((args.num_envs, action_dim), device=env.device)

    print("\n" + "=" * 78)
    print(" SANITY CHECK — 双臂托盘举升")
    print("=" * 78)

    # ── 一次性：打印 body / joint 名称，定位两侧夹爪手指 body ─────────
    robot = env.scene["robot"]
    body_names = list(robot.body_names)
    print("\nBODY NAMES :", body_names)
    print("JOINT NAMES:", list(robot.joint_names))
    # 注意：body 名形如 openarm_left_left_finger / openarm_left_right_finger，
    # 必须按"前缀=哪只手 + 后缀=finger"匹配，否则会把左右手的手指混在一起。
    left_finger_ids = [i for i, n in enumerate(body_names)
                       if n.lower().startswith("openarm_left_") and n.lower().endswith("finger")]
    right_finger_ids = [i for i, n in enumerate(body_names)
                        if n.lower().startswith("openarm_right_") and n.lower().endswith("finger")]
    print("left  finger bodies:", [(i, body_names[i]) for i in left_finger_ids])
    print("right finger bodies:", [(i, body_names[i]) for i in right_finger_ids])

    def finger_open_axis(ids):
        """两根手指 body 的世界连线（归一化）= 夹爪开合轴（世界系）。"""
        if len(ids) < 2:
            return None
        p = robot.data.body_pos_w[0, ids[:2]]
        sep = p[0] - p[1]
        n = torch.norm(sep)
        if n < 1e-9:
            return None
        return (sep / n)

    def finger_tray_clearance(ids, tray_pos, tray_quat):
        """手指 body 到托盘 AABB 表面的最近距离 (m)。负值 = 穿进碰撞体内部。"""
        if len(ids) < 2:
            return None
        # tray.usda: 半尺寸 0.18, 0.25, 0.015（局部系）
        half = torch.tensor([0.18, 0.25, 0.015], device=tray_pos.device)
        pts = robot.data.body_pos_w[0, ids]  # (2, 3)
        # 转到托盘局部
        from isaaclab.utils.math import quat_apply, quat_conjugate
        rel = pts - tray_pos.unsqueeze(0)
        q_inv = quat_conjugate(tray_quat.unsqueeze(0)).expand(len(ids), -1)
        local = quat_apply(q_inv, rel)
        # 点到轴对齐盒表面的 signed distance（内部为负）
        d = torch.abs(local) - half.unsqueeze(0)
        outside = torch.clamp(d, min=0.0)
        dist_outside = torch.norm(outside, dim=1)
        dist_inside = torch.max(d, dim=1).values
        signed = torch.where(dist_inside <= 0.0, dist_inside, dist_outside)
        return signed.min().item(), signed.cpu().tolist()

    def capture_band(open_val: float) -> float:
        rise = max(0.0, min(1.0, (open_val - 0.004) / 0.006))
        fall = max(0.0, min(1.0, (0.032 - open_val) / 0.006))
        return rise * fall

    def finger_open_mean(side: str) -> float:
        jids = [i for i, n in enumerate(robot.joint_names)
                if f"openarm_{side}_finger_joint" in n]
        if not jids:
            return float("nan")
        return robot.data.joint_pos[0, jids].mean().item()

    for step in range(args.steps):
        env.step(zero_action)

        if step % args.print_every == 0:
            scene = env.scene
            left_ee = scene["left_ee_frame"].data.target_pos_w[0, 0]
            right_ee = scene["right_ee_frame"].data.target_pos_w[0, 0]
            tray_pos = scene["tray"].data.root_pos_w[0]
            tray_quat = scene["tray"].data.root_quat_w[0]

            # 侧面抓取点：tray_pos + R * (0, ±0.23, 0)（与 rewards/observations 一致）
            from isaaclab.utils.math import quat_apply
            half_grasp_y = 0.23
            grasp_z_offset = 0.0
            left_local = torch.tensor([0.0, half_grasp_y, grasp_z_offset], device=tray_pos.device)
            right_local = torch.tensor([0.0, -half_grasp_y, grasp_z_offset], device=tray_pos.device)
            local_y = torch.tensor([0.0, 1.0, 0.0], device=tray_pos.device)
            tray_long = quat_apply(tray_quat.unsqueeze(0), local_y.unsqueeze(0))[0]
            left_tgt = tray_pos + quat_apply(tray_quat.unsqueeze(0), left_local.unsqueeze(0))[0]
            right_tgt = tray_pos + quat_apply(tray_quat.unsqueeze(0), right_local.unsqueeze(0))[0]

            # 手部三轴
            robot = scene["robot"]
            l_hand_id = robot.find_bodies("openarm_left_hand")[0][0]
            r_hand_id = robot.find_bodies("openarm_right_hand")[0][0]
            l_quat = robot.data.body_quat_w[0, l_hand_id]
            r_quat = robot.data.body_quat_w[0, r_hand_id]
            e = torch.eye(3, device=l_quat.device)
            l_axes = quat_apply(l_quat.unsqueeze(0).expand(3, -1), e)
            r_axes = quat_apply(r_quat.unsqueeze(0).expand(3, -1), e)

            # 托盘水平度（局部 +Z · 世界 +Z）
            x, y = tray_quat[1].item(), tray_quat[2].item()
            tray_flat = 1.0 - 2.0 * (x * x + y * y)

            print(f"\n--- step {step} ---")
            print(f"tray pos       : {tray_pos.cpu().tolist()}")
            print(f"tray flatness  : {tray_flat:.3f}  (期望 ≈ 1.0，<0.95 说明板明显倾斜)")
            print(f"tray long axis : {tray_long.cpu().tolist()}  (期望 ≈ (0,±1,0))")
            print(f"left  EE pos   : {left_ee.cpu().tolist()}")
            print(f"left  target   : {left_tgt.cpu().tolist()}")
            print(f"left  dist     : {torch.norm(left_ee - left_tgt):.3f}  "
                  f"(自然下垂期望 **> 0.20**)")
            print(f"right EE pos   : {right_ee.cpu().tolist()}")
            print(f"right target   : {right_tgt.cpu().tolist()}")
            print(f"right dist     : {torch.norm(right_ee - right_tgt):.3f}  "
                  f"(自然下垂期望 **> 0.20**)")

            # 大臂是否自然下垂：link1 局部 +Z 应接近世界 -Z（|world_z + 1| 小）
            for tag, link_name in (("left", "openarm_left_link1"), ("right", "openarm_right_link1")):
                bid = robot.find_bodies(link_name)[0][0]
                lq = robot.data.body_quat_w[0, bid]
                z_axis = quat_apply(lq.unsqueeze(0), torch.tensor([[0., 0., 1.]], device=lq.device))[0]
                hang = -z_axis[2].item()  # 1=完全朝下, 0=水平
                kind = "下垂✓" if hang > 0.85 else ("倾斜" if hang > 0.5 else "非下垂✗")
                print(f"{tag:5s} upper_arm hang: world_z={z_axis[2].item():+.2f}  "
                      f"down_score={hang:.2f}  -> {kind}")

            # TCP body vs FrameTransformer（修复后应 ≈ 0）
            for tag, side, frame_ee in (("L", "left", left_ee), ("R", "right", right_ee)):
                bid = robot.find_bodies(f"openarm_{side}_ee_tcp")[0][0]
                body_tcp = robot.data.body_pos_w[0, bid]
                delta = torch.norm(body_tcp - frame_ee).item()
                print(f"{tag} TCP body↔frame delta: {delta*1000:.1f} mm  (期望 < 1 mm)")
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

            # 夹爪开合轴 + 手指-托盘 clearance（穿模检测）
            for tag, ids, side in (("left", left_finger_ids, "left"),
                                   ("right", right_finger_ids, "right")):
                ax = finger_open_axis(ids)
                if ax is None:
                    continue
                az = abs(ax[2].item())
                kind = "竖直(能上下夹住板 ✓)" if az > 0.7 else (
                    "水平(无法上下夹住扁平板 ✗)" if az < 0.3 else "倾斜")
                print(f"{tag:5s} gripper open-axis (world): "
                      f"{[f'{x:+.2f}' for x in ax.cpu().tolist()]}  |z|={az:.2f}  -> {kind}")
                clr = finger_tray_clearance(ids, tray_pos, tray_quat)
                if clr is not None:
                    mn, each = clr
                    flag = "⚠ 穿模" if mn < -0.002 else ("贴边" if mn < 0.005 else "安全间隙")
                    print(f"{tag:5s} finger→tray clearance (m): min={mn:+.4f}  per-finger={each}  -> {flag}")
                fo = finger_open_mean(side)
                cb = capture_band(fo)
                print(f"{tag:5s} finger_open={fo:.4f}  capture_band={cb:.2f}  "
                      f"(spawn 期望 open≈0.044 & band≈0)")

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
