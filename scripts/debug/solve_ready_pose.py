# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""数值求解双臂"夹托盘侧面"的 ready 关节姿态（无需 ckpt / 无需训练）。

背景：inspect 实测该夹爪开合轴**无法竖直**（始终沿世界 Y 略带倾斜），所以放弃
"上下夹扁平板"，改成"水平夹托盘侧壁"。本脚本在**同一次启动**里对每条手臂做
随机搜索 + 坐标下降，找出使
    1) ee_tcp body 落在指定抓取点附近
    2) 夹爪开合轴尽量对齐某个期望方向（默认 Y=水平夹取；同时也报告 Z=竖直夹取）
的 7 个关节角，直接打印成可粘贴的 ready_joint_pos。

它会同时给出"水平夹取"和"竖直夹取"两套最优解及其残差，从而一次性回答
"这个夹爪到底能不能竖直夹住扁平板"。

用法：
    python scripts/debug/solve_ready_pose.py            # 默认 headless
    python scripts/debug/solve_ready_pose.py --samples 8000
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--samples", type=int, default=6000, help="每条臂随机采样次数")
parser.add_argument("--refine", type=int, default=400, help="坐标下降迭代次数")
parser.add_argument("--tray_x", type=float, default=0.40)
parser.add_argument("--grasp_y", type=float, default=0.23, help="抓取点 |y|（托盘半长）")
parser.add_argument("--tray_z", type=float, default=0.345, help="抓取点 z（托盘质心高度）")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True  # 纯运动学求解，不需要 GUI
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import gymnasium as gym

import openarm.tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def main():
    task_id = "Isaac-Lift-Tray-OpenArm-Bi-Play-v0"
    env_cfg = parse_env_cfg(task_id, device="cuda:0", num_envs=args.num_envs)
    env = gym.make(task_id, cfg=env_cfg).unwrapped
    env.reset()

    sim = env.sim
    dt = sim.get_physics_dt()
    robot = env.scene["robot"]
    device = robot.device

    body_names = list(robot.body_names)
    joint_names = list(robot.joint_names)

    def body_id(name):
        return body_names.index(name)

    def joint_ids(names):
        return [joint_names.index(n) for n in names]

    # —— 关节 / body 索引 ——
    arm = {
        "left": joint_ids([f"openarm_left_joint{i}" for i in range(1, 8)]),
        "right": joint_ids([f"openarm_right_joint{i}" for i in range(1, 8)]),
    }
    finger_jids = [j for j, n in enumerate(joint_names)
                   if "finger_joint" in n]
    tcp_bid = {"left": body_id("openarm_left_ee_tcp"),
               "right": body_id("openarm_right_ee_tcp")}
    fingers_bid = {
        "left": [i for i, n in enumerate(body_names)
                 if n.startswith("openarm_left_") and n.endswith("finger")],
        "right": [i for i, n in enumerate(body_names)
                  if n.startswith("openarm_right_") and n.endswith("finger")],
    }

    # —— 关节限位 ——
    lim = getattr(robot.data, "joint_pos_limits", None)
    if lim is None:
        lim = robot.data.soft_joint_pos_limits
    lim = lim[0]  # (J, 2)
    lo, hi = lim[:, 0], lim[:, 1]
    # 限位若为 ±inf（连续关节），夹到 ±pi
    big = torch.isinf(lo) | torch.isinf(hi) | (hi - lo > 2 * 3.1416)
    lo = torch.where(big, torch.full_like(lo, -3.1416), lo)
    hi = torch.where(big, torch.full_like(hi, 3.1416), hi)

    default_q = robot.data.default_joint_pos[0].clone()
    zeros_v = torch.zeros_like(default_q).unsqueeze(0)

    def set_and_fk(q_full):
        """写入全关节角 -> 步进一帧 -> 返回 (body_pos_w, body_quat_w)。"""
        q = q_full.unsqueeze(0)
        robot.write_joint_state_to_sim(q, zeros_v)
        robot.set_joint_position_target(q)
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(dt)
        return robot.data.body_pos_w[0], robot.data.body_quat_w[0]

    def open_axis(bpos, side):
        ids = fingers_bid[side]
        sep = bpos[ids[0]] - bpos[ids[1]]
        n = torch.norm(sep)
        return sep / n if n > 1e-9 else sep

    targets = {
        "left": torch.tensor([args.tray_x, args.grasp_y, args.tray_z], device=device),
        "right": torch.tensor([args.tray_x, -args.grasp_y, args.tray_z], device=device),
    }
    axis_Y = torch.tensor([0.0, 1.0, 0.0], device=device)
    axis_Z = torch.tensor([0.0, 0.0, 1.0], device=device)

    def score(q_full, side, want_axis, w_axis):
        bpos, _ = set_and_fk(q_full)
        tcp = bpos[tcp_bid[side]]
        pos_err = torch.norm(tcp - targets[side]).item()
        ax = open_axis(bpos, side)
        align = abs(torch.dot(ax, want_axis).item())  # 1=完全对齐
        # 主目标：先够到（pos_err 小），再对齐开合轴
        return pos_err + w_axis * (1.0 - align), pos_err, align

    def solve(side, want_axis, label):
        idx = arm[side]
        idx_t = torch.tensor(idx, device=device)
        best_q = default_q.clone()
        best_cost, best_pe, best_al = score(best_q, side, want_axis, 0.5)
        # —— 随机搜索 ——
        for _ in range(args.samples):
            q = default_q.clone()
            rand = lo[idx_t] + (hi[idx_t] - lo[idx_t]) * torch.rand(len(idx), device=device)
            q[idx_t] = rand
            q[finger_jids] = 0.044
            cost, pe, al = score(q, side, want_axis, 0.5)
            if cost < best_cost:
                best_cost, best_pe, best_al, best_q = cost, pe, al, q.clone()
        # —— 坐标下降细化 ——
        step = 0.30
        for it in range(args.refine):
            improved = False
            for k in idx:
                for s in (+step, -step):
                    q = best_q.clone()
                    q[k] = torch.clamp(q[k] + s, lo[k], hi[k])
                    cost, pe, al = score(q, side, want_axis, 0.5)
                    if cost < best_cost - 1e-5:
                        best_cost, best_pe, best_al, best_q = cost, pe, al, q.clone()
                        improved = True
            if not improved:
                step *= 0.5
                if step < 1e-3:
                    break
        vals = [best_q[j].item() for j in idx]
        print(f"\n[{side}] 目标开合轴={label}  pos_err={best_pe:.3f} m  "
              f"open-axis对齐={best_al:.2f} (1=完美)")
        for i, v in enumerate(vals, start=1):
            print(f'    "openarm_{side}_joint{i}": {v:.3f},')
        return best_pe, best_al

    print("\n" + "=" * 78)
    print(" READY POSE SOLVER — 抓取点:", {k: v.cpu().tolist() for k, v in targets.items()})
    print("=" * 78)

    summary = {}
    for side in ("left", "right"):
        peY, alY = solve(side, axis_Y, "Y(水平夹侧壁)")
        peZ, alZ = solve(side, axis_Z, "Z(竖直夹扁平板)")
        summary[side] = (peY, alY, peZ, alZ)

    print("\n" + "=" * 78)
    print(" 结论")
    print("=" * 78)
    for side, (peY, alY, peZ, alZ) in summary.items():
        print(f"[{side}] 水平夹取: pos_err={peY:.3f}, 对齐={alY:.2f}   |   "
              f"竖直夹取: pos_err={peZ:.3f}, 对齐={alZ:.2f}")
    print("说明：对齐≈1 且 pos_err<~0.05 才算该方向可行。")
    print("若竖直方向对齐始终上不去 -> 该夹爪无法竖直夹扁平板 -> 采用水平夹侧壁方案。")

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
