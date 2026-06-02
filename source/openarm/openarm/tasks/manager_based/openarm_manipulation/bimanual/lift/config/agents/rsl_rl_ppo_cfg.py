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

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

from isaaclab.utils import configclass


@configclass
class OpenArmTrayLiftPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO 设置（自然初态 + 四阶段课程版）：

    - 动作维 16，观测维 ~78，网络略深。
    - episode 10 s ≈ 500 policy steps（从下垂到抓取点需更长时间）。
    - num_steps_per_env=64 → 课程阶段切换见 lift_env_cfg.CurriculumCfg。
    - 建议 max_iterations=8000~10000 以跑完四阶段课程。
    """
    num_steps_per_env = 64
    max_iterations = 8000
    save_interval = 100
    experiment_name = "openarm_bi_tray_lift"
    run_name = ""
    resume = False
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        # 注：二值夹爪两维 raw action 对环境无梯度，过大的初始 std + 高 entropy bonus
        # 会让其幅值无界漂移；下面采用更保守的 init_noise_std / entropy_coef 抑制之。
        init_noise_std=0.5,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=8,
        num_mini_batches=8,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.016,
        max_grad_norm=1.0,
    )
