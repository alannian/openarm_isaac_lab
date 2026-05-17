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
    """PPO 设置：

    - 双臂任务动作维数 ~16，观测维数 ~70，需要稍深的网络。
    - 8 s 任务长度 + decimation=2 + dt=0.01 → episode ≈ 400 policy steps。
      num_steps_per_env=64 让一次 rollout 跨越约 1/6 episode，配合 4096 env
      给 ~262k 样本/iter，足够稳定的优势估计。
    - 初始 init_noise_std=1.0 提供充足探索；adaptive schedule 自动收紧。
    - entropy_coef=0.01 略高，鼓励双臂之间的探索协同。
    """
    num_steps_per_env = 64
    max_iterations = 5000
    save_interval = 100
    experiment_name = "openarm_bi_tray_lift"
    run_name = ""
    resume = False
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=8,
        num_mini_batches=8,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.016,
        max_grad_norm=1.0,
    )
