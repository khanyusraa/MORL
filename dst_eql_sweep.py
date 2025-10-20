import wandb
import numpy as np
import mo_gymnasium as mo_gym
from morl_baselines.multi_policy.envelope.envelope import Envelope
from morl_baselines.common.evaluation import eval_mo, hypervolume
from mo_gymnasium.wrappers import MORecordEpisodeStatistics

ref_point = np.array([-10.0, -10.0])
GAMMA = 0.98

def prep_environments():
    env = mo_gym.make("deep-sea-treasure-v0", render_mode="rgb_array")
    env = MORecordEpisodeStatistics(env, gamma=GAMMA)
    env.reset(seed=9)

    return env

def train(config=None):
    with wandb.init(config=config):
        config = wandb.config

        # Prepare training and evaluation environments
        env = prep_environments()
        eval_env = prep_environments()

        agent = Envelope(
            env=env,
            learning_rate=config.learning_rate,
            initial_epsilon=config.initial_epsilon,
            epsilon_decay_steps=config.epsilon_decay_steps,
            final_epsilon=config.final_epsilon,
            batch_size= 32,
            tau=config.tau,
            target_net_update_freq=config.target_net_update_freq,
            gamma=config.gamma,
            num_sample_w=config.num_sample_w,
            net_arch= [256, 256, 256, 256],
            per = True,
            per_alpha=config.per_alpha,
            gradient_updates=config.gradient_updates,
            buffer_size=config.buffer_size,
            initial_homotopy_lambda=config.initial_homotopy_lambda,
            final_homotopy_lambda=config.final_homotopy_lambda,
            homotopy_decay_steps=config.homotopy_decay_steps,
            learning_starts=config.learning_starts,
            max_grad_norm=config.max_grad_norm,
            seed=config.seed
        )

        pareto = env.unwrapped.pareto_front(GAMMA)

        agent.train(
            total_timesteps=100000,
            eval_env=eval_env, 
            eval_freq=10000,
            ref_point=ref_point,
            num_eval_episodes_for_front= 30,
            known_pareto_front=pareto
        )

# Sweep configuration
sweep_config = {
    "method": "bayes",
    "metric": {"name": "hypervolume", "goal": "maximize"},
    "parameters": {
        "learning_rate": {"min": 1e-5, "max": 1e-3},
        "initial_epsilon": {"min": 0.1, "max": 1.0},
        "epsilon_decay_steps": {"values": [1000, 5000, 10000, 20000]},
        "final_epsilon": {"min": 0.01, "max": 0.1},
        "tau": {"min": 0.001, "max": 0.05},
        "target_net_update_freq": {"values": [100, 500, 1000]},
        "gamma": {"min": 0.9, "max": 0.999},
        "num_sample_w": {"values": [4, 8, 16]},
        "per_alpha": {"min": 0.4, "max": 1.0},
        "gradient_updates": {"values": [1, 2, 4]},
        "buffer_size": {"values": [5000, 20000, 50000]},
        "initial_homotopy_lambda": {"min": 0.0, "max": 1.0},
        "final_homotopy_lambda": {"min": 0.0, "max": 1.0},
        "homotopy_decay_steps": {"values": [1000, 5000, 10000]},
        "learning_starts": {"values": [100, 1000, 5000]},
        "max_grad_norm": {"values": [0.5, 1.0, 5.0]},
        "seed": {"values": [0, 1, 2, 3, 4]},
    },
}

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="dst_eql")
    wandb.agent(sweep_id, function=train, count=20)