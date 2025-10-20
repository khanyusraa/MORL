import wandb
import numpy as np
import mo_gymnasium as mo_gym
from morl_baselines.single_policy.esr.eupg import EUPG
from morl_baselines.common.evaluation import eval_mo, hypervolume, expected_utility, maximum_utility_loss
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

        # Normalize weights
        weights = np.array([
            config.scalarization_weight_0,
            config.scalarization_weight_1,
        ], dtype=np.float32)
        weights = weights / weights.sum()

        def scalarization(rewards, weights=None):
            return (rewards * agent.weights).sum()

        agent = EUPG(
            env=env,
            scalarization=scalarization,
            weights=weights,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            buffer_size=config.buffer_size,
            seed=config.seed,
            log=True,
            log_every=5000
        )

        chunk_size = 10000
        n_chunks = 100000 // chunk_size

        for i in range(n_chunks):
            agent.train(total_timesteps=chunk_size)

            # Evaluate on eval_env
            _, _, results, _ = eval_mo(agent, eval_env, weights, scalarization)
            pareto_front = np.array(results)
            hv = hypervolume(ref_point, pareto_front)
            mul = maximum_utility_loss(pareto_front, ref_point, weights)
            eu = expected_utility(pareto_front, weights)

            wandb.log({
                "final_charts/hypervolume": hv,
                "final_charts/max_utility_loss": mul,
                "final_charts/expected_utility": eu,
            })

# Sweep configuration
sweep_config = {
    "method": "bayes",
    "metric": {"name": "hypervolume", "goal": "maximize"},
    "parameters": {
        "learning_rate": {"min": 1e-5, "max": 1e-3},
        "gamma": {"min": 0.9, "max": 0.999},
        "buffer_size": {"values": [5000, 50000, 100000]},
        "scalarization_weight_0": {"min": 0.01, "max": 1.0},
        "scalarization_weight_1": {"min": 0.01, "max": 1.0},
        "seed": {"values": [0, 1, 2, 3, 4]},
    },
}

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="dst_eupg")
    wandb.agent(sweep_id, function=train, count=20)