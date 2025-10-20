import wandb
import numpy as np
import mo_gymnasium as mo_gym
from morl_baselines.single_policy.esr.eupg import EUPG
from morl_baselines.common.evaluation import hypervolume, eval_mo

def train(config=None):
    with wandb.init(config=config):
        config = wandb.config

        env = mo_gym.make("minecart-v0")

        # Normalize weights
        weights = np.array([
            config.scalarization_weight_0,
            config.scalarization_weight_1,
            config.scalarization_weight_2
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
            buffer_size=50000,
        )

        agent.train(total_timesteps=400000, eval_freq=20000)

        _, _, results, _ = eval_mo(agent, env, weights, scalarization)

        pareto_front = np.array(results)

        ref_point = np.array([-1.0, -1.0, -100.0])

        hv = hypervolume(ref_point, pareto_front)
        wandb.log({"hypervolume": hv})

# Sweep Configuration
sweep_config = {
    "method": "bayes",
    "metric": {"name": "hypervolume", "goal": "maximize"},
    "parameters": {
        "learning_rate": {"min": 0.0004826859902260101, "max": 0.01967165138384503},
        "gamma": {"min": 0.475,"max":1.99},
        "scalarization_weight_0": {"min": 0.004242787522149127, "max": 1.851645803918992},
        "scalarization_weight_1": {"min": 0.0255826701683059, "max": 1.9955624610028624},
        "scalarization_weight_2": {"min": 0.36064079723012354, "max": 1.4425631889204942},
    },
}

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="eupg_minecart")
    wandb.agent(sweep_id, function=train, count=50)
