import glob
import os
import mo_gymnasium as mo_gym
import gymnasium as gym
import moviepy
from gymnasium.wrappers import RecordVideo
import stable_baselines3 as sb3
from mo_gymnasium.wrappers import MORecordEpisodeStatistics
import numpy as np
from morl_baselines.single_policy.esr.eupg import EUPG
from morl_baselines.multi_policy.envelope.envelope import Envelope
import argparse
from morl_baselines.common.evaluation import eval_mo, hypervolume, maximum_utility_loss, expected_utility
import wandb

ref_point = np.array([-10.0, -10.0])
GAMMA = 0.98

class DeepSeaTreasureStochasticityWrapper(gym.Wrapper):
    def __init__(self, env, level="low"):
        super().__init__(env)
        assert level in ["low", "medium", "high"]
        self.level = level
        self.step_count = 0

    def step(self, action):
        self.step_count += 1
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Copy to avoid modifying in place
        obs = obs.copy()
        info = dict(info)

        if self.level == "low":
            obs = self.apply_low(obs, action)
        elif self.level == "medium":
            obs = self.apply_medium(obs, action)
        elif self.level == "high":
            obs, info = self.apply_high(obs, action, info)

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.step_count = 0
        obs, info = self.env.reset(**kwargs)

        if self.level == "high":
            obs, info = self.apply_reset_high(obs, info)

        return obs, info

    def apply_low(self, obs, action):
        # Movement slip: with 10% probability, replace action
        if np.random.rand() < 0.1:
            alternatives = [a for a in range(self.action_space.n) if a != action]
            slipped_action = np.random.choice(alternatives)
            obs, _, _, _, _ = self.env.step(slipped_action)
        return obs

    def apply_medium(self, obs, action):
        obs = self.apply_low(obs, action)

        # Every 5 steps, force down move
        if self.step_count % 5 == 0:
            drift_action = 2 
            obs, _, _, _, _ = self.env.step(drift_action)
        return obs

    def apply_high(self, obs, action, info):
        obs = self.apply_medium(obs, action)
        return obs, info

    def apply_reset_high(self, obs, info):
        # Treasure placement variability
        if hasattr(self.unwrapped, "treasures"):
            shift = 2
            self.unwrapped.treasures = {
                k: (pos[0] + np.random.randint(-shift, shift+1),
                    pos[1] + np.random.randint(-shift, shift+1))
                for k, pos in self.unwrapped.treasures.items()
            }

        # Random obstacles
        if hasattr(self.unwrapped, "grid"):
            grid = self.unwrapped.grid.copy()
            num_obstacles = np.random.randint(1, 4)
            for _ in range(num_obstacles):
                x, y = np.random.randint(0, grid.shape[0]), np.random.randint(0, grid.shape[1])
                grid[x, y] = 1
            self.unwrapped.grid = grid

        return obs, info
    
def save_checkpoint(agent, step, algo, stoc):
    # Saves current agent progress due to limited computational resources
    os.makedirs("checkpoints", exist_ok=True)
    path = f"checkpoints/dst_{algo}_{stoc}_step{step}.zip"
    agent.save(path)
    print(f"Saved checkpoint at {step} timesteps to {path}")

def get_latest_checkpoint(algo, stoc):
    # Loads the most recent version of the agent to not lose progress during checkpointing
    cp_dir = "checkpoints"
    pattern = os.path.join(cp_dir, f"dst_{algo}_{stoc}_step*.zip")
    files = glob.glob(pattern)
    
    if not files:
        return None

    files.sort(key=lambda f: int(f.split("step")[-1].split(".zip")[0]))
    return files[-1]
    
def run_eupg(stochasticity_level, env, eval_env):
    # Initiliases the EUPG agent in the environment and begins training
    print(f"Running EUPG in the Deep Sea Treasure environment with {stochasticity_level} stochasticity")
    weights = np.array([
            0.10664248760800946,
            0.9910935761859672
        ], dtype=np.float32)
    weights = weights / weights.sum()

    def scalarization(rewards, weights=None):
        return (rewards * agent.weights).sum()

    agent = EUPG(
                env=env,
                scalarization=scalarization,
                weights=weights,
                learning_rate=0.00019651082554375255,
                gamma= 0.9594615907309146,
                buffer_size=5000,
                seed=3,
                log=True,
                log_every=20000
            )

    # Train the agent
    agent.train(
                total_timesteps=800000,
                eval_env=eval_env,
                eval_freq=20000
            )
    
    # Metric Logging
    
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

def run_eql(stochasticity_level, env, eval_env):
    # Initialises the EQL agent and begins training
    print(f"Running EQL in the Deep Sea Treasure environment with {stochasticity_level} stochasticity")
    pareto = env.unwrapped.pareto_front(GAMMA)

    latest_cp = get_latest_checkpoint("eql", stochasticity_level)
    if latest_cp:
        agent = Envelope.load(latest_cp, env=env)
        print(f"Resumed EQL Training from {latest_cp}")
    else:
        agent = Envelope(env= env,
                    learning_rate= 0.00023111672830355843, 
                    initial_epsilon= 0.6160551074666885, 
                    epsilon_decay_steps= 20000, 
                    final_epsilon=0.05593050447095275,
                    batch_size= 32, 
                    tau= 0.034136826173468825, 
                    target_net_update_freq= 100, 
                    gamma= 0.9922430751489348, 
                    num_sample_w= 4, 
                    net_arch= [256, 256, 256, 256], 
                    per= True, 
                    per_alpha= 0.8322490017603045,
                    gradient_updates= 4, 
                    buffer_size= 5000, 
                    initial_homotopy_lambda= 0.7150436175248289, 
                    final_homotopy_lambda= 0.832365561817894, 
                    homotopy_decay_steps= 1000, 
                    learning_starts= 1000, 
                    max_grad_norm=5,
                    seed= 0)
        print(f"New EQL agent created")

    total_timesteps = 800000
    checkpoint_interval = 200000
    steps_done = 0

    while steps_done < total_timesteps:
        next_chunk = min(checkpoint_interval, total_timesteps - steps_done)
        agent.train(
            total_timesteps=800000,
            eval_env=eval_env, 
            eval_freq=10000,
            ref_point=ref_point,
            num_eval_episodes_for_front= 30,
            num_eval_weights_for_eval=50,
            num_eval_weights_for_front=100,
            known_pareto_front=pareto,
            reset_learning_starts=False,
            reset_num_timesteps=True
        )
        steps_done += next_chunk
        save_checkpoint(agent, steps_done, "eql", stochasticity_level)

def prep_environments(stochasticity, algo, eval):
    # Creates the environments
    env = mo_gym.make("deep-sea-treasure-v0", render_mode="rgb_array")
    if (stochasticity != "none"):
        env = DeepSeaTreasureStochasticityWrapper(env, level=stochasticity)  
    env = MORecordEpisodeStatistics(env, gamma=GAMMA)
    if (not eval):
        if (algo == "eupg"):
            env = RecordVideo(env, "videos/dst_eupg", episode_trigger=lambda e: e % 500 == 0)
        elif (algo == "eql"):
            env = RecordVideo(env, "videos/dst_eql", episode_trigger=lambda e: e % 500 == 0)
    env.reset(seed=9)

    return env

def main():
    print('Starting experiment...')
    parser = argparse.ArgumentParser(description="Run MORL experiments")
    parser.add_argument(
        "stochasticity",
        choices=["none", "low", "medium", "high"],
        help="Set the stochasticity level"
    )
    parser.add_argument(
        "algorithm",
        choices=["eupg", "eql"],
        help="Choose which algorithm to run"
    )

    args = parser.parse_args()

    env = prep_environments(args.stochasticity, args.algorithm, False)
    eval_env = prep_environments(args.stochasticity, args.algorithm, True)
    
    if args.algorithm == "eupg":
        run_eupg(args.stochasticity, env, eval_env)
    elif args.algorithm == "eql":
        run_eql(args.stochasticity, env, eval_env)
    else:
        print("Unknown algorithm")

if __name__ == "__main__":
    main()