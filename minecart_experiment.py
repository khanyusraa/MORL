import glob
import os
import sys
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
from morl_baselines.common.morl_algorithm import MOPolicy
import wandb
import torch
import inspect

ref_point = np.array([-10.0, -10.0, -200.0])
GAMMA = 0.98
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f"Using device: {device}")

class MinecartSafeWrapper(gym.Wrapper):
  def step(self, action):
    obs, reward, terminated, truncated, info = super().step(action)
    reward = np.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
    return obs, reward, terminated, truncated, info

class MinecartStochasticityWrapper(gym.Wrapper):
  def __init__(self, env, level="low"):
    super().__init__(env)
    assert level in ["low", "medium", "high"]
    self.level = level

  def step(self, action):
    obs, reward, terminated, truncated, info = self.env.step(action)

    # Copy arrays to avoid modifying in place
    obs = obs.copy()
    reward = reward.copy()
    info = dict(info) 

    if self.level == "low":
      obs = self.apply_low(obs)
    elif self.level == "medium":
      obs, reward = self.apply_medium(obs, reward)
    elif self.level == "high":
      obs, reward, info = self.apply_high(obs, reward, info)

    return obs, reward, terminated, truncated, info

  def apply_low(self, obs):
    # Add velocity noise -0.02 to "speed" index (index 2)
    obs[2] += np.random.uniform(-0.02, 0.00)
    return obs

  def apply_medium(self, obs, reward):
    obs = self.apply_low(obs)
    # Ore yield variability -10%
    ore_factor = np.random.uniform(0.9, 1.0)
    reward[:2] *= ore_factor  # indices 0 & 1 are ore gains
    return obs, reward

  def apply_high(self, obs, reward, info):
    obs, reward = self.apply_medium(obs, reward)
    # Fuel cost noise +10%
    reward[2] *= np.random.uniform(1.0, 1.1)  # index 2 is fuel consumption
    return obs, reward, info
    
def save_checkpoint(agent, step, algo, stoc):
    # Saves current agent progress due to limited computational resources
    checkpoint_dir = "~/lustre/morl_experiments"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    path = os.path.join(checkpoint_dir, f"mc_{algo}_{stoc}_step{step}.zip")
    
    print("DEBUG: agent has attributes ->", dir(agent))
    print("DEBUG: has 'save'? ", hasattr(agent, "save"))
    print("DEBUG: has 'policy'? ", hasattr(agent, "policy"))
    print("DEBUG: has 'actor'? ", hasattr(agent, "actor"))

    print(f"Attempting to save to: {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    print(f"Agent class: {agent.__class__}")
    print(f"DEBUG: has 'q_net'? {hasattr(agent, 'q_net')}")
    print(f"DEBUG: has 'target_q_net'? {hasattr(agent, 'target_q_net')}")

    checkpoint = {
        "class_name": agent.__class__.__name__,
        "config": agent.get_config() if hasattr(agent, "get_config") else None,
        "step": step,
    }

    if hasattr(agent, "q_net"):
        checkpoint["q_net_state_dict"] = agent.q_net.state_dict()
    if hasattr(agent, "target_q_net"):
        checkpoint["target_q_net_state_dict"] = agent.target_q_net.state_dict()

    torch.save(checkpoint, path)

    if os.path.exists(path):
        print(f"Checkpoint successfully saved at {path}")
    else:
        print(f"Checkpoint save attempted but file not found at {path}")

def get_latest_checkpoint(algo, stoc):
  # Loads the most recent version of the agent to not lose progress during checkpointing
  pattern = os.path.join("/home/ykhan/lustre/morl_experiments", f"mc_{algo}_{stoc}_step*.zip")
  checkpoints = glob.glob(pattern)
  if not checkpoints:
    return None, 0

  def extract_step(path):
    fname = os.path.basename(path)
    try:
      return int(fname.split("step")[-1].split(".")[0])
    except ValueError:
      return -1

  checkpoints = [(cp, extract_step(cp)) for cp in checkpoints]
  checkpoints = [cp for cp in checkpoints if cp[1] >= 0]

  if not checkpoints:
    return None, 0

  latest_cp, latest_step = max(checkpoints, key=lambda x: x[1])
  return latest_cp, latest_step
    
def run_eupg(stochasticity_level, env, eval_env):
  # Initiliases the EUPG agent in the environment and begins training
  print(f"Running EUPG in the Minecart environment with {stochasticity_level} stochasticity")
  weights = np.array([
          0.94604,
          1.36589,
          0.46609
      ], dtype=np.float64)
  weights = weights / weights.sum()

  def scalarization(rewards, weights=None):
    return (rewards * agent.weights).sum()

  agent = EUPG(
            env=env,
            scalarization=scalarization,
            weights=weights,
            learning_rate=0.001,
            gamma=0.98,
            buffer_size=50000,
            seed=9,
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

  hv = hypervolume(np.array([-10.0, -10.0, -200.0]), pareto_front)
  mul = maximum_utility_loss(pareto_front, np.array([-10.0, -10.0, -200.0]), weights)
  eu = expected_utility(pareto_front, weights)

  wandb.log({
      "final_charts/hypervolume": hv,
      "final_charts/max_utility_loss": mul,
      "final_charts/expected_utility": eu,
  })

def run_eql(stochasticity_level, env, eval_env, chunk_size, array_index, total_steps=800000):
  # Initialises the EQL agent and begins training
  print(f"Running EQL in the Minecart environment with {stochasticity_level} stochasticity")
  pareto = env.unwrapped.pareto_front(GAMMA)

  # Finds latest checkpoint
  latest_cp, steps_done = get_latest_checkpoint("eql", stochasticity_level)
  if latest_cp:
    print(f"Loading checkpoint from {latest_cp}")
    checkpoint = torch.load(latest_cp, map_location="cpu")

    valid_args = inspect.signature(Envelope.__init__).parameters.keys()
    clean_config = {k: v for k, v in checkpoint["config"].items() if k in valid_args}

    agent = Envelope(env=env, **clean_config)

    # Load network weights
    agent.q_net.load_state_dict(checkpoint["q_net_state_dict"])
    agent.target_q_net.load_state_dict(checkpoint["target_q_net_state_dict"])
    print(f"Resumed EQL training from {latest_cp}")
  else:
      agent = Envelope(env= env,
                learning_rate= 0.00024, 
                initial_epsilon= 0.7293, 
                epsilon_decay_steps= 155463, 
                final_epsilon=0.05540,
                batch_size= 32, 
                tau= 0.1294, 
                target_net_update_freq= 3022, 
                gamma= 0.98, 
                num_sample_w= 32, 
                net_arch= [256, 256, 256, 256], 
                per= True, 
                per_alpha=0.3036,
                gradient_updates= 2, 
                buffer_size= 1486469, 
                initial_homotopy_lambda= 0.9021, 
                final_homotopy_lambda= 0.8728, 
                homotopy_decay_steps= 51843, 
                learning_starts= 167, 
                max_grad_norm=1.2262,
                seed= 9)
      print(f"New EQL agent created")

  agent.q_net.to(device)
  agent.target_q_net.to(device)
  steps_remaining = total_steps - steps_done
  steps_this_run = min(chunk_size, steps_remaining)

  print(f"Job {array_index}: Training for {steps_this_run} timesteps")
  print(f"{steps_done} timesteps have been completed thus far, target {total_steps} total")

  # Train the agent
  agent.train(
      total_timesteps=800000,
      eval_env=eval_env, 
      eval_freq=10000,
      ref_point=ref_point,
      num_eval_episodes_for_front=30,
      known_pareto_front=pareto
  )

  save_checkpoint(agent, steps_done+steps_this_run, "eql", stochasticity_level)

def prep_environments(stochasticity, algo, eval):
  # Creates the environments
  env = mo_gym.make("minecart-v0", render_mode="rgb_array")
  env = MinecartSafeWrapper(env)
  if (stochasticity != "none"):
    env = MinecartStochasticityWrapper(env, level=stochasticity)  
  env = MORecordEpisodeStatistics(env, gamma=GAMMA)
  if (not eval):
    if (algo == "eupg"):
      env = RecordVideo(env, "videos/minecart_eupg", episode_trigger=lambda e: e % 500 == 0)
    elif (algo == "eql"):
      env = RecordVideo(env, "videos/minecart_eql", episode_trigger=lambda e: e % 500 == 0)
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
  parser.add_argument(
      "chunk_size",
      type=int,
      help="Number of timesteps per chunk"
  )
  parser.add_argument(
      "array_index",
      type=int,
      help="The job number of the current run"
  )
  parser.add_argument(
      "total_steps",
      type=int,
      help="The total number of timesteps to run for"
  )

  args = parser.parse_args()

  env = prep_environments(args.stochasticity, args.algorithm, False)
  eval_env = prep_environments(args.stochasticity, args.algorithm, True)
  
  if args.algorithm == "eupg":
    run_eupg(args.stochasticity, env, eval_env)
  elif args.algorithm == "eql":
    run_eql(args.stochasticity, env, eval_env, args.chunk_size, args.array_index, args.total_steps)
  else:
    print("Unknown algorithm")

if __name__ == "__main__":
  main()