import sys
import mo_gymnasium as mo_gym
import gymnasium as gym
import moviepy
from gymnasium.wrappers import RecordVideo
import stable_baselines3 as sb3
from mo_gymnasium.wrappers import MORecordEpisodeStatistics
import numpy as np
from morl_baselines.multi_policy.envelope.envelope import Envelope

GAMMA = 0.99

env = mo_gym.make("minecart-v0", render_mode="rgb_array")
env = MORecordEpisodeStatistics(env, gamma=GAMMA)
env = RecordVideo(env, "videos/minecart_test", episode_trigger=lambda e: e%50==0)

eval_env = mo_gym.make("minecart-v0", render_mode="rgb_array")
eval_env = MORecordEpisodeStatistics(eval_env, gamma=GAMMA)

env.reset(seed=42)

agent = Envelope(
    env=env,
    net_arch=[256, 256],
    batch_size=128,
    gamma=GAMMA,
    seed=42,
    learning_rate=1e-3,
    initial_epsilon=1.0,
    final_epsilon=0.05,
    epsilon_decay_steps=25000,
    learning_starts=1000,
    gradient_updates=1,
    tau=0.005,
    target_net_update_freq=500,
    buffer_size=100000,
    envelope=True,
    num_sample_w=8,
    per=True,
    per_alpha=0.6,
    initial_homotopy_lambda=1.0,
    final_homotopy_lambda=0.1,
    homotopy_decay_steps=40000
)

# Train the agent
agent.train(
    total_timesteps=200000,
    eval_env=eval_env,
    eval_freq=10000,
    ref_point=np.array([0.0, 0.0, -100.0])
)

# Unwrap the RecordVideo wrapper
stats_env = env.env  # env is RecordVideo → stats_env is MORecordEpisodeStatistics

# Loop through logged episode stats
returns = list(stats_env.return_queue)
lengths = list(stats_env.length_queue)

print("\n--- Episode Details ---")
for i, (ret, length) in enumerate(zip(returns, lengths)):
    ore1, ore2, fuel_cost = ret
    print(f"Episode {i + 1}: Steps = {length}, Ore1 = {ore1:.2f}, Ore2 = {ore2:.2f}, Fuel Cost = {fuel_cost:.2f}")
