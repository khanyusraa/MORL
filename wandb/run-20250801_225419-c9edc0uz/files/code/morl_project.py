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
env = RecordVideo(env, "videos/minecart_test", episode_trigger=lambda e: e%10==0)

eval_env = mo_gym.make("minecart-v0", render_mode="rgb_array")
eval_env = MORecordEpisodeStatistics(eval_env, gamma=GAMMA)

env.reset(seed=42)

agent = Envelope(
    env=env,
    learning_rate=1e-3, 
    initial_epsilon=1.0,
    final_epsilon=0.05,
    epsilon_decay_steps=50_000,
    tau=0.005, 
    target_net_update_freq=500,
    buffer_size=100000,
    net_arch=[256, 256], #size of hidden layers
    batch_size=128,
    gamma= GAMMA,
    max_grad_norm=10.0, #gradient clipping
    num_sample_w=8, #no of weight vectors to sample for envelope target
    initial_homotopy_lambda=1.0, 
    final_homotopy_lambda=0.1,
    homotopy_decay_steps=150_000, 
    seed=42
)

agent.train(
    total_timesteps=200_000,
    eval_env=eval_env,
    eval_freq=10_000,
    ref_point=np.array([0.0, 0.0, -50.0])
)