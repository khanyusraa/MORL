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
env = RecordVideo(env, "videos/minecart_test", episode_trigger=lambda e: e%100==0)

eval_env = mo_gym.make("minecart-v0", render_mode="rgb_array")
eval_env = MORecordEpisodeStatistics(eval_env, gamma=GAMMA)

env.reset(seed=42)

agent = Envelope(
    env=env,
    net_arch=[256, 256], #size of hidden layers
    batch_size=128,
    gamma= GAMMA,
    seed=42
)

agent.train(
    total_timesteps=50000,
    eval_env=eval_env,
    eval_freq=10_000,
    ref_point=np.array([0.0, 0.0, -50.0])
)