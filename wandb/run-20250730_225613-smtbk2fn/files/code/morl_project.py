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

agent = Envelope(
    env=env,
    gamma=GAMMA,
    log=True
)

agent.train(total_timesteps=100000, eval_env=eval_env, ref_point=np.array([0, -50]))