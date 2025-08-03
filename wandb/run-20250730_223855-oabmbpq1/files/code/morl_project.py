import sys
import mo_gymnasium as mo_gym
import gymnasium as gym
import moviepy
from gymnasium.wrappers import RecordVideo
import stable_baselines3 as sb3
from mo_gymnasium.wrappers import MORecordEpisodeStatistics
import numpy as np
from morl_baselines.multi_policy.pareto_q_learning.pql import PQL

GAMMA = 0.99

env = mo_gym.make("deep-sea-treasure-v0")
env = MORecordEpisodeStatistics(env, gamma=GAMMA)

eval_env = mo_gym.make("deep-sea-treasure-v0")

agent = PQL(
    env=env,
    ref_point=np.array([0, -50]),
    gamma=GAMMA,
    log=True
)

agent.train(total_timesteps=100000, eval_env=eval_env, ref_point=np.array([0, -50]))