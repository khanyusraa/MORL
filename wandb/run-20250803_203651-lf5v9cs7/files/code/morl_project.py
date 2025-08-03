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
    
    # 🔧 Essential learning params
    learning_rate=1e-3,
    initial_epsilon=1.0,
    final_epsilon=0.05,
    epsilon_decay_steps=2000,
    learning_starts=100,
    gradient_updates=1,
    tau=0.005,
    target_net_update_freq=500,
    buffer_size=100_000,
    
    # ✅ Enable envelope-specific functionality
    envelope=True,
    num_sample_w=8,  # try 8 or 16

    # Optional: prioritized replay
    per=True,
    per_alpha=0.6,

    # Optional: homotopy (weight tradeoff blending)
    initial_homotopy_lambda=1.0,
    final_homotopy_lambda=0.1,
    homotopy_decay_steps=500
)

agent.train(
    total_timesteps=10000,
    eval_env=eval_env,
    eval_freq=5000,
    ref_point=np.array([0.0, 0.0, -1.0])
)

returns = list(env.return_queue)

print("\n--- Episode Reward Summary ---")
for idx, ret in enumerate(returns):
    print(f"Episode {idx + 1}: Reward Vector = {ret}")