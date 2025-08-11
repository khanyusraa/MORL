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

'''
agent = Envelope(
    env=env,
    batch_size=32, 
    gamma=GAMMA, #discount factor, lower will increase learning but bad for long-term
    seed=42, 
    learning_rate=0.0003, 
    initial_epsilon=1.0, #fully random at first
    final_epsilon=0.2, #minimal exploration probability
    epsilon_decay_steps=25000, #decays over these steps, higher is more prolonged
    learning_starts=1000, #higher prevents learning from biased data
    gradient_updates=1, #lower needs more environment steps for convergence
    tau=1.0, #higher updates target network faster but can be potentially unstable
    buffer_size=100000, #larger = better generalisation
    envelope=True, 
    num_sample_w=8, #higher is better approximation of Pareto Front, try 16 too
    per=True, #"prioritised experience replay" -> focuses on important experiences
    per_alpha=0.6, #closer to one, more prioritization
    initial_homotopy_lambda=1.0, #behaves like linear q-learning
    final_homotopy_lambda=0.1, #behaves like envelope q-learning
    homotopy_decay_steps=40000 
)

agent = Envelope(
    env=env, 
    learning_rate= 0.0003, 
    initial_epsilon= 0.2, 
    epsilon_decay_steps= 10000, 
    batch_size= 32, 
    tau= 1.0, 
    #clip_grand_norm= 1.0, 
    target_net_update_freq= 1000, 
    gamma= GAMMA, 
    #use_envelope= True, 
    num_sample_w= 8, 
    net_arch= [512, 512], 
    per= True, 
    gradient_updates= 1, 
    buffer_size= 100000, 
    initial_homotopy_lambda= 0.95, 
    final_homotopy_lambda= 1.0, 
    homotopy_decay_steps= None, 
    learning_starts= 100, 
    seed= 318)
    
    '''

agent = Envelope(env= env,
                 learning_rate= 0.00024, 
                 initial_epsilon= 0.7293, 
                 epsilon_decay_steps= 95463, 
                 batch_size= 32, 
                 tau= 0.1294, 
                 #clip_grand_norm= 1.2262, 
                 target_net_update_freq= 3022, 
                 gamma= 0.98, 
                 #use_envelope= True, 
                 num_sample_w= 3, 
                 net_arch= [256, 256, 256, 256], 
                 per= True, 
                 gradient_updates= 5, 
                 buffer_size= 1486469, 
                 initial_homotopy_lambda= 0.9021, 
                 final_homotopy_lambda= 0.8728, 
                 homotopy_decay_steps= 51843, 
                 learning_starts= 167, 
                 seed= 1)

# Train the agent
agent.train(
    total_timesteps=200000,
    eval_env=eval_env, #separate env with no noise for evaluation
    eval_freq=10000,
    ref_point=np.array([0.0, 0.0, -100.0]) #worst possible reward returns
)

stats_env = env.env # Unwrap the RecordVideo wrapper

returns = list(stats_env.return_queue)
lengths = list(stats_env.length_queue)

print("\n--- Episode Details ---")
for i, (ret, length) in enumerate(zip(returns, lengths)):
    ore1, ore2, fuel_cost = ret
    print(f"Episode {i + 1}: Steps = {length}, Ore1 = {ore1:.2f}, Ore2 = {ore2:.2f}, Fuel Cost = {fuel_cost:.2f}")
