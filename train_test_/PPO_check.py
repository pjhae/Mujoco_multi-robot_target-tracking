import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env import VecFrameStack


date = "0709"
trial = "A"
steps = "3440000"



## Make gym environment ##



env = make_vec_env("Hexy-v5", n_envs=1)
# env = gym.make("Hexy-v6")
# env = VecFrameStack(env, n_stack=3,  channels_order = "first")
# env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)



## Path ##

save_path='./save_model_'+date+'/'+trial+'/'

model = PPO.load(save_path+"Hexy_model_"+date+trial+"_"+steps+"_steps")

obs = env.reset()

## Rendering ##




while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()


