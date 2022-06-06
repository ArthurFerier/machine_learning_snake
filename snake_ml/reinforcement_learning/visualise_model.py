import gym
from stable_baselines3 import PPO

from snake_ml.reinforcement_learning.model import SnekEnv

models_dir = "models/get_to_food_fast_ppo"

env = SnekEnv()
env.reset()

model_path = f"{models_dir}/10350000.0"
model = PPO.load(model_path, env=env)

episodes = 1

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        print(rewards)