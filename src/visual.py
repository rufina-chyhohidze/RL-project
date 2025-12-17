import gymnasium as gym
from stable_baselines3 import PPO

model = PPO.load("results/extension/PPO_clip_range1")

env = gym.make("MountainCarContinuous-v0", render_mode="human")

obs, info = env.reset()
done = False
truncated = False

while not (done or truncated):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)

env.close()
