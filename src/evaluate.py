import gymnasium as gym
from stable_baselines3 import PPO, DQN, A2C, SAC
from pathlib import Path


ALGO_MAP = {
    "dqn": DQN,
    "ppo": PPO,
    "a2c": A2C,
    "sac": SAC,
}


def _infer_algo_from_path(model_path: str):
    name = Path(model_path).name.lower()
    for key, algo in ALGO_MAP.items():
        if key in name:
            return algo
    raise ValueError(
        f"Could not infer algorithm from model path '{model_path}'. "
        f"Make sure the filename contains one of {list(ALGO_MAP.keys())}."
    )


def evaluate(model_path, env_id, episodes=10):
    AlgoClass = _infer_algo_from_path(model_path)
    model = AlgoClass.load(model_path)

    env = gym.make(env_id)

    returns = []

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        ep_return = 0.0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_return += reward

        returns.append(ep_return)
        print(f"Episode {ep}: return = {ep_return}")

    env.close()

    avg_return = sum(returns) / len(returns)
    print(f"\nAverage return over {episodes} episodes: {avg_return}")
    return returns, avg_return
