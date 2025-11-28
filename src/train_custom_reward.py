from custom_env_wrapper import CustomRewardWrapper
from utils import load_config
import gymnasium as gym
from stable_baselines3 import PPO, DQN, A2C, SAC
from stable_baselines3.common.callbacks import CheckpointCallback
import os


ALGO_MAP = {
    "DQN": DQN,
    "PPO": PPO,
    "A2C": A2C,
    "SAC": SAC,
}


def train_custom(cfg_path="config/config_custom.yaml"):
    cfg = load_config(cfg_path)

    env_id = cfg["environment"]
    algo_name = cfg["algorithm"]
    timesteps = cfg["timesteps"]
    checkpoint_freq = int(cfg["checkpoint_freq"])
    num_trials = cfg["num_trials"]

    assert algo_name in ALGO_MAP, f"Unknown algorithm: {algo_name}"
    AlgoClass = ALGO_MAP[algo_name]

    os.makedirs("results/custom_reward", exist_ok=True)

    for trial in range(num_trials):
        base_env = gym.make(env_id)
        env = CustomRewardWrapper(base_env, cfg)

        log_dir = f"logs/custom_reward/{algo_name}_trial{trial}"
        checkpoint_dir = os.path.join(log_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        model = AlgoClass(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=log_dir,
        )

        cb = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=checkpoint_dir,
            name_prefix=f"{algo_name}_custom",
        )

        model.learn(total_timesteps=timesteps, callback=cb)

        model_save_path = f"results/custom_reward/{algo_name}_trial{trial}"
        model.save(model_save_path)
        env.close()


if __name__ == '__main__':
    train_custom()
