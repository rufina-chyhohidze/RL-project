from utils import load_config
import gymnasium as gym
from stable_baselines3 import PPO, DQN, A2C, SAC
from stable_baselines3.common.callbacks import CheckpointCallback
import os
from custom_env_wrapper import CustomRewardWrapper



ALGO_MAP = {
    "DQN": DQN,
    "PPO": PPO,
    "A2C": A2C,
    "SAC": SAC,
}


def train_extension(cfg_path="config/config_extension.yaml"):
    cfg = load_config(cfg_path)

    env_id = cfg["environment"]
    algo_name = cfg["algorithm"]
    timesteps = cfg["timesteps"]
    checkpoint_freq = cfg["checkpoint_freq"]

    hyperparam_name = cfg["hyperparam_name"]
    hyperparam_values = cfg["hyperparam_values"]

    assert algo_name in ALGO_MAP, f"Unknown algorithm: {algo_name}"
    AlgoClass = ALGO_MAP[algo_name]

    os.makedirs("results/extension", exist_ok=True)

    for idx, val in enumerate(hyperparam_values):
        if hyperparam_name in ["n_steps", "batch_size", "n_epochs"]:
            val = int(val)
        else:
            val = float(val)

        algo_kwargs = {hyperparam_name: val}

        base_env = gym.make(env_id)
        env = CustomRewardWrapper(base_env, cfg)

        log_dir = f"logs/extension/{algo_name}_{hyperparam_name}{idx}"
        checkpoint_dir = os.path.join(log_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)



        model = AlgoClass(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=log_dir,
            **algo_kwargs,
        )

        cb = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=checkpoint_dir,
            name_prefix=f"{algo_name}_ext_{hyperparam_name}{idx}",
        )

        model.learn(total_timesteps=timesteps, callback=cb)

        model_save_path = f"results/extension/{algo_name}_{hyperparam_name}{idx}"
        model.save(model_save_path)
        env.close()


if __name__ == '__main__':
    train_extension()
