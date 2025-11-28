import gymnasium as gym
from gymnasium import RewardWrapper


class CustomRewardWrapper(RewardWrapper):

    def __init__(self, env: gym.Env, cfg: dict):
        super().__init__(env)
        self.cfg = cfg

        # General scaling / penalties
        self.reward_scale = cfg.get("reward_scale", 1.0)
        self.step_penalty = cfg.get("step_penalty", 0.0)

        # MountainCarContinuous-specific shaping
        self.position_weight = cfg.get("position_weight", 10.0)
        self.velocity_weight = cfg.get("velocity_weight", 0.5)
        self.goal_position = cfg.get("goal_position", 0.45)

    def reward(self, reward: float) -> float:
        """
        reward: default environment reward.
        For MountainCarContinuous-v0, obs = [position, velocity].
        We shape the reward to:
        - encourage moving towards the goal position
        - encourage higher speed (absolute velocity)
        - optionally penalize each step a bit
        """
        # Start from scaled default reward
        shaped = self.reward_scale * reward

        state = getattr(self.env.unwrapped, "state", None)
        if state is not None:
            position, velocity = state

            shaped += self.position_weight * (position - self.goal_position)

            shaped += self.velocity_weight * abs(velocity)

        shaped -= self.step_penalty

        return shaped
