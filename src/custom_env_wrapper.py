import gymnasium as gym
from gymnasium import RewardWrapper


class CustomRewardWrapper(RewardWrapper):

    def __init__(self, env: gym.Env, cfg: dict):
        super().__init__(env)

        self.step_penalty = cfg.get("step_penalty", 0.0005)

        self.pos_reward_weight = cfg.get("pos_reward_weight", 1.0)
        self.vel_reward_weight = cfg.get("vel_reward_weight", 0.5)
        self.goal_position = cfg.get("goal_position", 0.45)

    def reward(self, reward: float) -> float:
        shaped = reward

        state = getattr(self.env.unwrapped, "state", None)
        if state is not None:
            position, velocity = state

            shaped += self.pos_reward_weight * max(0, position - (-0.5))

            shaped += self.vel_reward_weight * abs(velocity)

            if position > 0.3:
                shaped += 2.0  #bonus

        shaped -= self.step_penalty

        return shaped
