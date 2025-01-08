from gymnasium import Env
import safety_gymnasium
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

class SafetyGymnasium2GymnasiumVecEnv(Env):
    """
    Wrapper to convert Safety-Gymnasium environments to the Gymnasium API.
    """
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, seed=None):
        return self.env.reset(seed=seed)

    def step(self, actions):
        obs, rewards, costs, terminated, truncated, infos = self.env.step(actions)
        infos["cost"] = costs  # Add cost to infos
        return obs, rewards, terminated, truncated, infos
    
    def render(self):
        return self.env.render()

def make_safe_vec_env(env_id, n_envs=1, seed=31):
    """
    Factory function to create a vectorized Safety-Gymnasium environment.
    """
    # Use SB3's make_vec_env with the custom wrapper
    return make_vec_env(
        env_id=env_id,
        n_envs=n_envs,
        seed=seed,
        wrapper_class=SafetyGymnasium2GymnasiumVecEnv,
    )

def make_safe_env(env_id, n_stack=4, seed=31):
    """
    Factory function to create an individual Safety-Gymnasium environment.
    """
    env = safety_gymnasium.make(env_id)
    env = SafetyGymnasium2GymnasiumVecEnv(env)
    return env

def make_safe_eval_env(env_id, n_stack=4, seed=3100):
    """
    Factory function to create an individual Safety-Gymnasium environment for evaluation.
    """
    env = safety_gymnasium.make(env_id, render_mode="rgb_array", camera_id=1)
    return env

