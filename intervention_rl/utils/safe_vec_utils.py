# from gymnasium import Env
# import safety_gymnasium
# from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
# from stable_baselines3.common.env_util import make_vec_env


# class SafetyGymnasium2GymnasiumVecEnv(Env):
#     """
#     Wrapper to convert Safety-Gymnasium environments to the Gymnasium API.
#     """
#     def __init__(self, env, seed=31):
#         super().__init__()
#         self.env = env
#         self.seed = seed
#         self.observation_space = env.observation_space
#         self.action_space = env.action_space

#     def reset(self, seed=None):
#         if seed is not None:
#             self.seed = seed
#         return self.env.reset(seed=self.seed)

#     def step(self, actions):
#         # Safety-Gymnasium returns an extra value (`costs`), which we omit to conform to Gymnasium API.
#         obs, rewards, costs, terminated, truncated, infos = self.env.step(actions)
#         return obs, rewards, terminated, truncated, infos

#     def render(self, mode="rgb_array"):
#         return self.env.render(mode=mode)

#     def close(self):
#         self.env.close()


# class CachedSafetyEnv:
#     """
#     Cache for XML structures in Safety-Gymnasium to avoid redundant initialization.
#     """
#     _xml_cache = {}

#     @staticmethod
#     def create_env(env_id):
#         if env_id not in CachedSafetyEnv._xml_cache:
#             # Create and cache the MuJoCo model
#             env = safety_gymnasium.make(env_id)
#             CachedSafetyEnv._xml_cache[env_id] = env  # Cache the environment instance
#         else:
#             # Use the cached instance
#             env = CachedSafetyEnv._xml_cache[env_id]
#         return env


# def make_safe_vec_env(env_id, n_envs=1, seed=31, n_stack=4):
#     """
#     Factory function to create a vectorized Safety-Gymnasium environment with framestacking.
#     """
#     # Create the vectorized environment
#     def _make_env_fn():
#         # Each environment needs its own seed and wrapper
#         env = CachedSafetyEnv.create_env(env_id)
#         return SafetyGymnasium2GymnasiumVecEnv(env, seed=seed)

#     # Use DummyVecEnv to handle vectorized environments
#     vec_env = DummyVecEnv([_make_env_fn for _ in range(n_envs)])
#     # Add framestacking
#     vec_env = VecFrameStack(vec_env, n_stack=n_stack)
#     return vec_env


# def make_safe_env(env_id, seed=31, n_stack=4):
#     """
#     Factory function to create an individual Safety-Gymnasium environment with framestacking.
#     """
#     # Use the cached environment if available
#     env = CachedSafetyEnv.create_env(env_id)
#     # Wrap it in the custom wrapper
#     env = SafetyGymnasium2GymnasiumVecEnv(env, seed=seed)
#     # Wrap framestacking (using DummyVecEnv for compatibility)
#     env = VecFrameStack(DummyVecEnv([lambda: env]), n_stack=n_stack)
#     return env

from gymnasium import Env
import safety_gymnasium
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

class SafetyGymnasium2GymnasiumVecEnv(Env):
    """
    Wrapper to convert Safety-Gymnasium environments to the Gymnasium API.
    """
    def __init__(self, env, seed=31):
        super().__init__()
        self.env = env
        self.seed = seed
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, seed=None):
        if seed is not None:
            self.seed = seed
        return self.env.reset(seed=self.seed)

    def step(self, actions):
        obs, rewards, costs, terminated, truncated, infos = self.env.step(actions)
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
    env = SafetyGymnasium2GymnasiumVecEnv(env, seed=seed)
    return env

def make_safe_eval_env(env_id, n_stack=4, seed=3100):
    """
    Factory function to create an individual Safety-Gymnasium environment for evaluation.
    """
    env = safety_gymnasium.make(env_id, render_mode="rgb_array", camera_id=1)
    return env

