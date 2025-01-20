# Dict('accelerometer': Box(-inf, inf, (3,), float64),
#     'velocimeter': Box(-inf, inf, (3,), float64),
#     'gyro': Box(-inf, inf, (3,), float64),
#     'magnetometer': Box(-inf, inf, (3,), float64),
#     'circle_lidar': Box(0.0, 1.0, (16,), float64))

import numpy as np
import cv2
import imageio
import safety_gymnasium
from gymnasium import Env

import numpy as np

class MLPBlockerHeuristic:
    def __init__(self):
        pass
    
    def is_catastrophe(self, cost):   
        if cost > 0:
            return True
        return False

    # def is_block_zone(self, obs):
    #     lidar = obs[28:44]  # Indices 28 through 43
    #     vase = obs[44:60]   # Indices 44 through 59

    #     # Check lidar elements [0,3] or [12,15]
    #     if any(lidar[i] > 0.92 for i in range(0, 4)) or any(lidar[i] > 0.92 for i in range(12, 16)):
    #         return [-1, 0]

    #     # Check lidar elements [4,11]
    #     if any(lidar[i] > 0.92 for i in range(4, 12)):
    #         return [1, 0]

    #     # Check vase elements [0,3] or [12,15]
    #     if any(vase[i] > 0.92 for i in range(0, 4)) or any(vase[i] > 0.92 for i in range(12, 16)):
    #         return [-1, 0]

    #     # Check vase elements [4,11]
    #     if any(vase[i] > 0.92 for i in range(4, 12)):
    #         return [1, 0]

    #     # Default return if no conditions are met
    #     return [0, 0]


    def is_block_zone(self, obs):
        gremlins = obs[44:60]  # Indices 44 through 59
        hazards = obs[60:75]   # Indices 60 through 75

        # Check if any lidar or vase elements are greater than 0.94
        if any(value > 0.94 for value in gremlins) or any(value > 0.94 for value in hazards):
            return True

        return False
    
    def should_block(self, obs, cost):
        if self.is_catastrophe(cost) or self.is_block_zone(obs):
            return True
        return False

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

def make_safe_env(env_id):
    """
    Factory function to create an individual Safety-Gymnasium environment.
    """
    env = safety_gymnasium.make(env_id, render_mode="rgb_array", camera_id=1)
    env = SafetyGymnasium2GymnasiumVecEnv(env)
    return env

def main():
    # Create the environment
    env_id = 'SafetyPointButton1-v0'
    env = make_safe_env(env_id)

    obs = env.reset(seed=42)
    import ipdb; ipdb.set_trace()
    total_steps = 1000
    frames = []

    for i in range(total_steps):
        act = [0, 1]
        obs, reward, terminated, truncated, info = env.step(act)

        # Extract cost from infos
        cost = info.get("cost", 0)

        if cost > 0:
            print(f"Step {i}: Cost detected - {cost}")

        # Render the current frame
        render_obs = env.render()

        # Convert to BGR for OpenCV
        img = cv2.cvtColor(render_obs, cv2.COLOR_RGB2BGR)
        render_obs = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Extend the frame slightly
        frame_height, frame_width, _ = render_obs.shape
        extended_frame = np.zeros((frame_height + 120, frame_width, 3), dtype=np.uint8)
        extended_frame[:frame_height, :, :] = render_obs

        # Display observation values with wrapping every 6 values
        obs_values = obs[28:45]
        text_x = 10
        text_y = frame_height + 15
        for idx, value in enumerate(obs_values):
            text = f"{idx+28}:{value:.2f}"
            cv2.putText(extended_frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (255, 255, 255), 1)

            # Move to the next position
            if (idx + 1) % 6 == 0:  # New line after 6 values
                text_y += 10
                text_x = 10
            else:
                text_x += 40  # Horizontal spacing for the next value

        # Display cost at the bottom
        cost_text = f"Cost: {cost}"
        cv2.putText(extended_frame, cost_text, (10, frame_height + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        frames.append(extended_frame)

        if terminated or truncated:
            obs, info = env.reset()

    # Save the frames as a GIF with 30 fps
    imageio.mimsave('output_1.gif', frames, fps=30)

if __name__ == '__main__':
    main()
