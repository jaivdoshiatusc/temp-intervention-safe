import numpy as np

class MLPBlockerHeuristic:
    def __init__(self, clearance=0.5):
        self.clearance = clearance

    def obs_to_pixels(self, obs):
        return 0, 0
        # Scale factors as per the environment's specification
        scale_factors = {'x': 10, 'y': 6.666}

        # Extract position from the observation and apply scale factors
        x_pos_scaled = obs[0] * scale_factors['x']
        y_pos_scaled = obs[1] * scale_factors['y']

        # Convert x position: Center of the screen is 0 in the environment's coordinates
        x_pixel = (x_pos_scaled + self.VIEWPORT_WIDTH / self.SCALE / 2) * self.SCALE

        # Convert y position: Invert y axis for rendering (screen's 0,0 is at top-left)
        y_pixel = self.VIEWPORT_HEIGHT - (y_pos_scaled + self.VIEWPORT_HEIGHT / self.SCALE / 4) * self.SCALE

        return x_pixel, y_pixel
    
    def is_catastrophe(self, obs):   
        return False     
        x, y, vel_x, vel_y, angle, angular_velocity, leg_1_contact, leg_2_contact = obs
        adj_x, adj_y = self.obs_to_pixels(obs)

        # Catastrophe conditions
        in_box = (240 <= adj_x <= 360) and (120 <= adj_y <= 240)  # Render coordinate bounds
        angle_condition = -0.25 <= angle <= 0.25
        velocity_condition = vel_y >= -0.50

        return in_box and velocity_condition and angle_condition

    def is_block_zone(self, obs):
        return False
        x, y, vel_x, vel_y, angle, angular_velocity, leg_1_contact, leg_2_contact = obs
        adj_x, adj_y = self.obs_to_pixels(obs)

        # Catastrophe conditions
        in_box = (240 <= adj_x <= 360) and (110 <= adj_y <= 240)  # Render coordinate bounds
        angle_condition = -0.30 <= angle <= 0.30
        velocity_condition = vel_y >= -0.55

        return in_box and velocity_condition and angle_condition
    
    def should_block(self, obs, action):
        return False
        if self.is_catastrophe(obs) or self.is_block_zone(obs):
            return True
        return False
    