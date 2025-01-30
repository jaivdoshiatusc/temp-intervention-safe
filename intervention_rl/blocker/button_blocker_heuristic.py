import numpy as np

class ButtonBlockerHeuristic:
    def __init__(self, block_zone=0.92):
        self.block_zone = block_zone
    
    def is_catastrophe(self, cost):   
        if np.any(cost > 0):  # Checks if any element in `cost` is greater than 0
            return True
        return False

    def is_block_zone(self, obs):
        buttons = obs[12:28]  # Indices 12 through 27
        goal = obs[28:44]     # Indices 28 through 43
        hazards = obs[44:60]  # Indices 44 through 59
        # hazards = obs[60:75]   # Indices 60 through 75

        # Check specific ranges in buttons
        if (np.any(buttons[0:4] > self.block_zone) or np.any(buttons[12:16] > self.block_zone)):
            if not np.array_equal(buttons, goal):
                return [-1, 0]

        if np.any(buttons[4:12] > self.block_zone):
            if not np.array_equal(buttons, goal):
                return [1, 0]

        # Check gremlins
        if (np.any(hazards[0:4] > self.block_zone) or np.any(hazards[12:16] > self.block_zone)):
            return [-1, 0]

        if np.any(hazards[4:12] > self.block_zone):
            return [1, 0]

        # # Check hazards
        # if (np.any(hazards[0:4] > self.block_zone) or np.any(hazards[12:16] > self.block_zone)):
        #     return [-1, 0]

        # if np.any(hazards[4:12] > self.block_zone):
        #     return [1, 0]

        # Default return if no conditions are met
        return [2, 2]


    def override_block(self):
        return [-1, 0]

    # def is_block_zone(self, obs):
    #     lidar = obs[28:44]  # Indices 28 through 43
    #     vase = obs[44:60]   # Indices 44 through 59

    #     # Check if any lidar or vase elements are greater than 0.94
    #     if any(value > 0.94 for value in lidar) or any(value > 0.94 for value in vase):
    #         return True

    #     return False
    
    def should_block(self, obs, cost):
        if self.is_block_zone(obs) != [2,2]:
            return self.is_block_zone(obs)
        if self.is_catastrophe(cost):
            return self.override_block()
        return [2, 2]
