import numpy as np

class MLPBlockerHeuristic:
    def __init__(self, block_zone=0.92):
        self.block_zone = block_zone
    
    def is_catastrophe(self, cost):   
        if np.any(cost > 0):  # Checks if any element in `cost` is greater than 0
            return True
        return False

    def is_block_zone(self, obs):
        hazards = obs[40:56]  # Indices 28 through 43
        # vase = obs[44:60]   # Indices 44 through 59

        if any(hazards[i] > 0.86 for i in range(2, 6)) and any(hazards[i] > 0.86 for i in range(10, 14)):
            return [2, 2]

        if any(hazards[i] > 0.86 for i in range(2, 6)):
            return [1, 1]
        
        if any(hazards[i] > 0.86 for i in range(10, 14)):
            return [-1, -1]

        # # Check lidar elements [0,3] or [12,15]
        # if any(hazards[i] > 0.86 for i in range(0, 3)) or any(hazards[i] > 0.86 for i in range(13, 16)):
        #     return [-1, -1]

        # # Check lidar elements [4,11]
        # if any(hazards[i] > 0.86 for i in range(6, 11)):
        #     return [1, 1]

        # # Check lidar elements [0,3] or [12,15]
        # if any(hazards[i] > self.block_zone for i in range(0, 4)) or any(hazards[i] > self.block_zone for i in range(12, 16)):
        #     return [-1, 0]

        # # Check lidar elements [4,11]
        # if any(hazards[i] > self.block_zone for i in range(4, 12)):
        #     return [1, 0]
        
        # buttons = obs[24:40]
        # goal = obs[40:56]
        # hazards = obs[56:72]
        
        # # Check specific ranges in buttons
        # if (np.any(buttons[0:3] > 0.86) or np.any(buttons[13:16] > 0.86)):
        #     if not np.array_equal(buttons, goal):
        #         return [1, 1]
            
        # if np.any(buttons[6:11] > 0.86):
        #     if not np.array_equal(buttons, goal):
        #         return [-1, -1]

        # # Check lidar elements [0,3] or [12,15]
        # if any(hazards[i] > 0.86 for i in range(0, 3)) or any(hazards[i] > 0.86 for i in range(13, 16)):
        #     return [-1, -1]

        # # Check lidar elements [4,11]
        # if any(hazards[i] > 0.86 for i in range(6, 11)):
        #     return [1, 1]

        # # Check vase elements [0,3] or [12,15]
        # if any(vase[i] > self.block_zone for i in range(0, 4)) or any(vase[i] > self.block_zone for i in range(12, 16)):
        #     return [-1, 0]

        # # Check vase elements [4,11]
        # if any(vase[i] > self.block_zone for i in range(4, 12)):
        #     return [1, 0]

        # Default return if no conditions are met
        return [2, 2]
    
    def override_block(self):
        return [1, 1]

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
