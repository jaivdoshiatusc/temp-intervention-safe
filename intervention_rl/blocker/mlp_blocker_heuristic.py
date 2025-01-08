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
        lidar = obs[28:44]  # Indices 28 through 43
        vase = obs[44:60]   # Indices 44 through 59

        # Check if any lidar or vase elements are greater than 0.94
        if any(value > 0.94 for value in lidar) or any(value > 0.94 for value in vase):
            return True

        return False
    
    def should_block(self, obs, cost):
        if self.is_catastrophe(cost) or self.is_block_zone(obs):
            return True
        return False

