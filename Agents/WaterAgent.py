import random
from gymnasium import spaces
class WaterAgent:
    def __init__(self, pos):
        self.actions = ['move_up', 'move_down', 'move_left', 'move_right', 'idle', 'collect_water', 'water_crops']
        self.action_space = spaces.Discrete(7)
        self.pos = pos
        self.capacity = 3
        self.reward = 0
        self.dist_target = 0

        """
        0: position of agent
        1: number of water units holding
        """
        self.state = [pos, 0]

    def get_position(self):
        return self.state[0]
    
    def update_position(self, new_pos):
        self.state[0] = new_pos

    def collect_water(self):
        self.state[1] = self.capacity

    def holding_water(self):
        return self.state[1] > 0

    def get_water_units(self):
        return self.state[1]
    
    def reduce_water_units(self):
        self.state[1] -= 1

    def get_distance_target(self):
        return self.dist_target
    
    def update_distance_target(self, distance):
        self.dist_target = distance

    def drop_water(self):
        self.state[1] = 0

    def display_state(self):
        print("====Water Agent====")
        print(f"Reward: {self.reward}")
        print(f"Position: {self.state[0]}")
        print(f"Water Units: {self.state[1]}")
        print(f"Distance to target: {self.dist_target}")

