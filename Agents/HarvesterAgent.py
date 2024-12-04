import random
from gymnasium import spaces
from constants import CropTypes

class HarvesterAgent:
    def __init__(self, pos):
        # self.actions = ['move_up', 'move_down', 'move_left', 'move_right', 'face_up', 'face_down', 'face_left', 'face_right', 'idle', 'collect_seeds', 'plant_seed', 'drop_seeds']
        self.actions = ['move_up', 'move_down', 'move_left', 'move_right', 'idle', 'harvest_crops', 'drop_crops']
        self.action_space = spaces.Discrete(len(self.actions))
        self.pos = pos
        self.crop_units = 0
        self.yield_value = 0
        self.dist_target = 0
        self.reward = 0

        """
        0: position of agent
        1: type of crop carrying
        2: yield value of crops
        """
        self.state = [pos, CropTypes.EMPTY, 0]


    def get_position(self):
        return self.state[0]
    
    def get_distance_target(self):
        return self.dist_target
    
    def update_distance_target(self, distance):
        self.dist_target = distance
    
    def update_position(self, new_pos):
        self.state[0] = new_pos

    def update_crops(self, crop_type, crop_value):
        self.state[1] = crop_type
        self.state[2] = crop_value
    
    def get_crops_type(self):
        return self.state[1]
    
    def get_crops_value(self):
        return self.state[2]

    def is_holding_crops(self):
        return self.state[1] > 0

    def drop_crops(self):
        self.state[1] = 0
        self.state[2] = 0.0

    def display_state(self):
        print("====Harvester Agent====")
        print(f"Reward: {self.reward}")
        print(f"Position: {self.state[0]}")
        print(f"Crop Type: {self.state[1]}")
        print(f"Crop Value: {self.state[2]}")
        print(f"Distance to target: {self.dist_target}")
