import random
from gymnasium import spaces
from constants import CropTypes

class HarvesterAgent:
    def __init__(self, pos):
        # self.actions = ['move_up', 'move_down', 'move_left', 'move_right', 'face_up', 'face_down', 'face_left', 'face_right', 'idle', 'collect_seeds', 'plant_seed', 'drop_seeds']
        self.actions = ['move_up', 'move_down', 'move_left', 'move_right', 'idle', 'rotate_clockwise', 'rotate_anticlockwise', 'harvest_crops', 'drop_crops']
        self.action_space = spaces.Discrete(8)
        self.pos = pos
        self.facing = 2 # 0 - Up, 1 - Right, 2 - Down, 3 - Left
        self.holding_crops = False
        self.crop_units = 0
        self.crop_type = CropTypes.EMPTY # 0 - None, 1 - Type 1, 2 - Type 2, 3 - Type 3
        self.yield_value = 0

        """
        0: position of agent
        1: facing direction
        2: type of crop carrying
        3: yield value of crops
        """
        self.state = [pos, 0, 0, 0, 0]


    def get_position(self):
        return self.state[0]
    
    def get_facing(self):
        return self.state[1]
    
    def update_position(self, new_pos):
        self.state[0] = new_pos

    def update_facing(self, direction):
        self.state[1] = direction

    def update_crops(self, crop_type, crop_value):
        self.state[2] = crop_type
        self.state[3] = crop_value

    def rotate_clock(self):
        self.state[1] = (self.state[1]+1)%4

    def rotate_anticlock(self):
        self.state[1] = (self.state[1]-1)%4
    
    def get_crops_type(self):
        return self.state[2]
    
    def get_crops_value(self):
        return self.state[3]

    def is_holding_crops(self):
        return self.state[2] > 0

    def harvest(self, seed_type, seed_quantity):
        self.seed_type = seed_type
        self.seeds = min(seed_quantity, self.capacity)
        self.holding_crops = True

    def drop_crops(self):
        self.state[2] = 0
        self.state[3] = 0.0

    def display_state(self):
        directions = ['Up', 'Right', 'Down', 'Left']
        print("====Harvester Agent====")
        print(f"Position: {self.state[0]}")
        print(f"Facing: {directions[self.state[1]]}")
        print(f"Crop Type: {self.state[2]}")
        print(f"Crop Value: {self.state[3]}")
