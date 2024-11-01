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
        self.state = []

    def rotate_clock(self):
        self.facing = (self.facing+1)%4

    def rotate_anticlock(self):
        self.facing = (self.facing-1)%4

    def update_pos(self, pos):
        self.pos = pos

    def harvest(self, seed_type, seed_quantity):
        self.seed_type = seed_type
        self.seeds = min(seed_quantity, self.capacity)
        self.holding_crops = True

    def drop_crops(self):
        self.holding_crops = False
        dropped_crops = {
            "crop_type": self.crop_type,
            "crop_units": self.crop_units
        }
        self.crop_units = 0
        self.crop_type = 0
        return dropped_crops

    def update_dir(self, direc):
        self.facing = direc

    def display_state(self):
        directions = ['Up', 'Right', 'Down', 'Left']
        print("====Harvester Agent====")
        print(f"Position: {self.pos}")
        print(f"Facing: {directions[self.facing]}")
        print(f"Holding Seeds: {self.holding_crops}")
        print(f"Crop units: {self.crop_units}")
        print(f"Crop Type: {self.crop_type}")

    # def execute_action(self, action):
    #     if action = 
