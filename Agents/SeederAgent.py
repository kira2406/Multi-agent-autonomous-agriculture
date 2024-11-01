import random
from gymnasium import spaces
from constants import CropTypes

class SeederAgent:
    def __init__(self, pos):
        # self.actions = ['move_up', 'move_down', 'move_left', 'move_right', 'face_up', 'face_down', 'face_left', 'face_right', 'idle', 'collect_seeds', 'plant_seed', 'drop_seeds']
        self.actions = ['move_up', 'move_down', 'move_left', 'move_right', 'idle', 'rotate_clockwise', 'rotate_anticlockwise', 'collect_seeds', 'plant_seeds', 'drop_seeds']
        self.action_space = spaces.Discrete(8)
        self.pos = pos
        self.facing = 2 # 0 - Up, 1 - Right, 2 - Down, 3 - Left
        self.holding_seeds = False
        self.seeds = 0
        self.seed_type = CropTypes.EMPTY # 0 - None, 1 - Type 1, 2 - Type 2, 3 - Type 3
        self.capacity = 4

    def rotate_clock(self):
        self.facing = (self.facing+1)%4

    def rotate_anticlock(self):
        self.facing = (self.facing-1)%4

    def update_pos(self, pos):
        self.pos = pos

    def reduce_seeds(self):
        if self.seeds == 1:
            self.clear_seeds()
        else:
            self.seeds -= 1

    def clear_seeds(self):
        self.holding_seeds = False
        self.seed_type = 0
        self.seeds = 0


    def collect_seeds(self, seed_type, seed_quantity):
        self.seed_type = seed_type
        self.seeds = min(seed_quantity, self.capacity)
        self.holding_seeds = True

    def update_dir(self, direc):
        self.facing = direc

    def display_state(self):
        directions = ['Up', 'Right', 'Down', 'Left']
        print("====Seeder Agent====")
        print(f"Position: {self.pos}")
        print(f"Facing: {directions[self.facing]}")
        print(f"Holding Seeds: {self.holding_seeds}")
        print(f"Seeds: {self.seeds}")
        print(f"Seed Type: {self.seed_type}")

    # def execute_action(self, action):
    #     if action = 
