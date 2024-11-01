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

        """
        0: position of agent
        1: facing direction
        2: number of seeds holding
        3: sed type
        """
        self.state = [pos, 0, 0, 0]


    def get_position(self):
        return self.state[0]
    
    def get_facing(self):
        return self.state[1]
    
    def get_num_seeds(self):
        return self.state[2]
    
    
    def get_seed_type(self):
        return self.state[3]
    
    def update_position(self, new_pos):
        self.state[0] = new_pos

    def update_facing(self, direc):
        self.state[1] = direc

    def rotate_clock(self):
        self.state[1] = (self.state[1]+1)%4

    def rotate_anticlock(self):
        self.state[1] = (self.state[1]-1)%4

    def reduce_seeds(self):
        if self.state[2] == 1:
            self.clear_seeds()
        else:
            self.state[2] -= 1

    def clear_seeds(self):
        self.state[2] = 0
        self.state[3] = 0

    def is_holding_seeds(self):
        return self.state[2] > 0


    def collect_seeds(self, seed_type, seed_quantity):
        self.state[2] = seed_quantity
        self.state[3] = seed_type

    def display_state(self):
        print("====Seeder Agent====")
        print(self.state)

    # def execute_action(self, action):
    #     if action = 
