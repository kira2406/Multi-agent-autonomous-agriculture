import random
from gymnasium import spaces
from constants import CropTypes

class SeederAgent:
    def __init__(self, pos):
        # self.actions = ['move_up', 'move_down', 'move_left', 'move_right', 'face_up', 'face_down', 'face_left', 'face_right', 'idle', 'collect_seeds', 'plant_seed', 'drop_seeds']
        self.actions = ['move_up', 'move_down', 'move_left', 'move_right', 'idle', 'collect_seeds', 'plant_seeds', 'drop_seeds']
        self.action_space = spaces.Discrete(len(self.actions))
        self.pos = pos
        self.seeds = 0
        self.seed_type = CropTypes.EMPTY # 0 - None, 1 - Type 1, 2 - Type 2, 3 - Type 3
        self.capacity = 3
        self.dist_target = 0
        self.reward = 0

        """
        0: position of agent
        1: number of seeds holding
        2: seed type
        """
        self.state = [pos, 0, 0]


    def get_position(self):
        return self.state[0]
    
    def get_num_seeds(self):
        return self.state[1]
    
    def get_seed_type(self):
        return self.state[2]
    
    def get_distance_target(self):
        return self.dist_target
    
    def update_distance_target(self, distance):
        self.dist_target = distance
    
    def update_position(self, new_pos):
        self.state[0] = new_pos

    def reduce_seeds(self):
        if self.state[1] == 1:
            self.clear_seeds()
        else:
            self.state[1] -= 1

    def clear_seeds(self):
        self.state[1] = 0
        self.state[2] = 0

    def is_holding_seeds(self):
        return self.state[1] > 0


    def collect_seeds(self, seed_quantity, seed_type):

        self.state[1] = seed_quantity
        self.state[2] = seed_type

    def display_state(self):
        print("====Seeder Agent====")
        print(f"Reward: {self.reward}")
        print(f"Position: {self.state[0]}")
        print(f"Num of Seeds: {self.state[1]}")
        print(f"Seed Type: {self.state[2]}")
        print(f"Distance to target: {self.dist_target}")

    # def execute_action(self, action):
    #     if action = 
