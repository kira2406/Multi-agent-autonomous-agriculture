import random
from gymnasium import spaces
class WaterAgent:
    def __init__(self, pos):
        self.actions = ['move_up', 'move_down', 'move_left', 'move_right', 'idle', 'rotate_clockwise', 'rotate_anticlockwise', 'collect_water', 'water_crops', 'drop_water']
        self.action_space = spaces.Discrete(8)
        self.pos = pos
        self.facing = 2 # 0 - Up, 1 - Right, 2 - Down, 3 - Left
        self.holding_water = False
        self.water_units = 0
        self.capacity = 4

    def rotate_clock(self):
        self.facing = (self.facing+1)%4

    def rotate_anticlock(self):
        self.facing = (self.facing-1)%4

    def update_pos(self, pos):
        self.pos = pos

    def collect_water(self, water_units):
        self.water_units = min(water_units, self.capacity)
        self.holding_water = True

    def update_dir(self, direc):
        self.facing = direc


    def display_state(self):
        directions = ['Up', 'Right', 'Down', 'Left']
        print("====Water Agent====")
        print(f"Position: {self.pos}")
        print(f"Facing: {directions[self.facing]}")
        print(f"Holding Water: {self.holding_water}")
        print(f"Water Units: {self.water_units}")

    # def execute_action(self, action):
    #     if action = 
