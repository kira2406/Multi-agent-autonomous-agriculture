import pygame
import random
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from Agents.SeederAgent import SeederAgent
from Agents.WaterAgent import WaterAgent
from Agents.HarvesterAgent import HarvesterAgent
from pettingzoo.utils import ParallelEnv
from constants import CropStages, CropTypes, CropGrowthRates, CropInfo


# Screen dimensions
TILE_SIZE = 75
GRID_WIDTH, GRID_HEIGHT = 9, 8
SCREEN_WIDTH, SCREEN_HEIGHT = GRID_WIDTH * TILE_SIZE, GRID_HEIGHT * TILE_SIZE

START_IMG = pygame.transform.scale(pygame.image.load('assets/grid_cells/start.png'), (TILE_SIZE, TILE_SIZE))
MARKET_IMG = pygame.transform.scale(pygame.image.load('assets/grid_cells/market.png'), (TILE_SIZE, TILE_SIZE))
PLOT_IMG = pygame.transform.scale(pygame.image.load('assets/grid_cells/plot.png'), (TILE_SIZE, TILE_SIZE))
SEEDSTN1_IMG = pygame.transform.scale(pygame.image.load('assets/grid_cells/crop1.png'), (TILE_SIZE, TILE_SIZE))
SEEDSTN2_IMG = pygame.transform.scale(pygame.image.load('assets/grid_cells/crop2.png'), (TILE_SIZE, TILE_SIZE))
SEEDSTN3_IMG = pygame.transform.scale(pygame.image.load('assets/grid_cells/crop3.png'), (TILE_SIZE, TILE_SIZE))
GRASS_IMG = pygame.transform.scale(pygame.image.load('assets/grid_cells/grass.png'), (TILE_SIZE, TILE_SIZE))
WATER_IMG = pygame.transform.scale(pygame.image.load('assets/grid_cells/water.png'), (TILE_SIZE, TILE_SIZE))
SEEDER_AGENT_IMG = pygame.transform.scale(pygame.image.load('assets/agents/seeder_agent.png'), (TILE_SIZE, TILE_SIZE))
WATER_AGENT_IMG = pygame.transform.scale(pygame.image.load('assets/agents/water_agent.png'), (TILE_SIZE, TILE_SIZE))
HARVESTER_AGENT_IMG = pygame.transform.scale(pygame.image.load('assets/agents/harvester_agent.png'), (TILE_SIZE, TILE_SIZE))

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

DAYS_IN_YEAR = 365

# Custom RL Environment Class
class AgriEnv(ParallelEnv):
    def __init__(self):
        super(AgriEnv, self).__init__()

        self.agents = ['seeder_agent', 'water_agent', 'harvester_agent']
        
        self.initialize_environment()

        self.action_space = {
            self.agents[0]: self.seeder_agent.action_space,
            self.agents[1]: self.water_agent.action_space,  # Define actions for WaterAgent
            self.agents[2]: self.harvester_agent.action_space
        }

        self.current_day = 0

        # Observation space: Agent's position (x, y)
        self.observation_space = spaces.Box(low=0, high=GRID_WIDTH - 1, shape=(2,), dtype=np.int32)

        self.grid = [['grass' for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]  # Initial grass grid
        
        
        # Place start and market

        self.grid[0][0] = 'start'
        self.grid[0][1] = 'start'
        self.grid[0][2] = 'start'
        self.grid[0][3] = 'start'

        self.grid[0][5] = 'market'
        self.grid[0][6] = 'market'
        self.grid[0][7] = 'market'
        self.grid[0][8] = 'market'

        self.market_grids = [[0,5],[0,6],[0,7],[0,8]]

        # Place two groups of 2x6 plots
        self.plot_grids = [[3, 1],[4,1],[3,2],[4,2],[3,3],[4,3], [3, 5],[4,5],[3,6],[4,6],[3,7],[4,7]]
        for plot_grid in self.plot_grids:  # Rows 2 and 3
            self.grid[plot_grid[0]][plot_grid[1]] = 'plot'

        self.initialize_plots()

        self.grid[7][1] = 'seedStn1'
        self.grid[7][3] = 'seedStn2'
        self.grid[7][5] = 'seedStn3'
        self.grid[7][7] = 'water'

        self.seed_station_1 = [7, 1]
        self.seed_station_2 = [7, 3]
        self.seed_station_3 = [7, 5]
        self.water_tank = [7, 7]
        
        
        self.current_day = 0

        self.game_score = 0

        self.done = False

        

    def initialize_environment(self):
        self.seeder_agent = SeederAgent([0, 0])
        self.water_agent = WaterAgent([1, 0])
        self.harvester_agent = HarvesterAgent([2, 0])

        # Initialize Pygame and set up the screen
        pygame.init()
        pygame.font.init()

        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('9x8 Grid Environment')

    def reset(self):
        self.done = False
        self.current_day = 0

        self.initialize_environment()
        self.initialize_plots()

        return {0}

    def step(self, actions):

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        reward = {
            "seeder_agent": -1,
            "water_agent": -1,
            "total_reward": -1
        }

        for agent in self.agents:
            if agent == "seeder_agent":
                action = actions[agent]
                print(f"{agent}'s action: {self.seeder_agent.actions[action]}")
                self.step_seeder_agent(self.seeder_agent.actions[action])
                self.seeder_agent.display_state()
            elif agent == "water_agent":
                action = actions[agent]
                print(f"{agent}'s action: {self.water_agent.actions[action]}")
                self.step_water_agent(self.water_agent.actions[action])
                # self.water_agent.display_state()
            elif agent == "harvester_agent":
                action = actions[agent]
                print(f"{agent}'s action: {self.harvester_agent.actions[action]}")
                self.step_harvester_agent(self.harvester_agent.actions[action])
                self.harvester_agent.display_state()

        self.grow_plot_grids()
        print(self.plot_states)

        self.current_day += 1

        if self.current_day == DAYS_IN_YEAR:
            self.done = True

        return np.array(self.seeder_agent.pos), reward['total_reward'], self.done, {}
    
    def is_market(self, position):

        if [position[1], position[0]] in self.market_grids:
            return True
        return False

    def is_seed_station_1(self, position):
        """ Checks if the given grid cell is seed station 1"""
        if [position[1], position[0]] == self.seed_station_1:
            return True
        return False
        
    def is_seed_station_2(self, position):
        """ Checks if the given grid cell is seed station 2"""
        if [position[1], position[0]] == self.seed_station_2:
            return True
        return False
    
    def is_seed_station_3(self, position):
        """ Checks if the given grid cell is seed station 3"""
        if [position[1], position[0]] == self.seed_station_3:
            return True
        return False
        
    def is_water_tank(self, position):
        """ Checks if the given grid cell is a water tank"""
        if [position[1], position[0]] == self.water_tank:
            return True
        return False
    
    def is_obstacle(self, position):
        """ Checks if the given grid cell has an obstacle"""
        if self.is_seed_station_1(position) or self.is_seed_station_2(position) or self.is_seed_station_3(position) or self.is_water_tank(position):
            return True
        return False
    
    def initialize_plots(self):
        """ Initialize the plot grids to initial state"""
        
        self.plot_states = []
        self.plot_dict = {}
        for index, plot_grid in enumerate(self.plot_grids):
            self.plot_dict[tuple([plot_grid[1],plot_grid[0]])] = index
            self.plot_states.append([
                CropStages.NOT_PLANTED, # indicates the crop state
                CropTypes.EMPTY,
                0.0, # indicates water percentage
                0.0, # indicates growth percentage
                0.0, # indicates disease percentage
                0,    # days since water percentage is 0
                0     # yield value
                ])

    def is_plot_grid(self, position):
        """ Checks if the given grid cell is a plot grid or not """
        if [position[1], position[0]] in self.plot_grids:
            return True
        return False

    
    def is_within_bounds(self, position):
        """ Check if the agent is within the grid bounds """
        if self.is_plot_grid(position) or self.is_obstacle(position):
            return False
        if position[0] < GRID_WIDTH and position[0] >= 0 and position[1] >= 0 and position[1] < GRID_HEIGHT:
            return True
        else:
            return False
    
    def movement(self, agent, direction):
        """ Function to move the agent. Directions - Up, Down, Left, Right"""
        pos_x, pos_y = agent.pos[0], agent.pos[1]
        if direction == "move_up":
            next_pos = [pos_x, pos_y-1]
            if self.is_within_bounds(next_pos):
                agent.update_pos(next_pos)
                agent.update_dir(0)
                return True
        elif direction == "move_down":
            next_pos = [pos_x, pos_y+1]
            if self.is_within_bounds(next_pos):
                agent.update_pos(next_pos)
                agent.update_dir(2)
                return True
        elif direction == "move_left":
            next_pos = [pos_x-1, pos_y]
            if self.is_within_bounds(next_pos):
                agent.update_pos(next_pos)
                agent.update_dir(3)
                return True
        elif direction == "move_right":
            next_pos = [pos_x+1, pos_y]
            if self.is_within_bounds(next_pos):
                agent.update_pos(next_pos)
                agent.update_dir(1)
                return True
        return False

    
    def get_facing_cell(self, agent):
        """ Get the cell the agent is currently facing"""
        position = agent.pos
        pos_x, pos_y = position[0], position[1]
        if agent.facing == 0:
            return [pos_x, pos_y-1]
        elif agent.facing == 1:
            return [pos_x+1, pos_y]
        elif agent.facing == 2:
            return [pos_x, pos_y+1]
        elif agent.facing == 3:
            return [pos_x-1, pos_y]
        
    def pickup_seeds(self):
        """ Pickup seeds from the seed stations if the agent is facing the seed station """
        if self.seeder_agent.holding_seeds:
            return False
        facing_cell = self.get_facing_cell(self.seeder_agent)

        if self.is_seed_station_1(facing_cell):
            self.seeder_agent.collect_seeds(seed_quantity=5,seed_type=CropTypes.WHEAT)
            return True
        
        if self.is_seed_station_2(facing_cell):
            self.seeder_agent.collect_seeds(seed_quantity=5,seed_type=CropTypes.RICE)
            return True
        
        if self.is_seed_station_3(facing_cell):
            self.seeder_agent.collect_seeds(seed_quantity=5,seed_type=CropTypes.CORN)
            return True

        return False
    
    def drop_seeds(self):
        """ Drop seeds back into the seed station """
        if not self.seeder_agent.holding_seeds:
            return False
        facing_cell = self.get_facing_cell(self.seeder_agent)

        if self.is_seed_station_1(facing_cell) and self.seeder_agent.seed_type == CropTypes.WHEAT:
            self.seeder_agent.clear_seeds()
            return True
        elif self.is_seed_station_2(facing_cell) and self.seeder_agent.seed_type == CropTypes.RICE:
            self.seeder_agent.clear_seeds()
            return True
        
        elif self.is_seed_station_3(facing_cell) and self.seeder_agent.seed_type == CropTypes.CORN:
            self.seeder_agent.clear_seeds()
            return True

        return False
    
    def grow_plot_grids(self):
        """ Function to update the state of crops in plot grids after every step """
        for plot_id in range(len(self.plot_states)):
            if self.plot_states[plot_id] != CropStages.NOT_PLANTED:
                crop_state, crop_type, water_level, growth_level, disease_level, days_wo_water, yield_value = self.plot_states[plot_id]
                
                # grow the plant
                if growth_level < 1.5:
                    growth_level += CropInfo.get_growth_rate(crop_type)

                # set crop state after growing
                if growth_level >= 1.5:
                    crop_state = CropStages.DIED
                elif growth_level >= 1.2:
                    crop_state = CropStages.DRIED
                elif growth_level >= 1.0:
                    crop_state = CropStages.HARVEST

                if crop_state == CropStages.HARVEST:
                    yield_value = 1
                elif crop_state == CropStages.DRIED:
                    yield_value = 0.5
                elif crop_state == CropStages.DIED:
                    yield_value = 0
                elif crop_state == CropStages.GROWING:
                    yield_value = 0

                self.plot_states[plot_id] = [crop_state, crop_type, water_level, growth_level, disease_level, days_wo_water, yield_value]

                
    
    def plant_seeds(self):
        """ Function to plant seeds into the plot grids based on the plot which the agent is facing """
        
        facing_cell = tuple(self.get_facing_cell(self.seeder_agent))
        if self.is_plot_grid(facing_cell):
            plot_id = self.plot_dict[facing_cell]
            if self.plot_states[plot_id][0] == CropStages.NOT_PLANTED:
                self.plot_states[plot_id][0] = CropStages.GROWING
                self.plot_states[plot_id][1] = self.seeder_agent.seed_type
                self.seeder_agent.reduce_seeds()
                return True
        return False
    
    def harvest_crops(self):
        """ Function to harvest crops from the plot grids based on the plot which the agent is facing """
        
        facing_cell = tuple(self.get_facing_cell(self.harvester_agent))

        if not self.harvester_agent.holding_crops and self.is_plot_grid(facing_cell):
            plot_id = self.plot_dict[facing_cell]

            if self.plot_states[plot_id][0] != CropStages.NOT_PLANTED:
            
                # Moving the crops to the harvester
                self.harvester_agent.holding_crops = True
                self.harvester_agent.crop_type = self.plot_states[plot_id][1]
                self.harvester_agent.yield_value = self.plot_states[plot_id][6]

                # Clear the plot
                self.plot_states[plot_id] = [0, 0, 0.0, 0.0, 0.0, 0, 0]
                return True
        return False

    def drop_crops(self):
        """ Function to drop crops currently held by the harvester agent """
        
        facing_cell = tuple(self.get_facing_cell(self.harvester_agent))

        if self.harvester_agent.holding_crops:
            if self.is_market(facing_cell):
                market_value = [20, 40, 60]

                # Calculate game score by multiplying market_value*yield value
                print(f"Market: value={market_value[self.harvester_agent.crop_type-1]} yield_value:{self.harvester_agent.yield_value}")
                self.game_score += self.harvester_agent.yield_value*market_value[self.harvester_agent.crop_type-1]

                # Clear the harvester agent
                self.harvester_agent.holding_crops = False
                self.harvester_agent.crop_type = CropTypes.EMPTY
                self.harvester_agent.yield_value = 0

            return True
        return False


    
    def pickup_water(self):
        """ Pickup water_units of water from the water tank"""
        facing_cell = self.get_facing_cell(self.water_agent)

        if self.is_water_tank(facing_cell):
            self.water_agent.collect_water(water_units=4)
            return True

        return False

    

    
    def step_seeder_agent(self, action):
        """ Function defines the set of actions executed by the seeder_agent """
        seeder_agent_actions = self.seeder_agent.actions

        if action == seeder_agent_actions[0]: # move_up
            if self.movement(self.seeder_agent, action):
                reward = -1
            else:
                reward = -10
        elif action == seeder_agent_actions[1]: # move_down
            if self.movement(self.seeder_agent, action):
                reward = -1
            else:
                reward = -10
        elif action == seeder_agent_actions[2]: # move_left
            if self.movement(self.seeder_agent, action):
                reward = -1
            else:
                reward = -10
        elif action == seeder_agent_actions[3]: # move_right
            if self.movement(self.seeder_agent, action):
                reward = -1
            else:
                reward = -10
        elif action == seeder_agent_actions[4]: # idle
            reward = -1
        elif action == seeder_agent_actions[5]: # rotate_clockwise
            self.seeder_agent.rotate_clock()
        elif action == seeder_agent_actions[6]: # rotate_anticlockwise
            self.seeder_agent.rotate_anticlock()
        elif action == seeder_agent_actions[7]: # collect_seeds
            if self.pickup_seeds():
                reward = 5
            else:
                reward = -10
        elif action == seeder_agent_actions[8]: # plant_seeds
            if self.plant_seeds():
                reward = 20
            else:
                reward = -10
        elif action == seeder_agent_actions[9]: # drop_seeds
            if self.drop_seeds():
                reward = -1
            else:
                reward = -10

    def step_water_agent(self, action):
        """ Function defines the set of actions executed by the water_agent """

        water_agent_actions = self.water_agent.actions
        
        if action == water_agent_actions[0]: # move_up
            if self.movement(self.water_agent, action):
                reward = -1
            else:
                reward = -10
        elif action == water_agent_actions[1]: # move_down
            if self.movement(self.water_agent, action):
                reward = -1
            else:
                reward = -10
        elif action == water_agent_actions[2]: # move_left
            if self.movement(self.water_agent, action):
                reward = -1
            else:
                reward = -10
        elif action == water_agent_actions[3]: # move_right
            if self.movement(self.water_agent, action):
                reward = -1
            else:
                reward = -10
        elif action == water_agent_actions[4]: # idle
            reward = -1
        elif action == water_agent_actions[5]: # rotate_clockwise
            self.water_agent.rotate_clock()
        elif action == water_agent_actions[6]: # rotate_anticlockwise
            print(water_agent_actions[6])
            self.water_agent.rotate_anticlock()
        elif action == water_agent_actions[7]: # collect_water
            if self.pickup_water():
                reward = 5
            else:
                reward = -10
        elif action == water_agent_actions[8]: # water_crops
            print(self.get_facing_cell(self.water_agent))
        elif action == water_agent_actions[9]: # drop_water
            print(self.get_facing_cell(self.water_agent))


    def step_harvester_agent(self, action):
        """ Function defines the set of actions executed by the harvester_agent """

        harvester_agent_actions = self.harvester_agent.actions
        
        if action == harvester_agent_actions[0]: # move_up
            print(harvester_agent_actions[0])
            if self.movement(self.harvester_agent, action):
                reward = -1
            else:
                reward = -10
        elif action == harvester_agent_actions[1]: # move_down
            print(harvester_agent_actions[1])
            if self.movement(self.harvester_agent, action):
                reward = -1
            else:
                reward = -10
        elif action == harvester_agent_actions[2]: # move_left
            print(harvester_agent_actions[2])
            if self.movement(self.harvester_agent, action):
                reward = -1
            else:
                reward = -10
        elif action == harvester_agent_actions[3]: # move_right
            print(harvester_agent_actions[3])
            if self.movement(self.harvester_agent, action):
                reward = -1
            else:
                reward = -10
        elif action == harvester_agent_actions[4]: # idle
            print(harvester_agent_actions[4])
        elif action == harvester_agent_actions[5]: # rotate_clockwise
            print(harvester_agent_actions[5])
            self.harvester_agent.rotate_clock()
        elif action == harvester_agent_actions[6]: # rotate_anticlockwise
            print(harvester_agent_actions[6])
            self.harvester_agent.rotate_anticlock()
        elif action == harvester_agent_actions[7]: # harvest_crops
            print(harvester_agent_actions[7])
            print(self.harvest_crops())
        elif action == harvester_agent_actions[8]: # drop_crops
            print(harvester_agent_actions[8])
            print(self.drop_crops())


    def render(self):

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        self.screen.fill(WHITE)

        # Draw the grid
        for row in range(GRID_HEIGHT):
            for col in range(GRID_WIDTH):
                tile = self.grid[row][col]
                x, y = col * TILE_SIZE, row * TILE_SIZE
                if tile == 'start':
                    self.screen.blit(START_IMG, (x, y))
                elif tile == 'market':
                    self.screen.blit(MARKET_IMG, (x, y))
                elif tile == 'plot':
                    self.screen.blit(PLOT_IMG, (x, y))
                elif tile == 'seedStn1':
                    self.screen.blit(SEEDSTN1_IMG, (x, y))
                elif tile == 'seedStn2':
                    self.screen.blit(SEEDSTN2_IMG, (x, y))
                elif tile == 'seedStn3':
                    self.screen.blit(SEEDSTN3_IMG, (x, y))
                elif tile == 'water':
                    self.screen.blit(WATER_IMG, (x, y))
                else:
                    self.screen.blit(GRASS_IMG, (x, y))

        # Draw agent (for simplicity, representing as a black square)
        water_agent_pos = self.water_agent.pos  # Assume the position is stored here
        water_agent_x, water_agent_y = water_agent_pos[0] * TILE_SIZE, water_agent_pos[1] * TILE_SIZE
        # print("wateragent", self.water_agent.pos,self.water_agent.pos[0], self.water_agent.pos[1], water_agent_x, water_agent_y)
        self.screen.blit(WATER_AGENT_IMG, ( water_agent_x,  water_agent_y))

        seeder_agent_pos = self.seeder_agent.pos  # Assume the position is stored here
        seeder_agent_x, seeder_agent_y = seeder_agent_pos[0] * TILE_SIZE, seeder_agent_pos[1] * TILE_SIZE
        # print("seederagent", self.seeder_agent.pos,seeder_agent_pos[0],seeder_agent_pos[0], seeder_agent_x, seeder_agent_y)
        self.screen.blit(SEEDER_AGENT_IMG, ( seeder_agent_x,  seeder_agent_y))

        harvester_agent_pos = self.harvester_agent.pos  # Assume the position is stored here
        harvester_agent_x, harvester_agent_y = harvester_agent_pos[0] * TILE_SIZE, harvester_agent_pos[1] * TILE_SIZE
        # print("harvester_agent", self.harvester_agent.pos,harvester_agent_pos[0],harvester_agent_pos[0], harvester_agent_x, harvester_agent_y)
        self.screen.blit(HARVESTER_AGENT_IMG, ( harvester_agent_x,  harvester_agent_y))

        font = pygame.font.Font(None, 36)
        text = font.render(f'Day: {self.current_day}', True, BLACK)
        self.screen.blit(text, (10, 10))

        text2 = font.render(f'Game Score: {self.game_score}', True, BLACK)
        self.screen.blit(text2, (300, 10))

        pygame.display.update()

    def close(self):
        pygame.quit()