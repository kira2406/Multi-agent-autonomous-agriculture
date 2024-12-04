import pygame
import random
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from Agents.SeederAgent import SeederAgent
from Agents.WaterAgent import WaterAgent
from Agents.HarvesterAgent import HarvesterAgent
from pettingzoo.utils import ParallelEnv
from constants import CropStages, CropTypes, CropInfo, GridElements, Weather


# Screen dimensions for pygame window
TILE_SIZE = 75
GRID_WIDTH, GRID_HEIGHT = 8, 7
SCREEN_WIDTH, SCREEN_HEIGHT = GRID_WIDTH * TILE_SIZE, GRID_HEIGHT * TILE_SIZE

#screen cells loading of different images
START_IMG = pygame.transform.scale(pygame.image.load('assets/grid_cells/start.png'), (TILE_SIZE, TILE_SIZE))
S_START_IMG = pygame.transform.scale(pygame.image.load('assets/grid_cells/s_start.png'), (TILE_SIZE, TILE_SIZE))
W_START_IMG = pygame.transform.scale(pygame.image.load('assets/grid_cells/w_start.png'), (TILE_SIZE, TILE_SIZE))
H_START_IMG = pygame.transform.scale(pygame.image.load('assets/grid_cells/h_start.png'), (TILE_SIZE, TILE_SIZE))
MARKET_IMG = pygame.transform.scale(pygame.image.load('assets/grid_cells/market.png'), (TILE_SIZE, TILE_SIZE))
GARBAGE_IMG = pygame.transform.scale(pygame.image.load('assets/grid_cells/garbage.png'), (TILE_SIZE, TILE_SIZE))
PLOT_IMG = pygame.transform.scale(pygame.image.load('assets/grid_cells/plot.png'), (TILE_SIZE, TILE_SIZE))
PLOT_WHEAT_0_IMG = pygame.transform.scale(pygame.image.load('assets/grid_cells/plot_crop1_0.png'), (TILE_SIZE, TILE_SIZE))
PLOT_WHEAT_0_W_IMG = pygame.transform.scale(pygame.image.load('assets/grid_cells/plot_crop1_0_w.png'), (TILE_SIZE, TILE_SIZE))
PLOT_WHEAT_1_IMG = pygame.transform.scale(pygame.image.load('assets/grid_cells/plot_crop1_1.png'), (TILE_SIZE, TILE_SIZE))
PLOT_WHEAT_2_IMG = pygame.transform.scale(pygame.image.load('assets/grid_cells/plot_crop1_2.png'), (TILE_SIZE, TILE_SIZE))
PLOT_WHEAT_3_IMG = pygame.transform.scale(pygame.image.load('assets/grid_cells/plot_crop1_3.png'), (TILE_SIZE, TILE_SIZE))
PLOT_RICE_0_IMG = pygame.transform.scale(pygame.image.load('assets/grid_cells/plot_crop2_0.png'), (TILE_SIZE, TILE_SIZE))
PLOT_RICE_0_W_IMG = pygame.transform.scale(pygame.image.load('assets/grid_cells/plot_crop2_0_w.png'), (TILE_SIZE, TILE_SIZE))
PLOT_RICE_1_IMG = pygame.transform.scale(pygame.image.load('assets/grid_cells/plot_crop2_1.png'), (TILE_SIZE, TILE_SIZE))
PLOT_RICE_2_IMG = pygame.transform.scale(pygame.image.load('assets/grid_cells/plot_crop2_2.png'), (TILE_SIZE, TILE_SIZE))
PLOT_RICE_3_IMG = pygame.transform.scale(pygame.image.load('assets/grid_cells/plot_crop2_3.png'), (TILE_SIZE, TILE_SIZE))
PLOT_CORN_0_IMG = pygame.transform.scale(pygame.image.load('assets/grid_cells/plot_crop3_0.png'), (TILE_SIZE, TILE_SIZE))
PLOT_CORN_0_W_IMG = pygame.transform.scale(pygame.image.load('assets/grid_cells/plot_crop3_0_w.png'), (TILE_SIZE, TILE_SIZE))
PLOT_CORN_1_IMG = pygame.transform.scale(pygame.image.load('assets/grid_cells/plot_crop3_1.png'), (TILE_SIZE, TILE_SIZE))
PLOT_CORN_2_IMG = pygame.transform.scale(pygame.image.load('assets/grid_cells/plot_crop3_2.png'), (TILE_SIZE, TILE_SIZE))
PLOT_CORN_3_IMG = pygame.transform.scale(pygame.image.load('assets/grid_cells/plot_crop3_3.png'), (TILE_SIZE, TILE_SIZE))
SEEDSTN1_IMG = pygame.transform.scale(pygame.image.load('assets/grid_cells/crop1.png'), (TILE_SIZE, TILE_SIZE))
SEEDSTN2_IMG = pygame.transform.scale(pygame.image.load('assets/grid_cells/crop2.png'), (TILE_SIZE, TILE_SIZE))
SEEDSTN3_IMG = pygame.transform.scale(pygame.image.load('assets/grid_cells/crop3.png'), (TILE_SIZE, TILE_SIZE))
GRASS_IMG = pygame.transform.scale(pygame.image.load('assets/grid_cells/grass.png'), (TILE_SIZE, TILE_SIZE))
WATER_TANK_IMG = pygame.transform.scale(pygame.image.load('assets/grid_cells/water.png'), (TILE_SIZE, TILE_SIZE))
SEEDER_AGENT_IMG = pygame.transform.scale(pygame.image.load('assets/agents/seeder_agent.png'), (TILE_SIZE, TILE_SIZE))
SEEDER_AGENT_CORN_IMG = pygame.transform.scale(pygame.image.load('assets/agents/seeder_agent_corn.png'), (TILE_SIZE, TILE_SIZE))
SEEDER_AGENT_WHEAT_IMG = pygame.transform.scale(pygame.image.load('assets/agents/seeder_agent_wheat.png'), (TILE_SIZE, TILE_SIZE))
SEEDER_AGENT_RICE_IMG = pygame.transform.scale(pygame.image.load('assets/agents/seeder_agent_rice.png'), (TILE_SIZE, TILE_SIZE))
WATER_AGENT_IMG = pygame.transform.scale(pygame.image.load('assets/agents/water_agent.png'), (TILE_SIZE, TILE_SIZE))
WATER_AGENT_FILL_IMG = pygame.transform.scale(pygame.image.load('assets/agents/water_agent_fill.png'), (TILE_SIZE, TILE_SIZE))
HARVESTER_AGENT_IMG = pygame.transform.scale(pygame.image.load('assets/agents/harvester_agent.png'), (TILE_SIZE, TILE_SIZE))
HARVESTER_AGENT_CORN_IMG = pygame.transform.scale(pygame.image.load('assets/agents/harvester_agent_corn.png'), (TILE_SIZE, TILE_SIZE))
HARVESTER_AGENT_WHEAT_IMG = pygame.transform.scale(pygame.image.load('assets/agents/harvester_agent_wheat.png'), (TILE_SIZE, TILE_SIZE))
HARVESTER_AGENT_RICE_IMG = pygame.transform.scale(pygame.image.load('assets/agents/harvester_agent_rice.png'), (TILE_SIZE, TILE_SIZE))

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

DAYS_IN_YEAR = 500

# Defining RL Environment Class
class AgriEnv(ParallelEnv):
    def __init__(self, verbose = False):
        super(AgriEnv, self).__init__()
        self.verbose = verbose
        self.agents = ['seeder_agent', 'harvester_agent', 'water_agent']

        
        self.grid_state = np.zeros(GRID_HEIGHT * GRID_WIDTH, dtype=int)
        self.num_columns = GRID_WIDTH
        self.num_rows = GRID_HEIGHT

        self.STATION1_IDX = 0
        self.STATION2_IDX = 5
        self.STATION3_IDX = 48
        self.MARKET_IDX = 6
        self.GARBAGE_IDX = 7
        self.SEEDSTN1_IDX = 8
        self.WATERTANK_IDX = 40

        self.initialize_environment()


        self.current_day = 0

        self.plot_grids_indices = [18, 19, 21, 22, 26, 27, 29, 30, 42, 43, 45, 46, 50, 51, 53, 54]

        
        self.observation_space = spaces.Tuple((
            spaces.Box(low=-1.0, high=2.0, shape=(len(self.plot_grids_indices)*5,), dtype=np.float32),  # Plot grid state
            spaces.Box(low=-1.0, high=3.0, shape=(3,), dtype=np.float32),  # Seeder agent state
            spaces.Box(low=-1.0, high=3.0, shape=(2,), dtype=np.float32),   # Water agent state
            spaces.Box(low=-1.0, high=3.0, shape=(3,), dtype=np.float32)   # Harvester agent state
        ))
        
        self.info = {
            "times_picked_seeds": 0,
            "times_planted_seeds": 0,
            "times_dropped_seeds_w_planting": 0,
            "times_dropped_seeds_wo_planting": 0,
            "times_harvested":0,
            "times_sold":0,
        }
        
        
        # Place start and market
        self.grid_state[self.STATION1_IDX] = GridElements.STATION1
        self.grid_state[self.STATION2_IDX] = GridElements.STATION2
        self.grid_state[self.STATION3_IDX] = GridElements.STATION3
        self.grid_state[self.MARKET_IDX] = GridElements.MARKET
        self.grid_state[self.GARBAGE_IDX] = GridElements.GARBAGE

        
        for plot_idx in self.plot_grids_indices:
            self.grid_state[plot_idx] = GridElements.PLOT

        self.grid_state[self.SEEDSTN1_IDX] = GridElements.SEEDSTN1
        self.grid_state[self.WATERTANK_IDX] = GridElements.WATERTANK

        
        # Initialize plots
        self.initialize_plots()

        self.seeder_agent.update_distance_target(self.manhatten_dist(self.seeder_agent.get_position(), self.SEEDSTN1_IDX)-1)

        self.harvester_agent.update_distance_target(self.manhatten_dist(self.harvester_agent.get_position(), sum(self.plot_grids_indices)/len(self.plot_grids_indices))-1)

        self.water_agent.update_distance_target(self.manhatten_dist(self.seeder_agent.get_position(), self.WATERTANK_IDX)-1)
        
        
        self.current_day = 0

        self.game_score = 0

        self.done = False

        # Individual rewards for all agents
        self.agent_rewards = {}

        for a in self.agents:
            self.agent_rewards[a] = 0

        self.pygame_initialized = False

        # Placeholder for selected actions
        self.seeder_did_what = ""
        self.harvester_did_what = ""
        self.waterer_did_what = ""

    def initialize_pygame(self):
        # This function initializes pygame  window
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('7x8 Grid Environment')
         
        self.pygame_initialized = True

    def initialize_environment(self):
        """ Initialize agents """

        self.seeder_agent = SeederAgent(self.STATION1_IDX)
        self.harvester_agent = HarvesterAgent(self.STATION2_IDX)
        self.water_agent = WaterAgent(self.STATION3_IDX)

        self.seeder_agent.capacity = 4
        self.water_agent.capacity = 4

        self.action_space = {
            self.agents[0]: self.seeder_agent.action_space,
            self.agents[1]: self.harvester_agent.action_space,
            self.agents[2]: self.water_agent.action_space,
        }

    def reset(self):
        """ Resets the game environment """

        self.done = False
        self.current_day = 0
        self.game_score = 0

        self.initialize_environment()
        self.initialize_plots()

        return self.ret_observation(), {}
    
    def ret_observation(self):
        """ Returns plot states, agents states"""
        plot_state_obs = []
        for plot_state in self.plot_states:
            plot_state_obs.append([plot_state[0], plot_state[1], plot_state[2], plot_state[3], plot_state[6]])
        return [np.array(plot_state_obs).flatten().tolist(), self.seeder_agent.state, self.water_agent.state, self.harvester_agent.state]

    def step(self, actions):
        """ step the environment by executing the agent's actions"""

        if self.pygame_initialized:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()

        """ Set the agent's inreward fpr current step"""
        self.seeder_agent.reward = 0
        self.harvester_agent.reward = 0
        self.water_agent.reward = 0

        for agent in self.agents:
            if agent == "seeder_agent":
                action = actions[agent]
                seeder_agent_actions = self.seeder_agent.actions
                self.seeder_did_what = seeder_agent_actions[action]
                self.step_seeder_agent(self.seeder_agent.actions[action])
                
            elif agent == "water_agent":
                action = actions[agent]
                water_agent_actions = self.water_agent.actions
                self.waterer_did_what = water_agent_actions[action]
                self.step_water_agent(self.water_agent.actions[action])
            elif agent == "harvester_agent":
                action = actions[agent]
                harvester_agent_actions = self.harvester_agent.actions
                self.harvester_did_what = harvester_agent_actions[action]
                self.step_harvester_agent(self.harvester_agent.actions[action])

        # Execute a day in the environment
        self.grow_plot_grids()

        if self.verbose:
            self.seeder_agent.display_state()
            self.water_agent.display_state()
            self.harvester_agent.display_state()
            print(self.plot_states)

        # update step reward for each agent
        step_reward = {'seeder_agent': self.seeder_agent.reward,
                       'water_agent': self.water_agent.reward,
                       'harvester_agent': self.harvester_agent.reward}

        self.current_day += 1

        if self.current_day == DAYS_IN_YEAR:
            self.done = True

        return self.ret_observation(), step_reward, self.done, {}
    
    def manhatten_dist(self, grid_cell1, grid_cell2):
        """ Calculates manhattan distance between grid_cell1 and grid_cell 2"""

        dist1_y, dist1_x = self.get_coordinates(grid_cell1)
        dist2_y, dist2_x = self.get_coordinates(grid_cell2)
        return abs(dist1_x - dist2_x) + abs(dist1_y - dist2_y)
    
    def is_station1(self, pos):
        """ Returns True if grid cell is station 1 else False"""

        if pos < 0 or pos >= GRID_WIDTH*GRID_HEIGHT:
            return False
        if self.grid_state[pos] == GridElements.STATION1:
            return True
        return False
    
    def is_station2(self, pos):
        """ Returns True if grid cell is station 2 else False"""

        if pos < 0 or pos >= GRID_WIDTH*GRID_HEIGHT:
            return False
        if self.grid_state[pos] == GridElements.STATION2:
            return True
        return False
    
    def is_station3(self, pos):
        """ Returns True if grid cell is station 3 else False"""

        if pos < 0 or pos >= GRID_WIDTH*GRID_HEIGHT:
            return False
        if self.grid_state[pos] == GridElements.STATION3:
            return True
        return False
    
    def is_market(self, pos):
        """ Returns True if grid cell is market else False"""

        if pos < 0 or pos >= GRID_WIDTH*GRID_HEIGHT:
            return False
        if self.grid_state[pos] == GridElements.MARKET:
            return True
        return False
    
    def is_garbage(self, pos):
        """ Returns True if grid cell is garbage else False"""

        if pos < 0 or pos >= GRID_WIDTH*GRID_HEIGHT:
            return False
        if self.grid_state[pos] == GridElements.GARBAGE:
            return True
        return False

    def is_seed_station_1(self, pos):
        """ Returns True if grid cell is seed station 1 else False"""

        if pos < 0 or pos >= GRID_WIDTH*GRID_HEIGHT:
            return False
        if self.grid_state[pos] == GridElements.SEEDSTN1:
            return True
        return False
        
    def is_seed_station_2(self, pos):
        """ Returns True if grid cell is seed station 2 else False"""

        if pos < 0 or pos >= GRID_WIDTH*GRID_HEIGHT:
            return False
        if self.grid_state[pos] == GridElements.SEEDSTN2:
            return True
        return False
    
    def is_seed_station_3(self, pos):
        """ Returns True if grid cell is seed station 3 else False"""

        if pos < 0 or pos >= GRID_WIDTH*GRID_HEIGHT:
            return False
        if self.grid_state[pos] == GridElements.SEEDSTN3:
            return True
        return False
        
    def is_water_tank(self, pos):
        """ Returns True if grid cell is water tank else False"""

        if pos < 0 or pos >= GRID_WIDTH*GRID_HEIGHT:
            return False
        if self.grid_state[pos] == GridElements.WATERTANK:
            return True
        return False
    
    def is_obstacle(self, pos):
        """ Returns True if grid cell is blocked else False"""

        if self.is_seed_station_1(pos) or self.is_seed_station_2(pos) or self.is_seed_station_3(pos) or self.is_water_tank(pos):
            return True
        return False
    
    def initialize_plots(self):
        """ Initialize the plot grids to initial state"""
        
        self.plot_states = []
        self.plot_dict = {}
        for index, plot_index in enumerate(self.plot_grids_indices):
            self.plot_dict[plot_index] = index
            self.plot_states.append([
                CropStages.NOT_PLANTED, # indicates the crop state
                CropTypes.EMPTY, # indicates the type of crops
                0, # indicates watered or not
                0.0, # indicates growth percentage
                0.0, # indicates disease percentage
                0,    # days since water percentage is 0
                0     # yield value
                ])
        self.empty_plots = len(self.plot_states)
        self.dry_plots = 0
        self.harvest_ready_plots = 0

    def is_plot_grid(self, pos):
        """ Checks if the grid cell is a plot grid or not """

        if pos < 0 or pos >= GRID_WIDTH*GRID_HEIGHT:
            return False
        if self.grid_state[pos] == GridElements.PLOT:
            return True
        return False

    
    def is_within_bounds(self, pos):
        """ Check if the grid cell is within the grid bounds """

        if pos < 0 or pos >= GRID_WIDTH*GRID_HEIGHT:
            return False
        
        # Check if position is conflicting with other agents' position
        if pos == self.seeder_agent.get_position() or pos == self.harvester_agent.get_position() or pos == self.water_agent.get_position():
            return False
        
        return True
    
    def movement(self, agent, direction):
        """ Function to move the agent. Directions - Up, Down, Left, Right"""

        curr_grid = agent.get_position()
        next_pos = curr_grid

        if direction == "move_up":
            if curr_grid < self.num_columns:
                return False
            next_pos = curr_grid-self.num_columns
        elif direction == "move_down":
            if curr_grid >= (GRID_HEIGHT - 1) * self.num_columns:
                return False
            next_pos = curr_grid+self.num_columns
        elif direction == "move_left":
            # check if agent is in first column
            if curr_grid % self.num_columns == 0:
                return False
            next_pos = curr_grid - 1
        elif direction == "move_right":
            if (curr_grid+1) % self.num_columns == 0:
                return False
            next_pos = curr_grid + 1

        if self.is_within_bounds(next_pos):
            agent.update_position(next_pos)
            return True
        
        return False
        
    def pickup_seeds(self):
        """ Pickup seeds from the seed stations if the agent is at the seed station """

        if self.seeder_agent.is_holding_seeds():
            return -0.2
        

        curr_cell = self.seeder_agent.get_position()

        # check if the capacity is full
        available_capacity = self.seeder_agent.capacity - self.seeder_agent.get_num_seeds()
        if available_capacity <= 0:
            return -0.2

        # If the agent is at seed station 1
        if self.is_seed_station_1(curr_cell):
            # Agent picks up seeds
            self.seeder_agent.collect_seeds(seed_quantity=available_capacity,seed_type=CropTypes.WHEAT)
            self.info["times_picked_seeds"] += 1
            return 0.5

        return -0.2
    
    def drop_seeds(self):
        """
        Drop seeds back into the seed station and update rewards accordingly
        """
        # Penalize immediately if not holding seeds
        if not self.seeder_agent.is_holding_seeds():
            return -0.2

        curr_cell = self.seeder_agent.get_position()
        seed_type = self.seeder_agent.get_seed_type()
        num_seeds = self.seeder_agent.get_num_seeds()

        # Base reward/penalty
        reward = -0.01  # Default small penalty for dropping seeds

        # Check if agent is at the correct seed station for the held seed type
        if (
            (self.is_seed_station_1(curr_cell) and seed_type == CropTypes.WHEAT) or
            (self.is_seed_station_2(curr_cell) and seed_type == CropTypes.RICE) or
            (self.is_seed_station_3(curr_cell) and seed_type == CropTypes.CORN)
        ):
            if num_seeds == self.seeder_agent.capacity:
                # Large penalty for dropping without planting (inefficient strategy)
                reward = -0.2
                self.info["times_dropped_seeds_wo_planting"] = self.info.get("times_dropped_seeds_wo_planting", 0) + num_seeds
            else:
                # Moderate reward for freeing space when dropping is part of a valid strategy
                reward = 0.2
                self.info["times_dropped_seeds_w_planting"] = self.info.get("times_dropped_seeds_w_planting", 0) + num_seeds

            # Clear seeds and update logs
            self.seeder_agent.clear_seeds()
            return reward

        reward = -0.2  # Severe penalty for mismatched seed type/station
        return reward
    
    def grow_plot_grids(self):
        """ Function to update the state of crops in plot grids after every step """

        self.harvest_ready_plots = 0
        for plot_id in range(len(self.plot_states)):
            if self.plot_states[plot_id][0] != CropStages.NOT_PLANTED:
                crop_state, crop_type, is_watered, growth_level, disease_level, days_wo_water, yield_value = self.plot_states[plot_id]


                # grow the plant when water level is above 0.0
                if is_watered:
                    # print(f"{plot_id} water")
                    if growth_level < 2.0:
                        growth_level += CropInfo.get_growth_rate(crop_type)

                    # set crop state after growing
                    if growth_level >= 2.0:
                        crop_state = CropStages.DIED
                        self.harvest_ready_plots += 1
                    elif growth_level >= 1.5:
                        crop_state = CropStages.DRIED
                        self.harvest_ready_plots += 1
                    elif growth_level >= 1.0:
                        crop_state = CropStages.HARVEST
                        self.harvest_ready_plots += 1

                    if crop_state == CropStages.HARVEST:
                        yield_value = 1
                    elif crop_state == CropStages.DRIED:
                        yield_value = 0.5
                    elif crop_state == CropStages.DIED:
                        yield_value = 0
                    elif crop_state == CropStages.GROWING:
                        yield_value = 0

                self.plot_states[plot_id] = [crop_state, crop_type, is_watered, growth_level, disease_level, days_wo_water, yield_value]
        if self.verbose:
            print("harvest ready plots", self.harvest_ready_plots)
            print("dry_plots", self.dry_plots)
            print("empty_plots", self.empty_plots)
        
        
    def plant_seeds(self):
        """ Function to plant seeds into the plot grids """
        
        curr_cell = self.seeder_agent.get_position()

        # If the agent is not carrying seeds
        if self.seeder_agent.get_num_seeds() <= 0:
            return -0.2

        # If the agent is at the plots
        if self.is_plot_grid(curr_cell):
            plot_id = self.plot_dict[curr_cell]

            # If the plot is not planted
            if self.plot_states[plot_id][0] == CropStages.NOT_PLANTED:
                self.plot_states[plot_id][0] = CropStages.GROWING
                self.plot_states[plot_id][1] = self.seeder_agent.get_seed_type()
                self.seeder_agent.reduce_seeds()
                self.info["times_planted_seeds"] += 1
                self.empty_plots -= 1
                self.dry_plots += 1
                self.harvester_agent.reward += 1
                self.water_agent.reward += 1
                if self.empty_plots == 0:
                    return 1.5
                return 1
        return -0.2
    
    def harvest_crops(self):
        """ Function to harvest crops from the plot grids """

        if self.harvester_agent.is_holding_crops():
            return False
        
        curr_cell = self.harvester_agent.get_position()

        # If the agent is at the plots
        if self.is_plot_grid(curr_cell):
            plot_id = self.plot_dict[curr_cell]

            if self.plot_states[plot_id][0] != CropStages.NOT_PLANTED and self.plot_states[plot_id][0] != CropStages.GROWING:
            
                # Moving the crops to the harvester
                _, crop_type, _, _, _, _, crop_value = self.plot_states[plot_id]

                self.harvester_agent.update_crops(crop_type, crop_value)

                # Clear the plot
                self.plot_states[plot_id] = [CropStages.NOT_PLANTED, 0, 0.0, 0.0, 0.0, 0, 0]
                self.info["times_harvested"] += 1
                self.empty_plots += 1
                return True
        return False

    def drop_crops_market(self):
        """ Drop the crops at market """

        if self.harvester_agent.is_holding_crops():

            # check the value of crops, return False if crop value is 0
            if self.harvester_agent.get_crops_value() == 0:
                return -0.2
            market_value = [4, 8, 12]
            
            # Calculate game score by multiplying market_value*yield value
            if self.verbose:
                print(f"Market: value={market_value[self.harvester_agent.get_crops_type()-1]} yield_value:{self.harvester_agent.get_crops_value()}")
            self.game_score += market_value[self.harvester_agent.get_crops_type()-1]*self.harvester_agent.get_crops_value()
            reward = market_value[self.harvester_agent.get_crops_type()-1]*self.harvester_agent.get_crops_value()

            # Clear the harvester agent
            self.harvester_agent.drop_crops()
            self.info["times_sold"] += 1
            return reward
        return -0.2

    def drop_crops_garbage(self):
        """ Drop the crops at garbage """

        # If the agent is carrying crops
        if self.harvester_agent.is_holding_crops():
            
            # check the value of crops, return False if crop value is not 0
            if self.harvester_agent.get_crops_value() != 0:
                return -0.2
            if self.verbose:
                print(f"Garbage: yield_value:{self.harvester_agent.get_crops_value()}")
            # positive reward for throwing dead crops to garbage
            self.harvester_agent.drop_crops()
            return 0.2
        return -0.2
    


    
    def pickup_water(self):
        """ Pickup water_units of water from the water tank"""

        # If the agent is already carrying water
        if self.water_agent.holding_water():
            return -0.2

        curr_cell = self.water_agent.get_position()

        # If the agent is at the water tank
        if self.is_water_tank(curr_cell):
            self.water_agent.collect_water()
            return 0.5

        return -0.2
    
    def water_crops(self):
        """ Water the crops in the plots"""

        # If the agent is not carrying water
        if not self.water_agent.holding_water():
            return -0.2
        
        curr_cell = self.water_agent.get_position()

        # If the agent is at the plot
        if self.is_plot_grid(curr_cell):
            # Water the plots
            plot_id = self.plot_dict[curr_cell]
            # If plot is already watered
            if self.plot_states[plot_id][0] == CropStages.NOT_PLANTED or self.plot_states[plot_id][2] == 1:
                return -0.2
            else:
                # Set water level to 1
                self.plot_states[plot_id][2] = 1
                self.dry_plots -= 1
                # Reset the dry days counter
                self.plot_states[plot_id][5] = 0
                # Reduce the number of water units held by the agent
                self.water_agent.reduce_water_units()
                reward = 1.0

                self.seeder_agent.reward += 0.5
                if self.dry_plots == 0:
                    reward = 1.5
                return reward
        
        return -0.2
    
    def step_seeder_agent(self, action):
        """ Function defines the set of actions executed by the seeder_agent """
        seeder_agent_actions = self.seeder_agent.actions
        

        # Four movement actions
        if action in seeder_agent_actions[:4]:
            if self.movement(self.seeder_agent, action):
                curr_cell = self.seeder_agent.get_position()
                # If there are NO empty_plots
                reward = -0.01
                if self.empty_plots == 0:
                    # if the agent is moving into its station, give reward
                    if self.is_station1(curr_cell):
                        reward = 0.01
                    # if the agent is moving into plots, give penalty
                    if self.is_plot_grid(curr_cell):
                        reward = -0.2
                # If there are empty_plots
                else:
                    # if the agent is moving into its station, give penalty
                    if self.is_station1(curr_cell):
                        reward = -0.2
                    # if the agent is moving into plots, give slight reward
                    # if self.is_plot_grid(curr_cell):
                    #     reward = 0.1
                    # Small Reward agent for moving into seed station
                    if self.is_seed_station_1(curr_cell):
                        reward = 0.01

                # Give penalty for moving into unrelated grids
                if self.is_market(curr_cell) or self.is_garbage(curr_cell) or self.is_station2(curr_cell) or self.is_station3(curr_cell) or self.is_water_tank(curr_cell):
                    reward = -0.2
                    
            else:
                reward = -0.2
        elif action == seeder_agent_actions[4]: # idle
            if self.is_station1(self.seeder_agent.get_position()) and self.empty_plots == 0:
                reward = 0.2
            else:
                reward = -0.2
        elif action == seeder_agent_actions[5]: # collect seeds
            reward = self.pickup_seeds()
        elif action == seeder_agent_actions[6]: # plant_seeds
            reward = self.plant_seeds()
        elif action == seeder_agent_actions[7]: # drop_seeds
            reward =  self.drop_seeds()

        new_distance = 0
        if self.empty_plots == 0:
            new_distance = self.manhatten_dist(self.seeder_agent.get_position(), self.STATION1_IDX)
        else:
            if self.seeder_agent.get_num_seeds() > 0:
                distances = [self.manhatten_dist(self.seeder_agent.get_position(), pos) for pos in self.plot_grids_indices]
                new_distance = min(distances)
            else:
                new_distance = self.manhatten_dist(self.seeder_agent.get_position(), self.SEEDSTN1_IDX)

        self.seeder_agent.update_distance_target(new_distance)
        self.seeder_agent.reward += reward - new_distance*0.01


    def step_water_agent(self, action):
        """ Function defines the set of actions executed by the water_agent """

        water_agent_actions = self.water_agent.actions

        # Four movement actions
        if action in water_agent_actions[:4]:
            # Movement actions
            if self.movement(self.water_agent, action):
                # Get the current pos of water agent
                curr_cell = self.water_agent.get_position()
                # If there are NO dry_plots
                if self.dry_plots == 0:
                    # the water agent can chill at its station
                    if self.is_station3(curr_cell):
                        reward = 0.01
                    # the water agent cannot move into plots blocking others
                    if self.is_plot_grid(curr_cell):
                        reward = -0.2
                # If there are dry_plots
                else:
                    # the water agent is penalized for moving into its station
                    if self.is_station3(curr_cell):
                        reward = -0.2
                    # the water agent is rewarded for moving int owater tank.
                    if self.is_water_tank(curr_cell):
                        reward = 0.01


                # Penalize for moving into unrelated grids
                if self.is_market(curr_cell) or self.is_garbage(curr_cell) or self.is_station1(curr_cell) or self.is_station2(curr_cell) or self.is_seed_station_1(curr_cell):
                    reward = -0.2
                else:
                    # a small movement penalty
                    reward = -0.01
            # Invalid action by moving out of the grid or agent positions
            else:
                reward = -0.2
        elif action == water_agent_actions[4]: # idle
            # Penalty for idling at important grids
            curr_cell = self.water_agent.get_position()
            # If there are NO dry_plots, the water agent can chill at its station
            if self.is_station3(curr_cell) and self.dry_plots == 0:
                reward = 0.2
            else:
                reward = -0.2
            
        elif action == water_agent_actions[5]: # collect_water
            reward = self.pickup_water()
        elif action == water_agent_actions[6]: # water_crops
            reward = self.water_crops()

        new_distance = 0
        # If there are zero dry_plots, calculate distance to its station
        if self.dry_plots == 0:
            new_distance = self.manhatten_dist(self.water_agent.get_position(), self.STATION3_IDX)
        else:
            # If agent is carrying water, calculate distance to plots
            if self.water_agent.holding_water():
                not_watered_plots_indices = []
                for plot_idx in self.plot_grids_indices:
                    if self.plot_states[self.plot_dict[plot_idx]][2] == 0:
                        not_watered_plots_indices.append(plot_idx)
                
                distances = [self.manhatten_dist(self.water_agent.get_position(), pos) for pos in not_watered_plots_indices]
                if distances:
                    new_distance = min(distances)

            else:
                # If agent is not carrying water, calculate distance to water tank
                new_distance = self.manhatten_dist(self.water_agent.get_position(), self.WATERTANK_IDX)

        self.water_agent.update_distance_target(new_distance)
        self.water_agent.reward += reward - new_distance*0.01


    def step_harvester_agent(self, action):
        """ Function defines the set of actions executed by the harvester_agent """

        harvester_agent_actions = self.harvester_agent.actions
        
        reward = -0.01

        # Movement actions
        if action in harvester_agent_actions[:4]:
            reward = -0.01
            if self.movement(self.harvester_agent, action):
                # Get the current pos of water agent
                curr_cell = self.harvester_agent.get_position()
                # If there are NO plots ready to harvest
                if self.harvest_ready_plots == 0:
                    # If the agent is NOT carrying crops
                    if self.harvester_agent.get_crops_type() == 0:
                        # The agent gets reward for moving into its station
                        if self.is_station2(curr_cell):
                            reward = 0.01
                        # The agent gets penalty for moving into plots
                        if self.is_plot_grid(curr_cell):
                            reward = -0.2
                    # If the agent is carrying crops
                    else:
                        # The agent gets penalty for moving into its station
                        if self.is_station2(curr_cell):
                            reward = -0.2
                        # The agent gets reward for moving into market or garbage
                        if self.is_market(curr_cell) or self.is_garbage(curr_cell):
                            reward = 0.01

                    # The agent gets penalty for moving into plots blocking others
                    if self.is_plot_grid(curr_cell):
                        reward = -0.2
                # If there are plots ready to harvest
                else:
                    # If the agent is NOT carrying crops
                    if self.harvester_agent.get_crops_type() == 0:
                        # the agent gets reward for moving into plots
                        # if self.is_plot_grid(curr_cell):
                        #     reward = 0.1
                        # The agent gets penalty for moving into market or garbage
                        if self.is_market(curr_cell) or self.is_garbage(curr_cell):
                            reward = -0.2
                    # If the agent is carrying crops
                    else:
                        # Penalty for moving into it's station
                        if self.is_station2(curr_cell):
                            reward = -0.2
                        # The agent gets reward for moving into market or garbage
                        if self.is_market(curr_cell) or self.is_garbage(curr_cell):
                            reward = 0.01

                # If the agent is moving into unrelated grid cells, penalize
                if self.is_station1(curr_cell) or self.is_station3(curr_cell) or self.is_seed_station_1(curr_cell) or self.is_seed_station_2(curr_cell) or self.is_seed_station_3(curr_cell) or self.is_water_tank(curr_cell):
                    reward = -0.2
            # Penalize for invalid movement out of grid
            else:
                reward = -0.2
        elif action == harvester_agent_actions[4]: # idle
            curr_cell = curr_cell = self.harvester_agent.get_position()
            # If there are NO plots to harvest, the agent gets reward for idling at its station
            if self.is_station2(curr_cell) and self.harvest_ready_plots == 0:
                reward = 0.2
            # Penalty for idling in grid
            # If there are plots to harvest, the agent gets penalty for idling at its station
            else:
                reward = -0.2
            
        elif action == harvester_agent_actions[5]: # harvest_crops
            if self.harvest_crops():
                self.seeder_agent.reward += 0.5
                self.water_agent.reward += 0.5
                reward = 1
            else:
                reward = -0.2
        elif action == harvester_agent_actions[6]: # drop_crops
            if self.is_market(self.harvester_agent.get_position()):
                reward = self.drop_crops_market()
                self.seeder_agent.reward += reward//2
                self.water_agent.reward += reward//2
                reward = reward
            elif self.is_garbage(self.harvester_agent.get_position()):
                reward = self.drop_crops_garbage()
                self.seeder_agent.reward += reward/2
                self.water_agent.reward += reward/2

        new_distance = 0
        if self.harvest_ready_plots == 0 and self.harvester_agent.get_crops_type() == 0:
            new_distance = self.manhatten_dist(self.harvester_agent.get_position(), self.STATION2_IDX)
        else:
            if self.harvester_agent.get_crops_type() > 0:
                if self.harvester_agent.get_crops_value() > 0:
                    new_distance = self.manhatten_dist(self.harvester_agent.get_position(), self.MARKET_IDX)
                else:
                    new_distance = self.manhatten_dist(self.harvester_agent.get_position(), self.GARBAGE_IDX)
            else:
                distances = [self.manhatten_dist(self.harvester_agent.get_position(), pos) for pos in self.plot_grids_indices]
                new_distance = min(distances)

        self.harvester_agent.update_distance_target(new_distance)
        

        self.harvester_agent.reward += reward-new_distance*0.01
        # return reward


    def get_coordinates(self, position):
        """Convert a grid cell index back to (row, col)."""

        row = position // self.num_columns
        col = position % self.num_columns
        return row, col
    
    def render(self):
        """Render the AgriEnve environment"""

        if not self.pygame_initialized:
            self.initialize_pygame()

        # Handling windowed actions
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        # Fill screen with white
        self.screen.fill(WHITE)

        # Draw the various grid cell ont othe screen
        for i in range(GRID_HEIGHT*GRID_WIDTH):
            cell_pos = self.get_coordinates(i)
            x, y = cell_pos[1] * TILE_SIZE, cell_pos[0] * TILE_SIZE
            if self.grid_state[i] == GridElements.GRASS:
                self.screen.blit(GRASS_IMG, (x, y))
            elif self.grid_state[i] == GridElements.STATION1:
                self.screen.blit(S_START_IMG, (x, y))
            elif self.grid_state[i] == GridElements.STATION2:
                self.screen.blit(H_START_IMG, (x, y))
            elif self.grid_state[i] == GridElements.STATION3:
                self.screen.blit(W_START_IMG, (x, y))
            elif self.grid_state[i] == GridElements.MARKET:
                self.screen.blit(MARKET_IMG, (x, y))
            elif self.grid_state[i] == GridElements.GARBAGE:
                self.screen.blit(GARBAGE_IMG, (x, y))
            elif self.grid_state[i] == GridElements.SEEDSTN1:
                self.screen.blit(SEEDSTN1_IMG, (x, y))
            elif self.grid_state[i] == GridElements.SEEDSTN2:
                self.screen.blit(SEEDSTN2_IMG, (x, y))
            elif self.grid_state[i] == GridElements.SEEDSTN3:
                self.screen.blit(SEEDSTN3_IMG, (x, y))
            elif self.grid_state[i] == GridElements.WATERTANK:
                self.screen.blit(WATER_TANK_IMG, (x, y))
            elif self.grid_state[i] == GridElements.PLOT:
                plot_state = self.plot_states[self.plot_dict[i]]
                img = PLOT_IMG
                if plot_state[0] == -1 and plot_state[1] == 0:
                    img = PLOT_IMG
                elif plot_state[0] == 0 and plot_state[1] == 1 and plot_state[2] == 0:
                    img = PLOT_WHEAT_0_IMG
                elif plot_state[0] == 0 and plot_state[1] == 1 and plot_state[2] == 1:
                    img = PLOT_WHEAT_0_W_IMG
                elif plot_state[0] == 1 and plot_state[1] == 1:
                    img = PLOT_WHEAT_1_IMG
                elif plot_state[0] == 2 and plot_state[1] == 1:
                    img = PLOT_WHEAT_2_IMG
                elif plot_state[0] == 3 and plot_state[1] == 1:
                    img = PLOT_WHEAT_3_IMG
                elif plot_state[0] == 0 and plot_state[1] == 2 and plot_state[2] == 0:
                    img = PLOT_RICE_0_IMG
                elif plot_state[0] == 0 and plot_state[1] == 2 and plot_state[2] == 1:
                    img = PLOT_RICE_0_W_IMG
                elif plot_state[0] == 1 and plot_state[1] == 2:
                    img = PLOT_RICE_1_IMG
                elif plot_state[0] == 2 and plot_state[1] == 2:
                    img = PLOT_RICE_2_IMG
                elif plot_state[0] == 3 and plot_state[1] == 2:
                    img = PLOT_RICE_3_IMG
                elif plot_state[0] == 0 and plot_state[1] == 3 and plot_state[2] == 0:
                    img = PLOT_CORN_0_IMG
                elif plot_state[0] == 0 and plot_state[1] == 3 and plot_state[2] == 1:
                    img = PLOT_CORN_0_W_IMG
                elif plot_state[0] == 1 and plot_state[1] == 3:
                    img = PLOT_CORN_1_IMG
                elif plot_state[0] == 2 and plot_state[1] == 3:
                    img = PLOT_CORN_2_IMG
                elif plot_state[0] == 3 and plot_state[1] == 3:
                    img = PLOT_CORN_2_IMG
                
                self.screen.blit(img, (x, y))
            
        # Draw the water agent
        water_agent_pos = self.get_coordinates(self.water_agent.get_position())  
        water_agent_x, water_agent_y = water_agent_pos[1] * TILE_SIZE, water_agent_pos[0] * TILE_SIZE
        if self.water_agent.holding_water():
            self.screen.blit(WATER_AGENT_FILL_IMG, ( water_agent_x,  water_agent_y))
        else:
            self.screen.blit(WATER_AGENT_IMG, ( water_agent_x,  water_agent_y))
        
        # Draw the Seeder agent
        seeder_agent_pos = self.get_coordinates(self.seeder_agent.get_position())
        seeder_agent_x, seeder_agent_y = seeder_agent_pos[1] * TILE_SIZE, seeder_agent_pos[0] * TILE_SIZE
        seeder_agent_crops_type = self.seeder_agent.get_seed_type()
        if seeder_agent_crops_type == 0:
            self.screen.blit(SEEDER_AGENT_IMG, ( seeder_agent_x,  seeder_agent_y))
        elif seeder_agent_crops_type == 1:
            self.screen.blit(SEEDER_AGENT_WHEAT_IMG, ( seeder_agent_x,  seeder_agent_y))
        elif seeder_agent_crops_type == 2:
            self.screen.blit(SEEDER_AGENT_RICE_IMG, ( seeder_agent_x,  seeder_agent_y))
        elif seeder_agent_crops_type == 3:
            self.screen.blit(SEEDER_AGENT_CORN_IMG, ( seeder_agent_x,  seeder_agent_y))

        # Draw the Harvester agent
        harvester_agent_pos = self.get_coordinates(self.harvester_agent.get_position())
        harvester_agent_x, harvester_agent_y = harvester_agent_pos[1] * TILE_SIZE, harvester_agent_pos[0] * TILE_SIZE
        harvester_agent_crops_type = self.harvester_agent.get_crops_type()
        if harvester_agent_crops_type == 0:
            self.screen.blit(HARVESTER_AGENT_IMG, ( harvester_agent_x,  harvester_agent_y))
        elif harvester_agent_crops_type == 1:
            self.screen.blit(HARVESTER_AGENT_WHEAT_IMG, ( harvester_agent_x,  harvester_agent_y))
        elif harvester_agent_crops_type == 2:
            self.screen.blit(HARVESTER_AGENT_RICE_IMG, ( harvester_agent_x,  harvester_agent_y))
        elif harvester_agent_crops_type == 3:
            self.screen.blit(HARVESTER_AGENT_CORN_IMG, ( harvester_agent_x,  harvester_agent_y))

        # Display the current day on the screen
        font = pygame.font.Font(None, 20)
        text = font.render(f'Day: {self.current_day}', True, BLACK)
        self.screen.blit(text, (10, 10))

        # Display the current game score on the screen
        text2 = font.render(f'Game Score: {self.game_score}', True, BLACK)
        self.screen.blit(text2, (170, 10))

        # Display the agent's actions on the screen
        fontXS = pygame.font.Font(None, 15)
        s_act_text = fontXS.render(f'{self.seeder_did_what}', True, BLACK)
        h_act_text = fontXS.render(f'{self.harvester_did_what}', True, BLACK)
        w_act_text = fontXS.render(f'{self.waterer_did_what}', True, BLACK)
        self.screen.blit(s_act_text, (seeder_agent_x, seeder_agent_y+2))
        self.screen.blit(h_act_text, (harvester_agent_x, harvester_agent_y+2))
        self.screen.blit(w_act_text, (water_agent_x, water_agent_y+2))

        # Update the pygame screen
        pygame.display.update()

        return pygame.surfarray.array3d(self.screen).transpose([1, 0, 2])

    def close(self):
        """ Close the pygame window"""
        
        pygame.quit()