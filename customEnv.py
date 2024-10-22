import pygame
import random
import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Initialize PyGame
pygame.init()

# Screen dimensions
TILE_SIZE = 75
GRID_WIDTH, GRID_HEIGHT = 9, 8
SCREEN_WIDTH, SCREEN_HEIGHT = GRID_WIDTH * TILE_SIZE, GRID_HEIGHT * TILE_SIZE

START_IMG = pygame.transform.scale(pygame.image.load('assets/grid_cells/start.png'), (TILE_SIZE, TILE_SIZE))
MARKET_IMG = pygame.transform.scale(pygame.image.load('assets/grid_cells/market.png'), (TILE_SIZE, TILE_SIZE))
PLOT_IMG = pygame.transform.scale(pygame.image.load('assets/grid_cells/plot.png'), (TILE_SIZE, TILE_SIZE))
CROP1_IMG = pygame.transform.scale(pygame.image.load('assets/grid_cells/crop1.png'), (TILE_SIZE, TILE_SIZE))
CROP2_IMG = pygame.transform.scale(pygame.image.load('assets/grid_cells/crop2.png'), (TILE_SIZE, TILE_SIZE))
CROP3_IMG = pygame.transform.scale(pygame.image.load('assets/grid_cells/crop3.png'), (TILE_SIZE, TILE_SIZE))
GRASS_IMG = pygame.transform.scale(pygame.image.load('assets/grid_cells/grass.png'), (TILE_SIZE, TILE_SIZE))
WATER_IMG = pygame.transform.scale(pygame.image.load('assets/grid_cells/water.png'), (TILE_SIZE, TILE_SIZE))


# Agent settings
AGENT_SIZE = TILE_SIZE
AGENT_SPEED = TILE_SIZE  # Move one tile at a time

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Actions (Up, Down, Left, Right)
ACTIONS = {
    0: (-AGENT_SPEED, 0),  # Move left
    1: (AGENT_SPEED, 0),   # Move right
    2: (0, -AGENT_SPEED),  # Move up
    3: (0, AGENT_SPEED)    # Move down
}

DAYS_IN_YEAR = 365

# Custom RL Environment Class
class CustomEnv:
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0: Left, 1: Right, 2: Up, 3: Down
        self.action_space = spaces.Discrete(4)

        # Observation space: Agent's position (x, y)
        self.observation_space = spaces.Box(low=0, high=GRID_WIDTH - 1, shape=(2,), dtype=np.int32)

        self.agent_pos = [0, 0]  # Start position at top-left corner
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

        # Place two groups of 2x6 plots
        plot_grids = [[3, 1],[4,1],[3,2],[4,2],[3,3],[4,3], [3, 5],[4,5],[3,6],[4,6],[3,7],[4,7]]
        for plot_grid in plot_grids:  # Rows 2 and 3
            self.grid[plot_grid[0]][plot_grid[1]] = 'plot'

        self.grid[7][1] = 'crop1'
        self.grid[7][3] = 'crop2'
        self.grid[7][5] = 'crop3'
        self.grid[7][7] = 'water'
        
        
        self.current_day = 0

        self.done = False
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('9x8 Grid Environment')

    def reset(self):
        self.agent_pos = [0, 0]  # Start position at top-left
        self.done = False
        return np.array(self.agent_pos)

    def step(self, action):
        # Move agent
        self.agent_pos[0] += ACTIONS[action][0] // TILE_SIZE
        self.agent_pos[1] += ACTIONS[action][1] // TILE_SIZE

        # Ensure the agent stays within bounds
        self.agent_pos[0] = max(0, min(self.agent_pos[0], GRID_WIDTH - 1))
        self.agent_pos[1] = max(0, min(self.agent_pos[1], GRID_HEIGHT - 1))

        reward = -1  # Default step penalty
        # Check for interactions (e.g., reaching market)
        if self.grid[self.agent_pos[1]][self.agent_pos[0]] == 'market':
            reward = 100  # Reward for reaching the market
            self.done = True

        return np.array(self.agent_pos), reward, self.done, {}

    def render(self):
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
                elif tile == 'crop1':
                    self.screen.blit(CROP1_IMG, (x, y))
                elif tile == 'crop2':
                    self.screen.blit(CROP2_IMG, (x, y))
                elif tile == 'crop3':
                    self.screen.blit(CROP3_IMG, (x, y))
                elif tile == 'water':
                    self.screen.blit(WATER_IMG, (x, y))
                else:
                    self.screen.blit(GRASS_IMG, (x, y))

        # Draw agent (for simplicity, representing as a black square)
        pygame.draw.rect(self.screen, BLACK, (self.agent_pos[0] * TILE_SIZE, self.agent_pos[1] * TILE_SIZE, AGENT_SIZE, AGENT_SIZE))

        font = pygame.font.Font(None, 36)
        text = font.render(f'Day: {self.current_day}', True, BLACK)
        self.screen.blit(text, (10, 10))

        pygame.display.update()

    def close(self):
        pygame.quit()
