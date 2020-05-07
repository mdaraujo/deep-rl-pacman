import numpy as np
import gym

from colorama import Back, Style


class PacmanObservation:

    GHOST = 0
    WALL = 42
    EMPTY = 84
    ENERGY = 126
    BOOST = 168
    GHOST_ZOMBIE = 210
    PACMAN = 255

    def __init__(self, game_map):
        self.map = game_map

        # First dimension is for the image channels required by tf.nn.conv2d
        self.shape = (1, self.map.ver_tiles, self.map.hor_tiles)

        self.base_obs = np.full(self.shape, self.EMPTY, dtype=np.uint8)

        for y in range(self.map.ver_tiles):
            for x in range(self.map.hor_tiles):
                if self.map.is_wall((x, y)):
                    self.base_obs[0][y][x] = self.WALL

        self.current_obs = None

    def get_obs(self, game_state):

        self.current_obs = np.copy(self.base_obs)

        for x, y in game_state['energy']:
            self.current_obs[0][y][x] = self.ENERGY

        for x, y in game_state['boost']:
            self.current_obs[0][y][x] = self.BOOST

        for ghost in game_state['ghosts']:
            x, y = ghost[0]

            if ghost[1]:
                self.current_obs[0][y][x] = self.GHOST_ZOMBIE
            else:
                self.current_obs[0][y][x] = self.GHOST

        x, y = game_state['pacman']
        self.current_obs[0][y][x] = self.PACMAN

        return self.current_obs

    def get_space(self):
        return gym.spaces.Box(low=0, high=255, shape=self.shape, dtype=np.uint8)

    def render(self):
        for y in range(self.map.ver_tiles):
            for x in range(self.map.hor_tiles):
                color = None
                value = self.current_obs[0][y][x]

                if value == PacmanObservation.PACMAN:
                    color = Back.YELLOW
                elif value == PacmanObservation.GHOST:
                    color = Back.MAGENTA
                elif value == PacmanObservation.GHOST_ZOMBIE:
                    color = Back.BLUE
                elif value == PacmanObservation.EMPTY:
                    color = Back.BLACK
                elif value == PacmanObservation.WALL:
                    color = Back.WHITE
                elif value == PacmanObservation.ENERGY:
                    color = Back.RED
                elif value == PacmanObservation.BOOST:
                    color = Back.CYAN

                print(color, ' ', end='')
            print(Style.RESET_ALL)
