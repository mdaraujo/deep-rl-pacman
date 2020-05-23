from abc import ABC, abstractmethod
import numpy as np
import gym

from colorama import Back, Style


class PacmanObservation(ABC):

    def __init__(self, game_map):
        self._map = game_map
        self._shape = None

    @property
    def space(self):
        return gym.spaces.Box(low=0, high=255, shape=self._shape, dtype=np.uint8)

    @abstractmethod
    def get_obs(self, game_state):
        pass

    @abstractmethod
    def render(self):
        pass


class MultiChannelObs(PacmanObservation):

    PIXEL_IN = 255
    PIXEL_EMPTY = 0

    GHOST_IN = 64
    GHOST_ZOMBIE = 255
    GHOST_EMPTY = 0

    ENERGY_IN = 64
    BOOST_IN = 255
    ENERGY_EMPTY = 0

    # Channels Index
    WALL_CH = 0
    EMPTY_CH = 1
    ENERGY_CH = 2
    GHOST_CH = 3
    PACMAN_CH = 4

    def __init__(self, game_map):
        super().__init__(game_map)

        self._shape = (5, self._map.ver_tiles, self._map.hor_tiles)

        self._obs = np.full(self._shape, self.PIXEL_EMPTY, dtype=np.uint8)

        for y in range(self._map.ver_tiles):
            for x in range(self._map.hor_tiles):
                if self._map.is_wall((x, y)):
                    self._obs[self.WALL_CH][y][x] = self.PIXEL_IN

    def get_obs(self, game_state):

        # Reset channels
        for y in range(self._map.ver_tiles):
            for x in range(self._map.hor_tiles):

                self._obs[self.ENERGY_CH][y][x] = self.ENERGY_EMPTY
                self._obs[self.GHOST_CH][y][x] = self.GHOST_EMPTY
                self._obs[self.PACMAN_CH][y][x] = self.PIXEL_EMPTY

                if self._obs[self.WALL_CH][y][x] == self.PIXEL_IN:
                    self._obs[self.EMPTY_CH][y][x] = self.PIXEL_EMPTY
                else:
                    self._obs[self.EMPTY_CH][y][x] = self.PIXEL_IN

        for x, y in game_state['energy']:
            self._obs[self.ENERGY_CH][y][x] = self.ENERGY_IN
            self._obs[self.EMPTY_CH][y][x] = self.PIXEL_EMPTY

        for x, y in game_state['boost']:
            self._obs[self.ENERGY_CH][y][x] = self.BOOST_IN
            self._obs[self.EMPTY_CH][y][x] = self.PIXEL_EMPTY

        for ghost in game_state['ghosts']:
            x, y = ghost[0]

            if ghost[1]:
                self._obs[self.GHOST_CH][y][x] = self.GHOST_ZOMBIE
            else:
                self._obs[self.GHOST_CH][y][x] = self.GHOST_IN
            self._obs[self.EMPTY_CH][y][x] = self.PIXEL_EMPTY

        x, y = game_state['pacman']
        self._obs[self.PACMAN_CH][y][x] = self.PIXEL_IN
        self._obs[self.EMPTY_CH][y][x] = self.PIXEL_EMPTY

        return self._obs

    def render(self):
        for y in range(self._map.ver_tiles):
            for x in range(self._map.hor_tiles):
                color = None

                if self._obs[self.PACMAN_CH][y][x] == self.PIXEL_IN:
                    color = Back.YELLOW
                elif self._obs[self.GHOST_CH][y][x] == self.GHOST_IN:
                    color = Back.MAGENTA
                elif self._obs[self.GHOST_CH][y][x] == self.GHOST_ZOMBIE:
                    color = Back.BLUE
                elif self._obs[self.ENERGY_CH][y][x] == self.BOOST_IN:
                    color = Back.CYAN
                elif self._obs[self.ENERGY_CH][y][x] == self.ENERGY_IN:
                    color = Back.RED
                elif self._obs[self.WALL_CH][y][x] == self.PIXEL_IN:
                    color = Back.WHITE
                elif self._obs[self.EMPTY_CH][y][x] == self.PIXEL_IN:
                    color = Back.BLACK

                print(color, ' ', end='')
            print(Style.RESET_ALL)

        # np.set_printoptions(edgeitems=30, linewidth=100000)
        # print(self._obs)


class SingleChannelObs(PacmanObservation):

    GHOST = 0
    WALL = 42
    EMPTY = 84
    ENERGY = 126
    BOOST = 168
    GHOST_ZOMBIE = 210
    PACMAN = 255

    def __init__(self, game_map):
        super().__init__(game_map)

        # First dimension is for the image channels required by tf.nn.conv2d
        self._shape = (1, self._map.ver_tiles, self._map.hor_tiles)

        self.base_obs = np.full(self._shape, self.EMPTY, dtype=np.uint8)

        for y in range(self._map.ver_tiles):
            for x in range(self._map.hor_tiles):
                if self._map.is_wall((x, y)):
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

    def render(self):
        for y in range(self._map.ver_tiles):
            for x in range(self._map.hor_tiles):
                color = None
                value = self.current_obs[0][y][x]

                if value == self.PACMAN:
                    color = Back.YELLOW
                elif value == self.GHOST:
                    color = Back.MAGENTA
                elif value == self.GHOST_ZOMBIE:
                    color = Back.BLUE
                elif value == self.EMPTY:
                    color = Back.BLACK
                elif value == self.WALL:
                    color = Back.WHITE
                elif value == self.ENERGY:
                    color = Back.RED
                elif value == self.BOOST:
                    color = Back.CYAN

                print(color, ' ', end='')
            print(Style.RESET_ALL)
