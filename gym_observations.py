from abc import ABC, abstractmethod
import numpy as np
import gym

from colorama import Back, Style


class PacmanObservation(ABC):

    def __init__(self, game_map, max_lives):
        self._map = game_map
        self._max_lives = max_lives
        self._shape = None

        self.center_x, self.center_y = int(self._map.hor_tiles / 2), int(self._map.ver_tiles / 2)

        self.pac_x, self.pac_y = None, None

        self.walls = []

        for y in range(self._map.ver_tiles):
            for x in range(self._map.hor_tiles):
                if self._map.is_wall((x, y)):
                    self.walls.append((x, y))

    def _new_x(self, x):
        return ((x - self.pac_x) + self.center_x) % self._map.hor_tiles

    def _new_y(self, y):
        return ((y - self.pac_y) + self.center_y) % self._map.ver_tiles

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

    ENERGY_IN = 64
    BOOST_IN = 255
    ENERGY_EMPTY = 0

    # Channels Index
    WALL_CH = 0
    EMPTY_CH = 1
    ENERGY_CH = 2
    GHOST_CH = 3
    ZOMBIE_CH = 4

    def __init__(self, game_map, max_lives):
        super().__init__(game_map, max_lives)

        self._shape = (5, self._map.ver_tiles, self._map.hor_tiles)

        self._obs = np.full(self._shape, self.PIXEL_EMPTY, dtype=np.uint8)

    def get_obs(self, game_state):

        # Reset channels
        self._obs[self.EMPTY_CH][...] = self.PIXEL_IN
        self._obs[self.EMPTY_CH][self.center_y][self.center_x] = self.PIXEL_EMPTY
        self._obs[self.WALL_CH][...] = self.PIXEL_EMPTY
        self._obs[self.ENERGY_CH][...] = self.ENERGY_EMPTY
        self._obs[self.GHOST_CH][...] = self.PIXEL_EMPTY
        self._obs[self.ZOMBIE_CH][...] = self.PIXEL_EMPTY

        self.pac_x, self.pac_y = game_state['pacman']

        for x, y in self.walls:
            x, y = self._new_x(x), self._new_y(y)
            self._obs[self.WALL_CH][y][x] = self.PIXEL_IN
            self._obs[self.EMPTY_CH][y][x] = self.PIXEL_EMPTY

        for x, y in game_state['energy']:
            x, y = self._new_x(x), self._new_y(y)
            self._obs[self.ENERGY_CH][y][x] = self.ENERGY_IN
            self._obs[self.EMPTY_CH][y][x] = self.PIXEL_EMPTY

        for x, y in game_state['boost']:
            x, y = self._new_x(x), self._new_y(y)
            self._obs[self.ENERGY_CH][y][x] = self.BOOST_IN
            self._obs[self.EMPTY_CH][y][x] = self.PIXEL_EMPTY

        for ghost in game_state['ghosts']:
            x, y = ghost[0]
            x, y = self._new_x(x), self._new_y(y)

            if ghost[1]:
                self._obs[self.ZOMBIE_CH][y][x] = self.PIXEL_IN
            else:
                self._obs[self.GHOST_CH][y][x] = self.PIXEL_IN

            self._obs[self.EMPTY_CH][y][x] = self.PIXEL_EMPTY

        return self._obs

    def render(self):
        for y in range(self._map.ver_tiles):
            for x in range(self._map.hor_tiles):
                color = None

                if self._obs[self.GHOST_CH][y][x] == self.PIXEL_IN:
                    color = Back.MAGENTA
                elif self._obs[self.ZOMBIE_CH][y][x] == self.PIXEL_IN:
                    color = Back.BLUE
                elif self._obs[self.ENERGY_CH][y][x] == self.BOOST_IN:
                    color = Back.CYAN
                elif self._obs[self.ENERGY_CH][y][x] == self.ENERGY_IN:
                    color = Back.RED
                elif self._obs[self.WALL_CH][y][x] == self.PIXEL_IN:
                    color = Back.WHITE
                elif self._obs[self.EMPTY_CH][y][x] == self.PIXEL_IN:
                    color = Back.BLACK
                else:
                    color = Back.YELLOW

                print(color, ' ', end='')
            print(Style.RESET_ALL)

        # np.set_printoptions(edgeitems=30, linewidth=100000)
        # print(self._obs)


class SingleChannelObs(PacmanObservation):

    GHOST = 0
    WALL = 51
    EMPTY = 102
    ENERGY = 153
    BOOST = 204
    GHOST_ZOMBIE = 255

    def __init__(self, game_map, max_lives):
        super().__init__(game_map, max_lives)

        # First dimension is for the image channels required by tf.nn.conv2d
        self._shape = (1, self._map.ver_tiles, self._map.hor_tiles)

        self._obs = np.full(self._shape, self.EMPTY, dtype=np.uint8)

    def get_obs(self, game_state):

        self._obs[0][...] = self.EMPTY

        self.pac_x, self.pac_y = game_state['pacman']

        for x, y in self.walls:
            x, y = self._new_x(x), self._new_y(y)
            self._obs[0][y][x] = self.WALL

        for x, y in game_state['energy']:
            x, y = self._new_x(x), self._new_y(y)
            self._obs[0][y][x] = self.ENERGY

        for x, y in game_state['boost']:
            x, y = self._new_x(x), self._new_y(y)
            self._obs[0][y][x] = self.BOOST

        for ghost in game_state['ghosts']:
            x, y = ghost[0]
            x, y = self._new_x(x), self._new_y(y)

            if ghost[1]:
                self._obs[0][y][x] = self.GHOST_ZOMBIE
            else:
                self._obs[0][y][x] = self.GHOST

        return self._obs

    def render(self):
        for y in range(self._map.ver_tiles):
            for x in range(self._map.hor_tiles):
                color = None
                value = self._obs[0][y][x]

                if value == self.GHOST:
                    color = Back.MAGENTA
                elif value == self.GHOST_ZOMBIE:
                    color = Back.BLUE
                elif value == self.EMPTY:
                    color = Back.BLACK

                    if x == self.center_x and y == self.center_y:
                        color = Back.YELLOW

                elif value == self.WALL:
                    color = Back.WHITE
                elif value == self.ENERGY:
                    color = Back.RED
                elif value == self.BOOST:
                    color = Back.CYAN

                print(color, ' ', end='')
            print(Style.RESET_ALL)

        # np.set_printoptions(edgeitems=30, linewidth=100000)
        # print(self._obs)
