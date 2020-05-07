import random
import json
import numpy as np
import gym

from colorama import Back, Style
from stable_baselines.common.env_checker import check_env

from game import Game


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


class PacmanEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    keys = {0: 'w', 1: 'a', 2: 's', 3: 'd'}

    info_keywords = ('step', 'score', 'lives')

    def __init__(self, agent_name, mapfile, ghosts, level_ghosts, lives, timeout):

        self.agent_name = agent_name

        self._game = Game(mapfile, ghosts, level_ghosts, lives, timeout)

        self._pacman_obs = PacmanObservation(self._game.map)

        self.observation_space = self._pacman_obs.get_space()

        self.action_space = gym.spaces.Discrete(len(self.keys))

        self._current_score = 0

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError("Received invalid action={} which is not part of the action space.".format(action))

        self._game.keypress(self.keys[action])

        self._game.compute_next_frame()

        game_state = json.loads(self._game.state)

        reward = game_state['score'] - self._current_score

        self._current_score = game_state['score']

        info = {k: game_state[k] for k in self.info_keywords if k in game_state}

        done = not self._game.running

        return self._pacman_obs.get_obs(game_state), reward, done, info

    def reset(self):
        self._current_score = 0
        self._game.start(self.agent_name)
        self._game.compute_next_frame()
        game_state = json.loads(self._game.state)
        return self._pacman_obs.get_obs(game_state)

    def render(self, mode='human'):
        for y in range(self._game.map.ver_tiles):
            for x in range(self._game.map.hor_tiles):
                color = None
                value = self._pacman_obs.current_obs[0][y][x]

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


def main():
    """
    Testing gym pacman enviorment.
    """

    agent_name = "GymEnvTestAgent"
    mapfile = "data/map1.bmp"
    ghosts = 1
    level_ghosts = 1
    lives = 3
    timeout = 3000

    env = PacmanEnv(agent_name, mapfile, ghosts, level_ghosts, lives, timeout)
    print("Checking environment...")
    check_env(env, warn=True)

    print("\nObservation space:", env.observation_space)
    print("Shape:", env.observation_space.shape)
    # print("Observation space high:", env.observation_space.high)
    # print("Observation space low:", env.observation_space.low)

    print("Action space:", env.action_space)

    obs = env.reset()
    done = False

    action = 1  # a
    cur_x, cur_y = None, None

    while not done:
        env.render()

        y_arr, x_arr = np.where(obs[0] == PacmanObservation.PACMAN)
        y, x = y_arr[0], x_arr[0]

        # Using agent from client example
        if x == cur_x and y == cur_y:
            if action in [1, 3]:    # ad
                action = random.choice([0, 2])
            elif action in [0, 2]:  # ws
                action = random.choice([1, 3])
        cur_x, cur_y = x, y

        print("key:", PacmanEnv.keys[action])

        obs, reward, done, info = env.step(action)

        print("reward:", reward)
        print("info:", info)
        print()

        # # Stop game for debugging

        # if reward > 0:
        #     env.render()
        #     print("Received positive reward.")
        #     break

        # if np.isin(PacmanObservation.GHOST_ZOMBIE, obs[0]):
        #     env.render()
        #     print("Zombie")
        #     break


if __name__ == "__main__":
    main()
