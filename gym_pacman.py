import random
import json
import gym

from stable_baselines.common.env_checker import check_env

from game import Game, POINT_ENERGY, TIME_BONUS_STEPS
from gym_observations import SingleChannelObs, MultiChannelObs


class PacmanEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    keys = {0: 'w', 1: 'a', 2: 's', 3: 'd'}

    info_keywords = ('step', 'score', 'lives', 'ghosts')

    MAX_ENERGY_REWARD = 1.5
    MIN_ENERGY_REWARD = 0.5

    INITIAL_DIFFICULTY = 0
    MIN_WINS = 200

    def __init__(self, obs_type, positive_rewards, agent_name, mapfile,
                 max_ghosts, level_ghosts, lives, timeout, ghosts_rnd=True):

        self.positive_rewards = positive_rewards

        self.agent_name = agent_name

        self.max_ghosts = max_ghosts

        self.ghosts_rnd = ghosts_rnd

        self._game = Game(mapfile, self.max_ghosts, level_ghosts, lives, timeout)

        self._pacman_obs = obs_type(self._game.map, lives)

        self.observation_space = self._pacman_obs.space

        self.action_space = gym.spaces.Discrete(len(self.keys))

        self._current_score = 0

        self.current_lives = self._game._initial_lives

        self.total_energy = len(self._game.map.energy)

        self.energy_reward_increment = (self.MAX_ENERGY_REWARD - self.MIN_ENERGY_REWARD) / (self.total_energy - 1)

        self._current_energy_reward = self.MIN_ENERGY_REWARD

        self.difficulty = self.INITIAL_DIFFICULTY

        self.wins_count = 0

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError("Received invalid action={} which is not part of the action space.".format(action))

        self._game.keypress(self.keys[action])

        self._game.compute_next_frame()

        game_state = json.loads(self._game.state)

        info = {k: game_state[k] for k in self.info_keywords if k in game_state}
        info['ghosts'] = len(info['ghosts'])
        info['win'] = 0
        info['d'] = self.difficulty

        done = not self._game.running

        if game_state['lives'] < self.current_lives:
            self.current_lives = game_state['lives']

        reward = game_state['score'] - self._current_score

        # if reward == POINT_ENERGY:
        #     reward = self._current_energy_reward
        #     self._current_energy_reward += self.energy_reward_increment
        #     # print(self.total_energy - len(game_state['energy']), reward)

        self._current_score = game_state['score']

        if not self.positive_rewards:

            if game_state['lives'] == 0:
                # reward -= (self._game._timeout - game_state['step'] + 1) * (1.0 / TIME_BONUS_STEPS)
                reward -= 120

            reward -= 0.05

        if done:
            if self._game._timeout != game_state['step'] and game_state['lives'] > 0:
                info['win'] = 1

            if info['ghosts'] == self.difficulty:
                self.wins_count += info['win']

        if not done and info['ghosts'] == 0 and game_state['step'] >= 500:
            self._game.stop()
            done = True

            if info['ghosts'] == self.difficulty:
                self.wins_count += info['win']

        return self._pacman_obs.get_obs(game_state), reward, done, info

    def reset(self):
        self._current_score = 0
        self._current_energy_reward = self.MIN_ENERGY_REWARD

        if self.ghosts_rnd:

            if self.difficulty < self.max_ghosts and self.wins_count >= self.MIN_WINS:
                self.difficulty += 1
                self.wins_count = 0

            easier_episode_prob = self.difficulty / 20

            if random.random() < easier_episode_prob:
                if self.difficulty - 1 > self.INITIAL_DIFFICULTY:
                    n_ghosts = random.randint(self.INITIAL_DIFFICULTY, self.difficulty - 1)
                else:
                    n_ghosts = self.INITIAL_DIFFICULTY
            else:
                n_ghosts = self.difficulty

            self.set_n_ghosts(n_ghosts)

        self._game.start(self.agent_name)
        self._game.compute_next_frame()
        self.current_lives = self._game._initial_lives
        game_state = json.loads(self._game.state)
        return self._pacman_obs.get_obs(game_state)

    def set_n_ghosts(self, n_ghosts):
        self._game._n_ghosts = n_ghosts

    def render(self, mode='human'):
        self._pacman_obs.render()


def main():
    """
    Testing gym pacman enviorment.
    """

    agent_name = "GymEnvTestAgent"
    mapfile = "data/map1.bmp"
    ghosts = 4
    level_ghosts = 1
    lives = 3
    timeout = 3000

    obs_type = MultiChannelObs

    positive_rewards = False

    env = PacmanEnv(obs_type, positive_rewards, agent_name, mapfile, ghosts, level_ghosts, lives, timeout)
    print("Checking environment...")
    check_env(env, warn=True)

    print("\nObservation space:", env.observation_space)
    print("Shape:", env.observation_space.shape)
    # print("Observation space high:", env.observation_space.high)
    # print("Observation space low:", env.observation_space.low)

    print("Action space:", env.action_space)

    obs = env.reset()
    done = False

    sum_rewards = 0
    action = 1  # a
    cur_x, cur_y = None, None

    while not done:
        env.render()

        x, y = env._game._pacman

        # Using agent from client example
        if x == cur_x and y == cur_y:
            if action in [1, 3]:    # ad
                action = random.choice([0, 2])
            elif action in [0, 2]:  # ws
                action = random.choice([1, 3])
        cur_x, cur_y = x, y

        print("key:", PacmanEnv.keys[action])

        obs, reward, done, info = env.step(action)

        sum_rewards += reward

        print("reward:", reward)
        print("sum_rewards:", sum_rewards)
        print("info:", info)
        print()

        # # Stop game for debugging

        # if reward > 0:
        #     env.render()
        #     print("Received positive reward.")
        #     break

        # if (obs_type == SingleChannelObs and np.isin(SingleChannelObs.GHOST_ZOMBIE, obs[0])) \
        #         or (obs_type == MultiChannelObs and
        #             np.isin(MultiChannelObs.PIXEL_IN, obs[MultiChannelObs.ZOMBIE_CH])):
        #     env.render()
        #     print("Zombie")
        #     break

    # print("score:", sum_rewards + (env._game._timeout / TIME_BONUS_STEPS))


if __name__ == "__main__":
    main()
