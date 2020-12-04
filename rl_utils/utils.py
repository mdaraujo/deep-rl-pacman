import warnings
import os
import logging
import datetime
import csv
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines import PPO2, DQN
from stable_baselines.common.vec_env import VecEnv, DummyVecEnv
from stable_baselines.bench import Monitor

from gym_pacman import ENV_PARAMS

FIG_SIZE = (8.5, 5)

plt.rc('font', size=11)

EVAL_HEADER = ["TrainStep", "MeanScore", "StdScore", "MaxScore", "MinScore",
               "MeanReturn", "StdReturn", "MaxReturn", "MinReturn",
               "MeanLength", "StdLength", "MaxLength", "MinLength",
               "Wins", "EvalEpisodes",
               "EvalIdleStepsMean", "EvalIdleEps",
               "EvalGhostsMean", "EvalLevelMean", "EvalTime",
               "TrainGhostsMean", "TrainLevelMean"]


def evaluate_policy(model, env, n_eval_episodes=10, deterministic=True, fixed_params=False, render=False):
    """
    Runs policy for `n_eval_episodes` episodes and returns episodes returns, lengths and scores.
    This is made to work only with one env.

    Based on stable-baselines.common.evaluate_policy method
    """
    if isinstance(env, VecEnv):
        assert env.num_envs == 1, "You must pass only one environment when using this function"

    returns, lengths, scores, idle_steps, ghosts, levels = [], [], [], [], [], []

    n_wins = 0
    params_idx = 0
    curr_params_count = 0
    total_count = 0

    if not fixed_params:
        env.set_env_params(ENV_PARAMS[params_idx])

    while True:

        if not fixed_params:

            if curr_params_count >= ENV_PARAMS[params_idx].test_runs:
                curr_params_count = 0
                params_idx += 1

                if params_idx >= len(ENV_PARAMS):
                    break

            env.set_env_params(ENV_PARAMS[params_idx])

        elif total_count >= n_eval_episodes:
            break

        obs = env.reset()
        done, state = False, None
        episode_return = 0.0
        length = 0
        while not done:
            action, state = model.predict(obs, state=state, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            episode_return += reward
            length += 1
            if render:
                env.render()
        returns.append(episode_return)
        lengths.append(length)
        scores.append(info['score'])
        ghosts.append(info['ghosts'])
        levels.append(info['level'])
        idle_steps.append(info['idle'])
        n_wins += info['win']
        curr_params_count += 1
        total_count += 1

    return returns, lengths, scores, idle_steps, ghosts, levels, n_wins, total_count


def make_vec_env(pacman_env, n_envs=1, seed=None, start_index=0,
                 monitor_dir=None, wrapper_class=None,
                 env_kwargs=None, vec_env_cls=None, vec_env_kwargs=None):
    """
    Create a wrapped, monitored `VecEnv`.
    By default it uses a `DummyVecEnv` which is usually faster
    than a `SubprocVecEnv`.

    :param env_id: (str or Type[gym.Env]) the environment ID or the environment class
    :param n_envs: (int) the number of environments you wish to have in parallel
    :param seed: (int) the initial seed for the random number generator
    :param start_index: (int) start rank index
    :param monitor_dir: (str) Path to a folder where the monitor files will be saved.
        If None, no file will be written, however, the env will still be wrapped
        in a Monitor wrapper to provide additional information about training.
    :param wrapper_class: (gym.Wrapper or callable) Additional wrapper to use on the environment.
        This can also be a function with single argument that wraps the environment in many things.
    :param env_kwargs: (dict) Optional keyword argument to pass to the env constructor
    :param vec_env_cls: (Type[VecEnv]) A custom `VecEnv` class constructor. Default: None.
    :param vec_env_kwargs: (dict) Keyword arguments to pass to the `VecEnv` class constructor.
    :return: (VecEnv) The wrapped environment
    """
    env_kwargs = {} if env_kwargs is None else env_kwargs
    vec_env_kwargs = {} if vec_env_kwargs is None else vec_env_kwargs

    def make_env(rank):
        def _init():
            env = pacman_env

            if seed is not None:
                env.seed(seed + rank)
                env.action_space.seed(seed + rank)
            # Wrap the env in a Monitor wrapper
            # to have additional training information
            monitor_path = os.path.join(monitor_dir, str(rank)) if monitor_dir is not None else None
            # Create the monitor folder if needed
            if monitor_path is not None:
                os.makedirs(monitor_dir, exist_ok=True)
            env = Monitor(env, filename=monitor_path, info_keywords=('score', 'ghosts', 'level', 'win', 'd', 'map'))
            # Optionally, wrap the environment with the provided wrapper
            if wrapper_class is not None:
                env = wrapper_class(env)
            return env
        return _init

    # No custom VecEnv is passed
    if vec_env_cls is None:
        # Default: use a DummyVecEnv
        vec_env_cls = DummyVecEnv

    return vec_env_cls([make_env(i + start_index) for i in range(n_envs)], **vec_env_kwargs)


def get_results_columns(results):
    return np.mean(results), np.std(results), max(results), min(results)


def plot_error_bar(x, means, std, maxes, mins, title, x_label, y_label, outfile):
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # ax.set_xticks(x)
    # ax.set_xticklabels(x)

    ax.plot(x, means, color='tab:blue', marker='o', label='Average', zorder=3)

    ax.errorbar(x, means, std, color='tab:orange',
                fmt='|', label='Std Deviation', zorder=1)

    ax.scatter(x, maxes, color='tab:green',
               marker='o', label='Maximum', zorder=2)
    ax.scatter(x, mins, color='tab:green',
               marker='x', label='Minimum', zorder=2)

    ax.autoscale_view(True, True, True)
    ax.legend(loc='best', shadow=True, fancybox=True, framealpha=0.7)

    fig.tight_layout()
    fig.canvas.draw()
    fig.savefig(outfile)


def plot_line(x, y, title, x_label, y_label, outfile):
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ax.plot(x, y)

    ax.relim()
    ax.autoscale_view(True, True, True)
    fig.tight_layout()
    fig.canvas.draw()
    fig.savefig(outfile)


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def write_rows(outfile, rows, header, mode='w'):

    file_exists = os.path.isfile(outfile)

    with open(outfile, mode) as f:
        writer = csv.writer(f)

        if mode == 'w' or not file_exists:
            writer.writerow(header)

        for row in rows:
            new_row = [format(x, '7.1f') if isinstance(x, float) or isinstance(x, np.float32) else x for x in row]
            new_row = [format(x, '4d') if isinstance(x, int) else x for x in new_row]
            writer.writerow(new_row)


def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])


def get_alg(alg_name):

    if alg_name == "PPO":
        return PPO2
    elif alg_name == "DQN":
        return DQN

    return None


def get_formated_time(seconds):
    f_time = datetime.timedelta(seconds=seconds)
    f_time = str(datetime.timedelta(days=f_time.days, seconds=f_time.seconds))
    return f_time


def filter_tf_warnings():
    # Filter tensorflow version warnings
    # https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints/40426709
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

    # https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=Warning)

    logging.getLogger("tensorflow").setLevel(logging.ERROR)
