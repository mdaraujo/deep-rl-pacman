import warnings
import os
import logging
import datetime
import csv
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines import PPO2, DQN
from stable_baselines.common.vec_env import VecEnv

from gym_pacman import ENV_PARAMS


EVAL_HEADER = ["TrainStep", "MeanScore", "StdScore", "MaxScore", "MinScore",
               "MeanReturn", "StdReturn", "MaxReturn", "MinReturn",
               "MeanLength", "StdLength", "MaxLength", "MinLength",
               "EvalIdleStepsMean", "EvalIdleEps",
               "EvalGhostsMean", "EvalLevelMean", "Wins", "EvaluationTime",
               "TrainGhostsMean", "TrainLevelMean"]


def evaluate_policy(model, env, n_eval_episodes=10, deterministic=True, fixed_params=False, render=False):
    """
    Runs policy for `n_eval_episodes` episodes and returns episodes returns, lengths and scores.
    This is made to work only with one env.

    Based on stable-baselines.common.evaluate_policy method
    """
    if isinstance(env, VecEnv):
        assert env.num_envs == 1, "You must pass only one environment when using this function"

    change_params_ep = n_eval_episodes

    if ENV_PARAMS:
        change_params_ep /= len(ENV_PARAMS)

    params_idx = 0
    count = 0

    returns, lengths, scores, idle_steps, ghosts, levels = [], [], [], [], [], []

    n_wins = 0

    for _ in range(n_eval_episodes):

        if not fixed_params:

            if count >= change_params_ep:
                count = 0
                params_idx += 1

            env.set_env_params(ENV_PARAMS[params_idx])

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
        count += 1

    return returns, lengths, scores, idle_steps, ghosts, levels, n_wins


def get_results_columns(results):
    return np.mean(results), np.std(results), max(results), min(results)


def plot_error_bar(x, means, std, maxes, mins, title, x_label, y_label, outfile):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ax.plot(x, means, color='tab:blue', marker='o', label='Mean', zorder=3)

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
    fig, ax = plt.subplots()
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
