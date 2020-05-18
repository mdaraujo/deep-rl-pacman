import warnings
import os
import logging
import datetime
import csv
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines import PPO2, DQN


EVAL_HEADER = ["TrainStep", "MeanReturn", "StdReturn", "MaxReturn", "MinReturn",
               "MeanEpLength", "StdEpLength", "EvaluationTime"]


def plot_error_bar(x, means, std, title, x_label, y_label, outfile, mins=None, maxes=None):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ax.plot(x, means, color='tab:blue', marker='o', label='Mean', zorder=3)

    ax.errorbar(x, means, std, color='tab:orange',
                fmt='|', label='Std Deviation', zorder=1)

    if mins and maxes:
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
            new_row = [format(x, '6.1f') if isinstance(x, float) or isinstance(x, np.float32) else x for x in row]
            new_row = [format(x, '8d') if isinstance(x, int) else x for x in new_row]
            writer.writerow(new_row)


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
