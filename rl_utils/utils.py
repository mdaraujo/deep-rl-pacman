import warnings
import os
import logging
import datetime
import csv
import numpy as np

from stable_baselines import PPO2, DQN

FIG_SIZE = (10, 5)

EVAL_HEADER = ["TrainStep", "MeanReward", "StdReward", "MaxReward", "MinReward",
               "MeanEpLength", "StdEpLength", "EvaluationTime", "EvaluationEpisodes"]


def write_rows(outfile, rows, header, mode='w'):

    file_exists = os.path.isfile(outfile)

    with open(outfile, mode) as f:
        writer = csv.writer(f)

        if mode == 'w' or not file_exists:
            writer.writerow(header)

        for row in rows:
            new_row = [format(x, '.1f') if isinstance(x, float) or isinstance(x, np.float32) else x for x in row]
            writer.writerow(new_row)


def get_alg(alg_name):

    if alg_name == "PPO":
        return PPO2
    elif alg_name == "DQN":
        return DQN

    return None


def get_elapsed_time(current_time, start_time):
    elapsed_time_seconds = current_time - start_time
    elapsed_time_h = datetime.timedelta(seconds=elapsed_time_seconds)
    elapsed_time_h = str(datetime.timedelta(days=elapsed_time_h.days, seconds=elapsed_time_h.seconds))
    return elapsed_time_seconds, elapsed_time_h


def filter_tf_warnings():
    # Filter tensorflow version warnings
    # https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints/40426709
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

    # https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=Warning)

    logging.getLogger("tensorflow").setLevel(logging.ERROR)
