import os
import argparse
import time
import json
import numpy as np

from stable_baselines.common.evaluation import evaluate_policy

from gym_pacman import PacmanEnv
from gym_observations import SingleChannelObs, MultiChannelObs
from rl_utils.utils import get_alg, filter_tf_warnings, write_rows, EVAL_HEADER, get_formated_time


def main():
    parser = argparse.ArgumentParser(description='Test a model inside a directory.')
    parser.add_argument("logdir",
                        help="log directory")
    parser.add_argument("-l", "--latest", action="store_true",
                        help="Use latest dir inside 'logdir' (default: run for 'logdir')")
    parser.add_argument('-m', '--model_name', type=str, default="best_model",
                        help="Model file name (default: best_model)")
    parser.add_argument("-e", "--eval_episodes", type=int, default=30,
                        help="Number of evaluation episodes. (default: 30)")
    parser.add_argument("--map", help="path to the map bmp", default="data/map1.bmp")
    parser.add_argument("--ghosts", help="Number of ghosts", type=int, default=1)
    parser.add_argument("--level", help="difficulty level of ghosts", choices=[0, 1, 2, 3], default=1)
    parser.add_argument("--lives", help="Number of lives", type=int, default=3)
    parser.add_argument("--timeout", help="Timeout after this amount of steps", type=int, default=3000)
    args = parser.parse_args()

    log_dir = args.logdir

    all_subdirs = [os.path.join(log_dir, d) for d in sorted(os.listdir(log_dir))
                   if os.path.isdir(os.path.join(log_dir, d))]

    if args.latest:
        log_dir = sorted(all_subdirs)[-1]

    with open(os.path.join(log_dir, "params.json"), "r") as f:
        params = json.load(f)

    filter_tf_warnings()

    alg = get_alg(params['alg'])

    if params['obs_type'] == SingleChannelObs.__name__:
        obs_type = SingleChannelObs
    elif params['obs_type'] == MultiChannelObs.__name__:
        obs_type = MultiChannelObs
    else:
        raise ValueError("Invalid obs_type in params.json file.")

    print("\nUsing {} model at dir {}".format(args.model_name, log_dir))

    model = alg.load(os.path.join(log_dir, args.model_name))

    env = PacmanEnv(obs_type, params['agent_name'], args.map, args.ghosts, args.level, args.lives, args.timeout)

    eval_start_time = time.time()

    episode_rewards, episode_lengths = evaluate_policy(model, env,
                                                       n_eval_episodes=args.eval_episodes,
                                                       deterministic=False,
                                                       return_episode_rewards=True)

    eval_elapsed_time = get_formated_time(time.time() - eval_start_time)

    mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
    mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)

    header = EVAL_HEADER.copy()
    header[0] = 'ModelName'

    rows = [[args.model_name, mean_reward, std_reward,
             max(episode_rewards), min(episode_rewards),
             mean_ep_length, std_ep_length,
             eval_elapsed_time, args.eval_episodes]]

    write_rows(os.path.join(log_dir, 'test_evaluations.csv'),
               rows, header, mode='a')


if __name__ == "__main__":
    main()
