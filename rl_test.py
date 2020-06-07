import os
import argparse
import time
import json
import numpy as np

from gym_pacman import PacmanEnv
from gym_observations import SingleChannelObs, MultiChannelObs
from rl_utils.utils import get_alg, filter_tf_warnings, write_rows, EVAL_HEADER, get_formated_time
from rl_utils.utils import evaluate_policy, get_results_columns


def main():
    parser = argparse.ArgumentParser(description='Test a model inside a directory.')
    parser.add_argument("logdir",
                        help="log directory")
    parser.add_argument("-l", "--latest", action="store_true",
                        help="Use latest dir inside 'logdir' (default: run for 'logdir')")
    parser.add_argument('-m', '--model_name', type=str, default="best_model",
                        help="Model file name (default: best_model)")
    parser.add_argument("-e", "--eval_episodes", type=int, default=40,
                        help="Number of evaluation episodes. (default: 40)")
    parser.add_argument("--map", help="path to the map bmp", default="data/map1.bmp")
    parser.add_argument("--ghosts", help="Maximum number of ghosts", type=int, default=None)
    parser.add_argument("--level", help="difficulty level of ghosts", choices=['0', '1', '2', '3'], default=None)
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

    n_ghosts = params['ghosts']
    fixed_n_ghosts = False

    if args.ghosts:
        n_ghosts = args.ghosts
        fixed_n_ghosts = True

    ghosts_level = params['level']

    if args.level:
        ghosts_level = int(args.level)

    env = PacmanEnv(obs_type, params['positive_rewards'], params['agent_name'],
                    args.map, n_ghosts, ghosts_level, args.lives, args.timeout,
                    ghosts_rnd=False)

    eval_start_time = time.time()

    returns, lengths, scores, ghosts, n_wins = evaluate_policy(model, env,
                                                               args.eval_episodes,
                                                               deterministic=False,
                                                               fixed_n_ghosts=fixed_n_ghosts,
                                                               render=False)

    eval_elapsed_time = get_formated_time(time.time() - eval_start_time)

    header = EVAL_HEADER.copy()
    header[0] = 'ModelName'
    header[-1] = 'EvaluationEpisodes'

    mean_return, std_return, max_return, min_return = get_results_columns(returns)
    mean_length, std_length, max_length, min_length = get_results_columns(lengths)
    mean_score, std_score, max_score, min_score = get_results_columns(scores)

    rows = [[args.model_name, mean_score, std_score, max_score, min_score,
             mean_return, std_return, max_return, min_return,
             mean_length, std_length, max_length, min_length,
             np.mean(ghosts), n_wins, eval_elapsed_time, args.eval_episodes]]

    if args.ghosts:
        header.append('NGhosts')
        rows[0].append(n_ghosts)

    if args.level:
        header.append('Level')
        rows[0].append(ghosts_level)

    write_rows(os.path.join(log_dir, 'test_evaluations.csv'), rows, header, mode='a')


if __name__ == "__main__":
    main()
