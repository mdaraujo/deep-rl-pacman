import os
import argparse
import json
from datetime import datetime
from collections import OrderedDict

from stable_baselines.bench import Monitor

from gym_pacman import PacmanEnv
from gym_observations import SingleChannelObs, MultiChannelObs
from rl_utils.utils import get_alg, filter_tf_warnings
from rl_utils.callbacks import PlotEvalSaveCallback
from rl_utils.cnn_extractor import pacman_cnn

LOGS_BASE_DIR = 'logs'
AGENT_ID_FILE = 'agent.id'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--alg", type=str, default="DQN",
                        help="Algorithm name. PPO or DQN (default: DQN)")
    parser.add_argument("-t", "--timesteps", type=int, default=20000,
                        help="Total timesteps (default: 20000)")
    parser.add_argument("-f", "--eval_freq", type=int, default=None,
                        help="Number of callback calls between evaluations. (default: timesteps/10)")
    parser.add_argument("-e", "--eval_episodes", type=int, default=40,
                        help="Number of evaluation episodes. (default: 40)")
    parser.add_argument("-tb", "--tensorboard", default=None,
                        help="Tensorboard logdir. (default: None)")
    parser.add_argument("-s", "--single_channel_obs", action="store_true",
                        help='Use Single Channel Observation (default: Use Multi Channel Observation)')
    parser.add_argument("-pr", "--positive_rewards", action="store_true",
                        help='Use positive rewards (default: Use negative rewards)')
    parser.add_argument("-g", "--gamma", type=float, default=0.99,
                        help="Discount factor. (default: 0.99)")
    parser.add_argument("--map", help="path to the map bmp", default="data/map1.bmp")
    parser.add_argument("--ghosts", help="Maximum number of ghosts", type=int, default=4)
    parser.add_argument("--level", help="difficulty level of ghosts", choices=['0', '1', '2', '3'], default='3')
    parser.add_argument("--lives", help="Number of lives", type=int, default=3)
    parser.add_argument("--timeout", help="Timeout after this amount of steps", type=int, default=3000)
    args = parser.parse_args()

    now = datetime.now()

    eval_freq = args.eval_freq

    if not eval_freq:
        eval_freq = int(args.timesteps / 10)

    alg_name = args.alg

    agent_id = 1

    if os.path.isfile(AGENT_ID_FILE):
        with open(AGENT_ID_FILE, 'r') as f:
            agent_id = int(f.read())

    agent_name = alg_name + '_' + str(agent_id)

    with open(AGENT_ID_FILE, 'w') as f:
        f.write(str(agent_id + 1))

    obs_type = MultiChannelObs

    if args.single_channel_obs:
        obs_type = SingleChannelObs

    rewards_str = 'Neg'

    if args.positive_rewards:
        rewards_str = 'Pos'

    dir_name = "{}_{}_{}_{}_{}_{}".format(agent_id, alg_name,
                                          obs_type.__name__[:4],
                                          rewards_str,
                                          args.gamma,
                                          now.strftime('%y%m%d-%H%M%S'))

    log_dir = os.path.join(LOGS_BASE_DIR, dir_name)

    os.makedirs(log_dir, exist_ok=True)

    print("\nLog dir:", log_dir)

    params = OrderedDict()
    params['agent_name'] = agent_name
    params['alg'] = alg_name
    params['timesteps'] = args.timesteps
    params['eval_freq'] = eval_freq
    params['eval_episodes'] = args.eval_episodes
    params['obs_type'] = obs_type.__name__
    params['positive_rewards'] = args.positive_rewards
    params['gamma'] = args.gamma
    params['map'] = args.map
    params['ghosts'] = args.ghosts
    params['level'] = int(args.level)
    params['lives'] = args.lives
    params['timeout'] = args.timeout
    params['datetime'] = now.replace(microsecond=0).isoformat()

    with open(os.path.join(log_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=4)

    env = PacmanEnv(obs_type, args.positive_rewards, agent_name, args.map,
                    args.ghosts, int(args.level), args.lives, args.timeout)
    env = Monitor(env, filename=log_dir, info_keywords=('score', 'ghosts'))

    filter_tf_warnings()

    policy_kwargs = {'cnn_extractor': pacman_cnn, 'data_format': 'NCHW'}

    alg = get_alg(alg_name)

    if alg_name == "DQN":
        model = alg("CnnPolicy", env, policy_kwargs=policy_kwargs, prioritized_replay=True,
                    gamma=args.gamma, tensorboard_log=args.tensorboard, verbose=0)
    elif alg_name == "PPO":
        model = alg("CnnPolicy", env, policy_kwargs=policy_kwargs,
                    gamma=args.gamma, tensorboard_log=args.tensorboard, verbose=0)
        # model = alg("CnnPolicy", env, policy_kwargs=policy_kwargs, ent_coef=0.0, learning_rate=3e-4,
        #             gamma=args.gamma, tensorboard_log=args.tensorboard, verbose=0)

    eval_env = PacmanEnv(obs_type, args.positive_rewards, agent_name, args.map,
                         args.ghosts, int(args.level), args.lives, args.timeout,
                         ghosts_rnd=False)

    eval_callback = PlotEvalSaveCallback(eval_env, n_eval_episodes=args.eval_episodes,
                                         eval_freq=eval_freq,
                                         log_dir=log_dir, deterministic=False)

    with eval_callback:
        model.learn(total_timesteps=args.timesteps, callback=eval_callback, tb_log_name=agent_name)
