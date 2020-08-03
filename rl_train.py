import os
import argparse
import json
from datetime import datetime
from collections import OrderedDict

from stable_baselines.bench import Monitor

from gym_pacman import PacmanEnv, ENV_PARAMS
from gym_observations import SingleChannelObs, MultiChannelObs
from rl_utils.utils import get_alg, filter_tf_warnings, human_format, make_vec_env
from rl_utils.callbacks import PlotEvalSaveCallback
from rl_utils.cnn_extractor import pacman_cnn

LOGS_BASE_DIR = 'logs'
AGENT_ID_FILE = 'agent.id'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--alg", type=str, default="PPO",
                        help="Algorithm name. PPO or DQN (default: PPO)")
    parser.add_argument("-t", "--timesteps", type=int, default=20000,
                        help="Total timesteps (default: 20000)")
    parser.add_argument("-f", "--eval_freq", type=int, default=None,
                        help="Number of callback calls between evaluations. (default: timesteps/10)")
    parser.add_argument("-tb", "--tensorboard", default=None,
                        help="Tensorboard logdir. (default: None)")
    parser.add_argument("-s", "--single_channel_obs", action="store_true",
                        help='Use Single Channel Observation (default: Use Multi Channel Observation)')
    parser.add_argument("-pr", "--positive_rewards", action="store_true",
                        help='Use positive rewards (default: Use negative rewards)')
    parser.add_argument("-g", "--gamma", type=float, default=0.99,
                        help="Discount factor. (default: 0.99)")
    parser.add_argument('-l', '--log_dir', type=str, default=None,
                        help="Continue training of 'log_dir' (default: None)")
    parser.add_argument('-m', '--model_name', type=str, default="end_model",
                        help="Continue training of 'model_name' in 'log_dir' (default: end_model)")
    parser.add_argument("--map", help="path to the map bmp", default="data/map1.bmp")
    parser.add_argument("--ghosts", help="Maximum number of ghosts", type=int, default=4)
    parser.add_argument("--level", help="difficulty level of ghosts", choices=['0', '1', '2', '3'], default='1')
    parser.add_argument("--lives", help="Number of lives", type=int, default=3)
    parser.add_argument("--timeout", help="Timeout after this amount of steps", type=int, default=3000)
    args = parser.parse_args()

    now = datetime.now()

    # Read Params
    alg_name = args.alg
    timesteps = args.timesteps
    eval_freq = args.eval_freq
    positive_rewards = args.positive_rewards
    gamma = args.gamma
    ghosts = args.ghosts
    level = int(args.level)
    lives = args.lives
    timeout = args.timeout

    obs_type = MultiChannelObs

    if args.single_channel_obs:
        obs_type = SingleChannelObs

    map_files = []

    for mf in {param.map for param in ENV_PARAMS}:
        map_files.append(mf)

    previous_agent_name = ""

    # Override params
    if args.log_dir:
        with open(os.path.join(args.log_dir, "params.json"), "r") as f:
            params = json.load(f)
            previous_agent_name = params['agent_name']
            alg_name = params['alg']
            positive_rewards = params['positive_rewards']
            gamma = params['gamma']
            map_files = params['map_files']
            ghosts = params['ghosts']
            level = params['level']
            lives = params['lives']
            timeout = params['timeout']

            if params['obs_type'] == SingleChannelObs.__name__:
                obs_type = SingleChannelObs
            elif params['obs_type'] == MultiChannelObs.__name__:
                obs_type = MultiChannelObs
            else:
                raise ValueError("Invalid obs_type in params.json file.")

    # Process Params
    if not eval_freq:
        eval_freq = int(timesteps / 10)

    agent_id = 1

    if os.path.isfile(AGENT_ID_FILE):
        with open(AGENT_ID_FILE, 'r') as f:
            agent_id = int(f.read())

    agent_name = alg_name + '_' + str(agent_id)

    with open(AGENT_ID_FILE, 'w') as f:
        f.write(str(agent_id + 1))

    rewards_str = 'Neg'

    if positive_rewards:
        rewards_str = 'Pos'

    base_dir = "{}_{}_{}".format(alg_name, obs_type.__name__[:4], rewards_str)

    dir_name = "{}/{}_{}_{}_{}_{}_{}".format(base_dir,
                                             agent_id,
                                             previous_agent_name,
                                             base_dir,
                                             human_format(timesteps),
                                             gamma,
                                             now.strftime('%y%m%d-%H%M%S'))

    log_dir = os.path.join(LOGS_BASE_DIR, dir_name)

    os.makedirs(log_dir, exist_ok=True)

    print("\nLog dir:", log_dir)

    params = OrderedDict()
    params['agent_name'] = agent_name
    params['agent_id'] = agent_id
    params['alg'] = alg_name
    params['timesteps'] = timesteps
    params['eval_freq'] = eval_freq
    params['obs_type'] = obs_type.__name__
    params['positive_rewards'] = positive_rewards
    params['gamma'] = gamma
    params['map_files'] = map_files
    params['ghosts'] = ghosts
    params['level'] = level
    params['lives'] = lives
    params['timeout'] = timeout
    params['datetime'] = now.replace(microsecond=0).isoformat()
    params['previous_model_dir'] = args.log_dir
    if args.log_dir:
        params['previous_model_name'] = args.model_name
        params['previous_agent_name'] = previous_agent_name

    with open(os.path.join(log_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=4)

    pacman_env = PacmanEnv(obs_type, positive_rewards, agent_name,
                           ghosts, int(level), lives, timeout)

    env = make_vec_env(pacman_env, n_envs=1, monitor_dir=log_dir)

    filter_tf_warnings()

    policy_kwargs = {'cnn_extractor': pacman_cnn, 'data_format': 'NCHW'}

    alg = get_alg(alg_name)

    if alg_name == "DQN":
        model = alg("CnnPolicy", env, policy_kwargs=policy_kwargs, prioritized_replay=True,
                    gamma=gamma, tensorboard_log=args.tensorboard, verbose=0)
    elif alg_name == "PPO":
        model = alg("CnnPolicy", env, policy_kwargs=policy_kwargs,
                    gamma=gamma, tensorboard_log=args.tensorboard, verbose=0)
        # model = alg("CnnPolicy", env, policy_kwargs=policy_kwargs, ent_coef=0.0, learning_rate=3e-4,
        #             gamma=gamma, tensorboard_log=args.tensorboard, verbose=0)

    eval_env = PacmanEnv(obs_type, positive_rewards, agent_name,
                         ghosts, int(level), lives, timeout,
                         training=False)

    eval_callback = PlotEvalSaveCallback(eval_env,
                                         eval_freq=eval_freq,
                                         log_dir=log_dir,
                                         deterministic=False)

    with eval_callback:
        if args.log_dir:
            model = alg.load(os.path.join(args.log_dir, args.model_name))
            model.set_env(env)

        model.learn(total_timesteps=timesteps, callback=eval_callback, tb_log_name=agent_name)
