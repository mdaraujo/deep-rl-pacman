import os
import argparse
import json
from datetime import datetime
from collections import OrderedDict

from stable_baselines.bench import Monitor

from gym_pacman import PacmanEnv
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
    parser.add_argument("-e", "--eval_episodes", type=int, default=30,
                        help="Number of evaluation episodes. (default: 30)")
    parser.add_argument("--map", help="path to the map bmp", default="data/map1.bmp")
    parser.add_argument("--ghosts", help="Number of ghosts", type=int, default=1)
    parser.add_argument("--level", help="difficulty level of ghosts", choices=[0, 1, 2, 3], default=1)
    parser.add_argument("--lives", help="Number of lives", type=int, default=3)
    parser.add_argument("--timeout", help="Timeout after this amount of steps", type=int, default=3000)
    args = parser.parse_args()

    eval_freq = args.eval_freq

    if not eval_freq:
        eval_freq = int(args.timesteps / 10)

    now = datetime.now()

    alg_name = args.alg

    agent_id = 1

    if os.path.isfile(AGENT_ID_FILE):
        with open(AGENT_ID_FILE, 'r') as f:
            agent_id = int(f.read())

    agent_name = alg_name + '_' + str(agent_id)

    with open(AGENT_ID_FILE, 'w') as f:
        f.write(str(agent_id + 1))

    dir_name = "{}_{}".format(now.strftime('%y%m%d-%H%M%S'), agent_name)

    log_dir = os.path.join(LOGS_BASE_DIR, dir_name)

    os.makedirs(log_dir, exist_ok=True)

    print("\nLog dir:", log_dir)

    params = OrderedDict()
    params['agent_name'] = agent_name
    params['alg'] = alg_name
    params['timesteps'] = args.timesteps
    params['eval_freq'] = eval_freq
    params['eval_episodes'] = args.eval_episodes
    params['map'] = args.map
    params['ghosts'] = args.ghosts
    params['level'] = args.level
    params['lives'] = args.lives
    params['timeout'] = args.timeout
    params['datetime'] = now.replace(microsecond=0).isoformat()

    with open(os.path.join(log_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=4)

    env = PacmanEnv(agent_name, args.map, args.ghosts, args.level, args.lives, args.timeout)
    env = Monitor(env, filename=log_dir)

    filter_tf_warnings()

    policy_kwargs = {'cnn_extractor': pacman_cnn, 'data_format': 'NCHW', 'pad': 'SAME'}

    alg = get_alg(alg_name)

    tensorboard_log = None

    # tensorboard_log = "./logs_tb"

    if alg_name == "DQN":
        model = alg("CnnPolicy", env, policy_kwargs=policy_kwargs, prioritized_replay=True,
                    tensorboard_log=tensorboard_log, verbose=0)
    elif alg_name == "PPO":
        model = alg("CnnPolicy", env, policy_kwargs=policy_kwargs,
                    tensorboard_log=tensorboard_log, verbose=0)

    eval_env = PacmanEnv(agent_name, args.map, args.ghosts, int(args.level), args.lives, args.timeout)

    eval_callback = PlotEvalSaveCallback(eval_env, n_eval_episodes=args.eval_episodes,
                                         eval_freq=eval_freq,
                                         log_dir=log_dir, deterministic=True)

    with eval_callback:
        model.learn(total_timesteps=args.timesteps, callback=eval_callback)
