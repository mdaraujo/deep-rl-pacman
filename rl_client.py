import os
import argparse
import json
import asyncio
import websockets

from mapa import Map
from gym_pacman import PacmanEnv
from gym_observations import SingleChannelObs, MultiChannelObs
from rl_utils.utils import get_alg, filter_tf_warnings


async def agent_loop(model, obs_type, agent_name, server_address="localhost:8000"):
    async with websockets.connect("ws://{}/player".format(server_address)) as websocket:

        # Receive information about static game properties
        await websocket.send(json.dumps({"cmd": "join", "name": agent_name}))
        msg = await websocket.recv()

        print("Started.")

        game_properties = json.loads(msg)

        game_map = Map(game_properties['map'])

        pacman_obs = obs_type(game_map)

        keys = PacmanEnv.keys

        while True:
            r = await websocket.recv()
            state = json.loads(r)

            if not state['lives']:
                print("GAME OVER - Score: {}".format(state['score']))
                return

            obs = pacman_obs.get_obs(state)

            action, _states = model.predict(obs, deterministic=False)

            key = keys[action]

            await websocket.send(json.dumps({"cmd": "key", "key": key}))


def main():
    parser = argparse.ArgumentParser(description='Test a model inside a directory.')
    parser.add_argument("logdir",
                        help="log directory")
    parser.add_argument("-l", "--latest", action="store_true",
                        help="Use latest dir inside 'logdir' (default: run for 'logdir')")
    parser.add_argument('-m', '--model_name', type=str, default="best_model",
                        help="Model file name (default: best_model)")
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

    loop = asyncio.get_event_loop()

    SERVER = os.environ.get('SERVER', 'localhost')
    PORT = os.environ.get('PORT', '8000')

    loop.run_until_complete(agent_loop(model, obs_type, params['agent_name'], "{}:{}".format(SERVER, PORT)))


if __name__ == "__main__":
    main()
