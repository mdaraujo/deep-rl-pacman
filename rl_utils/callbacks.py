import os
import time
import json
from collections import OrderedDict
import numpy as np
from tqdm.auto import tqdm
from stable_baselines.common.callbacks import EvalCallback

from rl_utils.utils import get_elapsed_time, write_rows, EVAL_HEADER


class PlotEvalSaveCallback(EvalCallback):
    """
    Callback for evaluating the agent during training and saving the best and the last model.
    """

    def __init__(self, eval_env, n_eval_episodes, eval_freq, log_dir, deterministic):
        super().__init__(eval_env, None, n_eval_episodes, eval_freq,
                         log_dir, log_dir, deterministic,
                         render=False, verbose=0)

        self.log_dir = log_dir
        self.pbar = None
        self.start_time = None
        self.train_steps = []
        self.mean_rewards = []
        self.std_rewards = []
        self.max_rewards = []
        self.min_rewards = []
        self.mean_ep_lengths = []
        self.std_ep_lengths = []
        self.evals_elapsed_time = []

    def _on_training_start(self):
        super()._on_training_start()
        self.pbar = tqdm(total=self.locals['total_timesteps'])
        self.start_time = time.time()

    def _on_step(self):
        self.pbar.n = self.num_timesteps
        self.pbar.update(0)

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            eval_start_time = time.time()
            super()._on_step()
            _, eval_elapsed_time = get_elapsed_time(time.time(), eval_start_time)

            episode_rewards = self.evaluations_results[-1]
            episode_lengths = self.evaluations_length[-1]

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)

            self.train_steps.append(self.num_timesteps)
            self.mean_rewards.append(mean_reward)
            self.std_rewards.append(std_reward)
            self.max_rewards.append(max(episode_rewards)[0])
            self.min_rewards.append(min(episode_rewards)[0])
            self.mean_ep_lengths.append(mean_ep_length)
            self.std_ep_lengths.append(std_ep_length)
            self.evals_elapsed_time.append(eval_elapsed_time)

            eval_episodes = [self.n_eval_episodes for _ in range(len(self.mean_rewards))]

            rows = zip(self.train_steps, self.mean_rewards, self.std_rewards,
                       self.max_rewards, self.min_rewards,
                       self.mean_ep_lengths, self.std_ep_lengths,
                       self.evals_elapsed_time, eval_episodes)

            write_rows(os.path.join(self.log_dir, 'evaluations.csv'),
                       rows, EVAL_HEADER)

        return True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        self.model.save(os.path.join(self.log_dir, 'end_model'))

        best_train_step_index = self.mean_rewards.index(self.best_mean_reward)
        best_train_step = self.train_steps[best_train_step_index]

        _, elapsed_time = get_elapsed_time(time.time(), self.start_time)

        train_logs = OrderedDict()
        train_logs['end_train_step'] = self.num_timesteps
        train_logs['best_train_step'] = best_train_step
        train_logs['best_mean_reward'] = round(float(self.best_mean_reward), 2)
        train_logs['last_eval_train_step'] = self.train_steps[-1]
        train_logs['last_eval_mean_reward'] = round(float(self.mean_rewards[-1]), 2)
        train_logs['elapsed_time'] = elapsed_time

        with open(os.path.join(self.log_dir, "train_logs.json"), "w") as f:
            json.dump(train_logs, f, indent=4)

        self.pbar.close()
