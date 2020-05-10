import os
import time
import json
from collections import OrderedDict

import numpy as np
from tqdm.auto import tqdm
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.results_plotter import load_results, ts2xy

from rl_utils.utils import get_formated_time, write_rows, EVAL_HEADER, plot_line, plot_error_bar, moving_average


class PlotEvalSaveCallback(BaseCallback):
    """
    Callback for evaluating the agent during training and saving the best and the last model.
    """

    def __init__(self, eval_env, n_eval_episodes, eval_freq, log_dir, deterministic):
        super().__init__(verbose=0)
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.deterministic = deterministic
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
            self.eval_save_plot()

        return True

    def _on_training_end(self):
        # PPO may end before total_timesteps
        if self.num_timesteps < self.locals['total_timesteps']:
            self.eval_save_plot()

    def eval_save_plot(self):
        eval_start_time = time.time()

        episode_rewards, episode_lengths = evaluate_policy(self.model, self.eval_env,
                                                           n_eval_episodes=self.n_eval_episodes,
                                                           render=False,
                                                           deterministic=self.deterministic,
                                                           return_episode_rewards=True)

        eval_elapsed_time = get_formated_time(time.time() - eval_start_time)

        mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
        mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)

        if mean_reward > self.best_mean_reward:
            self.model.save(os.path.join(self.log_dir, 'best_model'))
            self.best_mean_reward = mean_reward

        self.train_steps.append(self.num_timesteps)
        self.mean_rewards.append(mean_reward)
        self.std_rewards.append(std_reward)
        self.max_rewards.append(max(episode_rewards))
        self.min_rewards.append(min(episode_rewards))
        self.mean_ep_lengths.append(mean_ep_length)
        self.std_ep_lengths.append(std_ep_length)
        self.evals_elapsed_time.append(eval_elapsed_time)

        eval_episodes = [self.n_eval_episodes for _ in range(len(self.mean_rewards))]

        rows = zip(self.train_steps, self.mean_rewards, self.std_rewards,
                   self.max_rewards, self.min_rewards,
                   self.mean_ep_lengths, self.std_ep_lengths,
                   self.evals_elapsed_time, eval_episodes)

        write_rows(os.path.join(self.log_dir, 'evaluations.csv'), rows, EVAL_HEADER)

        plot_error_bar(self.train_steps, self.mean_rewards, self.std_rewards,
                       'Evaluations Mean Reward on {} Episodes | Best: {:.1f}'.format(
                           self.n_eval_episodes, self.best_mean_reward),
                       'Training Step', 'Mean Episodes Reward',
                       os.path.join(self.log_dir, 'eval_rewards.png'),
                       self.min_rewards, self.max_rewards)

        plot_error_bar(self.train_steps, self.mean_ep_lengths, self.std_ep_lengths,
                       'Evaluations Mean Length on {} Episodes'.format(self.n_eval_episodes),
                       'Training Step', 'Mean Episodes Length',
                       os.path.join(self.log_dir, 'eval_lengths.png'))

        x, y = ts2xy(load_results(self.log_dir), 'timesteps')

        plot_line(x, y, 'Training Rewards | Total Episodes: {}'.format(len(y)),
                  'Training Step', 'Episode Reward',
                  os.path.join(self.log_dir, 'train_rewards.png'))

        n = 100
        if len(y) < n * 2:
            n = 10

        moving_y = moving_average(y, n=n)

        plot_line(x[n-1:], moving_y, 'Training Rewards Moving Mean | Total Episodes: {}'.format(len(y)),
                  'Training Step', 'Mean {} Episode Reward'.format(n),
                  os.path.join(self.log_dir, 'train_rewards_MM.png'))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        self.model.save(os.path.join(self.log_dir, 'end_model'))

        best_train_step_index = self.mean_rewards.index(self.best_mean_reward)
        best_train_step = self.train_steps[best_train_step_index]

        elapsed_time = time.time() - self.start_time

        process_time = time.process_time()

        train_logs = OrderedDict()
        train_logs['end_train_step'] = self.num_timesteps
        train_logs['best_train_step'] = best_train_step
        train_logs['best_mean_reward'] = round(float(self.best_mean_reward), 1)
        train_logs['last_eval_train_step'] = self.train_steps[-1]
        train_logs['last_eval_mean_reward'] = round(float(self.mean_rewards[-1]), 1)
        train_logs['elapsed_time'] = get_formated_time(elapsed_time)
        train_logs['process_time'] = get_formated_time(process_time)
        train_logs['process_time_s'] = round(process_time, 2)

        with open(os.path.join(self.log_dir, "train_logs.json"), "w") as f:
            json.dump(train_logs, f, indent=4)

        self.pbar.close()
