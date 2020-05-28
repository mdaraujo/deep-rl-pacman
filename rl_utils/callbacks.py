import os
import time
import json
from collections import OrderedDict

import numpy as np
from tqdm.auto import tqdm
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.results_plotter import load_results, ts2xy

from rl_utils.utils import get_formated_time, write_rows, EVAL_HEADER
from rl_utils.utils import plot_line, plot_error_bar, moving_average, evaluate_policy, get_results_columns


class PlotEvalSaveCallback(BaseCallback):
    """
    Callback for evaluating the agent during training and saving the best and the last model.
    """

    def __init__(self, eval_env, n_eval_episodes, eval_freq, log_dir, deterministic):
        super().__init__(verbose=0)
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.deterministic = deterministic
        self.log_dir = log_dir
        self.pbar = None
        self.start_time = None
        self.train_steps = []
        self.returns_columns = []
        self.lengths_columns = []
        self.scores_columns = []
        self.evals_elapsed_time = []
        self.evals_ghosts_mean = []
        self.evals_n_wins = []
        self.best_mean_score = -np.inf
        self.best_train_step = 0
        self.train_ghosts = []

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

        returns, lengths, scores, ghosts, n_wins = evaluate_policy(self.model, self.eval_env,
                                                                   self.n_eval_episodes,
                                                                   self.deterministic,
                                                                   render=False)

        eval_elapsed_time = get_formated_time(time.time() - eval_start_time)

        returns_columns = get_results_columns(returns)
        lengths_columns = get_results_columns(lengths)
        scores_columns = get_results_columns(scores)
        self.evals_ghosts_mean.append(np.mean(ghosts))
        self.evals_n_wins.append(n_wins)

        mean_score = scores_columns[0]

        if mean_score > self.best_mean_score:
            self.model.save(os.path.join(self.log_dir, 'best_model'))
            self.best_mean_score = mean_score
            self.best_train_step = self.num_timesteps

        self.train_steps.append(self.num_timesteps)
        self.returns_columns.append(returns_columns)
        self.lengths_columns.append(lengths_columns)
        self.scores_columns.append(scores_columns)
        self.evals_elapsed_time.append(eval_elapsed_time)

        mean_returns, std_returns, max_returns, min_returns = map(list, zip(*self.returns_columns))
        mean_lengths, std_lengths, max_lengths, min_lengths = map(list, zip(*self.lengths_columns))
        mean_scores, std_scores, max_scores, min_scores = map(list, zip(*self.scores_columns))

        train_results = load_results(self.log_dir)

        self.train_ghosts.append(train_results['ghosts'].mean())

        rows = zip(self.train_steps, mean_scores, std_scores, max_scores, min_scores,
                   mean_returns, std_returns, max_returns, min_returns,
                   mean_lengths, std_lengths, max_lengths, min_lengths,
                   self.evals_ghosts_mean, self.evals_n_wins,
                   self.evals_elapsed_time, self.train_ghosts)

        write_rows(os.path.join(self.log_dir, 'evaluations.csv'), rows, EVAL_HEADER)

        plot_error_bar(self.train_steps, mean_scores, std_scores, max_scores, min_scores,
                       'Evaluations Mean Score on {} Episodes | Best: {:.1f}'.format(
                           self.n_eval_episodes, self.best_mean_score),
                       'Training Step', 'Evaluation Mean Score',
                       os.path.join(self.log_dir, 'eval_scores.png'))

        plot_error_bar(self.train_steps, mean_returns, std_returns, max_returns, min_returns,
                       'Evaluations Mean Return on {} Episodes'.format(self.n_eval_episodes),
                       'Training Step', 'Evaluation Mean Return',
                       os.path.join(self.log_dir, 'eval_returns.png'))

        plot_error_bar(self.train_steps, mean_lengths, std_lengths, max_lengths, min_lengths,
                       'Evaluations Mean Length on {} Episodes'.format(self.n_eval_episodes),
                       'Training Step', 'Evaluation Mean Length',
                       os.path.join(self.log_dir, 'eval_lengths.png'))

        # Train returns
        x, train_returns = ts2xy(train_results, 'timesteps')

        plot_line(x, train_returns, 'Training Episodes Return | Total Episodes: {}'.format(len(train_returns)),
                  'Training Step', 'Episode Return',
                  os.path.join(self.log_dir, 'train_returns.png'))

        moving_n = 100
        if len(train_returns) < moving_n * 2:
            moving_n = 10

        moving_returns = moving_average(train_returns, n=moving_n)

        plot_line(x[moving_n-1:], moving_returns,
                  'Training Episodes Return Moving Mean | Total Episodes: {}'.format(len(train_returns)),
                  'Training Step', '{} Last Episodes Mean Return'.format(moving_n),
                  os.path.join(self.log_dir, 'train_returns_MM.png'))

        # Train scores
        train_scores = train_results['score'].tolist()

        plot_line(x, train_scores, 'Training Episodes Score | Total Episodes: {}'.format(len(train_scores)),
                  'Training Step', 'Episode Score',
                  os.path.join(self.log_dir, 'train_scores.png'))

        moving_scores = moving_average(train_scores, n=moving_n)

        plot_line(x[moving_n-1:], moving_scores,
                  'Training Episodes Score Moving Mean | Total Episodes: {}'.format(len(train_scores)),
                  'Training Step', '{} Last Episodes Mean Score'.format(moving_n),
                  os.path.join(self.log_dir, 'train_scores_MM.png'))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        self.model.save(os.path.join(self.log_dir, 'end_model'))

        elapsed_time = time.time() - self.start_time

        process_time = time.process_time()

        train_logs = OrderedDict()
        train_logs['end_train_step'] = self.num_timesteps
        train_logs['best_train_step'] = self.best_train_step
        train_logs['best_mean_score'] = round(float(self.best_mean_score), 1)
        train_logs['last_eval_train_step'] = self.train_steps[-1]
        train_logs['last_eval_mean_score'] = round(float(self.scores_columns[-1][0]), 1)
        train_logs['elapsed_time'] = get_formated_time(elapsed_time)
        train_logs['process_time'] = get_formated_time(process_time)
        train_logs['process_time_s'] = round(process_time, 2)

        with open(os.path.join(self.log_dir, "train_logs.json"), "w") as f:
            json.dump(train_logs, f, indent=4)

        self.pbar.close()
