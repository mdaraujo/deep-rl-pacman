import argparse
import os
import pandas as pd

from stable_baselines.results_plotter import load_results, ts2xy
from rl_utils.utils import plot_error_bar

from rl_utils.utils import plot_line, plot_error_bar, moving_average, evaluate_policy, get_results_columns


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("logdir", help="log directory")
    args = parser.parse_args()

    # column_names = ["Letter", "Number", "Symbol"]

    df = pd.read_csv(args.logdir + "evaluations.csv")

    print()

    plot_error_bar(df.TrainStep.tolist(), df.MeanReward.tolist(),
                   df.StdReward.tolist(), df.MaxReward.tolist(), df.MinReward.tolist(),
                   'Evaluations Mean Score on {} Episodes | Best: {:.1f}'.format(
                       30, max(df.MeanReward.tolist())),
                   'Training Step', 'Evaluation Mean Score',
                   os.path.join(args.logdir, 'eval_scores_new.png'))

    plot_error_bar(df.TrainStep.tolist(), df.MeanReward.tolist(),
                   df.StdReward.tolist(), df.MaxReward.tolist(), df.MinReward.tolist(),
                   'Evaluations Mean Return on {} Episodes | Best: {:.1f}'.format(
                       30, max(df.MeanReward.tolist())),
                   'Training Step', 'Evaluation Mean Return',
                   os.path.join(args.logdir, 'eval_returns_new.png'))

    train_results = load_results(args.logdir)

    # Train returns
    x, train_returns = ts2xy(train_results, 'timesteps')

    plot_line(x, train_returns, 'Training Episodes Return | Total Episodes: {}'.format(len(train_returns)),
              'Training Step', 'Episode Return',
              os.path.join(args.logdir, 'train_returns_new.png'))

    moving_n = 100
    if len(train_returns) < moving_n * 2:
        moving_n = 10

    moving_returns = moving_average(train_returns, n=moving_n)

    plot_line(x[moving_n-1:], moving_returns,
              'Training Episodes Return Moving Mean | Total Episodes: {}'.format(len(train_returns)),
              'Training Step', '{} Last Episodes Mean Return'.format(moving_n),
              os.path.join(args.logdir, 'train_returns_MM_new.png'))

    # Train scores
    if 'score' in train_results:
        train_scores = train_results['score'].tolist()

        plot_line(x, train_scores, 'Training Episodes Score | Total Episodes: {}'.format(len(train_scores)),
                  'Training Step', 'Episode Score',
                  os.path.join(args.logdir, 'train_scores_new.png'))

        moving_scores = moving_average(train_scores, n=moving_n)

        plot_line(x[moving_n-1:], moving_scores,
                  'Training Episodes Score Moving Mean | Total Episodes: {}'.format(len(train_scores)),
                  'Training Step', '{} Last Episodes Mean Score'.format(moving_n),
                  os.path.join(args.logdir, 'train_scores_MM_new.png'))

    # Train ghosts
    if 'ghosts' in train_results:
        train_ghosts = train_results['ghosts'].tolist()

        plot_line(x, train_ghosts, 'Training Episodes N Ghosts | Total Episodes: {}'.format(len(train_ghosts)),
                  'Training Step', 'Episode N Ghosts',
                  os.path.join(args.logdir, 'train_ghosts_new.png'))

        moving_ghosts = moving_average(train_ghosts, n=moving_n)

        plot_line(x[moving_n-1:], moving_ghosts,
                  'Training Episodes Ghosts Moving Mean | Total Episodes: {}'.format(len(train_ghosts)),
                  'Training Step', '{} Last Episodes Mean Ghosts'.format(moving_n),
                  os.path.join(args.logdir, 'train_ghosts_MM_new.png'))
