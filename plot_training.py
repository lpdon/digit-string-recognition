import csv
import math
from argparse import ArgumentParser
from typing import List, Any, Dict

import matplotlib.pyplot as plt
from numpy.ma import arange


def parse_args():
    parser = ArgumentParser("Script for plotting of training log.")
    parser.add_argument("--log", required=False, type=str, help="Path to the log file.")
    parser.add_argument("--columns", required=True, nargs='+', type=str,
                        help="Specifies which values should be plotted")
    parser.add_argument('--multi', default=False, action='store_true',
                        help='If true, plot every column into a separate plot.')
    return parser.parse_args()


def read_log(log_file: str) -> List[Dict[str, Any]]:
    with open(log_file, 'r') as csvfile:
        csvreader = csv.DictReader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        history = list(csvreader)
    return history


def plot(log_file: str, columns: List[str], multiple_plots: bool = False):
    history = read_log(log_file)
    selected_columns = [[row[sel_key] for row in history] for sel_key in columns]
    num_plot_rows = int(math.ceil(math.sqrt(len(columns)))) if multiple_plots else 1

    x = arange(len(history)) + 1
    for num, (col_name, col) in enumerate(zip(columns, selected_columns)):
        # Find the right spot on the plot
        if multiple_plots:
            plt.subplot(num_plot_rows, num_plot_rows, num + 1)
        plt.plot(x, col, label=col_name)
        plt.legend()
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    plot(args.log, columns=args.columns, multiple_plots=args.multi)
