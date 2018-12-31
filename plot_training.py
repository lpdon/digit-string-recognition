import csv
from argparse import ArgumentParser
from typing import List, Any, Dict

import matplotlib.pyplot as plt


def parse_args():
    parser = ArgumentParser("Script for plotting of training log.")
    parser.add_argument("--log", required=False, type=str, help="Path to the log file.")
    parser.add_argument("--columns", required=True, nargs='+', type=str,
                        help="Specifies which values should be plotted")
    return parser.parse_args()


def read_log(log_file: str) -> List[Dict[str, Any]]:
    with open(log_file, 'r') as csvfile:
        csvreader = csv.DictReader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        history = list(csvreader)
    return history


def plot(log_file: str, columns: List[str]):
    history = read_log(log_file)
    selected_columns = [[row[sel_key] for row in history] for sel_key in columns]
    for col_name, col in zip(columns, selected_columns):
        plt.plot(col, label=col_name)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    plot(args.log, columns=args.columns)
