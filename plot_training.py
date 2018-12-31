import csv
import math
from argparse import ArgumentParser
from typing import List, Any, Dict

import matplotlib.pyplot as plt
from numpy.ma import arange
import yaml


def create_parser():
    parser = ArgumentParser("Script for plotting of training log.")
    parser.add_argument("--log", required=False, type=str, help="Path to the log file.")
    parser.add_argument("--columns", required=False, nargs='+', type=List[str], default=[],
                        help="Specifies which values should be plotted")
    parser.add_argument('--multi', default=False, action='store_true',
                        help='If set, plot every column into a separate plot.')
    parser.add_argument('--refresh', default=False, action='store_true',
                        help='If set, plot will be refreshed every 10 seconds or '
                             'as specified with --refresh-time.')
    parser.add_argument('--refresh-time', type=int, default=10,
                        help='Specify the seconds to wait until the graph is refreshed if --refresh is set.')
    parser.add_argument("-c", "--config-file", type=str, required=False,
                        help="Path to a yaml configuration file.")
    return parser


def parse_args():
    parser = create_parser()
    args = parser.parse_args()

    if args.columns is None and args.config_file is None:
        parser.error("Columns or config file required.")

    if args.config_file:
        try:
            data = yaml.safe_load(open(args.config_file, "r"))
            delattr(args, 'config_file')
            arg_dict = args.__dict__
            for key, value in data.items():
                if isinstance(value, list):
                    for v in value:
                        arg_dict[key].append(v)
                else:
                    arg_dict[key] = value

        except yaml.YAMLError as exception:
            print(exception)

    return args


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
    plt.clf()
    for num, (col_name, col) in enumerate(zip(columns, selected_columns)):
        # Find the right spot on the plot
        if multiple_plots:
            plt.subplot(num_plot_rows, num_plot_rows, num + 1)
        plt.plot(x, col, label=col_name)
        plt.legend()


def plot_refresh(log_file: str, columns: List[str], multiple_plots: bool = False, refresh: bool = False,
                 refresh_time: int = 10):
    while True:
        plt.ion()
        plot(log_file, columns, multiple_plots)
        if not refresh:
            break
        plt.pause(refresh_time)
        # sleep(refresh_time)


if __name__ == "__main__":
    args = parse_args()
    plot_refresh(args.log, columns=args.columns, multiple_plots=args.multi, refresh=args.refresh,
                 refresh_time=args.refresh_time)
