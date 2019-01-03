from argparse import ArgumentParser, Namespace
from itertools import groupby
from pathlib import Path
from typing import Dict, Any, List, Tuple

import Levenshtein as lv
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from car_dataset import CAR
from model import StringNet
from timer import Timer
from util import concat, length_tensor, format_status_line, write_to_csv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_parser():
    parser = ArgumentParser("Training script for Digit String Recognition PyTorch-Model.")
    parser.add_argument("-d", "--data", type=str, required=False, default="",
                        help="Path to the root folder of the CAR-{A,B} dataset.")
    parser.add_argument("-e", "--epochs", type=int, default=50,
                        help="Number of epochs to train the model.")
    parser.add_argument("--target-size", "--is", nargs=2, type=int, default=(100, 300),
                        help="Y and X size to which the images should be resized.")
    parser.add_argument("--batch-size", "--bs", type=int, default=4,
                        help="Batch size for training and testing.")
    parser.add_argument("--train-val-split", "--val", type=float, default=0.8,
                        help="The ratio of the training data which is used for actual training. "
                             "The rest (1-ratio) is used for validation (development test set)")
    parser.add_argument("--seed", type=int, nargs='+', default=[666, ],
                        help="Seed used for the random number generator.")
    parser.add_argument("--lr", "--learning-rate", type=float, default=1e-4,
                        help="The initial learning rate.")
    parser.add_argument("-v", "--verbose", action='store_true', default=False, required=False,
                        help="Print more information.")
    parser.add_argument("-c", "--config-file", type=str, required=False,
                        help="Path to a yaml configuration file.")
    parser.add_argument("--log", required=False, type=str, help="Path to the log file destination.")
    parser.add_argument("--save_path", required=False, type=str, default="",
                        help="Path to the model destination. If empty, model won't be saved.")
    parser.add_argument("--load_path", required=False, type=str, default="",
                        help="Path to the saved model. If empty, model won't be loaded.")
    return parser


def parse_args():
    parser = create_parser()
    args = parser.parse_args()

    if args.data is None and args.config_file is None:
        parser.error("Dataset or config file required.")

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


def create_dataloader(args: Namespace, verbose: bool = False) -> Dict[str, DataLoader]:
    # Data augmentation and normalization for training
    # Just normalization for validation
    width, height = args.target_size
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((width, height)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.5, hue=0.5),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=10),
            transforms.ToTensor(),
            transforms.Normalize([0.6205, 0.6205, 0.6205], [0.1343, 0.1343, 0.1343])
        ]),
        'test': transforms.Compose([
            transforms.Resize((width, height)),
            transforms.ToTensor(),
            transforms.Normalize([0.6205, 0.6205, 0.6205], [0.1343, 0.1343, 0.1343])
        ]),
    }

    # Load dataset
    dataset = CAR(args.data, transform=data_transforms, train_val_split=args.train_val_split, verbose=verbose)
    if verbose:
        print(dataset)

    # Create training and validation dataloaders
    dataloaders_dict = {
        x: DataLoader(dataset.subsets[x],
                      batch_size=args.batch_size,
                      shuffle=True,
                      num_workers=4
                      ) for x in ['train', 'test', 'val']
    }
    return dataloaders_dict


def build_model(n_classes: int, seq_length: int, batch_size: int) -> StringNet:
    return StringNet(n_classes, seq_length, batch_size)


def loss_func():
    return nn.CTCLoss(blank=10, reduction='mean')


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Detect if we have a GPU available
    if torch.cuda.is_available():
        cudnn.deterministic = True
        cudnn.benchmark = False
        torch.cuda.manual_seed_all(seed)


def apply_ctc_loss(floss, output, target: List[List[int]]):
    target_lengths = length_tensor(target)
    target = concat(target)
    target = torch.Tensor(target)
    target = target.long()
    target = target.view((-1,))
    target = target.to(device)

    # Calculate lengths
    input_lengths = torch.full((output.shape[1],), output.shape[0], dtype=torch.long)

    return floss(output, target, input_lengths, target_lengths)


def postproc_output(output) -> List[str]:
    preds = output.argmax(2)
    preds = preds.transpose(0, 1)

    proc_preds = []

    for pred in preds:
        pred_str = [x[0] for x in groupby(pred)]
        pred_str = [str(int(p)) for p in pred_str if p != 10]
        pred_str = ''.join(pred_str)
        proc_preds.append(pred_str)

    return proc_preds


def calc_lv_dist(output, targets: List[str]):
    distances = []
    preds = postproc_output(output)
    for pred, gt in zip(preds, targets):
        distance = lv.distance(pred, gt)
        distances.append(distance)
    return distances


def calc_acc(output, targets: List[str]):
    acc = []
    preds = postproc_output(output)
    for pred, gt in zip(preds, targets):
        acc.append(pred == gt)
    return acc


def train(args: Namespace, seed: int = 0, verbose: bool = False) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    set_seed(seed)

    # Load dataset and create data loaders
    dataloaders = create_dataloader(args, verbose)

    seq_length = 12

    if args.load_path is not None and Path(args.load_path).is_file():
        print("Loading model weights from: " + args.load_path)
        model = torch.load(args.load_path)
    else:
        model = build_model(11, seq_length, args.batch_size).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    floss = loss_func()

    # Train here
    history = []
    phase = 'train'
    batch_timer = Timer()
    epoch_timer = Timer()
    total_batches = len(dataloaders[phase])
    for epoch in range(args.epochs):
        model.train()
        epoch_timer.start()
        batch_timer.reset()

        total_loss = num_samples = total_distance = total_accuracy = 0
        dummy_images = dummy_batch_targets = None

        for batch_num, (image, str_targets) in enumerate(dataloaders[phase]):
            batch_timer.start()
            # string to individual ints
            int_targets = [[int(c) for c in gt] for gt in str_targets]

            # Prepare image
            image = image.to(device)

            # Forward
            optimizer.zero_grad()
            output = model(image)
            loss = apply_ctc_loss(floss, output, int_targets)

            # Backward
            loss.backward(torch.ones_like(loss.data))

            # Update
            optimizer.step()

            distances = calc_lv_dist(output, str_targets)
            total_distance += sum(distances)
            accuracy = calc_acc(output, str_targets)
            total_accuracy += sum(accuracy)
            total_loss += loss.sum().item()
            num_samples += len(str_targets)

            if verbose:
                dummy_images = image
                dummy_batch_targets = str_targets
            batch_timer.stop()
            if batch_num % 10 == 0:
                print(batch_timer.format_status(num_total=total_batches - batch_num) + 20 * " ", end='\r', flush=True)

        epoch_timer.stop()
        if verbose:
            print("Train examples: ")
            print(model(dummy_images).argmax(2)[:, :10], dummy_batch_targets[:10])

        val_results = test(model, dataloaders['val'], verbose)
        history_item = {}
        history_item['epoch'] = epoch + 1
        history_item['avg_dist'] = total_distance / num_samples
        history_item['avg_loss'] = total_loss / num_samples
        history_item['accuracy'] = total_accuracy / num_samples
        history_item['time'] = epoch_timer.last()
        history_item.update({"val_" + key: value for key, value in val_results.items()})
        history.append(history_item)

        status_line = format_status_line(history_item)
        print(status_line)

        write_to_csv(history_item, args.log, write_header=epoch == 0, append=epoch != 0)

        if args.save_path is not None:
            torch.save(model, args.save_path)

    # Test here
    test_results = test(model, dataloaders['test'], verbose)
    status_line = format_status_line(test_results)
    print("Test         | " + status_line)
    test_results["total_training_time"] = epoch_timer.total()
    torch.save(model.state_dict(), './model.pth')
    return history, test_results


def test(model: nn.Module, dataloader: DataLoader, verbose: bool = False) -> Dict[str, Any]:
    model.eval()
    with torch.no_grad():
        dummy_images = dummy_batch_targets = None
        floss = loss_func()
        # Reset tracked metrics
        total_distance = samples = total_loss = total_accuracy = 0

        for image, str_targets in dataloader:
            # string to individual ints
            int_targets = [[int(c) for c in gt] for gt in str_targets]

            # Prepare image
            image = image.to(device)

            # Forward
            output = model(image)
            loss = apply_ctc_loss(floss, output, int_targets)

            total_loss += loss.sum().item()
            distances = calc_lv_dist(output, str_targets)
            total_distance += sum(distances)
            accuracy = calc_acc(output, str_targets)
            total_accuracy += sum(accuracy)
            samples += len(str_targets)

            if verbose:
                dummy_images = image
                dummy_batch_targets = str_targets
        if verbose:
            print("Validation example:")
            print(model(dummy_images).argmax(2)[:, :10], dummy_batch_targets[:10])
    return {'avg_dist': total_distance / samples, 'avg_loss': total_loss / samples,
            'accuracy': total_accuracy / samples}


if __name__ == "__main__":
    args = parse_args()
    if len(args.seed) == 1:
        train(args, seed=args.seed[0], verbose=args.verbose)
    else:
        # Get the results for every seed
        results = [train(args, seed=seed, verbose=args.verbose) for seed in args.seed]
        results = [result[1] for result in results]
        # Create dictionary to get a mapping from metric_name -> array of results of that metric
        # e.g. { 'accuracy': [0.67, 0.68] }
        metrics = next(iter(results)).keys()
        results = {key: np.asarray([result[key] for result in results]) for key in metrics}
        print(results)
        for key, values in results.items():
            avg = np.average(values)
            std = sum(np.abs(values - avg)) / len(values)
            print(key + ": ")
            print("\t Average:  {}".format(avg))
            print("\t STD:      {}".format(std))
