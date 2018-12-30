from argparse import ArgumentParser, Namespace
from itertools import groupby
from typing import Dict, Any, List

import Levenshtein as lv
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from car_dataset import CAR
from model import StringNet
from timer import Timer
from util import concat, length_tensor

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
            transforms.ToTensor(),
        ]),
        'test': transforms.Compose([
            transforms.Resize((width, height)),
            transforms.ToTensor(),
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


def build_model(n_classes: int, seq_length: int, batch_size: int) -> nn.Module:
    return StringNet(n_classes, seq_length, batch_size)


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
    target = Variable(target)
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


def train(args: Namespace, seed: int = 0, verbose: bool = False) -> Dict[str, Any]:
    set_seed(seed)

    # Load dataset and create data loaders
    dataloaders = create_dataloader(args, verbose)

    seq_length = 12

    model = build_model(11, seq_length, args.batch_size).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    floss = nn.CTCLoss(blank=10)

    # Train here
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
            image = Variable(image)
            image = image.to(device)

            # Forward
            optimizer.zero_grad()
            output = model(image)
            loss = apply_ctc_loss(floss, output, int_targets)

            # Backward
            loss.backward()

            # Update
            optimizer.step()

            distances = calc_lv_dist(output, str_targets)
            total_distance += sum(distances)
            accuracy = calc_acc(output, str_targets)
            total_accuracy += sum(accuracy)
            total_loss += loss.item()
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
        print("Epoch {}: loss: {} | avg_dist: {} | accuracy: {} | time {}s "
              "| val_dist: {} | val_loss: {}".format(epoch + 1,
                                                     round(total_loss / num_samples, 6),
                                                     round(total_distance / num_samples, 6),
                                                     round(total_accuracy / num_samples, 6),
                                                     int(epoch_timer.last()),
                                                     round(val_results['average_distance'], 6),
                                                     round(val_results['loss'], 6),
                                                     round(val_results['accuracy'], 6)))

    # Test here
    test_results = test(model, dataloaders['test'], verbose)
    print("Test   : test_dist:  {} | test_loss: {} | test_acc: {}".format(test_results['average_distance'],
                                                           test_results['loss'],
                                                           test_results['accuracy']))
    test_results["total_training_time"] = epoch_timer.total()
    return test_results


def test(model: nn.Module, dataloader: DataLoader, verbose: bool = False) -> Dict[str, Any]:
    model.eval()
    with torch.no_grad():
        dummy_images = dummy_batch_targets = None
        floss = nn.CTCLoss(blank=10)
        # Reset tracked metrics
        total_distance = samples = total_loss = total_accuracy = 0

        for image, str_targets in dataloader:
            # string to individual ints
            int_targets = [[int(c) for c in gt] for gt in str_targets]

            # Prepare image
            image = Variable(image)
            image = image.to(device)

            # Forward
            output = model(image)
            loss = apply_ctc_loss(floss, output, int_targets)

            total_loss += loss.item()
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
    return {'average_distance': total_distance / samples, 'loss': total_loss / samples,
            'accuracy' : total_accuracy / samples}


if __name__ == "__main__":
    args = parse_args()
    if len(args.seed) == 1:
        train(args, seed=args.seed[0], verbose=args.verbose)
    else:
        # Get the results for every seed
        results = [train(args, seed=seed, verbose=args.verbose) for seed in args.seed]
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
