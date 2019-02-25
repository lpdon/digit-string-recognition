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
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.transforms import transforms

from car_dataset import CAR, CAR_A_MEAN, CAR_A_STD
from cvl_dataset import CVL, CVL_MEAN, CVL_STD
from model import StringNet
from timer import Timer
from util import concat, length_tensor, format_status_line, write_to_csv, ImageWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mean_cache = {
    'CVL': CVL_MEAN,
    'CAR-A': CAR_A_MEAN,
    # 'CAR_B': CAR_B_MEAN
}
std_cache = {
    'CVL': CVL_STD,
    'CAR-A': CAR_A_STD,
    # 'CAR_B': CAR_B_STD
}

def create_parser():
    parser = ArgumentParser("Training script for Digit String Recognition PyTorch-Model.")
    parser.add_argument("-d", "--data", type=str, nargs='+', required=False, default="",
                        help="Path to the root folder of the CAR-{A,B} dataset.")
    parser.add_argument("-e", "--epochs", type=int, default=50,
                        help="Number of epochs to train the model.")
    parser.add_argument("--target-size", "--is", nargs=2, type=int, default=(50, 120),
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
    parser.add_argument("--mean", nargs='+', type=str, default=[CAR_A_MEAN,],
                        help="Mean of RGB values of images in the dataset. Will be used for normalization.")
    parser.add_argument("--std", nargs='+', type=str, default=[CAR_A_STD,],
                        help="Standard deviation of RGB of images in the dataset. Will be used for normalization.")
    return parser


def parse_args():
    parser = create_parser()
    args = parser.parse_args()

    # Use cached value if a cached string (CAR_A or CVL) is specified
    args.std = [std_cache[str(std)] if str(std) in std_cache else
                tuple([float(v) for v in std.split()]) if isinstance(std, str) else std
                for std in args.std]
    args.mean = [mean_cache[str(mean)] if str(mean) in mean_cache else
                 tuple([float(v) for v in mean.split()]) if isinstance(mean, str) else mean for mean in args.mean]

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

    # Wrap data in list if it is not already, since config does not return a list if only one element is specified.
    if not isinstance(args.data, list):
        args.data = [args.data]

    return args


def create_dataloader(data_paths, target_size, train_val_split, batch_size,
                      means=(CAR_A_MEAN,), stds=(CAR_A_STD,),
                      verbose: bool = False) -> Dict[str, DataLoader]:
    # Data augmentation and normalization for training
    # Just normalization for validation
    width, height = target_size
    if isinstance(data_paths, str):
        data_paths = [data_paths]
        means = [means]
        stds = [stds]
    assert len(data_paths) == len(means) == len(stds)
    datasets = []
    for data_path, mean, std in zip(data_paths, means, stds):
        print(mean, std)
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((width, height)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=4.0, hue=0.5),
                transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=10),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ]),
            'test': transforms.Compose([
                transforms.Resize((width, height)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ]),
        }

        # Load dataset
        if "car" in data_path.lower():
            dataset = CAR(data_path, transform=data_transforms, train_val_split=train_val_split, verbose=verbose)
        else:
            dataset = CVL(data_path, transform=data_transforms, train_val_split=train_val_split, verbose=verbose)
        if verbose:
            print(dataset)
        datasets.append(dataset)

    # Create training and validation dataloaders
    loader_names = ['train', 'test']
    if train_val_split < 1.0:
        loader_names.append('val')
    dataloaders_dict = {
        x: DataLoader(ConcatDataset([dataset.subsets[x] for dataset in datasets]),
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=4
                      ) for x in loader_names
    }
    return dataloaders_dict


def loss_func():
    return nn.CTCLoss(blank=10, reduction='sum')


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


def run(args: Namespace, seed: int = 0, verbose: bool = False) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    set_seed(seed)
    timer = Timer()
    timer.start()
    seq_length = 15  # TODO: make this a parameter

    if args.load_path is not None and Path(args.load_path).is_file():
        print("Loading model weights from: " + args.load_path)
        model = torch.load(args.load_path)
    else:
        model = StringNet(11, seq_length, args.batch_size).to(device)

    # Load dataset and create data loaders
    dataloaders = create_dataloader(args.data, target_size=args.target_size,
                                    train_val_split=args.train_val_split,
                                    means=args.mean, stds=args.std,
                                    batch_size=args.batch_size, verbose=verbose)

    # Train
    history = train(model, dataloaders['train'], dataloaders.get('val', None), lr=args.lr, epochs=args.epochs,
                    log_path=args.log, save_path=args.save_path, verbose=verbose)

    # Test
    # Replace None with this image writer to display failure cases: ImageWriter('output/', mean=args.mean, std=args.std)
    test_results = test(model, dataloaders['test'], verbose=verbose, failure_case_writer=None)
    print("Test         | " + format_status_line(test_results))

    timer.stop()
    test_results["total_training_time"] = timer.total()
    return history, test_results


def train(model: StringNet, train_data, val_data=None, lr=1e-4, epochs=100,
          log_path: str = None, save_path: str = None,
          verbose: bool = False) -> List[Dict[str, Any]]:
    # TODO: Early stopping

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    floss = loss_func()

    history = []
    batch_timer = Timer()
    epoch_timer = Timer()
    total_batches = len(train_data)
    for epoch in range(epochs):
        model.train()
        epoch_timer.start()
        batch_timer.reset()

        total_loss = num_samples = total_distance = total_accuracy = 0
        dummy_images = dummy_batch_targets = None

        for batch_num, (image, str_targets) in enumerate(train_data):
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

        if val_data is not None:
            val_results = test(model, val_data, verbose)
        else:
            val_results = {}

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

        if log_path is not None:
            write_to_csv(history_item, log_path, write_header=epoch == 0, append=epoch != 0)

        if save_path is not None:
            torch.save(model, save_path)

    return history


def test(model: nn.Module, dataloader: DataLoader, verbose: bool = False,
         failure_case_writer: ImageWriter = None) -> Dict[str, Any]:
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

            if failure_case_writer is not None:
                preds = postproc_output(output)
                for pred, gt, im in zip(preds, str_targets, image):
                    if not pred == gt:
                        failure_case_writer.write(im, "{}_vs._{}.png".format(pred if pred else "<empty>", gt))
        if verbose:
            print("Validation example:")
            print(model(dummy_images).argmax(2)[:, :10], dummy_batch_targets[:10])
    return {'avg_dist': total_distance / samples, 'avg_loss': total_loss / samples,
            'accuracy': total_accuracy / samples}


if __name__ == "__main__":
    args = parse_args()
    if len(args.seed) == 1:
        run(args, seed=args.seed[0], verbose=args.verbose)
    else:
        # Get the results for every seed
        results = [run(args, seed=seed, verbose=args.verbose) for seed in args.seed]
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
