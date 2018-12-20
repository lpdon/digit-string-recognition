from argparse import ArgumentParser, Namespace
from itertools import groupby
from typing import Dict, Any

import Levenshtein as lv
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from car_dataset import CAR
from model import StringNet


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = ArgumentParser("Training script for Digit String Recognition PyTorch-Model.")
    parser.add_argument("-d", "--data", type=str, required=True,
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
    parser.add_argument("--seed", type=int, default=666,
                        help="Seed used for the random number generator.")
    parser.add_argument("--lr", "--learning-rate", type=float, default=1e-4,
                        help="The initial learning rate.")
    parser.add_argument("-v", "--verbose", action='store_true', default=False, required=False,
                        help="Print more information.")
    return parser.parse_args()


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
    dataset = CAR(args.data, transform=data_transforms, train_val_split=args.train_val_split)
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
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        cudnn.deterministic = True
        cudnn.benchmark = False
        torch.cuda.manual_seed_all(seed)


def train(args: Namespace, verbose: bool = False):
    set_seed(args.seed)

    # Load dataset and create data loaders
    dataloaders = create_dataloader(args, verbose)

    # Detect if we have a GPU available
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cuda_avail = torch.cuda.is_available()

    seq_length = 10

    model = build_model(11, seq_length, args.batch_size).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    floss = nn.CTCLoss(blank=10)

    # Train here
    phase = 'train'
    for epoch in range(args.epochs):
        total_loss = num_loss = correct = samples = 0
        dummy_images = dummy_batch_targets = None
        model.train()

        for batch_imgs, batch_targets in dataloaders[phase]:
            image = batch_imgs
            target = batch_targets

            #string to individual ints
            new_target = []
            for gt in target:
              new_gt = [int(c) for c in gt.rstrip()]
              new_target += new_gt

            target = torch.Tensor(new_target)
            target = target.long()

            image = Variable(image)
            target = Variable(target)

            target = target.view((-1,))

            image = image.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(image)

            input_lengths = torch.full((output.shape[1],), output.shape[0], dtype=torch.long)
            target_lengths = torch.Tensor([len(t.rstrip()) for t in batch_targets]).type(torch.long)
            loss = floss(output, target, input_lengths, target_lengths)
            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            num_loss += 1

            pred = output.argmax(2)

            # correct += pred.eq(target.view_as(pred)).sum().item()
            samples += len(batch_targets)*seq_length

            dummy_images = image
            dummy_batch_targets = batch_targets

        val_results = test(model, dataloaders['val'])
        print(f"Epoch {epoch + 1:2}: loss: {round(total_loss / num_loss, 6):8.6} |"
              f" val_dist: {val_results['average_distance']:6.4}")

        print(model(dummy_images).argmax(2)[:, :10], dummy_batch_targets[:10])

    # Test here
    test_results = test(model, dataloaders['test'])
    print(f"Test   : test_dist:  {test_results['average_distance']:6.4}")


def test(model: nn.Module, dataloader: DataLoader) -> Dict[str, Any]:
    model.eval()
    with torch.no_grad():
        # Reset tracked metrics
        total_distance = samples = 0
        for batch_imgs, batch_targets in dataloader:
            image = batch_imgs
            image = Variable(image)

            if torch.cuda.is_available():
                image = image.cuda()

            output: torch.Tensor = model(image)

            preds = output.argmax(2)
            preds = preds.transpose(0, 1)
            for pred, gt in zip(preds, batch_targets):
                pred_str = [x[0] for x in groupby(pred)]
                pred_str = [str(int(p)) for p in pred_str if p != 10]
                pred_str = ''.join(pred_str)
                if total_distance == 0:
                    print(pred_str)
                    print(gt.rstrip())
                distance = lv.distance(pred_str, gt.rstrip())
                total_distance += distance
                samples += 1
    return {'average_distance': total_distance / samples}


if __name__ == "__main__":
    args = parse_args()
    train(args, args.verbose)
