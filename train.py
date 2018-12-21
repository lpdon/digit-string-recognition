from argparse import ArgumentParser, Namespace

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.transforms import Resize, transforms

import numpy as np

from car_dataset import CAR
from model import StringNet


def parse_args():
    parser = ArgumentParser("Training script for Digit String Recognition PyTorch-Model.")
    parser.add_argument("-d", "--data", type=str, required=True,
                        help="Path to the root folder of the CAR-{A,B} dataset.")
    parser.add_argument("-e", "--epochs", type=int, default=50, required=False,
                        help="Number of epochs to train the model.")
    parser.add_argument("--target-size", nargs=2, type=int, default=(100, 300),
                        help="Y and X size to which the images should be resized.")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for training and testing.")
    parser.add_argument("-v", "--verbose", action='store_true', default=False, required=False,
                        help="Print more information.")
    return parser.parse_args()


def create_dataloader(args: Namespace, verbose: bool = False):
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
    dataset = CAR(args.data, transform=data_transforms)
    if verbose:
        print(dataset)

    # Create training and validation dataloaders
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(dataset.subsets[x],
                                       batch_size=args.batch_size,
                                       shuffle=True,
                                       num_workers=4
                                       ) for x in ['train', 'test']
    }
    return dataloaders_dict


def build_model(n_classes, seq_length, batch_size):
    return StringNet(n_classes, seq_length, batch_size)


def train(args: Namespace, verbose: bool = False):
    # Load dataset
    # Load dataset and create data loaders
    dataloaders = create_dataloader(args, verbose)

    # Detect if we have a GPU available
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cuda_avail = torch.cuda.is_available()

    seq_length = 10
    model = build_model(11, seq_length, args.batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
    # floss = nn.NLLLoss()
    floss = nn.CTCLoss(blank=10, reduction="mean")

    if cuda_avail:
        model.cuda()

    # Train here
    phase = 'train'
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        num_loss = 0        
        correct = 0
        samples = 0

        dummy_images = None
        dummy_targets = None

        for batch_imgs, batch_targets in dataloaders[phase]:
            image = batch_imgs
            target = batch_targets

            #add padding
            padding = False
            new_target = []
            for i, gt in enumerate(target):
                gt = gt.rstrip()
                if padding:
                    while len(gt) < seq_length:
                        gt += ":"

                new_gt = []
                for j, c in enumerate(gt):
                    if j == 2:
                        break
                    new_gt.append(ord(c) - ord('0'))
                new_target += new_gt

            target = torch.Tensor(new_target)
            target = target.long()

            image = Variable(image)
            target = Variable(target)

            # target = target.view((len(batch_targets)*seq_length))

            if cuda_avail:
                image = image.cuda()
                target = target.cuda()

            optimizer.zero_grad()
            output = model(image)
            # print(output.shape)
            input_lengths = torch.full((output.shape[1],), output.shape[0], dtype=torch.long)
            target_lengths = torch.Tensor([min(2, len(t.rstrip())) for t in batch_targets]).type(torch.long).cuda()
            # print(input_lengths, target_lengths)
            loss = floss(output, target, input_lengths, target_lengths)
            # print(loss)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_loss += 1

            pred = output.max(1, keepdim=True)[1]

            # print(pred)
            # print(target)

            # correct += pred.eq(target.view_as(pred)).sum().item()
            samples += len(batch_targets)*seq_length

            dummy_images = image
            dummy_targets = target

        print("Epoch %d: loss: %f | acc: %f" % (epoch + 1, total_loss/num_loss, correct/samples))
        print(model(dummy_images).max(1, keepdim=True)[1], dummy_targets)

    # Test here
    # phase = 'test'
    # model.eval()

    # with torch.no_grad():
    #     for batch_imgs, batch_targets in dataloaders[phase]:
    #         image = batch_imgs
    #         target = batch_targets

    #         target = [int(i[0]) for i in target]

    #         target = torch.Tensor(target)
    #         target = target.long()

    #         image = Variable(image)
    #         target = Variable(target)

    #         if cuda_avail:
    #             image = image.cuda()
    #             target = target.cuda()

    #         output = model(image)
    #         loss = floss(output, target)

    #         total_loss += loss.item()
    #         num_loss += 1

    #         pred = output.max(1, keepdim=True)[1]
    #         correct += pred.eq(target.view_as(pred)).sum().item()
    #         samples += len(batch_targets)

    #     print("Test   : loss: %f | acc: %f" % (total_loss/num_loss, correct/samples))


if __name__ == "__main__":
    args = parse_args()
    train(args, args.verbose)
