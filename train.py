from argparse import ArgumentParser, Namespace

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.transforms import Resize, transforms

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

    model = build_model(11, 10, args.batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    floss = nn.NLLLoss()

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

        for batch_imgs, batch_targets in dataloaders[phase]:
            image = batch_imgs
            target = batch_targets

            #add padding
            new_target = []
            for i, gt in enumerate(target):
                gt = gt.rstrip()
                while len(gt) < 10:
                    gt += ":"

                new_gt = []
                for j, c in enumerate(gt):
                    new_gt.append(ord(c) - ord('0'))

                new_target.append(new_gt)

            target = new_target
            print(target)
            print(batch_targets)
            # assert(False)

            # target = [int(i[0]) for i in target]

            target = torch.Tensor(target)
            target = target.long()

            image = Variable(image)
            target = Variable(target)

            print(target)
            print(target.shape)

            if cuda_avail:
                image = image.cuda()
                target = target.cuda()

            optimizer.zero_grad()
            output = model(image)
            loss = floss(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_loss += 1

            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            samples += len(batch_targets)

        print("Epoch %d: loss: %f | acc: %f" % (epoch + 1, total_loss/num_loss, correct/samples))

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
