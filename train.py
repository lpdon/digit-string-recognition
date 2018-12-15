from argparse import ArgumentParser, Namespace
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor
import torch
import torch.nn as nn
from PIL import Image

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
    parser.add_argument("-v", "--verbose", action="store_true", default=False, required=False,
                        help="Print more information.")
    return parser.parse_args()


def build_model(d_in, d_out):
    return StringNet(d_in, d_out)


def train(args: Namespace, verbose: bool = False):
    # Load dataset
    width, height = args.target_size
    transform = transforms.Compose([Resize((width, height)), transforms.ToTensor()])

    dataset = CAR(args.data, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    if verbose:
        print(dataset)

    model = build_model((width, height), 10)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
    # floss = nn.CrossEntropyLoss()
    floss = nn.NLLLoss()

    cuda_avail = torch.cuda.is_available()

    if cuda_avail:
        model.cuda()

    model.train()

    # Train here
    for epoch in range(args.epochs):
        total_loss = 0
        num_loss = 0

        # for image, gt in dataset:
        for batch_idx, (image, gt) in enumerate(train_loader): 
            image = image.float()
            # print(image)

            # image = image.unsqueeze(0) 
            # print(image.size())

            # gt = int(str(gt)[0])
            # gt = tuple(map(int, gt))
            # print(gt)


            # gt = torch.Tensor([gt])
            gt = gt.long()

            print(gt)

            if cuda_avail:
                image = image.cuda()
                gt = gt.cuda()

            optimizer.zero_grad()
            output = model(image)
            loss = floss(output, gt)
            loss.backward()
            optimizer.step()

            # transforms.ToPILImage()(images[0]).show()

            # print(output)
            # print(gt.item())
            # print(loss.item())

            total_loss += loss.item()
            num_loss += 1

        print("%3d: %f" % (epoch, total_loss/num_loss))

    # Test here


if __name__ == "__main__":
    args = parse_args()
    train(args, args.verbose)
