from argparse import ArgumentParser, Namespace

import torch
from torchvision.transforms import Resize, transforms

from car_dataset import CAR


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


def build_model():
    pass


def train(args: Namespace, verbose: bool = False):

    # Load dataset and create data loaders
    dataloaders = create_dataloader(args, verbose)

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    build_model()

    # Train here
    phase = 'train'
    for epoch in range(args.epochs):
        for batch_imgs, batch_targets in dataloaders[phase]:
            for image, target in zip(batch_imgs, batch_targets):
                transforms.ToPILImage()(image).show()
                print("Label: " + target)
                input("Press enter to show next image..")

    # Test here


if __name__ == "__main__":
    args = parse_args()
    train(args, args.verbose)
