from argparse import ArgumentParser, Namespace
from torchvision.transforms import Resize

from car_dataset import CAR


def parse_args():
    parser = ArgumentParser("Training script for Digit String Recognition PyTorch-Model.")
    parser.add_argument("-d", "--data", type=str, required=True,
                        help="Path to the root folder of the CAR-{A,B} dataset.")
    parser.add_argument("-e", "--epochs", type=int, default=50, required=False,
                        help="Number of epochs to train the model.")
    parser.add_argument("--target-size", nargs=2, type=int, default=(100, 300),
                        help="Y and X size to which the images should be resized.")
    parser.add_argument("-v", "--verbose", action='store_true', default=False, required=False,
                        help="Print more information.")
    return parser.parse_args()


def build_model():
    pass


def train(args: Namespace, verbose: bool = False):
    # Load dataset
    width, height = args.target_size
    transform = Resize((width, height))
    dataset = CAR(args.data, transform=transform)
    if verbose:
        print(dataset)

    build_model()

    # Train here
    for epoch in range(args.epochs):
        for image, gt in dataset:
            image.show()
            print("Label: " + gt)
            input("Press enter to show next image..")

    # Test here


if __name__ == "__main__":
    args = parse_args()
    train(args, args.verbose)
