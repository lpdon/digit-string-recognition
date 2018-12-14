from argparse import ArgumentParser

from car_dataset import CAR


def parse_args():
    parser = ArgumentParser("Training script for Digit String Recognition PyTorch-Model.")
    parser.add_argument("-d", "--data", type=str, required=True,
                        help="Path to the root folder of the CAR-{A,B} dataset.")
    parser.add_argument("-e", "--epochs", type=int, default=50, required=False,
                        help="Number of epochs to train the model.")
    parser.add_argument("-v", "--verbose", action='store_true', default=False, required=False,
                        help="Print more information.")
    return parser.parse_args()


def build_model():
    pass


def train(dataset_root: str, epochs: int, verbose: bool=False):
    # Load dataset
    dataset = CAR(dataset_root)
    if verbose:
        print(dataset)

    build_model()

    # Train here
    for epoch in range(epochs):
        for image, gt in dataset:
            image.show()
            input("Press enter to show next image..")
    # Test here


if __name__ == "__main__":
    args = parse_args()
    train(args.data, args.epochs, args.verbose)
