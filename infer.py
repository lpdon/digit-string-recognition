from argparse import ArgumentParser

import torch
from torchvision.transforms import transforms

from car_dataset import CAR_A_STD, CAR_A_MEAN
from dataset import default_loader
from train import postproc_output

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run(args):
    width, height = args.target_size

    model = torch.load(args.load_path)

    data_transform = transforms.Compose([
        transforms.Resize((width, height)),
        transforms.ToTensor(),
        transforms.Normalize(mean=args.mean, std=args.std)
    ])
    image = default_loader(args.image)
    image = data_transform(image)
    image = image.to(device)
    image = image.view(1, 3, width, height)
    output = model(image)
    pred = postproc_output(output)
    print(pred)


def create_parser():
    parser = ArgumentParser("Inference script for Digit String Recognition PyTorch-Model.")
    parser.add_argument("-i", "--image", type=str, required=True,
                        help="Path to the image which should be used as input.")
    parser.add_argument("--target-size", "--is", nargs=2, type=int, default=(50, 120),
                        help="Y and X size to which the images should be resized.")
    parser.add_argument("-v", "--verbose", action='store_true', default=False, required=False,
                        help="Print more information.")
    parser.add_argument("--load_path", required=False, type=str, default="model.pt",
                        help="Path to the saved model. If empty, model won't be loaded.")
    parser.add_argument("--mean", nargs=3, type=float, default=CAR_A_MEAN,
                        help="Mean of RGB values of images in the dataset. Will be used for normalization.")
    parser.add_argument("--std", nargs=3, type=float, default=CAR_A_STD,
                        help="Standard deviation of RGB of images in the dataset. Will be used for normalization.")
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    run(args)
