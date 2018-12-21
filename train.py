from argparse import ArgumentParser, Namespace
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from car_dataset import CAR
from model import StringNet
from CTCLoss import *

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


def build_model() -> nn.Module:
    return StringNet(n_classes=11)


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

    model = build_model()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    floss = nn.CTCLoss()

    if cuda_avail:
        model.cuda()

    # Train here
    phase = 'train'
    model.train()
    
 
   
    
    for epoch in range(args.epochs):
        total_loss = num_loss = correct = samples = 0
        counter = 0
        for batch_imgs, batch_targets in dataloaders[phase]:
            
            model.zero_grad()
            model.hidden = model.reset_hidden(1)
            model.reset_cell(1)
            
            image = batch_imgs
            target = batch_targets
            #print("batch targets: " + str(target))

            
            target = [int(i) for i in target[0].strip()]
            

            #print("target: " + str(target))
            target = torch.Tensor(target)
            target = target.long()

            image = Variable(image)
            target = Variable(target)

            if cuda_avail:
                image = image.cuda()
                target = target.cuda()

           # optimizer.zero_grad()
            
            output = model(image)

          
            #Tensor of size (T, N, C) where C = number of characters in alphabet including blank
            output = output.view(output.size()[0], 1, 11)
           # print("output after reshape:", output.size())
            
            #targets: Tensor of size (N, S) or (sum(target_lengths))
            target = target.view(len(batch_targets), target.size()[0])
           # print("target after reshape:", target.size())
            
            #input_lengths: Tuple or tensor of size (N)
            input_lengths  = torch.tensor([output.size()[1]])
            input_lengths[0] = output.size()[0]
           # print("input_lengths:", input_lengths)
            
            #target_lengths: Tuple or tensor of size (N)
            target_lengths = torch.tensor([target.size()[1]])
            target_lengths[0] = target.size()[1]
           # print("target_lengths:", target_lengths.size())
 
            pred = output.argmax(2).squeeze()
            
           
       
            #out_np = output.data.numpy()
            #predictions = floss.decode_best_path(out_np)
            #print ("best_path_predictions[0]: " + str(out_np))
            
            
            
            loss = floss(output, target, input_lengths, target_lengths)
            
            if (counter % 20 == 0):
               print("gt size: " +str(target.size()) + ", value: " + str(target))
               print("pred size: " +str (pred.size()) + ", value: " + str(pred))
               print("loss: ", loss.item())
         
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_loss += 1

           
           # correct += pred.eq(target).sum().item()
            samples += len(batch_targets)
            
            counter += 1
            
            
        val_results = test(model, dataloaders['val'])
        print(
            f"Epoch {epoch + 1:2}: loss: {round(total_loss / num_loss, 6):8.6} | train_acc: {correct / samples:6.4} | "
              f"val_acc: {val_results['accuracy']:6.4}")

    # Test here
    test_results = test(model, dataloaders['test'])
    print(f"Test   : test_acc:  {test_results['accuracy']:6.4}")


def test(model: nn.Module, dataloader: DataLoader) -> Dict[str, Any]:
    model.eval()
    with torch.no_grad():
        # Reset tracked metrics
        correct = samples = 0
        for batch_imgs, batch_targets in dataloader:
            image = batch_imgs
            target = batch_targets

            target = [int(i[0]) for i in target]

            target = torch.Tensor(target)
            target = target.long()

            image = Variable(image)
            target = Variable(target)

            if torch.cuda.is_available():
                image = image.cuda()
                target = target.cuda()

            output = model(image)

            pred = output.argmax(1)
            correct += pred.eq(target).sum().item()
            samples += len(batch_targets)
    return {'accuracy': correct / samples}


if __name__ == "__main__":
    args = parse_args()
    train(args, args.verbose)
