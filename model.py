import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class StringNet(nn.Module):
    def __init__(self, n_classes):
        """
    In the constructor we instantiate two nn.Linear modules and assign them as
    member variables.

    D_in: input dimension
    D_out: output dimension
    """
        super(StringNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=0)
        self.pool1 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=0)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=0)
        self.pool2 = nn.MaxPool2d(2)

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=0)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=0)
        self.pool3 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(128 * 9 * 34, 128)  # depend on the flatten output,
        # checked if line 49. Dont know if there is an auto solution
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        """
    In the forward function we accept a Variable of input data and we must
    return a Variable of output data. We can use Modules defined in the
    constructor as well as arbitrary operators on Variables.
    """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool3(x)

        # print(x.shape, x.size(0))

        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        x = F.log_softmax(x, dim=1)

        return x
