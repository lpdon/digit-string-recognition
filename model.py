import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ResBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, padding=1, stride=2)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.conv2 = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.res_conv = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=1, padding=0, stride=2)
        self.bn_res = nn.BatchNorm2d(output_channels)

    def forward(self, x):
        res = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x.add(self.bn_res(self.res_conv(res))))

        return x


class StringNet(nn.Module):
    def __init__(self, n_classes: int, seq_length: int, batch_size: int,
                 lstm_hidden_dim: int = 100, bidirectional: bool = True, lstm_layers: int = 2,
                 lstm_dropout: float = 0.5, fc2_dim: int = 100):
        super(StringNet, self).__init__()

        self.n_classes = n_classes
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.lstm_hidden_dim = lstm_hidden_dim
        self.bidirectional = bidirectional
        self.lstm_layers = lstm_layers

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(3, stride=1)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.res_block1 = ResBlock(64, 128)
        self.res_block2 = ResBlock(128, 256)
        self.res_block3 = ResBlock(256, 512)

        self.fc2 = nn.Linear(512 * 6, n_classes)

    def forward(self, x):
        current_batch_size = x.shape[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = res1 = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = F.relu(x.add(res1))

        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)

        features = x.permute(0, 3, 1, 2).view(current_batch_size, self.seq_length, -1)

        outs = self.fc2(features).permute(1, 0, 2)
        outs = F.log_softmax(outs, 2)

        return outs

    @property
    def directions(self) -> int:
        return 2 if self.bidirectional else 1
