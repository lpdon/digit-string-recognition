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
                 lstm_hidden_dim: int = 100, bidirectional: bool = False, lstm_layers: int = 2,
                 lstm_dropout: float = 0.5, fc2_dim: int = 100):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
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

        self.lstm_forward = nn.LSTM(3072, self.lstm_hidden_dim, num_layers=self.lstm_layers, bias=True,
                                    dropout=lstm_dropout)

        if self.bidirectional:
            self.lstm_backward = nn.LSTM(3072, self.lstm_hidden_dim, num_layers=self.lstm_layers, bias=True,
                                         dropout=lstm_dropout)

        self.fc1 = nn.Linear(self.lstm_hidden_dim * self.directions, fc2_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(fc2_dim, n_classes)

    def init_hidden(self, input_length):
        # The axes semantics are (num_layers * num_directions, minibatch_size, hidden_dim)
        return (torch.zeros(self.lstm_layers * self.directions, input_length, self.lstm_hidden_dim).to(device),
                torch.zeros(self.lstm_layers * self.directions, input_length, self.lstm_hidden_dim).to(device))

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must
        return a Variable of output data. We can use Modules defined in the
        constructor as well as arbitrary operators on Variables.
        """
        current_batch_size = x.shape[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = res1 = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = F.relu(x.add(res1))        

        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)

        features = x.permute(3, 0, 1, 2).view(self.seq_length, current_batch_size, -1)
        hidden = self.init_hidden(current_batch_size)

        outs1, _ = self.lstm_forward(features, hidden)
        if self.bidirectional:
            outs2, _ = self.lstm_backward(features.flip(0), hidden)
            outs = outs1.add(outs2.flip(0))
        else:
            outs = outs1

        # Decode the hidden state of the last time step
        outs = self.fc1(outs)
        outs = self.dropout(outs)
        outs = self.fc2(outs)
        outs = F.log_softmax(outs, 2)

        return outs

    @property
    def directions(self) -> int:
        return 2 if self.bidirectional else 1
