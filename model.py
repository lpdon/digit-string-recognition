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
    def __init__(self, n_classes, seq_length, batch_size):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(StringNet, self).__init__()

        self.n_classes = n_classes
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.hidden_dim = 100
        self.bidirectional = False
        self.lstm_layers = 2

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(3, stride=1)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.res_block1 = ResBlock(64, 128)
        self.res_block2 = ResBlock(128, 128)

        self.res_block3 = ResBlock(128, 256)
        self.res_block4 = ResBlock(256, 256)
        self.res_block5 = ResBlock(256, 256)

        self.res_block6 = ResBlock(256, 512)
        self.res_block7 = ResBlock(512, 512)
        self.res_block8 = ResBlock(512, 512)

        self.lstm_forward = nn.LSTM(1024, self.hidden_dim, num_layers=self.lstm_layers, bias=True, 
                                    dropout=0.5)

        self.lstm_backward = nn.LSTM(1024, self.hidden_dim, num_layers=self.lstm_layers, bias=True, 
                                    dropout=0.5)

        self.fc2 = nn.Linear(self.hidden_dim * self.directions, n_classes)
        # self.fc2 = nn.Linear(self.hidden_dim, n_classes)
        self.dropout5 = nn.Dropout(p=0.5)

    def init_hidden(self, input_length):
        # The axes semantics are (num_layers * num_directions, minibatch_size, hidden_dim)
        return (torch.zeros(self.lstm_layers * self.directions, input_length, self.hidden_dim).to(device),
                torch.zeros(self.lstm_layers * self.directions, input_length, self.hidden_dim).to(device))

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
        x = sum1 = F.relu(x.add(res1))        

        x = self.res_block1(x)
        x = self.res_block2(x)

        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.res_block5(x)
        
        x = self.res_block6(x)
        x = self.res_block7(x)
        x = self.res_block8(x)

        x = x.view(x.size(0), -1)  # flatten

        features = x.view(1, current_batch_size, -1).repeat(self.seq_length, 1, 1)
        hidden = self.init_hidden(current_batch_size)

        # print(features.shape)

        outs1, _ = self.lstm_forward(features, hidden)
        outs2, _ = self.lstm_backward(features.flip(0), hidden)

        # print(outs1)
        # print(outs2)
        # print(outs1.shape)
        # assert False

        outs = outs1.add(outs2.flip(0))

        # print(outs.shape)
        # assert False

        # outs = outs[:, :, :self.hidden_dim].add(outs[:, :, self.hidden_dim:])
        # print(outs.shape)

        # Decode the hidden state of the last time step
        outs = self.fc2(outs)
        outs = F.log_softmax(outs, 2)

        return outs

    @property
    def directions(self) -> int:
        return 2 if self.bidirectional else 1
