import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
        self.hidden_dim = 64
        self.bidirectional = True
        self.lstm_layers = 1

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=0)
        self.dropout1 = nn.Dropout(p=0.8)
        self.pool1 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=0)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=0)
        self.dropout2 = nn.Dropout(p=0.8)
        self.pool2 = nn.MaxPool2d(2)

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=0)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=0)
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=0)
        self.dropout3 = nn.Dropout(p=0.8)
        self.pool3 = nn.MaxPool2d(2)

        self.conv8 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=0)
        self.conv9 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=0)
        self.conv10 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=0)
        self.dropout4 = nn.Dropout(p=0.8)
        self.pool4 = nn.MaxPool2d(2)

        self.lstm = nn.LSTM(256 * 1 * 13, self.hidden_dim, num_layers=self.lstm_layers, bias=True,
                            bidirectional=self.bidirectional)

        self.fc2 = nn.Linear(self.hidden_dim * self.directions, n_classes)

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
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout1(x)
        x = self.pool1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.dropout2(x)
        x = self.pool2(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = self.dropout3(x)
        x = self.pool3(x)

        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = self.dropout4(x)
        x = self.pool4(x)

        x = x.view(x.size(0), -1)  # flatten

        features = x.view(1, current_batch_size, -1).repeat(self.seq_length, 1, 1)
        hidden = self.init_hidden(current_batch_size)

        outs, hidden = self.lstm(features, hidden)

        # Decode the hidden state of the last time step
        outs = self.fc2(outs)
        outs = F.log_softmax(outs, 2)

        return outs

    @property
    def directions(self) -> int:
        return 2 if self.bidirectional else 1
