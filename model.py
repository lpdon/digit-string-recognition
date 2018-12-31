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
    self.hidden_dim = 200

    self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=0)
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=0)
    self.pool1 = nn.MaxPool2d(2)

    self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=0)
    self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=0)
    self.pool2 = nn.MaxPool2d(2)

    self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=0)
    self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=0)
    self.pool3 = nn.MaxPool2d(2)

    self.fc1 = nn.Linear(128 * 9 * 34, 128 * seq_length)  # depend on the flatten output,
                                        #checked if line 49. Dont know if there is an auto solution

    self.lstm = nn.LSTM(128*seq_length, self.hidden_dim, bias=True, bidirectional=True)

    self.fc2 = nn.Linear(self.hidden_dim*2, n_classes)


  def init_hidden(self, input_length):
    # The axes semantics are (num_layers * num_directions, minibatch_size, hidden_dim)
      return (torch.zeros(2, input_length, self.hidden_dim).to(device),
              torch.zeros(2, input_length, self.hidden_dim).to(device))


  def forward(self, x):
    """
    In the forward function we accept a Variable of input data and we must
    return a Variable of output data. We can use Modules defined in the
    constructor as well as arbitrary operators on Variables.
    """
    current_batch_size = x.shape[0]
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = self.pool1(x)

    x = F.relu(self.conv3(x))
    x = F.relu(self.conv4(x))
    x = self.pool2(x)

    x = F.relu(self.conv5(x))
    x = F.relu(self.conv6(x))
    x = self.pool3(x)

    x = x.view(x.size(0), -1) #flatten
    x = F.relu(self.fc1(x))

    features = x.view(1, current_batch_size, -1).repeat(self.seq_length, 1, 1)
    hidden = self.init_hidden(current_batch_size)

    outs, hidden = self.lstm(features, hidden)

    # Decode the hidden state of the last time step
    outs = self.fc2(outs)
    outs = F.log_softmax(outs, 2)

    return outs
