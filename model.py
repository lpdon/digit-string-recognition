import torch
import torch.nn as nn
import torch.nn.functional as F


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

    self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=0)
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=0)
    self.pool1 = nn.MaxPool2d(2)

    self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=0)
    self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=0)
    self.pool2 = nn.MaxPool2d(2)

    self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=0)
    self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=0)
    self.pool3 = nn.MaxPool2d(2)

    self.fc1 = nn.Linear(128*9*34, 128*seq_length) #depend on the flatten output, 
                                        #checked if line 49. Dont know if there is an auto solution
        
    self.lstm = nn.LSTM(128*seq_length, self.hidden_dim)

    self.fc2 = nn.Linear(self.hidden_dim, n_classes)


  def init_hidden(self, input_length):
    # The axes semantics are (num_layers, minibatch_size, hidden_dim)
    if torch.cuda.is_available():
      return (torch.zeros(1, input_length, self.hidden_dim).cuda(),
              torch.zeros(1, input_length, self.hidden_dim).cuda())
    else:
      return (torch.zeros(1, input_length, self.hidden_dim),
              torch.zeros(1, input_length, self.hidden_dim))

  
  def forward(self, x):
    """
    In the forward function we accept a Variable of input data and we must 
    return a Variable of output data. We can use Modules defined in the 
    constructor as well as arbitrary operators on Variables.
    """
    input_length = len(x)

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

    x = x.view(x.size(0), -1) #flatten
    x = F.relu(self.fc1(x))

    x = [x for _ in range(self.seq_length)]

    features = torch.cat(x).view(self.seq_length, input_length, -1)
    lstm_out, hidden = self.lstm(features, self.init_hidden(input_length))

    x = self.fc2(lstm_out.view(input_length, self.seq_length, -1))
    x = F.log_softmax(x, dim=1)

    x = x.view((input_length * self.seq_length, -1))

    return x
