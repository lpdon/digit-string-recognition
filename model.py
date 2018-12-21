import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable
import torch

class StringNet(nn.Module):
    def __init__(self, n_classes):
        """
    In the constructor we instantiate two nn.Linear modules and assign them as
    member variables.

    D_in: input dimension
    D_out: output dimension
    """
        super(StringNet, self).__init__()
        
        self.debug = False
        
        self.classes = n_classes

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=0)
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=0)
		
		# batch norm (before activation)
        #self.conv2_bn = nn.BatchNorm2d(32) # batch normalization
		
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=0)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=0)
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=0)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=0)
        self.pool3 = nn.MaxPool2d(2)
        
        
		
	    # LSTM
        
        self.lstm_input_size = 512 * 8  # number of features = H * cnn_output_chanel = 8 * 512 = 4096
        self.lstm_hidden_size = 32
        self.lstm_num_layers = 1
        self.lstm_hidden = None
        self.lstm_cell = None

        self.lstm = nn.LSTM(self.lstm_input_size, self.lstm_hidden_size, self.lstm_num_layers)
        
        
        # FC: convert to 11-d probability vector
        self.fc_output_size = self.classes
        self.fc = nn.Linear(self.lstm_hidden_size, self.fc_output_size)
        

        # softmax:
        self.softmax = nn.Softmax()
        
        

    def forward(self, x: Tensor) -> Tensor:
        """
    In the forward function we accept a Variable of input data and we must
    return a Variable of output data. We can use Modules defined in the
    constructor as well as arbitrary operators on Variables.
    """
        if(self.debug) : print("input: ", x.size())
        out = F.relu(self.conv1(x))
        if(self.debug) : print("after conv1: ", out.size())
        
        out = self.pool1(out)
        if(self.debug) : print("after pool1: ", out.size())
        
        out = F.relu(self.conv2(out))
        if(self.debug) : print("after conv2: ", out.size())
        
        out = F.relu(self.conv3(out))
        if(self.debug) :  print("after conv3: ", out.size())
        
        
        #out = self.conv2_bn(out)
        #print("after batch norm: ", out.size())
        
        out = F.relu(self.conv4(out))
        if(self.debug) : print("after conv4: ", out.size())
        
        out = F.relu(self.conv5(out))
        if(self.debug) : print("after conv5: ", out.size())
        
        out = self.pool2(out)
        if(self.debug) : print("after pool2: ", out.size())
        
        out = F.relu(self.conv6(out))
        if(self.debug) : print("after conv6: ", out.size())
        
        out = F.relu(self.conv7(out))
        if(self.debug) : print("after conv7: ", out.size())
        
        out = self.pool3(out)
        if(self.debug) : print("after pool3: ", out.size())

        out = out.permute(3, 0, 2, 1) # D(out) = (W, batch_size, H, cnn_output_chanel)
       # print("after permute: ", out.size())
        #out.contiguous()
        out = out.contiguous().view(-1, 1, self.lstm_input_size) # D(out) = (seq_len, batch_size, lstm_input_size) where seq_len = W, lstm_input_size = H * cnn_output_chanel

        if(self.debug) : print("before LSTM: ", out.size())
        # LSTM
        out, (self.lstm_hidden, self.lstm_cell) = self.lstm(out, (self.lstm_hidden, self.lstm_cell)) # D(out) = (seq_len, batch,  hidden_size)

        if(self.debug) : print("after LSTM: ", out.size())
         # reshape
        out.contiguous()
        out = out.view(-1, self.lstm_hidden_size) # D(out) = (batch_size * seq_len, hidden_size)

        if(self.debug) : print("after reshape: ", out.size())
        
        # fc layer
        out = self.fc(out) # D(out) = (batch_size * seq_len, classes)
        if(self.debug) : print("after fc: ", out.size())
         
        out = F.log_softmax(out, dim=1)
        if(self.debug) : print("after softmax: ", out.size())
        
        return out

    def reset_hidden(self, batch_size):
        # reset hidden state for time 0
        h0 = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size) # random init
        #h0 = h0.cuda() if self.use_cuda else h0
        self.lstm_hidden = Variable(h0)

    def reset_cell(self, batch_size):
        # reset cell state for time 0
        c0 = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size) # random init
        #c0 = c0.cuda() if self.use_cuda else c0
        self.lstm_cell = Variable(c0)