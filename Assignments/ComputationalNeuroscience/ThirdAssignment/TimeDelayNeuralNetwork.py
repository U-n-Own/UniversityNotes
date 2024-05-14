import torch
from torch import nn


""" Time Delay Neural Network implementation in PyTorch.

- Basically is a 1-D Convolutional Net with pooling and dilation.

Inputs of this network are 1-D or (3D?) tensors of shape (batch_size, sequence_length, input_size).

The nets was created for acoustic features.


"""


# Time Delay Neural Network with pytorch

class TimeDelayNeuralNetwork(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, kernel_size=3, dilation=1, dropout=0.5, pool_size=2, pool_stride=2, padding=0):
        super(TimeDelayNeuralNetwork, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.dropout = dropout
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.padding = padding
        
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size, dilation=dilation, padding=padding)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size, dilation=dilation, padding=padding)
        
        # Calculate output size after convolution and pooling
        #conv_output_size = ((input_size + 2 * padding - dilation * (kernel_size - 1) - 1) // pool_stride) + 1
        #pool_output_size = ((conv_output_size - pool_size) // pool_stride) + 1
        
        #self.fc_input_size = hidden_size * pool_output_size
        
        self.pool = nn.MaxPool1d(pool_size, stride=pool_stride)
        #self.pool = nn.AvgPool1d(pool_size, stride=pool_stride)
        
        self.dropout = nn.Dropout(dropout)
        
        self.nonlinearity = nn.ReLU()
        
        #self.fc = nn.Linear(self.fc_input_size, output_size)
        
        self.fc = nn.Linear(hidden_size, output_size)
         
         
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.nonlinearity(x)
        x = self.dropout(x)
        
        """
        x = self.conv2(x)
        x = self.nonlinearity(x)
        x = self.dropout(x)
        """
        
        x = torch.flatten(x, 1)
        
        x = self.fc(x)
        
        return x
    
    