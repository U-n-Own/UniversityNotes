import torch
from torch import nn


""" We will use Recurrent Neural Network implementation by PyTorch.

"""

class RNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.5):
        super(RNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, dropout=dropout, nonlinearity='relu')
        
        self.linear = nn.Linear(hidden_size, output_size)
        
        
    def forward(self, x, h0=None):
        
        #h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, h_n = self.rnn(x, h0)
        
        out = self.linear(out)
    
        return out, h_n