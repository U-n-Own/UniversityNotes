import pandas as pd
import seaborn as sns
import numpy as np
import random

import torch
import torch.nn as nn
from tqdm import tqdm

import torch.nn.functional as F

# import mnist from torchvision
from torchvision import datasets, transforms


class SimpleCNN(nn.Module):
    def __init__(self, embedding_size=32):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)

        self.fc1 = nn.Linear(32*5*5, 128)
        self.fc2 = nn.Linear(128, embedding_size) # the final emb size is 32

        self.embedding_size = embedding_size
    
    def forward_once(self, input):
        x = self.conv1(input)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        
        x = x.view(-1, 32*5*5)
        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        return x
    
class SiameseNetwork(nn.Module):
    def __init__(self, embedding_size=32):
        super(SiameseNetwork, self).__init__()
        
        self.cnn = SimpleCNN(embedding_size)
        
    def forward(self, input1, input2):
        output1 = self.cnn.forward_once(input1)
        output2 = self.cnn.forward_once(input2)
        
        return output1, output2
