import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# Contrastive loss

class   ContrastiveLoss(nn.Module):
    """Contrastive loss function.

    Description: This loss function is used to train siamese networks.
    It takes two inputs and a label (0 or 1) and returns the contrastive loss.
    The loss is defined as the mean of the squared euclidean distance between the two inputs
    if the label is 0, and the squared euclidean distance between the two inputs if the label is 1.
    The loss is then multiplied by the label (0 or 1) and summed with the loss multiplied by 1-label.
   
    Loss has a p(xi,xj) (Images embeddings)
       
    p(xi,xj) = sigmoid(W|f(xi) - f(xj)|): probability of beign the same class
       
    Loss(Batch) = sum mathbb{1}_{y=1} * log p(xi,xj) + (1 - mathbb{1}_{y=1}) * log(1 - p(xi,xj))
    """
    
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        
    def forward(self, output1, output2, label):
        
        # Compute contrastive loss
        # take 0.5 probability one of the two couple using random
        distance = F.pairwise_distance(output1, output2)     
        # Compute loss
        loss = torch.mean((1 - label) * torch.pow(distance, 2) +
                          label * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))
        return loss
      
       
# Triplet loss

class TripletLoss(nn.Module):

    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = torch.sum(torch.pow(anchor - positive, 2), 1)
        distance_negative = torch.sum(torch.pow(anchor - negative, 2), 1)
        loss = torch.relu(distance_positive - distance_negative + self.margin) # relu è max(0, x)
        return torch.mean(loss)
    
