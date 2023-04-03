""" Implementing a RBM for reconstucting images from MNIST dataset """

import numpy as np
import random 

class RBM:
    
    def __init__(self, num_visible, num_hidden, epochs, learning_rate): 
        # write document for init
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.learning_rate = learning_rate    
        self.epochs = epochs
        
        # Initialize the model parameters (weights and biases)
        
        # Initialize matrix W from  a normal distribution with mean 0 and stddev sq
        #self.weights = np.random.normal(mean = 0, stddev = np.sqrt(1.0/num_visible), size = (num_visible, num_hidden))
        self.weights = np.random.uniform(low = -1, high = 1, size = (num_visible, num_hidden))
        
        # these are our b and c in the picture avaialble in the assignment document
        self.visible_bias = np.zeros(num_visible)
        self.hidden_bias = np.zeros(num_hidden)
        
        
    def train(self, data):
        # Train the RBM using the CD-1 algorithm
        pass
    
    def inference(self, data):
        # Perform inference on the RBM given the data
        # Reconstructed data is returned
        pass
        
    def sigmoid(self, x):
        # Implement the sigmoid activation function
       return 1.0/(1.0 + np.exp(-x))
   
    def gibbs_sampling(self, num_steps, visible):
        # Implement the gibbs sampling algorithm
        # Used to approximate joint distribution of hidden and visible units
        
        # Q is a matrix of size (num_visible, num_hidden) 
        Q = np.zeros((self.num_visible, self.num_hidden))
        
        # repeat until convergence
        for i in range(num_steps):
            # sample the hidden units given the visible units
            hidden = self.sample_hidden(visible)
            # sample the visible units given the hidden units
            visible = self.sample_visible(hidden) 
            
            Q += np.outer(visible, np.transpose(hidden))
        
        return 1/num_steps * Q
    
    
    def sample_hidden(self, visible, batch_size):
        # Sample the hidden layer activations given the visible layer activations
        # Direct sampling

        # sigmoid(b+W^T*v)
        prob_hidden = self.sigmoid(np.dot(visible, (self.weights)) + self.hidden_bias)
        
        # sample the hidden units given the probability
        hidden = np.random.binomial(1, prob_hidden)
        
        
        return prob_hidden
     
    def sample_visible(self, hidden):
        # Sample the visible layer activations given the hidden layer activations
        # Direct sampling
        
        # calculate the probability of visible units given the hidden units
        
        prob_visible = self.sigmoid(np.dot(hidden, np.transpose(self.weights)) + self.visible_bias)
        
        return prob_visible     
    
        
    def contrastive_divergence_1(self, data, learning_rate, num_epochs, num_steps=1):
        # Implement the CD-1 algorithm to train the RBM
        
        # Positive divergence

        
        # Negative divergence
        pass