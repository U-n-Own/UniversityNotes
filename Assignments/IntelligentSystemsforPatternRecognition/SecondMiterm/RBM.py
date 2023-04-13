""" Implementing a RBM for reconstucting images from MNIST dataset """

import numpy as np
import random 

class RBM:
    
    def __init__(self, num_visible, num_hidden, epochs, learning_rate): 
        """
        Initializing the RBM parameters
        :param num_visible: number of visible units
        :param num_hidden: number of hidden units
        :param epochs: number of epochs of training
        :param learning_rate: learning rate of the model
        """
    
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.learning_rate = learning_rate    
        self.epochs = epochs
        
        # Initialize the model parameters (weights and biases)
        
        # Initialize matrix W from  a normal distribution with mean 0 and stddev sq
        self.weights = np.random.normal(loc=0, scale=0.01, size=(num_hidden, num_visible))

        # These are our b and c in the picture avaialble in the assignment document
        
        # We should set the visible bias to log[pi/(1 − pi)] where pi is the proportion
        # of training vectors in which unit i is on
        self.visible_bias = np.zeros(num_visible)
        self.hidden_bias = np.zeros(num_hidden)

        self.epochs_error = np.array([])
        self.error_history = np.array([])
        
        self.epochs_error_list = []
        self.epochs_history_list = []

    def sigmoid(self, x):
       return 1.0/(1.0 + np.exp(-x))
    
    def _gibbs_sampling(self, visible, num_steps=1):
        """ 
        TODO: 
        Currently not used:
            Implement the gibbs sampling algorithm
            Used to approximate joint distribution of hidden and visible units

        """ 
        
        # Q is a matrix of size (num_visible, num_hidden) 
        Q = np.zeros((self.num_visible, self.num_hidden))
        
        # repeat until convergence
        for i in range(num_steps):

            
            Q += np.outer(visible, np.transpose(hidden))
        
        return 1/num_steps * Q
    
    
    def sample_hidden(self, visible, batch_size=10):
        # Sample the hidden layer activations given the visible layer : data

        # sigmoid(b+W^T*v)
        prob_hidden = self.sigmoid(np.dot(visible, (self.weights)) + self.hidden_bias)
        
        return prob_hidden
     
    def sample_visible(self, hidden):
        # Sample the visible layer activations given the hidden layer activations
        
        # calculate the probability of visible units given the hidden units
        
        prob_visible = self.sigmoid(np.dot(hidden, (self.weights.T)) + self.visible_bias)
        
        return prob_visible     
    
    def reconstruct_data(self, data):
        
        hidden = self.sigmoid(np.dot(data, (self.weights.T)) + self.hidden_bias)
        
        recon_probs = self.sigmoid(hidden.dot(self.weights)+ self.visible_bias)
        
        return recon_probs
   
    def update(self, wake, dream, negative, positive, data, recon_data):
        """
        
        Updates the weights and biases of the RBM model.

        """
        self.weights += self.learning_rate * (wake - dream)
        self.hidden_bias += self.learning_rate * (positive - negative)
        self.visible_bias += self.learning_rate * (data-recon_data)
                
    def contrastive_divergence_1(self, data, num_steps=1):
        """ 
        Implement the CD-1 algorithm to train the RBM
        
        returns: parameters for the next update
        
        """ 
        
        # Positive divergence : gradient of the log-likelihood of the data given the model

        # This is equation 7 in the Hinton paper
        positive_hi_prob = self.sigmoid(np.dot(data, self.weights.T) + self.hidden_bias)
                    
        wake_phase = np.dot(positive_hi_prob.T, data)         
        
        probs_hidden = np.array(positive_hi_prob)
        
        # Sample the hidden layer activations given the visible layer
        hidden_units_sample = (np.random.rand(*probs_hidden.shape) < probs_hidden).astype(float)
        
        
        # Negative divergence : gradient of the log-likelihood of the model given the data

        # This is equation 8 in the Hinton paper
        
        # We're doing gibbs sampling here for 1 step, first by sampling binary visible units from positive probabilities
        # In gibbs sampling we evaluate pv1: the probability of the visible units given the hidden units
        
        # v1: the sample of the visible units given the hidden units.
        # ph1: the probability of the hidden units given the visible units and finally, 
        # h1: the sample of the hidden units given the visible units. The reconstruction is given by v1.
        pv1 = self.sigmoid(np.dot(hidden_units_sample, self.weights) + self.visible_bias)
        v1 = (np.random.rand(self.num_visible) < pv1).astype('float')
        ph1 = self.sigmoid(np.dot(v1, self.weights.T) + self.hidden_bias)
        h1 = (np.random.rand(self.num_hidden) < ph1).astype('float')
            
        # From Hinton paper: When the hidden units are being driven by data, always use stochastic binary states. When they are
        # being driven by reconstructions, always use probabilities without sampling. 
        # So the step to sample here is not needed
        
        #dream = np.dot(recon_data_prob.T, neg_hi_probs)
        dream = np.dot(h1.T, v1)
            
        reconstruction_error = np.sum((data - v1) ** 2)

        reshaped_v1= v1.reshape(self.num_visible,)
        reshaped_data = data.reshape(self.num_visible,) 
        reshaped_positive_hi_prob = positive_hi_prob.reshape(self.num_hidden,)
        reshaped_h1 = h1.reshape(self.num_hidden,)
            
        return  wake_phase, dream, reconstruction_error, reshaped_v1, reshaped_data, reshaped_positive_hi_prob, reshaped_h1