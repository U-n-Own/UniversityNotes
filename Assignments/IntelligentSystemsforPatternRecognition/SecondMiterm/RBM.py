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
        #self.weights = np.random.normal(mean = 0, stddev = np.sqrt(1.0/num_visible), size = (num_visible, num_hidden))
        self.weights = np.random.uniform(low = -1, high = 1, size = (num_hidden, num_visible))
        
        # these are our b and c in the picture avaialble in the assignment document
        self.visible_bias = np.zeros(num_visible)
        self.hidden_bias = np.zeros(num_hidden)
        
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
    
    
    def sample_hidden(self, visible, batch_size=10):
        # Sample the hidden layer activations given the visible layer : data

        # sigmoid(b+W^T*v)
        prob_hidden = self.sigmoid(np.dot(visible, (self.weights).T) + self.hidden_bias)
        
        return prob_hidden
     
    def sample_visible(self, hidden):
        # Sample the visible layer activations given the hidden layer activations
        
        # calculate the probability of visible units given the hidden units
        
        prob_visible = self.sigmoid(np.dot(hidden, (self.weights)) + self.visible_bias)
        
        return prob_visible     
   
   
    def reconstruct(self, data, hidden, prob_visible):
        # Reconstruct the data given the data
        #A “reconstruction” is produced by setting each vi to 1 with a probability given by P(vi = 1|h) = σ(bi + ∑_j w_ij*h_j)

        reconstructed_data = np.zeros(data.shape)
        
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if data[i][j] == 0:
                    reconstructed_data[i][j] = prob_visible[i][j]
                else:
                    reconstructed_data[i][j] = data[i][j]
        
        return reconstructed_data
   
   
    def update(self, wake, dream, negative, positive, data, recon_data):
        
        reconstruction_error = np.sum((wake - dream)**2)

        self.weights += self.learning_rate * (wake - dream)
        self.hidden_bias += self.learning_rate * (np.sum(positive, axis=0) - np.sum(negative, axis=0))
        self.visible_bias += self.learning_rate * (np.sum(data, axis=0) - np.sum(recon_data, axis=0))
        
        return reconstruction_error
                
    def contrastive_divergence_1(self, data, num_steps=1):
        """ 
        Implement the CD-1 algorithm to train the RBM
        """ 
        
        # Positive divergence : 
        positive_hi_prob = self.sample_hidden(visible=data)
            
        # transform in probabilities to fire the hidden units 1 or 0
        # put at zero the probabilities that are less than 0.5
        probs_hidden = np.array(positive_hi_prob)
        
        wake_phase = np.dot(data.T, probs_hidden)         

        # extract an array of n_hidden from a binomial and put to 1 the probs corresponding if the prob is higher
        hidden_units_sample = np.random.binomial(1, probs_hidden, size=probs_hidden.shape)
                    
        activation_mask = probs_hidden >= hidden_units_sample
        
        for i, activated in enumerate(activation_mask):
            probs_hidden[i][activated] = 1
            probs_hidden[i][~activated] = 0
        
        
        # Negative divergence

        recon_data_prob = self.sample_visible(hidden=probs_hidden)

        reconstruction_sample = np.random.binomial(1, recon_data_prob, size=recon_data_prob.shape)
        
        activation_mask = recon_data_prob >= reconstruction_sample
        
        for i, activated in enumerate(activation_mask):
            recon_data_prob[i][activated] = 1
            recon_data_prob[i][~activated] = 0
        
        neg_hi_probs = self.sample_hidden(visible=recon_data_prob) 

        dream = np.dot(recon_data_prob.T, neg_hi_probs)

        
        return wake_phase, dream, probs_hidden, recon_data_prob, neg_hi_probs