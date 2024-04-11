import numpy as np
import matplotlib.pyplot as plt

class HopfieldNet():
    """
    @Author: Vincenzo Gargano
    
    Code for a simple Hopfield network storing binary patterns of digits
    """
    
    def __init__(self, patterns, dimension=1024):
        """
        Initialize the Hopefield network with the patterns to store
        """
        self.patterns = patterns
        
        # Symmetric weights matrix
        self.weights = self._memorization(dimension)

        
    def _memorization(self, dimension=32):
        """
        Compute the weights matrix, basically the forward step
        """
        
        weights = np.zeros((dimension,dimension))
        
        # project weights matrix in a bigger space
        #weights = np.zeros((2048,2048))
         
        for pattern in self.patterns:
            weights += (1/dimension)*np.outer(pattern, pattern)
        
        np.fill_diagonal(weights, 0)
        
        return weights

    def energy(self, pattern):
        """
        Compute the energy of the network given a pattern
        """
        return -0.5*np.dot(np.dot(pattern.T, self.weights), pattern)
    
    def overlap(self, pattern):
        """Compute the overlap function
        """
    def recall(self, pattern, steps=100, dimension=32, bias=0.7):
        """
        Recall the pattern from the network, given a pattern choose randomly a
        neuron and save the update then do it for each neuron at least once
        """
        bias = bias
        energy = []
        new_pattern = pattern.copy()
        
        for step_num in range(steps):
            previous_pattern = new_pattern.copy()
            for _ in range(dimension):
                # choose a random neuron not yet updated                    
                i = np.random.randint(0, dimension)
                
                # Update the i-th neuron
                pattern[i] = np.sign(np.dot(self.weights[i], pattern) + bias)
                 
                new_pattern = pattern.copy()
                
                # compute the energy of the pattern
                energy.append(self.energy(pattern))
                
        # if the pattern is equal to the one at previous step, stop
            if np.array_equal(previous_pattern, new_pattern):
                print('Pattern converged after {} steps'.format(step_num))
                return pattern, energy
    
        print('Pattern did not converge after {} steps'.format(steps))
            
        return pattern, energy