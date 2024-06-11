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
        
        # debug shapes for outer product
        for pattern in self.patterns:
            weights += (1/dimension)*np.outer(pattern, pattern)
            
        np.fill_diagonal(weights, 0)
        
        return weights

    def energy(self, pattern):
        """
        Compute the energy of the network given a pattern
        """
        return -0.5*np.dot(np.dot(pattern.T, self.weights), pattern)
    
    def overlap(self, pattern, recall_pattern):
        """
        Compute the overlap function: 
        
        m = 1/N * sum_i s_i * x_i
        """
        
        return (1/len(pattern))*np.dot(pattern, recall_pattern)
    
      
    def recall(self, pattern, steps=100, dimension=32, bias=0.7, by_frame=False, compute_energy=False, true_pattern=None):
        """
        Recall the pattern from the network, given a pattern choose randomly a
        neuron and save the update then do it for each neuron at least once
        """
        bias = bias
        energy = []
        new_pattern = pattern.copy()
        overlap = []
        
        # If by_frame is True, save each state of the pattern during the recall phase to see the evolution using visualization trough video
        # with frame equal to the number of steps
        frames = []
        
        for step_num in range(steps):
            
            previous_pattern = new_pattern.copy()
            update_order = np.random.permutation(dimension)
            print('Step number: {}'.format(step_num)) 
            for update in range(dimension):
                # choose a random permutation of the neurons
                
                pattern[update_order[update]] = np.sign(np.dot(self.weights[update_order[update]], pattern) + bias)

                #pattern[i] = np.sign(np.dot(self.weights[i], pattern) + bias)
                
                new_pattern = pattern.copy()
                
                if by_frame:
                    frames.append(new_pattern)
                
                if compute_energy:
                    # compute overlap
                    flattened_old = previous_pattern.flatten()
                    flattened_new = new_pattern.flatten()
                    true_pattern = true_pattern.flatten()
                    
                    overlap.append(self.overlap(flattened_new, true_pattern))
                    # compute the energy of the pattern
                    energy.append(self.energy(pattern))
                
        # if the pattern is equal to the one at previous step, stop
            if np.array_equal(previous_pattern, new_pattern):
                print('Pattern converged after {} steps'.format(step_num))
                return pattern, energy, overlap, frames
    
        print('Pattern did not converge after {} steps'.format(steps))
        
        return pattern, energy, overlap, frames