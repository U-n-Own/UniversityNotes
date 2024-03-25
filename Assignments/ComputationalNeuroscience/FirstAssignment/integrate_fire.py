# Import libraries for simlation of Hodgkin-Huxley model and Izhikevich model

import numpy as np
from scipy.integrate import odeint

# Class for neuron types
class IntegrateAndFireNeuron():
    """Class for a simple integrate and fire neuron model. 
    The neuron has a membrane potential and a threshold.
    When the membrane potential reaches the threshold, the neuron fires and the membrane potential is reset
    to a resting potential. The membrane potential is updated according to the equation:
    
    Equivalent circuit: Only a capacitor and a current I(t) is considered.
    """
    
    def __init__(self) -> None:
        
        # Membrane potential
        self.u = -65
        # Threshold
        self.threshold = -55
        # Resting potential
        self.resting_potential = -70
        # Time constant
        self.tau = 10 
        # Time step
        self.dt = 0.1

    def du_dt(self, u, t, I):
        """Function to calculate the derivative of the membrane potential"""
        t_index = int((t - 0.0001) / self.dt)
        return (self.resting_potential - u + I[t_index]) / self.tau
    
    def simulate(self, I, T):
        
        t = np.arange(0, T, self.dt)
        u = odeint(self.du_dt, self.u, t, args=(I,))
        spikes = u >= self.threshold
        u[spikes] = self.resting_potential
        return t, u.flatten(), spikes.flatten()