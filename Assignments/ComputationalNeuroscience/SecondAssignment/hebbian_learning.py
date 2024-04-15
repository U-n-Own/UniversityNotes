# Define a class for different hebbian learning implementations
import numpy as np
# integrating
from scipy.integrate import odeint
from scipy.integrate import solve_ivp


class HebbianLearning():
    """Class for different hebbian learning implementations
    
    Discretized Hebbian Learning
    
    Following this iterated map with euler integration:
    
    From this DE : \tau_w dw/dn = v*u

    dw/dn = 1/tau_w v*u = f(w)
    w(n+1) = w(n) + h f(w(n))
    w(n+1) = w(n) + h/tau_w v(n)*u(n)
    
    so 
    
    delta w(n) = eta v(n)*u(n)
    
    Then Oja's rule is given by:
    
    dw/dn = 1/tau_w (v*u - alpha v^2 w)
    
    Discretized Oja's rule:
    
    w(n+1) = w(n) + h/tau_w (v(n)*u(n) - alpha v(n)^2 w(n))
    """
    
    # Initialize the class
    def __init__(self, tau_w=0.1, eta=0.1, weights_dim=2, alpha=0.1):
        self.tau_w = tau_w
        self.eta = eta
        
        # for Oja's rule
        self.alpha = alpha
        
        # random weight initialization (dimension is given by the number of neurons in the network)
        #self.w = np.random.rand(weights_dim)
        self.w = np.random.rand(weights_dim)
        
    # Update the weights
    def _update_hebbian_simple(self, v, u):
        self.w += self.eta * v * u
        
    def _update_oja(self, v, u):
        self.w += self.eta * ((self.alpha * v) * (u - self.alpha * v * self.w))
            
    # Simulate the learning
    def train_hebbian_simple(self, data, epochs=100):
        
        w_shape = self.w.shape
        # Initialize the weight history
        w_history = np.zeros((len(data), *w_shape))
   
        # Simulate with a time step dynamics of tau_w
        for epoch in range(epochs):
            for u in data:
                v = self.predict(u)
                self._update_hebbian_simple(v, u)
                w_history[epoch] = self.w
            
        return w_history

    # Changing the Hebbian learning rule to be Oja's rule
    def train_oja(self, data, epochs=100):
           
        w_shape = self.w.shape 
        # Initialize the weight history
        w_history = np.zeros((len(data), *w_shape))

        # Simulate with a time step dynamics of tau_w
        for epoch in range(epochs):
            for u in data:
                v = self.predict(u)
                self._update_oja(v, u)
                w_history[epoch] = self.w

        return w_history
    
    def predict(self, u):
        # matrix w 
        return np.dot(u, self.w.T)
    
    