import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

class EchoStateNetwork:
    """Echo State Network class.
    
    Parameters
    ----------
    N_h : int
        Number of hidden units.
    N_x : int
        Number of input units.
    rho_enforced : float
        Spectral radius of the recurrent weight matrix.
    omega_in : float
        Input scaling.
    washout : int
        Number of time steps to washout.
    W_x : np.ndarray
        Input weight matrix.
    W_h : np.ndarray
        Recurrent weight matrix.
    H : list
        List to store hidden states.
    W_out : np.ndarray
        Output weight matrix.
    state : np.ndarray
        Hidden state.
    bias : np.ndarray
        Bias vector.
    """
    def __init__(self, N_h=500, N_x=1, rho_enforced=1.1, omega_in=1.1, washout=1000):
        self.N_h = N_h
        self.N_x = N_x
        self.rho_enforced = rho_enforced
        self.omega_in = omega_in
        self.washout = washout
        self.state = np.zeros((N_h, 1))
        self.bias = np.random.uniform(-1, 1, (N_h, N_x))
        self.W_x = self._initialize_W_x()
        self.W_h = self._initialize_W_h()
        self.H = []

    def _initialize_W_x(self):
        W_x = np.random.uniform(-self.omega_in, self.omega_in, (self.N_h, self.N_x))
        return self.omega_in * (W_x / np.linalg.norm(W_x))

    def _initialize_W_h(self):
        W_r = np.random.uniform(-1, 1, (self.N_h, self.N_h))
        recurrent_rho = max(abs(np.linalg.eigvals(W_r)))
        return W_r * (self.rho_enforced / recurrent_rho)

    def fit(self, train_input, target, epochs=4000):
        
        for t in range(epochs):
            self.state = np.tanh((self.W_x @ train_input[t]).reshape(-1,1) + (self.W_h @ self.state) + self.bias)
            self.H.append(self.state)
            
        self.H = self.H[self.washout:]
        self.H = np.array(self.H).reshape(-1, self.N_h) 
        # Washout target
        
        target = target[self.washout:epochs]
        
        self.W_out = np.linalg.pinv(self.H) @ target
        
    def predict(self):
        return self.H @ self.W_out

    def plot_prediction(self, target, prediction):
        plt.figure(figsize=(30,12))
        plt.plot(target)
        plt.plot(prediction)
        plt.show()

    def calculate_error(self, target, prediction):
        return np.mean((prediction - target) ** 2)