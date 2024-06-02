import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge

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
    def __init__(self, N_h=500, N_x=1, rho_enforced=1.1, omega_in=1.1, washout=1000, alternative_method=None):
        self.N_h = N_h
        self.N_x = N_x
        self.rho_enforced = rho_enforced
        self.omega_in = omega_in
        self.washout = washout
        self.state = np.zeros((N_h, 1))
        self.bias = np.random.uniform(-1, 1, (N_h, N_x))
        self.W_x = self._initialize_W_x()
        self.W_h = self._initialize_W_h()
        self.W_out = None
        self.H = []
        self.alternative_method = alternative_method

    def _initialize_W_x(self):
        W_x = np.random.uniform(-self.omega_in, self.omega_in, (self.N_h, self.N_x))
        return self.omega_in * (W_x / np.linalg.norm(W_x))

    def _initialize_W_h(self):
        W_r = np.random.uniform(-1, 1, (self.N_h, self.N_h))
        recurrent_rho = max(abs(np.linalg.eigvals(W_r)))
        return W_r * (self.rho_enforced / recurrent_rho)

    def fit(self, train_input, target, epochs=4000, delay=0, test=False):
        """
        Here we will do all the necessary step to make work our ESN
        
        So basically we:
        
        1. Run the ESN on the input train and collect the hidden states
        2. Then we discard a transient called washout because the network needs to stabilize
        3. Then we train the *Readout* layer using the collected hidden states and the target
        by using the pseudo-inverse, but we can use any other method 
        like ridge regression or Linear Regression
        """  
        for t in range(epochs):
            self.state = np.tanh((self.W_x @ train_input[t]).reshape(-1,1) + (self.W_h @ self.state) + self.bias)
            self.H.append(self.state.copy())
        
        if delay > 0:
            #self.H = self.H[self.washout:-delay]
            self.H = self.H[self.washout:]
            target = target[self.washout:epochs]
        else:
            self.H = self.H[self.washout:]
            target = target[self.washout:epochs]
            
        self.H = np.array(self.H).reshape(-1, self.N_h) 
        
        # Washout target
        target = target[self.washout:epochs]
        self.H = self.H[self.washout:]
        
        if self.alternative_method == 'classification':
            self._fit_classificator(target)
        elif self.alternative_method == 'ridge':
            self.regressor = Ridge(alpha=1e-6)
            self.regressor.fit(self.H, target)
        else:
            self.W_out = np.linalg.pinv(self.H) @ target

        
    def predict(self):
        return self.H @ self.W_out
    
    def predict_test(self, inputs):
        
        predictions = np.zeros(inputs.shape[0])
        for t in range(len(inputs)):
            self.state = np.tanh((self.W_x @ inputs[t]).reshape(-1,1) + (self.W_h @ self.state) + self.bias)
            predictions[t] = self.regressor.predict(self.state.T)
            
        return predictions
            

    def plot_prediction(self, target, prediction):
        plt.figure(figsize=(30,12))
        plt.plot(target)
        plt.plot(prediction)
        plt.show()

    def calculate_error(self, target, prediction):
        return np.mean((prediction - target) ** 2)
    
    # For sequential MNIST we dont use pinv but will use log regressior
    def _fit_classificator(self, target):
        """
        For the sequential MNIST we will use a logistic regression to train the last layer
        and we want the logits to get the probabilities and the predictions for each
        """
        # Train with logistic regression the last layer
        self.W_out = LogisticRegression(max_iter=1000).fit(self.H, target)

        # Now compute logits
        self.logits = self.H @ self.W_out.coef_.T + self.W_out.intercept_

        # Now compute the probabilities
        self.probabilities = np.exp(self.logits) / np.sum(np.exp(self.logits), axis=1, keepdims=True)

        # Now compute the predictions
        self.predictions = np.argmax(self.probabilities, axis=1)

        # Now compute the accuracy
        self.accuracy = np.mean(self.predictions == target)

        print('Accuracy:', self.accuracy)