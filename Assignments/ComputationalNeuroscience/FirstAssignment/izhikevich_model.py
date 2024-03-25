
import numpy as np
from scipy.integrate import odeint
# RungeKutta    


# Class for neuron types
class IzhikevichModel():
    """Class for Izhikevich neuron model. 
    @author Vincenzo Gargano
    
    2 Variables
        (u,w): u is the fast voltage variable (n shaped nullcline) 
        and w is the slow recovery variable (sigmoidean nullcline)
    5 Parameters: 
        (a,b, I): Determining the behaviour of the system
        (c,d): Affecting after-spike behaviour
        
        Equations: dudt = I + u^2 - w
                   dwdt = a(bu - w)
                   
        with reset condition if:u >= 1
                                u <- c, w <- w + d  
    """

    def __init__(self, a, b, c, d, I, w, u = -65, excitability_class=None, tau = 0.25) -> None:
        
        self.a = a #0.02
        self.b = b #0.2
        self.c = c #-65
        self.d = d #2
        self.I = I #5
        
        self.u = u
        self.w = w
        
        self.dt = tau
     
        # None for all the other if chages i change behavior of u
        self.excitability_class = excitability_class
     
    def dudt(self, I, t):
        """
        dudt is the first equation of the Izhikevich model actually not multiplying here by dt.
        """
        if self.excitability_class == None or self.excitability_class != 1:
            return 0.04*self.u**2 + 5*self.u + 140 - self.w + I
        elif self.excitability_class == 1:
            return self.dudt_c1(I, t)
         
    def dudt_c1(self, I, t):
        """
        for class 1 excitability
        """    
        return 0.04*(self.u**2) + 4.1*self.u + 108 - self.w + I
    
    def dwdt(self):
        """
        dwdt is the second equation of the Izhikevich model actually not multiplying here by dt.
        """
        if self.excitability_class == None or self.excitability_class != 2:
            return self.a*(self.b*self.u - self.w)
        elif self.excitability_class == 2:
            return self.a*(self.b*self.u+65)
    
    
    def simulate(self, I, T):
        """Simulate the Izhikevich model by using Leap-Frog method."""
        threshold = 30
        #t = np.arange(0, T, 0.1)
        t = np.arange(0, min(T, len(I)*0.1), 0.1)
        
        u = [] 
        w = []
        spikes = []
        
        for i in range(len(t)):
            
            self.u += self.dt*self.dudt(I[i], t = t[i])
            self.w += self.dt*self.dwdt()
            
            if self.u >= threshold:
                
                self.u = self.c
                self.w += self.d
                
                u.append(30)
        
            else:
                u.append(self.u)
            w.append(self.w)
                
        return t, u, w, spikes
    
        
        
    
     