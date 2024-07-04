# Creating a class to simulate a cyclical game of rock, paper, scissors
# The game is a competition in space we will use the mean field model to simulate the game
# The games take places in a grid with NxN cells each cell can be empty or occupied by r,p or s.

# Each timestep the two sites are chosen, the occupant of first can invade the second with a probability
# that will be defined later : P_p (paper can only invade rock) and so on, if invader invade replaces and so on
# if the two sites are the same then nothing happens, if the two sites are empty nothing happens
# if the two sites are the same and not empty nothing happens, probability of invasion of other events is 0.

# We choose randomly the first site and the next we will choose between
# the 8 neighbors of the first site since we are using a 2D grid


# Let's use a lattice model to simulate the game, we will use a 2D array to represent the grid


import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
from matplotlib import colors

class RockPaperScissors:
    
    def __init__(self, N, P_r, P_p, P_s, t, s_quant=0.33, p_quant=0.33, r_quant=0.34):
        self.N = N
        # variables for different initial states of quantities of rock, paper,
        # scissors in the grid summing to 1 so that the grid initially contains % of each
        self.s_quant = s_quant
        self.p_quant = p_quant
        self.r_quant = r_quant
        
        self.P_r = P_r
        self.P_p = P_p
        self.P_s = P_s
        # A unit of time t is N individual time steps (which will be one epoch of the simulation)
        self.t = t
        self.grid = np.zeros((N,N))
        
        
        # Set starting n_p, n_r, n_s according to the initial quantities of each state 
        self.dn_p = self.p_quant
        self.dn_r = self.r_quant 
        self.dn_s = self.s_quant
        
        # set to violet, red and blue the colors of the grid
        self.cmap = ListedColormap(['yellow', 'red', 'blue'])
        self.norm = colors.BoundaryNorm(boundaries=[0.5, 1.5, 2.5, 3.5], ncolors=3, clip=True)

        self.evolution = []

    def initialize(self):
        # initialize all the grid with s,p,r given as probability choosing 
        # the s_quant, p_quant, r_quant
        for i in range(self.N):
            for j in range(self.N):
                # choose a random number between 0 and 1
                rand_num = random.random()
                # set random state of the grid with the quantity of each state
                # so if they are all 0.33 each will have the same probability of being chosen
                # so divide in 3 regions which choose the state if random 0-33 choose s, 33-66 choose p, 66-100 choose r
                if rand_num < self.s_quant:
                    # set to 1 for scissors
                    self.grid[i][j] = 1
                elif rand_num < self.s_quant + self.p_quant:
                    # set to 2 for paper
                    self.grid[i][j] = 2
                else:
                    # set to 3 for rock
                    self.grid[i][j] = 3
         
             
    def update_grid_state(self):
        """Update the grid according to the rules of the game
        
        Rock invades Scissors: rock is coded as 1 and scissors as 2
        Scissors invade Paper: scissors is coded as 2 and paper as 3
        Paper invades Rock: paper is coded as 3 and rock as 1
        
        First site is choosen randomly and the next site is one of the 8 neighbors of the first site
        Repeat if the results do not pass the conditions of the game
        """

        # Short range dispersal strategy (only 8 neighbors are considered)
        for i in range(self.N):
            # choose the first site randomly
            x = random.randint(0, self.N-1)
            y = random.randint(0, self.N-1)
            
            x_next = random.randint(-1, 1)
            y_next = random.randint(-1, 1)
            
            # if the next site is out of bounds, choose again
            while x+x_next < 0 or x+x_next >= self.N or y+y_next < 0 or y+y_next >= self.N:
                x_next = random.randint(-1, 1)
                y_next = random.randint(-1, 1)
                
            current_site = self.grid[x][y]
            next_site = self.grid[x+x_next][y+y_next]

            match (current_site, next_site):
                case (1, 2):  # Rock beats Scissors
                    if random.random() < self.P_r:
                        self.grid[x+x_next][y+y_next] = next_site
                case (2, 3):  # Scissors beats Paper
                    if random.random() < self.P_s:
                        self.grid[x+x_next][y+y_next] = next_site
                case (3, 1):  # Paper beats Rock
                    if random.random() < self.P_p:  
                        self.grid[x+x_next][y+y_next] = next_site
                case _:
                    # No action needed for other combinations or same types
                    pass
                        
    def update_grid_state_lr(self):
        # This version is a long range dispersal where the next site is chosen randomly from the grid
        
        for i in range(self.N):
        
            # choose the first site randomly
            x = random.randint(0, self.N-1)
            y = random.randint(0, self.N-1)
            
            # choose the next site randomly
            x_next = random.randint(0, self.N-1)
            y_next = random.randint(0, self.N-1)
            
            current_site = self.grid[x][y]
            next_site = self.grid[x_next][y_next]

            match (current_site, next_site):
                case (1, 2):
                    if random.random() < self.P_r:
                        self.grid[x_next][y_next] = next_site
                case (2, 3):
                    if random.random() < self.P_s:
                        self.grid[x_next][y_next] = next_site
                case (3, 1):
                    if random.random() < self.P_p:
                        self.grid[x_next][y_next] = next_site
                case _:
                    pass
    
    
    def plot_grid(self):
        plt.figure()
        plt.imshow(self.grid, cmap=self.cmap, norm=self.norm)
        plt.axis('off')
        plt.show()        

    def count_states(self):
        # count the number of each state in the grid
        s_count = 0
        p_count = 0
        r_count = 0
        for i in range(self.N):
            for j in range(self.N):
                if self.grid[i][j] == 1:
                    s_count += 1
                elif self.grid[i][j] == 2:
                    p_count += 1
                elif self.grid[i][j] == 3:
                    r_count += 1
        return s_count, p_count, r_count
        
    def mean_field_eq(self):
        # Mean Field Equation for the game
        # example for dn_r/dt = n_r(P_r*n_s - P_p*n_p)
        
        # calculate the rate of change of each state
        self.dn_p = self.p_quant * (self.P_r * self.r_quant - self.P_s * self.s_quant)
        self.dn_r = self.r_quant * (self.P_s * self.s_quant - self.P_p * self.p_quant)
        self.dn_s = self.s_quant * (self.P_p * self.p_quant - self.P_r * self.r_quant)
        
        return self.dn_p, self.dn_r, self.dn_s
        
    def dynamics(self, delta_time=0.1):
        """Representing in a simplex plot the evolution of the game with each edge of the simplex representing the quantity of each state
        and the vertices the rate of change of the populations we will see the trajectoryù
        here until reaching a fixpoint in accord to the Mean Field Equation
        """       
        
        dn_p, dn_r, dn_s = self.mean_field_eq()
        
        self.p_quant += dn_p * delta_time
        self.r_quant += dn_r * delta_time
        self.r_quant += dn_s * delta_time
        
        self.evolution.append([self.p_quant, self.r_quant, self.s_quant])
       
    def to_simplex_coords(self, x1, x2, x3):
        """Convert 3D coordinates to 2D simplex coordinates."""
        return 0.5 * (2 * x2 + x3) / (x1 + x2 + x3), (np.sqrt(3) / 2) * x3 / (x1 + x2 + x3)

    def plot_simplex(self, evolution):
        """Plot the evolution of the system on a 2D simplex."""
        fig, ax = plt.subplots()
    
        # Draw the simplex
        simplex_points = np.array([
            self.to_simplex_coords(3, 0, 0),
            self.to_simplex_coords(0, 3, 0),
            self.to_simplex_coords(0, 0, 3),
            self.to_simplex_coords(3, 0, 0)  # Closing the triangle
        ])
        ax.plot(simplex_points[:, 0], simplex_points[:, 1], 'k-')

        # Plot the evolution
        for step in evolution:
            x1, x2, x3 = step
            simplex_coord = self.to_simplex_coords(x1, x2, x3)
            ax.plot(*simplex_coord, 'bo')
    
        ax.set_aspect('equal')
        ax.axis('off')  # Remove axes
        plt.title('Evolution on the Simplex')
        plt.show()         
    

    def play_game(self):
        
        # do not show the numbers on the axes
        plt.axis('off')
        
        self.initialize()
        plt.ion()
        
        for i in range(self.t):
            plt.clf()
            plt.imshow(self.grid, cmap=self.cmap, norm=self.norm)
            plt.draw()
            self.update_grid_state()
            self.dynamics()
        plt.ioff()
        plt.show()