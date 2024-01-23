In order to run a snippets write a code block with `run-language`

Eg : Python
```run-python
#pytorch tutorial on tensors

# Define a tensor and let's see some transformations

#A tensor is a mathematical object that represents a multilinear #function between vector spaces. It is a generalized form of vectors #and matrices that can have any number of dimensions.

#Let's define a 2x3 tensor with pytorch


import torch

# Define a 2x3 tensor
tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])

print(tensor)

```

## More complex example

Write code that draws n samples from dirchilet distribution and output an histogram

```run-python
import numpy as np
import matplotlib.pyplot as plt

# Define the parameters of the Dirichlet distribution
alpha = [1, 2, 3]

# Generate random samples from the Dirichlet distribution
samples = np.random.dirichlet(alpha, size=1000)

# Plot the resulting distribution
fig, ax = plt.subplots()
ax.hist(samples, bins=25, density=True)
ax.set_xlabel('Values')
ax.set_ylabel('Density')
ax.set_title('Dirichlet Distribution')
plt.show()

```

### Asking GPT to make a 3D Plot

The implementation works but python scripts do not... why??!?!?

```run-python

print(sys.version)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create a grid of points
x, y = np.linspace(-3, 3, 100), np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)

# Define the Gaussian function
def gaussian(x, y):
    return np.exp(-(x**2 + y**2))

# Compute the values of the Gaussian function at the grid points
Z = gaussian(X, Y)

# Plot the 3D Gaussian
ax.plot_surface(X, Y, Z, cmap='viridis')

# Add labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Gaussian')

# Show the plot
plt.show()

```
