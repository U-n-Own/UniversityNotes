#define a random matrix

using LinearAlgebra
using Plots

# A = rand(1:10, 200, 200)

# create a symmetric part of A

A_sympart = 1/2 * (A + A')

# show A_sympart with eye
heatmap(A_sympart, aspect_ratio=1)

issymmetric(A_sympart)

#check eigenvalues of A_sympart
eigvals(A_sympart)

# is antisymmetric
isantisymmetric(A_sympart)

# is transpose equal to -A=