# VOFML-19

The purpose of this repository is to share an example of machine learning Finite Volume scheme for the 2D transport of 2-fluid flow, following the methodology of [(Després and Jourdren, JCP 2020)](https://doi.org/10.1016%2Fj.jcp.2020.109275).
The scheme takes as input the values of the volume fraction in a 3×3 stencil and the Courant number and returns the volume fraction of the flux.
This particular example is a neural network of hidden layer sizes (80, 40, 20, 10), 

The `data` directory stores the main content of this repository: the value of the coefficients of the neural network as csv files.
Each file stores a matrix: `Wi.csv` is the matrix of coefficients at layer i, while `bi.csv` is the vector of intercepts at layer i.

The rest of the repository is a Julia package presenting an example of use of these coefficients.
