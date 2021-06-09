# VOFML-19: an example of machine-learned VOF scheme

This repository contains an example of machine learning Finite Volume scheme for the 2D transport of 2-fluid flow, following the "VOFML" methodology of [(Després and Jourdren, JCP 2020)](https://doi.org/10.1016%2Fj.jcp.2020.109275) ([open access version](https://hal.archives-ouvertes.fr/hal-02447631)).

The scheme is a function taking as input the values of the volume fractions in a 3×3 stencil and the Courant number and returning the volume fraction of the flux.
This particular example of scheme has been trained by B. Després in 2019 during the writing of the above-cited paper.
It is a neural network of hidden layer sizes (80, 40, 20, 10), 

## Structure of the repository

The `data` directory stores the main content of this repository: the value of the coefficients of the neural network as csv files.
Each file stores a matrix: `Wi.csv` is the matrix of coefficients at layer i, while `bi.csv` is the vector of intercepts at layer i.

The rest of the repository is a Julia code using these coefficients in a neural network.
The Julia code has nothing special and is only meant as an example.
The neural network could easily be implemented in any other language.
