# Von_Karman_Vortex_Street using Lattice Boltzmann Method

## What is this?
This is demonstartating a simulation flow past a cylinder using Lattice Boltzmann Method, Where the whole area of flow is divided into grids and the flow in these grids are studied .A particle can go in 9 directions, The collision of the particles with other particles moving in different direction is studied in oreder to get a flow.
At Re=80, Von Karman Vortex Street is observed

## What is Von Karman Vortex Street?
Periodic shedding of vortices from either side of the cylinder, 
forming a repeating pattern in the wake of the flow.

## Method
- D2Q9 lattice scheme
- BGK collision operator
- Reynolds number: Re=80
- Outputs: velocity fields (u, v) and density snapshots

## Results
![Video Project 1](https://github.com/user-attachments/assets/cbc09f05-1bc3-4e30-8a2f-10a6cf8b4c56)

## Dependencies
JAX, numpy, matplotlib

## How to run
python LBM_On_Cylinder.py

## Work in Progress
Building a Physics-Informed Neural Network (PINN) on top of this 
simulation to recover pressure from sparse LBM velocity fields 
using Navier-Stokes residuals


