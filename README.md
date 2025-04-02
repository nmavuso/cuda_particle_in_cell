# GPU-Accelerated PIC Plasma Simulation

## Overview

This project implements a **Particle-In-Cell (PIC) plasma simulation** on GPUs using CUDA. It includes:
- GPU-accelerated charge deposition and field updates.
- A simple Jacobi solver for the Poisson equation.
- A basic particle pusher (using an Euler update).

## Prerequisites 

NVIDIA GPU

nvcc 

## Files

- `main.cu`: Contains the CUDA code for the simulation.
- `Makefile`: Build script for compiling the project.
- `README.md`: This file.

## Installation

Clone the repository and compile the project:

```bash
git clone https://github.com/nmavuso/cuda_particle_in_cell.git
cd cuda_particle_in_cell
make

