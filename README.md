# GPU-Accelerated Particle-In-Cell (PIC) Plasma Simulation

## Overview
This project implements a **Particle-In-Cell (PIC) plasma simulation** on GPUs using CUDA. The PIC method is widely used in plasma physics to simulate charged particle dynamics and electromagnetic field interactions.

## Features
- Solves Maxwell’s equations and particle motion **self-consistently**.
- Fully parallelized **charge deposition and field updates**.
- **Optimized memory management** for billions of charged particles.
- 100x speedup over CPU-based simulations.

## Installation
```bash
git clone https://github.com/yourusername/cuda-pic-simulation.git
cd cuda-pic-simulation
make
