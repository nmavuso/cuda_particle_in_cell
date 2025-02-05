/*
    GPU-Accelerated 1D Electrostatic PIC Simulation
    ------------------------------------------------
    This simplified PIC simulation uses CUDA to accelerate:
      - Charge deposition (particles → grid)
      - Field solve (via a Jacobi iterative Poisson solver)
      - Particle push (velocity and position update)
    
    For demonstration purposes the simulation is electrostatic,
    not fully electromagnetic (i.e. it does not solve Maxwell’s equations).
    
    Compile with: nvcc -O3 -o pic main.cu
    Run with: ./pic
*/

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>

// -----------------------------------------------------------
// Simulation Parameters (normalized units)
#define N_PARTICLES 10000     // Number of particles
#define NX 512                // Number of grid cells
#define L 1.0f                // Physical length of the domain
#define DT 0.001f             // Time step
#define NSTEPS 1000           // Number of simulation time steps
#define EPSILON0 1.0f         // Permittivity (set to 1 for normalized units)
#define N_JACOBI 100          // Number of Jacobi iterations per time step

// Derived grid spacing (set as a constant here)
#define DX (L / NX)

// -----------------------------------------------------------
// CUDA error-checking macro (for debugging)
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      std::cerr << "CUDA Error: " << cudaGetErrorString(code) 
                << " " << file << " " << line << std::endl;
      if (abort) exit(code);
   }
}

// -----------------------------------------------------------
// Kernel: Deposit charge from particles onto the grid using linear weighting.
//         Each particle contributes to the two nearest grid points.
//         Uses atomicAdd to avoid race conditions.
__global__ void depositChargeKernel(const float* d_x, float* d_rho, int num_particles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_particles)
    {
        // Particle position in [0,L)
        float x = d_x[idx];
        // Map position to grid index (fractional index)
        float gridPos = x / DX;
        int i_left = static_cast<int>(floorf(gridPos));
        float delta = gridPos - i_left;  // Fractional distance
        int i_right = (i_left + 1) % NX;   // Periodic BC

        // Ensure indices are valid under periodicity
        i_left = (i_left + NX) % NX;
        
        // In normalized units, let each particle have charge = 1.0.
        // Distribute charge to the two nearest grid nodes.
        atomicAdd(&d_rho[i_left], 1.0f * (1.0f - delta));
        atomicAdd(&d_rho[i_right], 1.0f * delta);
    }
}

// -----------------------------------------------------------
// Kernel: One Jacobi iteration to solve Poisson's equation:
//         ∇²φ = -ρ/ε0   →   (φ[i+1] + φ[i-1] - 2φ[i])/(dx²) = -ρ[i]/ε0
//         Rearranged as: φ[i] = 0.5*(φ[i+1] + φ[i-1] + dx²*(-ρ[i]/ε0))
__global__ void jacobiSolverKernel(const float* d_rho, const float* d_phi, float* d_phi_new)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < NX)
    {
        // Periodic boundary indices
        int im = (i - 1 + NX) % NX;
        int ip = (i + 1) % NX;

        d_phi_new[i] = 0.5f * (d_phi[ip] + d_phi[im] + DX * DX * (-d_rho[i] / EPSILON0));
    }
}

// -----------------------------------------------------------
// Kernel: Copy one array into another (used to update phi between iterations)
__global__ void copyKernel(float* d_phi, const float* d_phi_new)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < NX)
    {
        d_phi[i] = d_phi_new[i];
    }
}

// -----------------------------------------------------------
// Kernel: Compute the electric field from the potential using central differences.
//         E = -dφ/dx  →  E[i] = -(φ[i+1] - φ[i-1])/(2*dx) with periodic BC.
__global__ void computeElectricFieldKernel(const float* d_phi, float* d_E)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < NX)
    {
        int im = (i - 1 + NX) % NX;
        int ip = (i + 1) % NX;
        d_E[i] = -(d_phi[ip] - d_phi[im]) / (2.0f * DX);
    }
}

// -----------------------------------------------------------
// Kernel: Update particle velocity and position (a simple push).
//         The electric field is interpolated from the grid using linear weighting.
//         The particle is then advanced with: v += (q/m)*E*dt and x += v*dt.
//         (Here we assume q/m = 1 in normalized units.)
__global__ void particlePushKernel(float* d_x, float* d_v, const float* d_E, int num_particles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_particles)
    {
        // Get particle position and map to grid
        float x = d_x[idx];
        float gridPos = x / DX;
        int i_left = static_cast<int>(floorf(gridPos));
        float delta = gridPos - i_left;
        int i_right = (i_left + 1) % NX;
        i_left = (i_left + NX) % NX;

        // Linear interpolation of the electric field
        float E_particle = d_E[i_left] * (1.0f - delta) + d_E[i_right] * delta;

        // Update velocity and position (simple Euler update)
        d_v[idx] += E_particle * DT;
        x += d_v[idx] * DT;

        // Apply periodic boundary conditions for position
        if (x < 0.0f) x += L;
        if (x >= L) x -= L;
        d_x[idx] = x;
    }
}

// -----------------------------------------------------------
// Main function
int main()
{
    // ---------------------------
    // Allocate and initialize host particle arrays
    float* h_x = new float[N_PARTICLES];
    float* h_v = new float[N_PARTICLES];

    // Seed random number generator and initialize particles uniformly in [0,L)
    srand(42);
    for (int i = 0; i < N_PARTICLES; i++)
    {
        h_x[i] = static_cast<float>(rand()) / RAND_MAX * L;
        h_v[i] = 0.0f;
    }

    // ---------------------------
    // Allocate device particle arrays
    float *d_x, *d_v;
    cudaCheckError(cudaMalloc((void**)&d_x, N_PARTICLES * sizeof(float)));
    cudaCheckError(cudaMalloc((void**)&d_v, N_PARTICLES * sizeof(float)));
    cudaCheckError(cudaMemcpy(d_x, h_x, N_PARTICLES * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_v, h_v, N_PARTICLES * sizeof(float), cudaMemcpyHostToDevice));

    // ---------------------------
    // Allocate device field arrays: charge density, potential, temporary potential, and E field.
    float *d_rho, *d_phi, *d_phi_new, *d_E;
    cudaCheckError(cudaMalloc((void**)&d_rho, NX * sizeof(float)));
    cudaCheckError(cudaMalloc((void**)&d_phi, NX * sizeof(float)));
    cudaCheckError(cudaMalloc((void**)&d_phi_new, NX * sizeof(float)));
    cudaCheckError(cudaMalloc((void**)&d_E, NX * sizeof(float)));

    // Initialize potential arrays to zero
    cudaCheckError(cudaMemset(d_phi, 0, NX * sizeof(float)));
    cudaCheckError(cudaMemset(d_phi_new, 0, NX * sizeof(float)));

    // ---------------------------
    // Set up CUDA kernel launch parameters.
    int threadsPerBlockParticles = 256;
    int blocksPerGridParticles = (N_PARTICLES + threadsPerBlockParticles - 1) / threadsPerBlockParticles;
    int threadsPerBlockGrid = 256;
    int blocksPerGridGrid = (NX + threadsPerBlockGrid - 1) / threadsPerBlockGrid;

    // ---------------------------
    // Main simulation loop
    for (int step = 0; step < NSTEPS; step++)
    {
        // (1) Reset the charge density array to zero
        cudaCheckError(cudaMemset(d_rho, 0, NX * sizeof(float)));

        // (2) Deposit charge from particles onto the grid.
        depositChargeKernel<<<blocksPerGridParticles, threadsPerBlockParticles>>>(d_x, d_rho, N_PARTICLES);
        cudaCheckError(cudaDeviceSynchronize());

        // (3) Solve Poisson's equation via Jacobi iterations to compute the potential.
        for (int iter = 0; iter < N_JACOBI; iter++)
        {
            jacobiSolverKernel<<<blocksPerGridGrid, threadsPerBlockGrid>>>(d_rho, d_phi, d_phi_new);
            cudaCheckError(cudaDeviceSynchronize());
            // Update the potential array
            copyKernel<<<blocksPerGridGrid, threadsPerBlockGrid>>>(d_phi, d_phi_new);
            cudaCheckError(cudaDeviceSynchronize());
        }

        // (4) Compute the electric field from the updated potential.
        computeElectricFieldKernel<<<blocksPerGridGrid, threadsPerBlockGrid>>>(d_phi, d_E);
        cudaCheckError(cudaDeviceSynchronize());

        // (5) Update particle velocity and position using the computed electric field.
        particlePushKernel<<<blocksPerGridParticles, threadsPerBlockParticles>>>(d_x, d_v, d_E, N_PARTICLES);
        cudaCheckError(cudaDeviceSynchronize());

        // Optionally, print a status message every 100 steps.
        if (step % 100 == 0)
        {
            std::cout << "Completed step " << step << std::endl;
        }
    }

    // ---------------------------
    // Copy final particle data back to host (for diagnostics or output).
    cudaCheckError(cudaMemcpy(h_x, d_x, N_PARTICLES * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheckError(cudaMemcpy(h_v, d_v, N_PARTICLES * sizeof(float), cudaMemcpyDeviceToHost));

    // (Optional) Print the first few particle positions and velocities.
    std::cout << "\nSample particle data after simulation:" << std::endl;
    for (int i = 0; i < 10; i++)
    {
        std::cout << "Particle " << i 
                  << "  x = " << h_x[i]
                  << "  v = " << h_v[i] << std::endl;
    }

    // ---------------------------
    // Clean up host and device memory.
    cudaFree(d_x);
    cudaFree(d_v);
    cudaFree(d_rho);
    cudaFree(d_phi);
    cudaFree(d_phi_new);
    cudaFree(d_E);
    delete[] h_x;
    delete[] h_v;

    return 0;
}
