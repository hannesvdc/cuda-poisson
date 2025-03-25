#include <iostream>
#include <cuda_runtime.h>
#include <fstream>
#include <algorithm>

// Index helper function
int getArrayIndex(int x_index, int y_index, int N) {
    return N * y_index + x_index;
}

__global__
void initializeRHS(float *f, float dx, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // global x-index = block_size * which block am I + which thread am I?
    int y = blockIdx.y * blockDim.y + threadIdx.y; // global x-index = block_size * which block am I + which thread am I?

    float x_coord = x * dx;
    float y_coord = y * dx;
    int index = getArrayIndex(x, y, N);
    f[index] = -sinf(2.0 * M_PI * x_coord) * sinf(M_PI * y_coord);
}

__global__
void jacobi_step(float *u_old, float *u_new, float *f, float dx, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // global x-index = block_size * which block am I + which thread am I?
    int y = blockIdx.y * blockDim.y + threadIdx.y; // global x-index = block_size * which block am I + which thread am I?

    if ( x > 0 && x < N-1 && y > 0 && y < N-1 ) {
        int index = getArrayIndex( x, y, N);
        int left  = getArrayIndex( x-1, y, N);
        int right = getArrayIndex( x+1, y, N);
        int up    = getArrayIndex( x, y+1, N);
        int down  = getArrayIndex( x, y-1, N);

        // Update into new array with Jacobi
        u_new[index] = 0.25f * (u_old[left] + u_old[right] + u_old[up] + u_old[down] - dx * dx * f[index]);
    }
}

void writeSolutionToCSV(const char* filename, float* u, int N) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file for writing.\n";
        return;
    }

    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {
            int index = y * N + x;
            file << u[index];
            if (x != N - 1) file << ",";
        }
        file << "\n";
    }

    file.close();
}

int main() {
    // Allocate u_old, u_new and f in regular memory.
    int N = 256;
    int n_elements = N * N;
    size_t size_bytes = n_elements * sizeof(float);
    float *u_old = new float[n_elements]();
    float dx = 1.0f / (N-1);

    // Device memory
    float *u_old_d, *u_new_d, *f_d;
    cudaMalloc(&u_old_d, size_bytes);
    cudaMalloc(&u_new_d, size_bytes);
    cudaMalloc(&f_d, size_bytes);

    // Define the Grid and Block sizes for the number of threads
    int nThreadsPerDim = 16;
    int nBlocksPerDim = 16;
    dim3 blockDim(nThreadsPerDim, nThreadsPerDim);
    dim3 gridDim(nBlocksPerDim, nBlocksPerDim);

    // Copy source term f and initial u_old to device
    cudaMemcpy(u_old_d, u_old, size_bytes, cudaMemcpyHostToDevice);
    initializeRHS<<<gridDim, blockDim>>>(f_d, dx, N);
    std::cout << "Launching grid of size (" << gridDim.x << ", " << gridDim.y << ") "
              << "with blocks of size (" << blockDim.x << ", " << blockDim.y << ")\n";


    // Call the Jacobi solver iteratively
    int max_iter = 100000;
    for (int iter = 0; iter < max_iter; ++iter) {
        jacobi_step<<<gridDim, blockDim>>>(u_old_d, u_new_d, f_d, dx, N);

        // Swap pointers (u_old becomes input for next iteration)
        std::swap(u_old_d, u_new_d);
    }

    // Move the solution u_old_d to the CPU
    cudaDeviceSynchronize();
    cudaMemcpy(u_old, u_old_d, size_bytes, cudaMemcpyDeviceToHost);

    // Store result in CSV file
    writeSolutionToCSV("solution.csv", u_old, N);

    // Free memory
    cudaFree(u_old_d);
    cudaFree(u_new_d);
    cudaFree(f_d);
    delete[] u_old;

    return 0;
}