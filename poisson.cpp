#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <fstream>

// Index helper function
int getArrayIndex(int x_index, int y_index, int N) {
    return N * y_index + x_index;
}

/*
 This function implements the Gauss-Seidel algorithm to solve the poission equation Delta u = f.
 We assume Dirichlet boundary conditions everywhere, i.e., u(x,y) = 0 when x = 0 or y = 0 or x = 1 or y = 1.
*/
void solvePoissonEquationJacobi(float *u_old, float *u_new, float *f, float dx, int N) {
    int max_iter = 100000;

    for ( int iter = 0; iter < max_iter; ++iter ) {
        if (iter % 1000 == 0) {
            std::cout <<"Iteration #: " << iter << std::endl;
        }

        // Boundary conditions are implicitly taken care of.
        for (int x_index = 1; x_index < N-1; ++x_index) {
            for (int y_index = 1; y_index < N-1; ++y_index) {
                int index = getArrayIndex( x_index, y_index, N);
                int left  = getArrayIndex( x_index-1, y_index, N);
                int right = getArrayIndex( x_index+1, y_index, N);
                int up    = getArrayIndex( x_index, y_index+1, N);
                int down  = getArrayIndex( x_index, y_index-1, N);

                // Update in place using Gauss-Seidel
                u_new[index] = 0.25 * (u_old[left] + u_old[right] + u_old[up] + u_old[down] - dx * dx * f[index]);
            }
        }

        // Swap u_old and u_new using the built-in function.
        std::swap(u_old, u_new);
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
    // Allocate u and fill with zeros
    // x_index = index % N
    // y_index = index / N (integer division)
    int N = 256;
    float *u_old = new float[N * N]();
    float *u_new = new float[N * N]();
    float dx = 1.0f / (N-1);

    // Setup the source term f(x, y) = sin(pi x) sin(pi y)
    float *f = new float[N * N];
    for (int x_index = 0; x_index < N; ++x_index) {
        for (int y_index = 0; y_index < N; ++y_index) {
            int index = getArrayIndex(x_index, y_index, N);
            float x = x_index * dx;
            float y = y_index * dx;
            f[index] = -sin(2.0 * M_PI * x) * sin(M_PI * y);
        }
    }

    // Solve the Poisson equation and save to file for visualization.
    solvePoissonEquationJacobi(u_old, u_new, f, dx, N);
    writeSolutionToCSV("solution.csv", u_old, N);

    // Delete all pointers
    delete[] u_old;
    delete[] u_new;
    delete[] f;

    return 0;
}
