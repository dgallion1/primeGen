#include <iostream>
#include <iomanip>
#include <limits>

#include <cuda_runtime.h>

__global__ void harmonic_sum_kernel(double *partial_sums, long long n) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0.0;

    // Each thread handles part of the sum
    for (long long i = idx + 1; i <= n; i += blockDim.x * gridDim.x) {
        sum += 1.0 / (double) i;
    }

    partial_sums[idx] = sum; // Store each thread's partial sum
}

double harmonic_sum(long long n) {
    double total_sum = 0.0;
    const int threads_per_block = 256;
     int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    const int num_blocks = prop.multiProcessorCount*2;  // Adjust based on your GPU
    std::cout << "Number of blocks: " << num_blocks << std::endl;
    double *partial_sums;
    double *d_partial_sums;

    // Allocate memory on host and device
    partial_sums = new double[threads_per_block * num_blocks];
    cudaMalloc(&d_partial_sums, threads_per_block * num_blocks * sizeof(double));

    // Initialize the GPU memory to zero (just in case)
    cudaMemset(d_partial_sums, 0, threads_per_block * num_blocks * sizeof(double));

    // Launch the kernel on the GPU
    harmonic_sum_kernel<<<num_blocks, threads_per_block>>>(d_partial_sums, n);

    // Check if kernel launch was successful
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return 0.0;
    }

    // Synchronize to ensure all threads are done
    cudaDeviceSynchronize();

    // Copy the partial sums back to the host
    cudaMemcpy(partial_sums, d_partial_sums, threads_per_block * num_blocks * sizeof(double), cudaMemcpyDeviceToHost);

    // Sum the partial sums from all threads
    for (int i = 0; i < threads_per_block * num_blocks; ++i) {
        total_sum += partial_sums[i];
    }

    // Free memory
    delete[] partial_sums;
    cudaFree(d_partial_sums);

    return total_sum;
}

int main() {
    long long n = 10E7;  // Number of terms
    double result,aa;
    aa=1.123456789123456789;
    result = harmonic_sum(n);
    std::cout << std::fixed;
    std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);
    std::cout << n << ") Harmonic sum: " << result << std::endl;
    return 0;
}
