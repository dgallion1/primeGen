#include <iostream>
#include <vector>
#include <primesieve.hpp>
#include <cuda_runtime.h>
#include <cstdlib>
#include <algorithm>
#include <iomanip>
#include <nvml.h>
#include <unistd.h>
#include <chrono>
#include <thread>

__device__ double atomicAddDouble(double* address, double value) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(__longlong_as_double(assumed) + value));
    } while (assumed != old);
    return __longlong_as_double(old);
}



__global__ void computeSum(double* partialSums, const char* signs, long long start, long long end, long long maxN) {
    extern __shared__ double shared[];  // Shared memory allocation
    long long i = blockIdx.x * blockDim.x + threadIdx.x + start;
    int tid = threadIdx.x;

    // Initialize shared memory with the computed term for each thread
    double term = 0.0;
    if (i <= maxN && i <= end) {
        term = 1.0 / i;
        if (signs[i - start] == 0 && i > 4) {
            term = -term;
            //printf("Neg %d thread %d: %f\n", blockIdx.x, tid, term);
        }
    }
    shared[tid] = term;

    // Synchronize to make sure all threads have written to shared memory
    __syncthreads();

    // Reduction in shared memory to compute the block-level partial sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && shared[tid] != 0.0 && s < 10) {
            //printf("Block %d thread %d: %f  next: %f index:  %d+%d\n", blockIdx.x, tid, shared[tid],  shared[tid + s], tid, s);
            shared[tid] += shared[tid + s];
            //printf("Blxxk %d thread %d: %f\n", blockIdx.x, tid, shared[tid]);
        }
        // Synchronize to ensure all threads have completed this step of reduction
        __syncthreads();
    }

    // Write the result from the first thread of each block to global memory
    if (tid == 0) {
        partialSums[blockIdx.x] = shared[0];
        //printf("Block %d partial sum: %f\n", blockIdx.x, shared[0]);
    }
}


void calculateSumInChunks(long long highestPrime, long long maxN, int chunkSize) {
    double totalResult = 0.0;
    long long start = 0;

    // Set CUDA device
    if (cudaSetDevice(0) != cudaSuccess) {
        std::cerr << "Error setting CUDA device." << std::endl;
        return;
    }

    // Allocate signs array on GPU (reuse for each chunk)
    char* d_signs;
    if (cudaMalloc((void**)&d_signs, chunkSize * sizeof(char)) != cudaSuccess) {
        std::cerr << "Error allocating memory for signs on GPU." << std::endl;
        return;
    }

    // Create CUDA streams
    const int numStreams = 4;
    cudaStream_t streams[numStreams];
    for (int i = 0; i < numStreams; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    // Process chunks of the range from 0 to highestPrime
    while (start < highestPrime) {
        long long end = std::min(start + chunkSize, highestPrime);
        std::vector<long long> primes;
        primesieve::generate_primes(start + 1, end, &primes);
        int numPrimes = primes.size();

        // Error checking for empty prime chunk
        if (numPrimes == 0) {
            std::cerr << "No primes found in the current chunk: start = " << start << ", end = " << end << std::endl;
            start = end;
            continue;
        }

        // Create signs array on host
        std::vector<char> signs(chunkSize, 1);
        bool currentSign = true;  // Start with positive sign
        long long previousPrimeIndex = 0;
        for (int i = 0; i < numPrimes; ++i) {
            long long primeIndex = primes[i] - (start + 1);
            for (long long j = previousPrimeIndex; j < primeIndex; ++j) {
                signs[j] = currentSign ? 1 : 0;
            }
            currentSign = !currentSign;  // Toggle the sign after each prime
            previousPrimeIndex = primeIndex;
        }
        // Fill the remaining signs after the last prime in the chunk
        for (long long j = previousPrimeIndex; j < chunkSize; ++j) {
            signs[j] = currentSign ? 1 : 0;
        }

        // Use multiple streams to copy data and launch kernels
        int streamIdx = (start / chunkSize) % numStreams;

        // Copy signs array to device asynchronously
        if (cudaMemcpyAsync(d_signs, signs.data(), chunkSize * sizeof(char), cudaMemcpyHostToDevice, streams[streamIdx]) != cudaSuccess) {
            std::cerr << "Error copying signs to GPU." << std::endl;
            cudaFree(d_signs);
            for (int i = 0; i < numStreams; ++i) {
                cudaStreamDestroy(streams[i]);
            }
            return;
        }

        int blks = 512;
        // Allocate partial sums array on GPU for this specific kernel launch
        int blocksPerGrid = std::min((chunkSize + blks - 1) / blks, 65535);
        double* d_partialSums;
        if (cudaMalloc((void**)&d_partialSums, blocksPerGrid * sizeof(double)) != cudaSuccess) {
            std::cerr << "Error allocating memory for partial sums on GPU." << std::endl;
            cudaFree(d_signs);
            for (int i = 0; i < numStreams; ++i) {
                cudaStreamDestroy(streams[i]);
            }
            return;
        }

        // Launch kernel asynchronously
        computeSum<<<blocksPerGrid, blks, blks * sizeof(double), streams[streamIdx]>>>(d_partialSums, d_signs, start + 1, end, maxN);

        // Wait for the stream to complete
        cudaStreamSynchronize(streams[streamIdx]);

        // Copy partial sums from GPU to host
        std::vector<double> partialSums(blocksPerGrid, 0.0);
        if (cudaMemcpy(partialSums.data(), d_partialSums, blocksPerGrid * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
            std::cerr << "Error copying partial sums from GPU." << std::endl;
            cudaFree(d_partialSums);
            cudaFree(d_signs);
            for (int i = 0; i < numStreams; ++i) {
                cudaStreamDestroy(streams[i]);
            }
            return;
        }

        // Sum partial sums on the host
        for (double partialSum : partialSums) {
            totalResult += partialSum;
        }

        // Free device memory for partial sums
        cudaFree(d_partialSums);

        // Update the start for the next chunk
        start = end;
    }

    // Clean up GPU memory
    cudaFree(d_signs);
    for (int i = 0; i < numStreams; ++i) {
        cudaStreamDestroy(streams[i]);
    }

    std::cout << std::fixed << std::setprecision(15) << "Resulting sum: " << totalResult << std::endl;

    // Reset CUDA device
    cudaDeviceReset();
}



int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <highest prime to generate>" << std::endl;
        return 1;
    }

    long long highestPrime = std::atoll(argv[1]);
    if (highestPrime <= 0) {
        std::cerr << "Please provide a positive number for the highest prime to generate." << std::endl;
        return 1;
    }

    // Set the maximum value of N for the calculation
    long long maxN = highestPrime;

    // Define chunk size for handling large number of primes (reduced to fit memory limits)
    int chunkSize = 10000; // Increased chunk size to improve GPU utilization

    // Call calculateSumInChunks to handle large prime arrays in chunks
    calculateSumInChunks(highestPrime, maxN, chunkSize);

    return 0;
}
