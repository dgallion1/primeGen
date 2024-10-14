#include <iostream>
#include <vector>
#include <primesieve.hpp>
#include <cuda_runtime.h>
#include <cstdlib>
#include <algorithm>
#include <nvml.h>

__global__ void computeSum(double* result, const char* signs, long long start, long long end, long long maxN) {
    long long i = blockIdx.x * blockDim.x + threadIdx.x + start;
    if (i > maxN || i > end) return;

    double term = 1.0 / i;
    if (!signs[i - start] && i > 4) {
        term = -term;
    }

    //printf("Index: %lld, Term: %f, Sign: %d\n", i, term, signs[i - start]);
    atomicAdd(result, term);
}

void calculateSumInChunks(long long highestPrime, long long maxN, int chunkSize) {
    double totalResult = 0.0;
    long long start = 0;

    // Initialize NVML and check driver version
    if (nvmlInit() != NVML_SUCCESS) {
        std::cerr << "Failed to initialize NVML." << std::endl;
        return;
    }

    char driverVersion[80];
    if (nvmlSystemGetDriverVersion(driverVersion, sizeof(driverVersion)) != NVML_SUCCESS) {
        std::cerr << "Failed to get NVIDIA driver version." << std::endl;
        nvmlShutdown();
        return;
    }
    std::cout << "NVIDIA Driver Version: " << driverVersion << std::endl;

    // Get and print CUDA runtime version
    int runtimeVersion = 0;
    if (cudaRuntimeGetVersion(&runtimeVersion) != cudaSuccess) {
        std::cerr << "Error getting CUDA runtime version." << std::endl;
        nvmlShutdown();
        return;
    }
    std::cout << "CUDA Runtime Version: " << runtimeVersion << std::endl;

    // Shutdown NVML after checking
    nvmlShutdown();

    // Set CUDA device
    if (cudaSetDevice(0) != cudaSuccess) {
        std::cerr << "Error setting CUDA device." << std::endl;
        return;
    }

    // Check available GPU memory
    size_t freeMem = 0, totalMem = 0;
    cudaMemGetInfo(&freeMem, &totalMem);
    std::cout << "Free GPU Memory: " << freeMem / (1024 * 1024) << " MB, Total GPU Memory: " << totalMem / (1024 * 1024) << " MB" << std::endl;

    // Allocate result on host using pinned memory
    double* result;
    if (cudaHostAlloc((void**)&result, sizeof(double), cudaHostAllocMapped) != cudaSuccess) {
        std::cerr << "Error allocating pinned memory for result." << std::endl;
        return;
    }

    double* d_result;
    if (cudaHostGetDevicePointer((void**)&d_result, result, 0) != cudaSuccess) {
        std::cerr << "Error getting device pointer for result." << std::endl;
        cudaFreeHost(result);
        return;
    }

    // Allocate signs array on GPU (reuse for each chunk)
    char* d_signs;
    if (cudaMalloc((void**)&d_signs, chunkSize * sizeof(char)) != cudaSuccess) {
        std::cerr << "Error allocating memory for signs on GPU." << std::endl;
        cudaFreeHost(result);
        return;
    }

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

        // Copy signs array to device
        if (cudaMemcpy(d_signs, signs.data(), chunkSize * sizeof(char), cudaMemcpyHostToDevice) != cudaSuccess) {
            std::cerr << "Error copying signs to GPU." << std::endl;
            cudaFree(d_signs);
            cudaFreeHost(result);
            return;
        }

        // Set result memory on GPU to 0
        *result = 0.0;

        // Launch kernel
        int threadsPerBlock = 64;  // Further reduced to minimize GPU usage
        int blocksPerGrid = std::min((chunkSize + threadsPerBlock - 1) / threadsPerBlock, 65535);  // Limit the number of blocks
        computeSum<<<blocksPerGrid, threadsPerBlock>>>(d_result, d_signs, start + 1, end, maxN);
        if (cudaDeviceSynchronize() != cudaSuccess) {
            std::cerr << "Error synchronizing CUDA kernel." << std::endl;
            cudaFree(d_signs);
            cudaFreeHost(result);
            return;
        }

        // Accumulate result
        totalResult += *result;

        // Move to the next chunk
        start = end;
    }

    // Clean up GPU memory
    cudaFree(d_signs);
    cudaFreeHost(result);

    std::cout << "Resulting sum: " << totalResult << std::endl;

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
    int chunkSize = 10000; // Reduced to avoid GPU memory allocation issues

    // Call calculateSumInChunks to handle large prime arrays in chunks
    calculateSumInChunks(highestPrime, maxN, chunkSize);

    return 0;
}