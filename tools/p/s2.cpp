#include <iostream>
#include <omp.h> // OpenMP library

// Function to calculate harmonic sum
double harmonic_sum(long long n) {
    double sum = 0.0;

    // Parallelize the loop using OpenMP
    #pragma omp parallel for reduction(+:sum)
    for (long long i = 1; i <= n; ++i) {
        sum += 1.0 / i;
    }

    return sum;
}

int main() {
    long long n = 100000000000; // Number of terms in harmonic sum
    double result;

    // Start timing
    double start = omp_get_wtime();
    
    // Calculate harmonic sum
    result = harmonic_sum(n);
    
    // End timing
    double end = omp_get_wtime();

    std::cout << "Harmonic sum: " << result << std::endl;
    std::cout << "Time taken: " << (end - start) << " seconds." << std::endl;

    return 0;
}
