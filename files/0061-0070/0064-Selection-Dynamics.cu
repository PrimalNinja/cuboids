%%writefile cuboids.cu
#include <iostream>
#include <cuda_runtime.h>
#include <ctime>

#define TEST_ID 27.02
#define BRAIN_COUNT 64

__global__ void evolutionKernel(int* batchEnergies, int* winnerIdx, int* maxEnergy) {
    if (threadIdx.x == 0) {
        int bestVal = -2147483648; // Minimum possible int
        int bestIdx = 0;
        for (int i = 0; i < BRAIN_COUNT; i++) {
            if (batchEnergies[i] > bestVal) {
                bestVal = batchEnergies[i];
                bestIdx = i;
            }
        }
        *winnerIdx = bestIdx;
        *maxEnergy = bestVal;
    }
}

int main() {
    srand(time(0));
    int *d_energies, *d_winner, *d_max;
    cudaMallocManaged(&d_energies, BRAIN_COUNT * sizeof(int));
    cudaMallocManaged(&d_winner, sizeof(int));
    cudaMallocManaged(&d_max, sizeof(int));

    std::cout << "--- F1 TELEMETRY [ID: " << TEST_ID << "] ---" << std::endl;
    std::cout << "Starting 3 Generations of Competitive Selection..." << std::endl;

    for (int gen = 1; gen <= 3; gen++) {
        // Simulate brain performance (higher = better resonance)
        for(int i=0; i<BRAIN_COUNT; i++) d_energies[i] = rand() % 5000;

        evolutionKernel<<<1, 1>>>(d_energies, d_winner, d_max);
        cudaDeviceSynchronize();

        std::cout << "Gen " << gen << " | Winner ID: " << *d_winner 
                  << " | Resonance: " << *d_max << std::endl;
    }

    std::cout << "STATUS: EVOLUTIONARY ANCHOR ESTABLISHED" << std::endl;
    std::cout << "------------------------------------------" << std::endl;

    cudaFree(d_energies); cudaFree(d_winner); cudaFree(d_max);
    return 0;
}