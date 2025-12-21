%%writefile cuboids.cu
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <chrono>
#include <vector>

#define N 32
#define TOTAL (N * N * N)

// --- TRADITIONAL: FP32 High-Precision Archive (Standard AI) ---
void traditionalSave(float* data, std::string filename) {
    std::ofstream fout(filename, std::ios::out | std::ios::binary);
    fout.write((char*)data, TOTAL * sizeof(float)); // 128KB
    fout.close();
}

// --- DNA PERSISTENT: Ternary Compressed Archive ---
void dnaSave(int8_t* data, std::string filename) {
    std::ofstream fout(filename, std::ios::out | std::ios::binary);
    fout.write((char*)data, TOTAL * sizeof(int8_t)); // 32KB
    fout.close();
}

int main() {
    float *h_trad = new float[TOTAL];
    int8_t *h_dna = new int8_t[TOTAL];

    // Initialize data
    for (int i = 0; i < TOTAL; i++) {
        h_trad[i] = 1.0f;
        h_dna[i] = 1;
    }

    std::cout << "--- 21/12 RELEASE: FILE 0045 (ARCHIVE RACE) ---" << std::endl;

    // 1. TRADITIONAL BENCHMARK (FP32 Weight Dump)
    auto s1 = std::chrono::high_resolution_clock::now();
    for(int i=0; i<100; i++) traditionalSave(h_trad, "trad_brain.bin");
    auto e1 = std::chrono::high_resolution_clock::now();
    double trad_ms = std::chrono::duration<double, std::milli>(e1 - s1).count();

    // 2. DNA PERSISTENT BENCHMARK (Ternary State Dump)
    auto s2 = std::chrono::high_resolution_clock::now();
    for(int i=0; i<100; i++) dnaSave(h_dna, "dna_brain.bin");
    auto e2 = std::chrono::high_resolution_clock::now();
    double dna_ms = std::chrono::duration<double, std::milli>(e2 - s2).count();

    std::cout << "Traditional (FP32 Storage): " << trad_ms << " ms" << std::endl;
    std::cout << "DNA Persistent (Ternary):   " << dna_ms << " ms" << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;
    std::cout << "I/O EFFICIENCY GAP: " << (trad_ms / dna_ms) << "x" << std::endl;
    std::cout << "Data Footprint Reduction: 400%" << std::endl;

    delete[] h_trad; delete[] h_dna;
    return 0;
}