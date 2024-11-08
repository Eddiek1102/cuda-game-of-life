/*
Author: Edward Kwak
Class: ECE 4122
Last Modified Date: 11/8/2024

Description: 

*/

#include "cuda_kernels.cuh"
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <random>

// Kernel to update the grid based on the Game of Life rules
__global__ void updateGrid(bool* currentGrid, bool* nextGrid, int gridWidth, int gridHeight) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < gridWidth && y < gridHeight) {
        int count = 0;
        for (int i = -1; i <= 1; ++i) {
            for (int j = -1; j <= 1; ++j) {
                if (i == 0 && j == 0) continue;
                int newX = (x + i + gridWidth) % gridWidth;
                int newY = (y + j + gridHeight) % gridHeight;
                count += currentGrid[newY * gridWidth + newX];
            }
        }

        int i = y * gridWidth + x;
        if (currentGrid[i]) nextGrid[i] = count == 2 || count == 3;
        else nextGrid[i] = count == 3;
    }
}

// Function to initialize the grid with random values
void randomizeGrid(bool* grid, int gridWidth, int gridHeight) {
    std::mt19937 mt(static_cast<unsigned>(std::time(nullptr)));
    std::uniform_int_distribution<int> distribution(0, 1);

    for (int i = 0; i < gridWidth * gridHeight; i++) grid[i] = distribution(mt) == 1;
}

// Host function to run the Game of Life using CUDA
void runProgram(bool* d_currentGrid, bool* d_nextGrid, int gridWidth, int gridHeight, dim3 numThreads, dim3 numBlocks) {
    updateGrid  <<<numBlocks, numThreads >>> (d_currentGrid, d_nextGrid, gridWidth, gridHeight);
    cudaDeviceSynchronize();
}
