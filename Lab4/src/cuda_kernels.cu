#include "cuda_kernels.cuh"
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <random>

// Kernel to update the grid based on the Game of Life rules
__global__ void updateGridKernel(bool* currentGrid, bool* nextGrid, int gridWidth, int gridHeight) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < gridWidth && y < gridHeight) {
        int count = 0;
        for (int i = -1; i <= 1; ++i) {
            for (int j = -1; j <= 1; ++j) {
                if (i == 0 && j == 0) continue;
                int nx = (x + i + gridWidth) % gridWidth;
                int ny = (y + j + gridHeight) % gridHeight;
                count += currentGrid[ny * gridWidth + nx];
            }
        }

        int idx = y * gridWidth + x;
        if (currentGrid[idx]) {
            nextGrid[idx] = (count == 2 || count == 3);
        }
        else {
            nextGrid[idx] = (count == 3);
        }
    }
}

// Function to initialize the grid with random values
void seedRandomGrid(bool* grid, int gridWidth, int gridHeight) {
    std::mt19937 mt(static_cast<unsigned>(std::time(nullptr)));
    std::uniform_int_distribution<int> distribution(0, 1);
    for (int i = 0; i < gridWidth * gridHeight; i++) {
        grid[i] = (distribution(mt) == 1);
    }
}

// Host function to run the Game of Life using CUDA
void runGameOfLife(bool* d_currentGrid, bool* d_nextGrid, int gridWidth, int gridHeight, dim3 threadsPerBlock, dim3 numBlocks) {
    updateGridKernel << <numBlocks, threadsPerBlock >> > (d_currentGrid, d_nextGrid, gridWidth, gridHeight);
    cudaDeviceSynchronize();
}
