/*
Author: Edward Kwak
Class: ECE 4122
Last Modified Date: 11/8/2024

Description: This file contains implementations for functions used to run John Conway's Game
             of Life with CUDA. Function prototypes are located in cuda_kernels.cuh.

*/

#include "cuda_kernels.cuh"
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <random>


/*
    Description:
        - Assigns each thread to a cell (x, y) in the grid. The kernel counts the number of 
        live neighbros for each cell and determines the cell's next state.
    
    Parameters:
        - currentGrid: Points to device memory holding the current grid state.
        - nextGrid: Points to device memory for storing the next grid state.
        - gridWidth: Width of the grid.
        - gridHeight: Height of the grid.
*/
__global__ void updateGrid(bool* currentGrid, bool* nextGrid, int gridWidth, int gridHeight) {
    // Calculate the (x, y) coordinates of the cell this thread will process.
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if (x, y) is within the grid boundaries.
    if (x < gridWidth && y < gridHeight) {
        // Neighbor count for live cells.
        int count = 0;
        
        // Loop through the 3x3 neighbors.
        for (int i = -1; i <= 1; ++i) {
            for (int j = -1; j <= 1; ++j) {
                // Skip the cell itself.
                if (i == 0 && j == 0) continue;

                // Calculate the wrapped neighbor coordinates.
                int newX = (x + i + gridWidth) % gridWidth;
                int newY = (y + j + gridHeight) % gridHeight;

                // Increment coutn if neighbor cell is alive.
                count += currentGrid[newY * gridWidth + newX];
            }
        }
        
        // Determine the index for the current cell.
        int i = y * gridWidth + x;

        // Determine state of cell (live/dead).
        if (currentGrid[i]) nextGrid[i] = count == 2 || count == 3;
        else nextGrid[i] = count == 3;
    }
}

/*
    Description:
        - Populates the grid with live/dead cells.
    
    Parameters:
        - grid: Points to the grid.
        - gridWidth: Width of the grid.
        - gridHeight: Height of the grid.
*/
void randomizeGrid(bool* grid, int gridWidth, int gridHeight) {
    // Random number generator.
    std::mt19937 mt(static_cast<unsigned>(std::time(nullptr)));
    std::uniform_int_distribution<int> distribution(0, 1);

    // Populate the grid.
    for (int i = 0; i < gridWidth * gridHeight; i++) grid[i] = distribution(mt) == 1;
}

/*
    Desc:
        - Launches CUDA kernel 'updateGrid' to compute the next state of each cell in the grid. Calls
        cudaDeviceSynchronize() to ensure all threads have completed their work before control returns to host.
    
    Params: 
        - d_currentGrid: Points to device memory holding current grid state.
        - d-nextGrid: Points to the device memory for next grid state.
        - gridWidth: Width of the grid.
        - gridHeight: Height of the grid.
        - numThreads: Number of threads per block.
        - numBlocks: Number of blocks.
*/
void runProgram(bool* d_currentGrid, bool* d_nextGrid, int gridWidth, int gridHeight, dim3 numThreads, dim3 numBlocks) {
    // Launch the kernel.
    updateGrid  <<<numBlocks, numThreads >>> (d_currentGrid, d_nextGrid, gridWidth, gridHeight);
    // Ensure all threads have completed.
    cudaDeviceSynchronize();
}
