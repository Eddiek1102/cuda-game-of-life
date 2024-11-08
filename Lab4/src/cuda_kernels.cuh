/*
Author: Edward Kwak
Class: ECE 4122
Last Modified Date: 11/8/2024

Description: Header file that contains function prototypes for implementing Game of Life using CUDA.

*/

#ifndef CUDA_KERNELS_CUH
#define CUDA_KERNELS_CUH

#include <cuda_runtime.h>


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
__global__ void updateGrid(bool* currentGrid, bool* nextGrid, int grid_width, int grid_height);

/*
    Description:
        - Populates the grid with live/dead cells.
    
    Parameters:
        - grid: Points to the grid.
        - gridWidth: Width of the grid.
        - gridHeight: Height of the grid.
*/
void randomizeGrid(bool* grid, int grid_width, int grid_height);

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
void runProgram(bool* d_currentGrid, bool* d_nextGrid, int grid_width, int grid_height, dim3 numThreads, dim3 numBlocks);

#endif
