/*
Author: Edward Kwak
Class: ECE 4122
Last Modified Date: 11/8/2024

Description:

*/

#ifndef CUDA_KERNELS_CUH
#define CUDA_KERNELS_CUH

#include <cuda_runtime.h>

// Kernel to update the grid based on the Game of Life rules
__global__ void updateGrid(bool* currentGrid, bool* nextGrid, int grid_width, int grid_height);

// Host function to initialize the grid with random values
void randomizeGrid(bool* grid, int grid_width, int grid_height);

// Host function to run the Game of Life using CUDA
void runProgram(bool* d_currentGrid, bool* d_nextGrid, int grid_width, int grid_height, dim3 numThreads, dim3 numBlocks);

#endif
