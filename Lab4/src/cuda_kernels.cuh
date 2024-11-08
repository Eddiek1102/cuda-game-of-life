#ifndef CUDA_KERNELS_CUH
#define CUDA_KERNELS_CUH

#include <cuda_runtime.h>

// Kernel to update the grid based on the Game of Life rules
__global__ void updateGridKernel(bool* currentGrid, bool* nextGrid, int grid_width, int grid_height);

// Host function to initialize the grid with random values
void seedRandomGrid(bool* grid, int grid_width, int grid_height);

// Host function to run the Game of Life using CUDA
void runGameOfLife(bool* d_currentGrid, bool* d_nextGrid, int grid_width, int grid_height, dim3 threadsPerBlock, dim3 numBlocks);

#endif
