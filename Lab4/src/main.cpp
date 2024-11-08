/*
Author: Edward Kwak
Class: ECE 4122
Last Modified Date: 11/8/2024

Description: Main file for Conway's Game of Life using CUDA for processing.

             Sample command line input: 
                ./Lab4 -n = 8 -c 5 -x 800 -y 600 -t NORMAL
                    -n = number of threads per block
                    -c = cell size
                    -x = window width
                    -y = window height
                    -t = memory type (NORMAL, PINNED, MANAGED)

*/

/*
    PREPROCESSOR DIRECTIVES
*/
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cuda_runtime.h>
#include <SFML/Graphics.hpp>
#include "cuda_kernels.cuh"


/*
    MAIN FUNCTION
*/
int main(int argc, char* argv[]) {
    // Default spec.
    int windowWidth = 800;
    int windowHeight = 600;
    int cellSize = 5;
    int gridWidth = windowWidth / cellSize;
    int gridHeight = windowHeight / cellSize;
    int threadsPerBlock = 32;
    std::string memoryType = "NORMAL";

    // Parse command line arguments & change default specs when necessary.
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-x" && i + 1 < argc) {
            windowWidth = std::stoi(argv[++i]);
        }
        else if (arg == "-y" && i + 1 < argc) {
            windowHeight = std::stoi(argv[++i]);
        }
        else if (arg == "-c" && i + 1 < argc) {
            cellSize = std::stoi(argv[++i]);
        }
        else if (arg == "-n" && i + 1 < argc) {
            threadsPerBlock = std::stoi(argv[++i]);
        }
        else if (arg == "-t" && i + 1 < argc) {
            memoryType = argv[++i];
        }
    }

    // Set grid dimensions.
    gridWidth = windowWidth / cellSize;
    gridHeight = windowHeight / cellSize;

    bool* d_currentGrid;
    bool* d_nextGrid;

    if (memoryType == "PINNED") {
        cudaMallocHost(&d_currentGrid, gridWidth * gridHeight * sizeof(bool));
        cudaMallocHost(&d_nextGrid, gridWidth * gridHeight * sizeof(bool));
    }
    else if (memoryType == "MANAGED") {
        cudaMallocManaged(&d_currentGrid, gridWidth * gridHeight * sizeof(bool));
        cudaMallocManaged(&d_nextGrid, gridWidth * gridHeight * sizeof(bool));
    }
    else if (memoryType == "NORMAL") {
        cudaMalloc(&d_currentGrid, gridWidth * gridHeight * sizeof(bool));
        cudaMalloc(&d_nextGrid, gridWidth * gridHeight * sizeof(bool));
    }
    else {
        std::cerr << "Invalid memory type.\n";
    }

    // Seed the initial grid with random values
    bool* h_grid = new bool[gridWidth * gridHeight];
    seedRandomGrid(h_grid, gridWidth, gridHeight);
    cudaMemcpy(d_currentGrid, h_grid, gridWidth * gridHeight * sizeof(bool), cudaMemcpyHostToDevice);
    delete[] h_grid;

    // Allocate memory on the host for visualization
    bool* h_currentGrid = new bool[gridWidth * gridHeight];

    // Define the CUDA thread and block dimensions
    dim3 threadsPerBlockDim(16, 16);
    dim3 numBlocks((gridWidth + threadsPerBlockDim.x - 1) / threadsPerBlockDim.x, (gridHeight + threadsPerBlockDim.y - 1) / threadsPerBlockDim.y);

    // Set up the SFML window
    sf::RenderWindow window(sf::VideoMode(windowWidth, windowHeight), "Lab 4: CUDA Game of Life");
    //window.setFramerateLimit(60);

    int generationCount = 0;
    float totalProcessingTime = 0.0f;

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed || (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Escape)) {
                window.close();
            }
        }

        // Initialize Events to keep track of generation time.
        cudaEvent_t generationStart;
        cudaEvent_t generationEnd;
        cudaEventCreate(&generationStart);
        cudaEventCreate(&generationEnd);

        // Time at start of generation.
        cudaEventRecord(generationStart);

        // Run CUDA kernel to update the grid
        runGameOfLife(d_currentGrid, d_nextGrid, gridWidth, gridHeight, threadsPerBlockDim, numBlocks);

        // Time at end of generation.
        cudaEventRecord(generationEnd);
        cudaEventSynchronize(generationEnd);

        // Calculate duration of generation & add to total processing time.
        float processingTime = 0.0f;        
        cudaEventElapsedTime(&processingTime, generationStart, generationEnd);
        totalProcessingTime += processingTime;

        // Increase the generation count
        generationCount++;

        // Swap the grids
        std::swap(d_currentGrid, d_nextGrid);

        // Copy the entire grid from device to host for visualization
        cudaMemcpy(h_currentGrid, d_currentGrid, gridWidth * gridHeight * sizeof(bool), cudaMemcpyDeviceToHost);

        // Draw the current grid
        window.clear();
        for (int y = 0; y < gridHeight; ++y) {
            for (int x = 0; x < gridWidth; ++x) {
                if (h_currentGrid[y * gridWidth + x]) {
                    sf::RectangleShape cell(sf::Vector2f(cellSize, cellSize));
                    cell.setPosition(x * cellSize, y * cellSize);
                    cell.setFillColor(sf::Color::White);
                    window.draw(cell);
                }
            }
        }
        window.display();

        // Display timing information every 100 frames
        if (generationCount >= 100) {
            std::cout << "100 generations took " << totalProcessingTime << " microseconds.\n";
            generationCount = 0;
            totalProcessingTime = 0.0f;
        }
    }

    // Free the allocated host memory
    delete[] h_currentGrid;

    // Free GPU memory
    if (memoryType == "PINNED") {
        cudaFreeHost(d_currentGrid);
        cudaFreeHost(d_nextGrid);
    }
    else {
        cudaFree(d_currentGrid);
        cudaFree(d_nextGrid);
    }

    return 0;
}
