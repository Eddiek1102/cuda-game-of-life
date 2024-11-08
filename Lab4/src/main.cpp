#include <iostream>
#include <vector>
#include <string>
#include <chrono>

#include <cuda_runtime.h>
#include <SFML/Graphics.hpp>
#include "cuda_kernels.cuh"


int main(int argc, char* argv[]) {
    // Default spec.
    int windowWidth = 800;
    int windowHeight = 600;
    int cellSize = 5;
    int gridWidth = windowWidth / cellSize;
    int gridHeight = windowHeight / cellSize;
    int threadsPerBlock = 32;
    std::string memoryType = "NORMAL";

    // Parse command line arguments
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

    gridWidth = windowWidth / cellSize;
    gridHeight = windowHeight / cellSize;

    // Allocate memory for the grids based on memory type
    bool* d_currentGrid, * d_nextGrid;
    if (memoryType == "PINNED") {
        cudaMallocHost(&d_currentGrid, gridWidth * gridHeight * sizeof(bool));
        cudaMallocHost(&d_nextGrid, gridWidth * gridHeight * sizeof(bool));
    }
    else if (memoryType == "MANAGED") {
        cudaMallocManaged(&d_currentGrid, gridWidth * gridHeight * sizeof(bool));
        cudaMallocManaged(&d_nextGrid, gridWidth * gridHeight * sizeof(bool));
    }
    else {
        cudaMalloc(&d_currentGrid, gridWidth * gridHeight * sizeof(bool));
        cudaMalloc(&d_nextGrid, gridWidth * gridHeight * sizeof(bool));
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
    dim3 numBlocks((gridWidth + threadsPerBlockDim.x - 1) / threadsPerBlockDim.x,
        (gridHeight + threadsPerBlockDim.y - 1) / threadsPerBlockDim.y);

    // Set up the SFML window
    sf::RenderWindow window(sf::VideoMode(windowWidth, windowHeight), "CUDA-based Game of Life");
    window.setFramerateLimit(60);

    // Main loop
    unsigned long frameCounter = 0;
    double processTime = 0.0;

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed || (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Escape)) {
                window.close();
            }
        }

        // Start timing
        auto start_time = std::chrono::high_resolution_clock::now();

        // Run CUDA kernel to update the grid
        runGameOfLife(d_currentGrid, d_nextGrid, gridWidth, gridHeight, threadsPerBlockDim, numBlocks);

        // Swap the grids
        std::swap(d_currentGrid, d_nextGrid);

        // Stop timing
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        processTime += duration;
        frameCounter++;

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
        if (frameCounter == 100) {
            std::cout << "100 generations took " << processTime << " microseconds." << std::endl;
            frameCounter = 0;
            processTime = 0.0;
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
