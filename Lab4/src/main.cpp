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


#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cuda_runtime.h>
#include <SFML/Graphics.hpp>
#include "cuda_kernels.cuh"

//####################################################################//
//####################################################################//
//                       FUNCTION PROTOTYPES                          //
//####################################################################//
//####################################################################//

/*
    Description:
        - Parses command line arguments and overrides default specs.
    
    Parameters:
        - argc: Number of arguments.
        - argv: Arguments.
        - windowWidth: Width of the window.
        - windowHeight: Height of the window.
        - cellSize: Size of cells.
        - numThreads: Number of threads per block.
        - memoryType: NORMAL/PINNED/MANAGED
*/
void parseCommandLineArgs(int argc, char* argv[], int& windowWidth, int& windowHeight, int& cellSize, int& numThreads, std::string& memoryType);

/*
    Description:
        - Frees allocated CUDA memory for the current and next grid states based on the memory type.
    
    Parameters:
        - d_currentGrid: Points to the device memory for the current grid state.
        - d_nextGrid: Points to the device memory for the next grid state.
        - memoryType: Memory allocation type (NORMAL/PINNED/MANAGED)
*/
void freeCudaMemory(bool* d_currentGrid, bool* d_nextGrid, std::string memoryType);


//####################################################################//
//####################################################################//
//                              MAIN                                  //
//####################################################################//
//####################################################################//

int main(int argc, char* argv[]) {
    // Default spec.
    int windowWidth = 800;
    int windowHeight = 600;
    int cellSize = 5;
    int gridWidth = windowWidth / cellSize;
    int gridHeight = windowHeight / cellSize;
    int numThreads = 32;
    std::string memoryType = "NORMAL";

    // Parse command line arguments and override defaults.
    parseCommandLineArgs(argc, argv, windowWidth, windowHeight, cellSize, numThreads, memoryType);

    // Grid dimensions
    gridWidth = windowWidth / cellSize;
    gridHeight = windowHeight / cellSize;

    // Device pointers for current & next grid state.
    bool* d_currentGrid;
    bool* d_nextGrid;

    // Allocate CUDA memory based on the memory type.
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
        std::cerr << "Invalid memory type: " << memoryType << "\n"; 
    }

    // Host grid for initializing with live/dead cells.
    bool* h_grid = new bool[gridWidth * gridHeight];
    randomizeGrid(h_grid, gridWidth, gridHeight);
    cudaMemcpy(d_currentGrid, h_grid, gridWidth * gridHeight * sizeof(bool), cudaMemcpyHostToDevice);
    // Free temp host grid.
    delete[] h_grid;

    // Host array to store the current grid state for rendering.
    bool* h_currentGrid = new bool[gridWidth * gridHeight];

    // Define CUDA thread block and grid sizes.
    dim3 numThreadsDimensions(numThreads, numThreads);
    dim3 numBlocks((gridWidth + numThreadsDimensions.x - 1) / numThreadsDimensions.x, (gridHeight + numThreadsDimensions.y - 1) / numThreadsDimensions.y);

    // Create SFML window.
    sf::RenderWindow window(sf::VideoMode(windowWidth, windowHeight), "Lab 4: CUDA Game of Life");
    
    // Set framerate limit.
    // window.setFramerateLimit(60);

    // Keep track of number of generations processed & cumulative processing time.
    int generationCount = 0;
    float totalProcessingTime = 0.0f;

    // Game loop
    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed || (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Escape)) {
                std::cout << "Program Terminated\n";
                window.close();
            }
        }

        // CUDA events for timing and generation processing.
        cudaEvent_t generationStart;
        cudaEvent_t generationEnd;
        cudaEventCreate(&generationStart);
        cudaEventCreate(&generationEnd);

        // Start timing the processing.
        cudaEventRecord(generationStart);

        // Call the CUDA kernel to update the grid.
        runProgram(d_currentGrid, d_nextGrid, gridWidth, gridHeight, numThreadsDimensions, numBlocks);

        // Swap pointers for current and next grid state.
        std::swap(d_currentGrid, d_nextGrid);

        // Copy the updated grid from device to host for rendering.
        cudaMemcpy(h_currentGrid, d_currentGrid, gridWidth * gridHeight * sizeof(bool), cudaMemcpyDeviceToHost);

        // Stop timing the processing.
        cudaEventRecord(generationEnd);
        cudaEventSynchronize(generationEnd);

        // Calculate the processing time for generation & add to cumulative processing time.
        float processingTime = 0.0f;        
        cudaEventElapsedTime(&processingTime, generationStart, generationEnd);
        totalProcessingTime += processingTime;

        // Increment generation count.
        generationCount++;

        // Clear window for rendering & render.
        window.clear();
        for (int y = 0; y < gridHeight; ++y) {
            for (int x = 0; x < gridWidth; ++x) {
                if (h_currentGrid[y * gridWidth + x]) {
                    sf::RectangleShape cell(sf::Vector2f(cellSize, cellSize));
                    cell.setPosition(x * cellSize, y * cellSize);
                    cell.setFillColor(sf::Color::Red);
                    window.draw(cell);
                }
            }
        }
        window.display();

        // Every 100 generations, report the average processing time.
        if (generationCount >= 100) {
            std::cout << "100 generations took " << totalProcessingTime << " microseconds.\n";
            generationCount = 0;
            totalProcessingTime = 0.0f;
        }
    }

    // Free the host memory.
    delete[] h_currentGrid;

    // Free CUDA memory.
    freeCudaMemory(d_currentGrid, d_nextGrid, memoryType);

    return 0;
}


//####################################################################//
//####################################################################//
//                      FUNCTION IMPLEMENTATIONS                      //
//####################################################################//
//####################################################################//

void parseCommandLineArgs(int argc, char* argv[], int& windowWidth, int& windowHeight, int& cellSize, int& numThreads, std::string& memoryType) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-n" && i + 1 < argc) {
            numThreads = std::stoi(argv[++i]);
        }
        else if (arg == "-c" && i + 1 < argc) {
            cellSize = std::stoi(argv[++i]);
        }
        else if (arg == "-x" && i + 1 < argc) {
            windowWidth = std::stoi(argv[++i]);
        }
        else if (arg == "-y" && i + 1 < argc) {
            windowHeight = std::stoi(argv[++i]);
        }
        else if (arg == "-t" && i + 1 < argc) {
            memoryType = argv[++i];
        }
        else {
            std::cerr << "Invalid/unknown command line arguments: " << arg << "\n";
        }
    }
}

void freeCudaMemory(bool* d_currentGrid, bool* d_nextGrid, std::string memoryType) {
    if (memoryType == "PINNED") {
        cudaFreeHost(d_currentGrid);
        cudaFreeHost(d_nextGrid);
        return;
    }
    cudaFree(d_currentGrid);
    cudaFree(d_nextGrid);
}