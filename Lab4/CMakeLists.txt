# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Set the required CUDA version (optional)
set(CMAKE_CUDA_STANDARD 11)

# Project name
project(Lab4 LANGUAGES CXX CUDA)

# Include SFML
include_directories(${PROJECT_SOURCE_DIR}/../SFML/include)

link_directories(${PROJECT_SOURCE_DIR}/../SFML/lib)

# Set CUDA architecture
#set(CUDA_ARCHITECTURES 52)

# Specify the target executable
add_executable(Lab4 src/main.cpp src/cuda_kernels.cu)

# Link SFML libraries
target_link_libraries(Lab4 sfml-graphics sfml-window sfml-system)

# Specify C++11 standard
#set_property(TARGET Lab4 PROPERTY CXX_STANDARD 17)

set_target_properties(
    Lab4 PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY "${COMMON_OUTPUT_DIR}/bin"
)
# Compile Info
set_target_properties(Lab4 PROPERTIES CUDA_ARCHITECTURES "50;72;90")
target_compile_features(Lab4 PUBLIC cxx_std_11)
target_compile_features(Lab4 PUBLIC cuda_std_11)