cmake_minimum_required(VERSION 3.0)
project(BLOSSOM)

# Set C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set maximum optimization level
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++2b")

# Add source files
set(SOURCES
    src/main.cpp
)

# Add headers directory for include
include_directories(src)

# Create executable
add_executable(blossom ${SOURCES})

# Example: Link external libraries if needed
# target_link_libraries(abm_sim some_library)

# Optionally, set output directory for executable
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
