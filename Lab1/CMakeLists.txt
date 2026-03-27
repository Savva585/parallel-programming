cmake_minimum_required(VERSION 3.12)

project(pr-a04-omp CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)
add_executable(pr-a04-matrix-omp
    main.cpp
    matrix.cpp
    matrix.h
)

target_link_libraries(pr-a04-matrix-omp PRIVATE OpenMP::OpenMP_CXX)