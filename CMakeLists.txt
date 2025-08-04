
cmake_minimum_required(VERSION 3.24)

### Setup project name and language
project(SpM VERSION 1.2
        DESCRIPTION "A CMake project template"
        HOMEPAGE_URL "https://github.com/DavidAce/CMakeTemplate"
        LANGUAGES C CXX)


### Set up options
option(SPM_CMAKE_DEBUG     "Print info during CMake configuration"                OFF)
option(SPM_ENABLE_OPENMP   "Enables OpenMP"                                       OFF)


if(SPM_DEBUG_CMAKE)
    ### Print operating system details
    include(cmake/PrintHostInfo.cmake)
endif()

### Add all source files
add_executable(Spectral
        source/main.cpp
        source/cli.cpp
        source/spm.cpp
        source/svd.cpp
        source/admm.cpp
        source/io.cpp
        # ...
        )

target_include_directories(Spectral PRIVATE source)


### Find dependencies
find_package(h5pp       1.11.2  REQUIRED)
find_package(spdlog     1.13.0  REQUIRED)
find_package(pcg-cpp            REQUIRED)
find_package(CLI11      2.3.2   REQUIRED)
find_package(nlohmann_json      3.11.3  REQUIRED)
find_package(Eigen3     3.4.0   REQUIRED)


### Create a helper target that links dependencies (this target can be used to build tests and benchmarks)
add_library(spm-libs INTERFACE)

# Link the libraries to the helper target
target_link_libraries(spm-libs INTERFACE h5pp::h5pp)
target_link_libraries(spm-libs INTERFACE spdlog::spdlog )
target_link_libraries(spm-libs INTERFACE pcg-cpp::pcg-cpp )
target_link_libraries(spm-libs INTERFACE CLI11::CLI11)
target_link_libraries(spm-libs INTERFACE nlohmann_json::nlohmann_json)
target_link_libraries(spm-libs INTERFACE Eigen3::Eigen)



if(SPM_ENABLE_OPENMP)
    find_package(OpenMP COMPONENTS CXX REQUIRED)
    target_link_libraries(spm-libs INTERFACE OpenMP::OpenMP_CXX)
endif()



### Link targets to the main executable
target_link_libraries(Spectral PUBLIC spm-libs)

################################################################
### Get git version number                                   ###
### Generates a header gitversion/gitversion.h               ###
### Include it using #include <gitversion.h>                 ###
### Gives a namespace GIT:: with several git identifiers     ###
################################################################
include(cmake/gitversion.cmake)

### CTest
if(SPM_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
    if(SPM_ENABLE_COVERAGE)
        # Coverage is used to check what fraction of the source code is tested
        # We use this option in the github actions when compiling on Ubuntu with gcc
        target_compile_options(spm-libs INTERFACE --coverage)
        target_link_options(spm-libs INTERFACE --coverage)
    endif()
endif()


if(SPM_CMAKE_DEBUG)
    # Print summary of CMake configuration
    include(cmake/PrintTargetInfo.cmake)
    print_and_write_project_summary(Spectral)
endif()
