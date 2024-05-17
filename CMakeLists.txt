cmake_minimum_required(VERSION 3.10)

project(canny-edge-detection CXX C)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(EXISTS "${CMAKE_SOURCE_DIR}/libs/lodepng/lodepng.cpp")
  file(RENAME libs/lodepng/lodepng.cpp libs/lodepng/lodepng.c)
endif()

add_library(lodepng libs/lodepng/lodepng.h libs/lodepng/lodepng.c)
include_directories(libs/lodepng/)
link_directories(libs/lodepng/)
target_include_directories(lodepng PUBLIC libs/lodepng)

find_package(OpenMP)
find_package(OpenCL)
find_package( OpenCV REQUIRED )
if(OpenMP_C_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
endif()

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(sequential sequential/main.c universal/universal.c)
target_link_libraries(sequential OpenMP::OpenMP_C)
target_link_libraries(sequential lodepng m)

add_executable(openmp openmp/main.c universal/universal.c)
target_link_libraries(openmp OpenMP::OpenMP_C)
target_link_libraries(openmp lodepng m)

add_executable(opencl opencl/main.c universal/universal.c)
target_link_libraries(opencl ${OpenCL_LIBRARIES})
target_link_libraries(opencl lodepng m)
target_link_libraries(opencl OpenMP::OpenMP_C)
set_target_properties(opencl PROPERTIES C_STANDARD 11
                                                 C_STANDARD_REQUIRED ON
                                                 C_EXTENSIONS OFF)

add_executable(opencv opencv/main.cpp)
target_link_libraries(opencv ${OpenCV_LIBS})
target_link_libraries(opencv OpenMP::OpenMP_C)

target_compile_definitions(opencl PRIVATE CL_TARGET_OPENCL_VERSION=120)

