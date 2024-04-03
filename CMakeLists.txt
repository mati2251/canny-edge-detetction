cmake_minimum_required(VERSION 3.10)

project(canny-edge-detection)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(EXISTS "${CMAKE_SOURCE_DIR}/libs/lodepng/lodepng.cpp")
  file(RENAME libs/lodepng/lodepng.cpp libs/lodepng/lodepng.c)
endif()

add_library(lodepng libs/lodepng/lodepng.h libs/lodepng/lodepng.c)
include_directories(libs/lodepng/)
link_directories(libs/lodepng/)
target_include_directories(lodepng PUBLIC libs/lodepng)

add_executable(sequential sequential/main.c )
target_link_libraries(sequential lodepng m)
