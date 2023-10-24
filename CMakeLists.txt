cmake_minimum_required(VERSION 3.27)

set(CMAKE_CXX_COMPILER "clang++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-unnamed-lambda -fsycl-targets=nvptx64-nvidia-cuda")

project(GameOfLife LANGUAGES CXX)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(GameOfLife src/main.cpp)
target_link_libraries( GameOfLife ${OpenCV_LIBS} )