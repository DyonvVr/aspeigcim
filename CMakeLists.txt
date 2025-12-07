cmake_minimum_required(VERSION 3.19)
project(aspeigcim)

set(CMAKE_CXX_STANDARD 17)

include_directories(/usr/include/boost)

add_executable(aspeigcim src/aspeigcim/classes/chamil.cpp)
