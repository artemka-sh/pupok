cmake_minimum_required(VERSION 4.1)

project(DisplayImage VERSION 1.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set (CMAKE_CXX_STANDARD 23)
find_package(OpenCV REQUIRED )
INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS} )
add_executable( ${PROJECT_NAME} main.cpp )



target_link_libraries ( ${PROJECT_NAME} ${OpenCV_LIBS} )


