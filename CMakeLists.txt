cmake_minimum_required(VERSION 3.28.3)

project(pupok VERSION 1.0)


set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
endif()

add_executable( ${PROJECT_NAME} main.cpp )
#file(GLOB_RECURSE SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
#add_executable(${PROJECT_NAME} ${SRC_FILES} ${HEADER_FILES})

#target_include_directories(${PROJECT_NAME}
#        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
#)

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_23)
target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:
        -march=native
        -mtune=native
        -O3
        -ffast-math
        >
)



find_package(OpenCV REQUIRED )
INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS} )




target_link_libraries ( ${PROJECT_NAME} ${OpenCV_LIBS} )


