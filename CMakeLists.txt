cmake_minimum_required(VERSION 3.2)
project(genSeqReg CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set (CMAKE_CXX_STANDARD 11)

# Activate all errors
if (MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin )

include_directories("${PROJECT_SOURCE_DIR}/cmdline")
set(LIBS ${LIBS} "${PROJECT_SOURCE_DIR}/cmdline/cmdline.h")

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_subdirectory("src")
