cmake_minimum_required(VERSION 3.2)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_library(cmdline INTERFACE)
target_include_directories(cmdline INTERFACE "${PROJECT_SOURCE_DIR}/cmdline")

add_executable(genClassDS genClassDS.cpp)
target_link_libraries(genClassDS PRIVATE Threads::Threads cmdline)
