cmake_minimum_required(VERSION 3.2)

add_executable(genClassDS genClassDS.cpp ${LIBS})
target_link_libraries(genClassDS Threads::Threads)
