cmake_minimum_required(VERSION 3.2)

# add_executable(genSeqReg genSeqReg.cpp ${LIBS})
# add_executable(genSeqClass genSeqClass.cpp ${LIBS})
# add_executable(genSeqClassReg genSeqClassReg.cpp ${LIBS})
add_executable(genClassDS genClassDS.cpp ${LIBS})
target_link_libraries(genSeqClassReg Threads::Threads)
