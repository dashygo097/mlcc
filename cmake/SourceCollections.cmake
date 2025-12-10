# Collect source and header files

file(GLOB_RECURSE MLCC_SOURCES 
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c 
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc 
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

file(GLOB_RECURSE MLCC_HEADERS 
  ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h 
  ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hh 
  ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp
)
