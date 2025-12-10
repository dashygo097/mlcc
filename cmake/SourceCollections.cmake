# Collect source and header files

file(GLOB_RECURSE MLCC_SOURCES 
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/*.c 
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/*.cc 
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/*.cpp
)

file(GLOB_RECURSE MLCC_HEADERS 
  ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h 
  ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hh 
  ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp
)
