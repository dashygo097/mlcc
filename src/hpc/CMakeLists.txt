cmake_minimum_required(VERSION 3.21)

include(${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backends/hpc EXCLUDE_FROM_ALL)

file(GLOB_RECURSE HPC_DIALECT_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/IR/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/Transforms/*.cpp
)

add_mlcc_dialect(
  NAME HPC
  TD_FILE ${CMAKE_CURRENT_SOURCE_DIR}/lib/Ops.td
  PASS_TD_FILE ${CMAKE_CURRENT_SOURCE_DIR}/lib/Passes.td
  SOURCES ${HPC_DIALECT_SOURCES}
)

file(GLOB_RECURSE HPC_RUNTIME_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/Runtime/*.cpp
)

add_mlcc_runtime(
  NAME HPC
  SOURCES ${HPC_RUNTIME_SOURCES}
  LINK_LIBS ${HPC_LIBRARY}
  INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/backends/hpc/include
)

message(STATUS "[HPC] dialect sources: ${HPC_DIALECT_SOURCES}")
message(STATUS "[HPC] runtime sources: ${HPC_RUNTIME_SOURCES}")

# opt
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/opt)
