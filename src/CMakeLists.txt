cmake_minimum_required(VERSION 3.21)

message(STATUS "[src] Subproject Project")
  
# Make sure we have the MLIR CMake commands
if(NOT COMMAND mlir_tablegen)
  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  include(AddMLIR)
  include(TableGen)
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/AddDialect.cmake)

# HPC  
file(GLOB_RECURSE HPC_DIALECT_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/hpc/IR/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/hpc/Transforms/*.cpp
)

add_mlcc_dialect(
  NAME HPC
  TD_FILE ${CMAKE_CURRENT_SOURCE_DIR}/hpc/Ops.td
  PASS_TD_FILE ${CMAKE_CURRENT_SOURCE_DIR}/hpc/Passes.td
  SOURCES ${HPC_DIALECT_SOURCES}
)

file(GLOB_RECURSE HPC_RUNTIME_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/hpc/Runtime/*.cpp
)

add_mlcc_runtime(
  NAME HPC
  SOURCES ${HPC_RUNTIME_SOURCES}
  LINK_LIBS ${HPC_LIBRARY}
  INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/3rdparty/hpc/include
)

message(STATUS "[src] HPC dialect sources: ${HPC_DIALECT_SOURCES}")
message(STATUS "[src] HPC runtime sources: ${HPC_RUNTIME_SOURCES}")

# mlcc-opt
add_llvm_executable(mlcc-opt mlcc-opt.cpp)

llvm_update_compile_flags(mlcc-opt)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

target_link_libraries(mlcc-opt PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  ${MLCC_LIBRARY}
  mlcc_hpc_dialect
  mlcc_hpc_runtime
  MLIROptLib
  MLIRPass
  MLIRTransforms
  MLIRSupport
  MLIRIR
  ${HPC_LIBRARY}
)

target_include_directories(mlcc-opt PRIVATE
  ${MLCC_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}/include
)

set_target_properties(mlcc-opt PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

mlir_check_all_link_libraries(mlcc-opt)

message(STATUS "[src] Created executable: mlcc-opt")
