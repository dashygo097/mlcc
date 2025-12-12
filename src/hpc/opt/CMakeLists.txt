cmake_minimum_required(VERSION 3.21)

# mlcc-opt
add_llvm_executable(hpc-mlcc-opt ${CMAKE_CURRENT_SOURCE_DIR}/mlcc-opt.cpp)

llvm_update_compile_flags(hpc-mlcc-opt)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

target_link_libraries(hpc-mlcc-opt PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  ${MLCC_LIBRARY}

  MLIROptLib
  MLIRPass
  MLIRTransforms
  MLIRSupport
  MLIRIR

  mlcc_hpc_dialect
  mlcc_hpc_runtime
  ${HPC_LIBRARY}
)

target_include_directories(hpc-mlcc-opt PRIVATE
  ${MLCC_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}/include
)

set_target_properties(hpc-mlcc-opt PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

mlir_check_all_link_libraries(hpc-mlcc-opt)

message(STATUS "[hpc-mlcc-opt] Created executable")
