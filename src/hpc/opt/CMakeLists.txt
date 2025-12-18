cmake_minimum_required(VERSION 3.21)

# mlcc-opt
add_llvm_executable(hpc-opt ${CMAKE_CURRENT_SOURCE_DIR}/mlcc-opt.cpp)

target_link_libraries(hpc-opt PRIVATE
  mlcc_hpc_dialect
  mlcc_hpc_runtime
  mlcc
  hpc

  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIROptLib
  MLIRTransforms
  
  # Dialects
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRArithDialect
  MLIRSCFDialect

  # Conversions
  MLIRFuncToLLVM
  MLIRMemRefToLLVM
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
)

target_include_directories(hpc-opt PRIVATE
  ${MLCC_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}/include
)

set_target_properties(hpc-opt PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

mlir_check_all_link_libraries(hpc-opt)

message(STATUS "[hpc-mlcc-opt] Created executable")
