cmake_minimum_required(VERSION 3.21)

set(MLCC_HPC_PLUSPLUS_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/mlcc++.cpp)

if(EXISTS ${MLCC_HPC_PLUSPLUS_SOURCE})
  find_package(Clang QUIET CONFIG)
  
  if(Clang_FOUND)
    add_llvm_executable(hpc-mlcc++ ${MLCC_HPC_PLUSPLUS_SOURCE})

    llvm_update_compile_flags(hpc-mlcc++)

    get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
    get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

    target_link_libraries(hpc-mlcc++ PRIVATE
      # Clang
      clangTooling
      clangFrontend
      clangAST
      clangBasic
      clangLex
      clangParse
      clangSema
      
      # MLIR
      ${dialect_libs}
      ${conversion_libs}
      mlcc_hpc_dialect
      mlcc_hpc_runtime
      MLIRParser
      MLIRPass
      MLIRTransforms
      MLIRIR
      MLIRSupport
      MLIRTargetLLVMIRExport
      MLIRBuiltinToLLVMIRTranslation
      MLIRLLVMToLLVMIRTranslation
      MLIRExecutionEngineUtils
      
      # HPC
      ${HPC_LIBRARY}
      
      # LLVM
      LLVMTarget
      LLVMPasses
    )

    target_include_directories(hpc-mlcc++ PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_BINARY_DIR}/include
      ${CMAKE_SOURCE_DIR}/include
      ${CLANG_INCLUDE_DIRS}
      ${MLIR_INCLUDE_DIRS}
    )

    set_target_properties(hpc-mlcc++ PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )

    message(STATUS "[mlcc-hpc++] Created HPC DOT compiler")
  else()
    message(WARNING "[mlcc-hpc++] Clang not found, skipping")
  endif()
else()
  message(WARNING "[mlcc-hpc++] Source file not found")
endif()
