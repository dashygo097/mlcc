cmake_minimum_required(VERSION 3.21)

# Detect if we're being built as part of the main project or standalone
if(NOT TARGET mlcc)
  # Standalone mode - need to find the MLCC library
  project(mlcc-tools LANGUAGES C CXX)
  
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  
  # Find MLCC library (assumes it's been installed or built in ../build)
  set(MLCC_DIR ${CMAKE_SOURCE_DIR}/../build CACHE PATH "Path to MLCC build directory")
  
  # Find LLVM/MLIR
  find_package(LLVM REQUIRED CONFIG)
  find_package(MLIR REQUIRED CONFIG)
  
  include_directories(${LLVM_INCLUDE_DIRS})
  include_directories(${MLIR_INCLUDE_DIRS})
  
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  
  include(AddLLVM)
  include(AddMLIR)
  include(TableGen)
  
  # Find libmlcc.a
  find_library(MLCC_LIBRARY 
    NAMES mlcc
    PATHS ${MLCC_DIR}/lib
    REQUIRED
  )
  
  # Find HPC library
  find_library(HPC_LIBRARY
    NAMES hpc
    PATHS ${MLCC_DIR}/3rdparty/hpc/lib
    REQUIRED
  )
  
  set(MLCC_INCLUDE_DIRS
    ${MLCC_DIR}/../include
    ${MLCC_DIR}/include
    ${MLCC_DIR}/../3rdparty/hpc/include
  )
  
  include_directories(${MLCC_INCLUDE_DIRS})
  
  message(STATUS "[src] Standalone mode - using MLCC from:  ${MLCC_DIR}")
else()
  # Subproject mode - MLCC library is already available
  message(STATUS "[src] Subproject mode - using parent MLCC library")
  
  # Make sure we have the MLIR CMake commands
  if(NOT COMMAND mlir_tablegen)
    list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
    include(AddMLIR)
    include(TableGen)
  endif()
endif()

# ==============================================================================
# Generate TableGen files for HPC dialect
# ==============================================================================

# Set output directory for generated files
set(HPC_TABLEGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/include/hpc)
file(MAKE_DIRECTORY ${HPC_TABLEGEN_OUTPUT_DIR})

# Add to include path
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

message(STATUS "[src] TableGen output directory: ${HPC_TABLEGEN_OUTPUT_DIR}")

# Generate dialect files from Ops.td
set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_SOURCE_DIR}/hpc/Ops.td)

mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=hpc)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=hpc)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)

add_public_tablegen_target(HPCOpsIncGen)

# Copy generated files to the expected location
add_custom_command(TARGET HPCOpsIncGen POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_BINARY_DIR}/Dialect.h.inc
    ${HPC_TABLEGEN_OUTPUT_DIR}/Dialect.h.inc
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_BINARY_DIR}/Dialect.cpp.inc
    ${HPC_TABLEGEN_OUTPUT_DIR}/Dialect.cpp.inc
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_BINARY_DIR}/Ops.h.inc
    ${HPC_TABLEGEN_OUTPUT_DIR}/Ops.h.inc
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_BINARY_DIR}/Ops.cpp.inc
    ${HPC_TABLEGEN_OUTPUT_DIR}/Ops.cpp.inc
  COMMENT "Copying HPC dialect generated files"
)

# Generate pass files from Passes.td
set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_SOURCE_DIR}/hpc/Passes.td)

mlir_tablegen(Passes.h.inc -gen-pass-decls -name HPC)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix HPC)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix HPC)

add_public_tablegen_target(HPCPassesIncGen)

# Copy generated pass files
add_custom_command(TARGET HPCPassesIncGen POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_BINARY_DIR}/Passes.h.inc
    ${HPC_TABLEGEN_OUTPUT_DIR}/Passes.h.inc
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_BINARY_DIR}/Passes.capi.h.inc
    ${HPC_TABLEGEN_OUTPUT_DIR}/Passes.capi.h.inc
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_BINARY_DIR}/Passes.capi.cpp.inc
    ${HPC_TABLEGEN_OUTPUT_DIR}/Passes.capi.cpp.inc
  COMMENT "Copying HPC pass generated files"
)

# ==============================================================================
# Collect HPC dialect sources
# ==============================================================================

set(HPC_DIALECT_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/hpc/IR/Dialect.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/hpc/IR/Verifiers.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/hpc/Transforms/Lower2LLVM.cpp
)

set(HPC_RUNTIME_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/hpc/Runtime/Runtime.cpp
)

# Verify sources exist
foreach(src ${HPC_DIALECT_SOURCES} ${HPC_RUNTIME_SOURCES})
  if(NOT EXISTS ${src})
    message(FATAL_ERROR "Source file not found: ${src}")
  endif()
endforeach()

message(STATUS "[src] HPC dialect sources: ${HPC_DIALECT_SOURCES}")
message(STATUS "[src] HPC runtime sources: ${HPC_RUNTIME_SOURCES}")

# ==============================================================================
# Build HPC dialect library (used by executables)
# ==============================================================================

add_library(mlcc_hpc_dialect STATIC ${HPC_DIALECT_SOURCES})

# Add dependencies on TableGen targets
add_dependencies(mlcc_hpc_dialect HPCOpsIncGen HPCPassesIncGen)

target_include_directories(mlcc_hpc_dialect PUBLIC
  ${MLCC_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}/include
)

target_link_libraries(mlcc_hpc_dialect PUBLIC
  ${MLCC_LIBRARY}
  MLIRParser
  MLIRIR
  MLIRSupport
  MLIRDialect
  MLIRPass
  MLIRTransforms
  MLIRFuncDialect
  MLIRArithDialect
  MLIRLLVMDialect
  MLIRLLVMCommonConversion
  MLIRMemRefDialect
  LLVMSupport
  LLVMCore
)

set_target_properties(mlcc_hpc_dialect PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

message(STATUS "[src] Created library: libmlcc_hpc_dialect.a")

# ==============================================================================
# Build HPC runtime library
# ==============================================================================

add_library(mlcc_hpc_runtime STATIC ${HPC_RUNTIME_SOURCES})

target_link_libraries(mlcc_hpc_runtime PUBLIC ${HPC_LIBRARY})

target_include_directories(mlcc_hpc_runtime PUBLIC
  ${MLCC_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/3rdparty/hpc/include
)

set_target_properties(mlcc_hpc_runtime PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

message(STATUS "[src] Created library: libmlcc_hpc_runtime.a")

# ==============================================================================
# Build mlcc-opt executable
# ==============================================================================

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
