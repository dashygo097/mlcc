# Find and configure all external dependencies

# --- 3rdparty libraries (always included) ---
include_directories(3rdparty)

# Add googletest and benchmark for MLCC tests 
if(ENABLE_TESTING)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/googletest")
    add_subdirectory(3rdparty/googletest EXCLUDE_FROM_ALL)
  endif()
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/benchmark")
    add_subdirectory(3rdparty/benchmark EXCLUDE_FROM_ALL)
  endif()
endif()

# --- LLVM ---
find_package(LLVM REQUIRED CONFIG)
include(AddLLVM)
include(TableGen)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig. cmake in: ${LLVM_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})

# LLVM definitions
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})
set(HAS_LLVM TRUE)

llvm_map_components_to_libnames(llvm_libs support core irreader)

# --- MLIR ---
find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(AddMLIR)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
set(HAS_MLIR TRUE)
