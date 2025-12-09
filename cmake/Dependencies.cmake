# Find and configure all external dependencies

# --- 3rdparty libraries (always included) ---
include_directories(3rdparty)
if(ENABLE_TESTS)
  add_subdirectory(3rdparty/googletest EXCLUDE_FROM_ALL)
  add_subdirectory(3rdparty/benchmark EXCLUDE_FROM_ALL)
endif()

# --- LLVM ---
find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})

# LLVM definitions
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})
set(HAS_LLVM TRUE)

llvm_map_components_to_libnames(llvm_libs support core irreader)

# --- MLIR ---
find_package(MLIR REQUIRED CONFIG)
include_directories(${MLIR_INCLUDE_DIRS})
set(HAS_MLIR TRUE)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
