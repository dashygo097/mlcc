cmake_minimum_required(VERSION 3.21)
project(mlc LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add cmake module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Generate and include build configuration
if(EXISTS "${CMAKE_BINARY_DIR}/config.cmake")
  include("${CMAKE_BINARY_DIR}/config.cmake")
endif()

# Include modular cmake components
include(Platforms)
include(CompilerSetup)
include(Helpers)
include(LLVMSetup)
include(Dependencies)
include(SourceCollections)
include(LibraryTarget)
include(BuildSummaries)

# Tools (compiler executables)
if(ENABLE_TOOLS_BUILDING)
  add_subdirectory(tools)
endif()

# Tests
if(ENABLE_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
