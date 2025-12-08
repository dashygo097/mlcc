# Define the main library target

# Build mlc library
add_library(mlc ${MLC_SOURCES} ${MLC_HEADERS} ${MLC_CUDA_HEADERS} ${MLC_CUDA_SOURCES})
  
set_target_properties(mlc PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)
  
target_include_directories(mlc PUBLIC 
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
  
# Export compile features
target_compile_features(mlc PUBLIC cxx_std_17)
