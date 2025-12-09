# Define the main library target

# Build mlc library
add_library(mlcc ${MLCC_SOURCES} ${MLCC_HEADERS})
  
set_target_properties(mlcc PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)
  
target_include_directories(mlcc PUBLIC 
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
  
# Export compile features
target_compile_features(mlcc PUBLIC cxx_std_17)
