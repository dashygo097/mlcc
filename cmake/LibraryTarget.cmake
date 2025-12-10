
list(LENGTH MLCC_SOURCES MLCC_SOURCE_COUNT)

if(MLCC_SOURCE_COUNT GREATER 0)
  # Build static library
  add_library(mlcc STATIC ${MLCC_SOURCES})
  
  target_include_directories(mlcc PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE: include>
  )
  
  target_compile_features(mlcc PUBLIC cxx_std_17)
  
  # Link against MLIR/LLVM
  target_link_libraries(mlcc PUBLIC
    MLIRParser
    MLIRIR
    MLIRSupport
    MLIRDialect
    MLIRPass
    MLIRTransforms
    MLIRFuncDialect
    MLIRArithDialect
    MLIRMemRefDialect
    LLVMSupport
    LLVMCore
  )
  
  # Depend on TableGen targets
  if(DEFINED MLCC_TABLEGEN_TARGETS)
    add_dependencies(mlcc ${MLCC_TABLEGEN_TARGETS})
  endif()
  
  set_target_properties(mlcc PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
  )
  
  message(STATUS "Created MLCC library:  ${CMAKE_BINARY_DIR}/lib/libmlcc.a")
  
else()
  # Header-only library
  add_library(mlcc INTERFACE)
  
  target_include_directories(mlcc INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
  )
  
  target_compile_features(mlcc INTERFACE cxx_std_17)
  
  message(STATUS "Created MLCC library: header-only")
endif()

# Export variables for use by src/
set(MLCC_INCLUDE_DIRS 
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}/include
  CACHE INTERNAL "MLCC include directories"
)

set(MLCC_LIBRARY mlcc CACHE INTERNAL "MLCC library target")
