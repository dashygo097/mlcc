# AddDialect.cmake - Helper functions for adding MLIR dialects to MLCC

# Function to add a dialect with TableGen generation
# Usage:
#   add_mlcc_dialect(
#     NAME <dialect_name>
#     TD_FILE <path_to_ops_td>
#     SOURCES <source_files...>
#     [PASS_TD_FILE <path_to_passes_td>]
#     [DEPENDS <dependencies...>]
#   )
function(add_mlcc_dialect)
  cmake_parse_arguments(
    DIALECT
    ""
    "NAME;TD_FILE;PASS_TD_FILE"
    "SOURCES;DEPENDS"
    ${ARGN}
  )

  if(NOT DIALECT_NAME)
    message(FATAL_ERROR "add_mlcc_dialect: NAME is required")
  endif()

  if(NOT DIALECT_TD_FILE)
    message(FATAL_ERROR "add_mlcc_dialect: TD_FILE is required")
  endif()

  if(NOT DIALECT_SOURCES)
    message(FATAL_ERROR "add_mlcc_dialect: SOURCES is required")
  endif()

  string(TOLOWER ${DIALECT_NAME} dialect_lower)
  string(TOUPPER ${DIALECT_NAME} dialect_upper)

  # Set up TableGen output directory
  set(TABLEGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/include/${dialect_lower})
  file(MAKE_DIRECTORY ${TABLEGEN_OUTPUT_DIR})

  message(STATUS "[${DIALECT_NAME}] TableGen output: ${TABLEGEN_OUTPUT_DIR}")

  # Generate dialect operations
  set(LLVM_TARGET_DEFINITIONS ${DIALECT_TD_FILE})

  mlir_tablegen(${dialect_lower}_Dialect.h.inc -gen-dialect-decls -dialect=${dialect_lower})
  mlir_tablegen(${dialect_lower}_Dialect.cpp.inc -gen-dialect-defs -dialect=${dialect_lower})
  mlir_tablegen(${dialect_lower}_Ops.h.inc -gen-op-decls)
  mlir_tablegen(${dialect_lower}_Ops.cpp.inc -gen-op-defs)

  set(ops_target ${dialect_upper}OpsIncGen)
  add_public_tablegen_target(${ops_target})

  # Copy generated files
  add_custom_command(TARGET ${ops_target} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${CMAKE_CURRENT_BINARY_DIR}/${dialect_lower}_Dialect.h.inc
      ${TABLEGEN_OUTPUT_DIR}/Dialect.h.inc
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${CMAKE_CURRENT_BINARY_DIR}/${dialect_lower}_Dialect.cpp.inc
      ${TABLEGEN_OUTPUT_DIR}/Dialect.cpp.inc
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${CMAKE_CURRENT_BINARY_DIR}/${dialect_lower}_Ops.h.inc
      ${TABLEGEN_OUTPUT_DIR}/Ops.h.inc
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${CMAKE_CURRENT_BINARY_DIR}/${dialect_lower}_Ops.cpp.inc
      ${TABLEGEN_OUTPUT_DIR}/Ops.cpp.inc
    COMMENT "Copying ${DIALECT_NAME} dialect generated files"
  )

  set(tablegen_targets ${ops_target})

  # Generate passes if specified
  if(DIALECT_PASS_TD_FILE)
    set(LLVM_TARGET_DEFINITIONS ${DIALECT_PASS_TD_FILE})

    mlir_tablegen(${dialect_lower}_Passes.h.inc -gen-pass-decls -name ${DIALECT_NAME})
    mlir_tablegen(${dialect_lower}_Passes.capi.h.inc -gen-pass-capi-header --prefix ${DIALECT_NAME})
    mlir_tablegen(${dialect_lower}_Passes.capi.cpp.inc -gen-pass-capi-impl --prefix ${DIALECT_NAME})

    set(pass_target ${dialect_upper}PassesIncGen)
    add_public_tablegen_target(${pass_target})

    # Copy generated pass files
    add_custom_command(TARGET ${pass_target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_BINARY_DIR}/${dialect_lower}_Passes.h.inc
        ${TABLEGEN_OUTPUT_DIR}/Passes.h.inc
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_BINARY_DIR}/${dialect_lower}_Passes.capi.h.inc
        ${TABLEGEN_OUTPUT_DIR}/Passes.capi.h.inc
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_BINARY_DIR}/${dialect_lower}_Passes.capi.cpp.inc
        ${TABLEGEN_OUTPUT_DIR}/Passes.capi.cpp.inc
      COMMENT "Copying ${DIALECT_NAME} pass generated files"
    )

    list(APPEND tablegen_targets ${pass_target})
  endif()

  # Verify sources exist
  foreach(src ${DIALECT_SOURCES})
    if(NOT EXISTS ${src})
      message(FATAL_ERROR "Source file not found: ${src}")
    endif()
  endforeach()

  message(STATUS "[${DIALECT_NAME}] Sources: ${DIALECT_SOURCES}")

  # Create dialect library
  set(library_name mlcc_${dialect_lower}_dialect)
  add_library(${library_name} STATIC ${DIALECT_SOURCES})

  # Add TableGen dependencies
  add_dependencies(${library_name} ${tablegen_targets})

  # Add external dependencies if specified
  if(DIALECT_DEPENDS)
    add_dependencies(${library_name} ${DIALECT_DEPENDS})
  endif()

  target_include_directories(${library_name} PUBLIC
    ${MLCC_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/include
  )

  target_link_libraries(${library_name} PUBLIC
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

  set_target_properties(${library_name} PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
  )

  message(STATUS "[${DIALECT_NAME}] Created library: lib${library_name}.a")

  # Export library name to parent scope
  set(${dialect_upper}_DIALECT_LIBRARY ${library_name} PARENT_SCOPE)
  set(${dialect_upper}_TABLEGEN_TARGETS ${tablegen_targets} PARENT_SCOPE)
endfunction()

# Function to add a runtime library for a dialect
# Usage:
#   add_mlcc_runtime(
#     NAME <dialect_name>
#     SOURCES <source_files...>
#     [LINK_LIBS <libraries...>]
#     [INCLUDE_DIRS <directories...>]
#   )
function(add_mlcc_runtime)
  cmake_parse_arguments(
    RUNTIME
    ""
    "NAME"
    "SOURCES;LINK_LIBS;INCLUDE_DIRS"
    ${ARGN}
  )

  if(NOT RUNTIME_NAME)
    message(FATAL_ERROR "add_mlcc_runtime: NAME is required")
  endif()

  if(NOT RUNTIME_SOURCES)
    message(FATAL_ERROR "add_mlcc_runtime: SOURCES is required")
  endif()

  string(TOLOWER ${RUNTIME_NAME} runtime_lower)
  string(TOUPPER ${RUNTIME_NAME} runtime_upper)

  # Verify sources exist
  foreach(src ${RUNTIME_SOURCES})
    if(NOT EXISTS ${src})
      message(FATAL_ERROR "Source file not found: ${src}")
    endif()
  endforeach()

  message(STATUS "[${RUNTIME_NAME}] Runtime sources: ${RUNTIME_SOURCES}")

  # Create runtime library
  set(library_name mlcc_${runtime_lower}_runtime)
  add_library(${library_name} STATIC ${RUNTIME_SOURCES})

  target_include_directories(${library_name} PUBLIC
    ${MLCC_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${RUNTIME_INCLUDE_DIRS}
  )

  if(RUNTIME_LINK_LIBS)
    target_link_libraries(${library_name} PUBLIC ${RUNTIME_LINK_LIBS})
  endif()

  set_target_properties(${library_name} PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
  )

  message(STATUS "[${RUNTIME_NAME}] Created runtime library: lib${library_name}.a")

  # Export library name to parent scope
  set(${runtime_upper}_RUNTIME_LIBRARY ${library_name} PARENT_SCOPE)
endfunction()
