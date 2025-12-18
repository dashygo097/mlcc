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
  add_mlir_library(${library_name} STATIC ${RUNTIME_SOURCES})

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
