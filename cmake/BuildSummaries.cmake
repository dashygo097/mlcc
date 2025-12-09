# Print build configuration summary

include(Helpers)

print_info("◆ BUILD SUMMARY " "0")
print_info("======================================\n" "0")

# Platform
print_info("[INFO] Platform: ${CMAKE_SYSTEM_NAME} (${CMAKE_SYSTEM_PROCESSOR})\n" "91")
print_info("[INFO] Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}\n" "96")

# Feature flags
print_info("[INFO] Features:\n" "95")

if(CCACHE_PROGRAM)
  print_info("  ✓ ccache: ${CCACHE_PROGRAM}\n" "92")
endif()

if(HAS_LLVM)
print_info("  ✓ llvm: ${LLVM_DIR}\n" "92")
else()
  print_info("  ✗ llvm support\n" "91")
endif()

if(HAS_MLIR)
  print_info("  ✓ mlir: ${MLIR_DIR}\n" "92")
else()
  print_info("  ✗ mlir support\n" "91")
endif()

# Source files
print_info("\n" "0")
make_paths_relative(REL_SOURCES MLCC_SOURCES)
make_paths_relative(REL_HEADERS MLCC_HEADERS)

make_preview_string(REL_HEADERS 3)
print_info("[TRACE] Headers: ${PREVIOUS_SCOPE_VAR}\n" "94")

make_preview_string(REL_SOURCES 3)
print_info("[TRACE] Sources: ${PREVIOUS_SCOPE_VAR}\n" "96")

print_info("◆ END OF SUMMARIES " "90")
print_info("===================================\n" "90")
print_info("\n" "0")
