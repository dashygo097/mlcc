#include "../Dialect.hpp"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::hpc;

// Include generated dialect implementation
#include "hpc/Dialect.cpp.inc"

//===----------------------------------------------------------------------===//
// HPC Dialect
//===----------------------------------------------------------------------===//

void HPCDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "hpc/Ops.cpp.inc"
      >();
}

// Include generated operation implementations
#define GET_OP_CLASSES
#include "hpc/Ops.cpp.inc"
