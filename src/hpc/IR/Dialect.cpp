#include "../Dialect.hpp"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::hpc;

#include "hpc/Dialect.cpp.inc"

void HPCDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "hpc/Ops.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "hpc/Ops.cpp.inc"
