#pragma once

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Include generated dialect declaration
#include "hpc/Dialect.h.inc"

// Declare the dialect namespace
namespace mlir {
namespace hpc {

// Forward declarations - actual definitions come from TableGen
class AxpyOp;
class CopyOp;
class ScalOp;
class DotOp;

} // namespace hpc
} // namespace mlir

// Include generated operation definitions
#define GET_OP_CLASSES
#include "hpc/Ops.h.inc"
