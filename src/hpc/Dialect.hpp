#pragma once

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "hpc/Dialect.h.inc"

namespace mlir {
namespace hpc {

class AxpyOp;
class CopyOp;
class ScalOp;
class DotOp;

} // namespace hpc
} // namespace mlir

#define GET_OP_CLASSES
#include "hpc/Ops.h.inc"
