#include "../Dialect.hpp"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::hpc;

//===----------------------------------------------------------------------===//
// Operation Verifiers
//===----------------------------------------------------------------------===//

LogicalResult AxpyOp::verify() {
  auto srcType = getSrc().getType().dyn_cast<MemRefType>();
  auto dstType = getDst().getType().dyn_cast<MemRefType>();

  if (!srcType || !dstType)
    return emitOpError("operands must be memref types");

  if (srcType.getElementType() != dstType.getElementType())
    return emitOpError("source and destination must have same element type");

  if (getAlpha().getType() != srcType.getElementType())
    return emitOpError("alpha must match memref element type");

  return success();
}

LogicalResult CopyOp::verify() {
  auto srcType = getSrc().getType().dyn_cast<MemRefType>();
  auto dstType = getDst().getType().dyn_cast<MemRefType>();

  if (!srcType || !dstType)
    return emitOpError("operands must be memref types");

  if (srcType.getElementType() != dstType.getElementType())
    return emitOpError("source and destination must have same element type");

  return success();
}

LogicalResult ScalOp::verify() {
  auto dstType = getDst().getType().dyn_cast<MemRefType>();

  if (!dstType)
    return emitOpError("operand must be memref type");

  if (getAlpha().getType() != dstType.getElementType())
    return emitOpError("alpha must match memref element type");

  return success();
}

LogicalResult DotOp::verify() {
  auto src1Type = getSrc1().getType().dyn_cast<MemRefType>();
  auto src2Type = getSrc2().getType().dyn_cast<MemRefType>();

  if (!src1Type || !src2Type)
    return emitOpError("operands must be memref types");

  if (src1Type.getElementType() != src2Type.getElementType())
    return emitOpError("operands must have same element type");

  if (getResult().getType() != src1Type.getElementType())
    return emitOpError("result type must match memref element type");

  return success();
}
