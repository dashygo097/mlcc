#include "../Dialect.hpp"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::hpc;

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
