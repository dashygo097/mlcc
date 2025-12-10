#include "../Dialect.hpp"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::hpc;

// Include generated implementation
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

//===----------------------------------------------------------------------===//
// AxpyOp
//===----------------------------------------------------------------------===//

void AxpyOp::build(OpBuilder &builder, OperationState &state, IntegerAttr n,
                   Value alpha, Value src, Value dst) {
  state.addAttribute("n", n);
  state.addOperands({alpha, src, dst});
}

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

void AxpyOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), getDst());
  effects.emplace_back(MemoryEffects::Read::get(), getSrc());
}

//===----------------------------------------------------------------------===//
// CopyOp
//===----------------------------------------------------------------------===//

void CopyOp::build(OpBuilder &builder, OperationState &state, IntegerAttr n,
                   Value src, Value dst) {
  state.addAttribute("n", n);
  state.addOperands({src, dst});
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

void CopyOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), getDst());
  effects.emplace_back(MemoryEffects::Read::get(), getSrc());
}

//===----------------------------------------------------------------------===//
// ScalOp
//===----------------------------------------------------------------------===//

void ScalOp::build(OpBuilder &builder, OperationState &state, IntegerAttr n,
                   Value alpha, Value dst) {
  state.addAttribute("n", n);
  state.addOperands({alpha, dst});
}

LogicalResult ScalOp::verify() {
  auto dstType = getDst().getType().dyn_cast<MemRefType>();

  if (!dstType)
    return emitOpError("operand must be memref type");

  if (getAlpha().getType() != dstType.getElementType())
    return emitOpError("alpha must match memref element type");

  return success();
}

void ScalOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), getDst());
  effects.emplace_back(MemoryEffects::Read::get(), getDst());
}

//===----------------------------------------------------------------------===//
// DotOp
//===----------------------------------------------------------------------===//

void DotOp::build(OpBuilder &builder, OperationState &state, Type resultType,
                  IntegerAttr n, Value src1, Value src2) {
  state.addAttribute("n", n);
  state.addOperands({src1, src2});
  state.addTypes(resultType);
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
