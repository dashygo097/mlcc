#pragma once

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Include generated files (now in build/include/hpc/)
#include "hpc/Dialect.h.inc"

namespace mlir {
namespace hpc {

//===----------------------------------------------------------------------===//
// HPC Operations
//===----------------------------------------------------------------------===//

/// AXPY operation:  Y = alpha * X + Y
class AxpyOp
    : public Op<AxpyOp, OpTrait::ZeroResults, MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;

  static llvm::StringRef getOperationName() { return "hpc.axpy"; }

  static void build(OpBuilder &builder, OperationState &state, IntegerAttr n,
                    Value alpha, Value src, Value dst);

  LogicalResult verify();

  void
  getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
                 &effects);

  // Accessors
  IntegerAttr getNAttr() { return (*this)->getAttrOfType<IntegerAttr>("n"); }
  Value getAlpha() { return getOperand(0); }
  Value getSrc() { return getOperand(1); }
  Value getDst() { return getOperand(2); }
};

/// COPY operation: Y = X
class CopyOp
    : public Op<CopyOp, OpTrait::ZeroResults, MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;

  static llvm::StringRef getOperationName() { return "hpc.copy"; }

  static void build(OpBuilder &builder, OperationState &state, IntegerAttr n,
                    Value src, Value dst);

  LogicalResult verify();

  void
  getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
                 &effects);

  // Accessors
  IntegerAttr getNAttr() { return (*this)->getAttrOfType<IntegerAttr>("n"); }
  Value getSrc() { return getOperand(0); }
  Value getDst() { return getOperand(1); }
};

/// SCAL operation: X = alpha * X
class ScalOp
    : public Op<ScalOp, OpTrait::ZeroResults, MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;

  static llvm::StringRef getOperationName() { return "hpc.scal"; }

  static void build(OpBuilder &builder, OperationState &state, IntegerAttr n,
                    Value alpha, Value dst);

  LogicalResult verify();

  void
  getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
                 &effects);

  // Accessors
  IntegerAttr getNAttr() { return (*this)->getAttrOfType<IntegerAttr>("n"); }
  Value getAlpha() { return getOperand(0); }
  Value getDst() { return getOperand(1); }
};

/// DOT operation: result = X · Y
class DotOp : public Op<DotOp, OpTrait::OneResult, OpTrait::Pure> {
public:
  using Op::Op;

  static llvm::StringRef getOperationName() { return "hpc.dot"; }

  static void build(OpBuilder &builder, OperationState &state, Type resultType,
                    IntegerAttr n, Value src1, Value src2);

  LogicalResult verify();

  // Accessors
  IntegerAttr getNAttr() { return (*this)->getAttrOfType<IntegerAttr>("n"); }
  Value getSrc1() { return getOperand(0); }
  Value getSrc2() { return getOperand(1); }
  Value getResult() { return Op::getResult(0); }
};

} // namespace hpc
} // namespace mlir

#define GET_OP_CLASSES
#include "hpc/Ops.h.inc"

#endif // SRC_HPC_DIALECT_HH
