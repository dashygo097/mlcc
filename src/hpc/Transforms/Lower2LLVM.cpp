#include "../Dialect.hpp"
#include "../Passes.hpp"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {

LLVM::LLVMFuncOp getOrInsertFunction(PatternRewriter &rewriter, ModuleOp module,
                                     StringRef name,
                                     LLVM::LLVMFunctionType type) {
  if (auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(name))
    return func;

  PatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  return rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), name, type);
}

Value extractMemRefPtr(Location loc, Value memref, PatternRewriter &rewriter) {
  return rewriter.create<LLVM::ExtractValueOp>(loc, memref,
                                               ArrayRef<int64_t>{1});
}

struct AxpyOpLowering : public OpConversionPattern<hpc::AxpyOp> {
  using OpConversionPattern<hpc::AxpyOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hpc::AxpyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();

    auto srcType = op.getSrc().getType().cast<MemRefType>();
    Type elemType = srcType.getElementType();

    std::string funcName =
        elemType.isF32() ? "hpc_axpy_seq_f32" : "hpc_axpy_seq_f64";

    auto i64Type = rewriter.getI64Type();
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

    auto funcType = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(rewriter.getContext()),
        {i64Type, ptrType, ptrType, elemType}, false);

    auto func = getOrInsertFunction(rewriter, module, funcName, funcType);

    Value srcPtr = extractMemRefPtr(loc, adaptor.getSrc(), rewriter);
    Value dstPtr = extractMemRefPtr(loc, adaptor.getDst(), rewriter);

    if (!srcPtr || !dstPtr)
      return rewriter.notifyMatchFailure(op, "failed to extract pointers");

    Value n = rewriter.create<LLVM::ConstantOp>(loc, i64Type, op.getNAttr());

    rewriter.create<LLVM::CallOp>(
        loc, TypeRange{}, // void return
        funcName, ValueRange{n, dstPtr, srcPtr, adaptor.getAlpha()});

    rewriter.eraseOp(op);
    return success();
  }
};

struct CopyOpLowering : public OpConversionPattern<hpc::CopyOp> {
  using OpConversionPattern<hpc::CopyOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hpc::CopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();

    auto srcType = op.getSrc().getType().cast<MemRefType>();
    Type elemType = srcType.getElementType();

    std::string funcName =
        elemType.isF32() ? "hpc_copy_seq_f32" : "hpc_copy_seq_f64";

    auto i64Type = rewriter.getI64Type();
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

    auto funcType = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(rewriter.getContext()),
        {i64Type, ptrType, ptrType}, false);

    auto func = getOrInsertFunction(rewriter, module, funcName, funcType);

    Value srcPtr = extractMemRefPtr(loc, adaptor.getSrc(), rewriter);
    Value dstPtr = extractMemRefPtr(loc, adaptor.getDst(), rewriter);

    if (!srcPtr || !dstPtr)
      return rewriter.notifyMatchFailure(op, "failed to extract pointers");

    Value n = rewriter.create<LLVM::ConstantOp>(loc, i64Type, op.getNAttr());

    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, // void return
                                  funcName, ValueRange{n, dstPtr, srcPtr});

    rewriter.eraseOp(op);
    return success();
  }
};

struct ScalOpLowering : public OpConversionPattern<hpc::ScalOp> {
  using OpConversionPattern<hpc::ScalOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hpc::ScalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();

    auto dstType = op.getDst().getType().cast<MemRefType>();
    Type elemType = dstType.getElementType();

    std::string funcName =
        elemType.isF32() ? "hpc_scal_seq_f32" : "hpc_scal_seq_f64";

    auto i64Type = rewriter.getI64Type();
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

    auto funcType = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(rewriter.getContext()),
        {i64Type, ptrType, elemType}, false);

    auto func = getOrInsertFunction(rewriter, module, funcName, funcType);

    Value dstPtr = extractMemRefPtr(loc, adaptor.getDst(), rewriter);

    if (!dstPtr)
      return rewriter.notifyMatchFailure(op, "failed to extract pointer");

    Value n = rewriter.create<LLVM::ConstantOp>(loc, i64Type, op.getNAttr());

    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, // void return
                                  funcName,
                                  ValueRange{n, dstPtr, adaptor.getAlpha()});

    rewriter.eraseOp(op);
    return success();
  }
};

struct DotOpLowering : public OpConversionPattern<hpc::DotOp> {
  using OpConversionPattern<hpc::DotOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hpc::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();

    auto src1Type = op.getSrc1().getType().cast<MemRefType>();
    Type elemType = src1Type.getElementType();
    Type resultType = op.getResult().getType();

    std::string funcName =
        elemType.isF32() ? "hpc_dot_seq_f32" : "hpc_dot_seq_f64";

    auto i64Type = rewriter.getI64Type();
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

    auto funcType = LLVM::LLVMFunctionType::get(
        resultType, {i64Type, ptrType, ptrType}, false);

    auto func = getOrInsertFunction(rewriter, module, funcName, funcType);

    Value src1Ptr = extractMemRefPtr(loc, adaptor.getSrc1(), rewriter);
    Value src2Ptr = extractMemRefPtr(loc, adaptor.getSrc2(), rewriter);

    if (!src1Ptr || !src2Ptr)
      return rewriter.notifyMatchFailure(op, "failed to extract pointers");

    Value n = rewriter.create<LLVM::ConstantOp>(loc, i64Type, op.getNAttr());

    auto callOp = rewriter.create<LLVM::CallOp>(
        loc,
        resultType, // return type
        funcName, ValueRange{n, src1Ptr, src2Ptr});

    rewriter.replaceOp(op, callOp.getResult());
    return success();
  }
};

struct LowerHPCToLLVMPass
    : public PassWrapper<LowerHPCToLLVMPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerHPCToLLVMPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addLegalDialect<LLVM::LLVMDialect, memref::MemRefDialect>();
    target.addIllegalDialect<hpc::HPCDialect>();

    RewritePatternSet patterns(&getContext());
    patterns.add<AxpyOpLowering, CopyOpLowering, ScalOpLowering, DotOpLowering>(
        &getContext());

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }

  StringRef getArgument() const override { return "lower-hpc-to-llvm"; }
  StringRef getDescription() const override {
    return "Lower HPC dialect to LLVM calls to libhpc.a";
  }
};

} // namespace

namespace mlir {
namespace hpc {

std::unique_ptr<OperationPass<ModuleOp>> createLowerHPCToLLVMPass() {
  return std::make_unique<LowerHPCToLLVMPass>();
}

void registerPasses() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createLowerHPCToLLVMPass();
  });
}

} // namespace hpc
} // namespace mlir

#define GEN_PASS_REGISTRATION
#include "hpc/Passes.h.inc"
