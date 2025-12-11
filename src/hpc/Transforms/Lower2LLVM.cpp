#include "../Dialect.hpp"
#include "../Passes.hpp"

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {

// Helpers
LLVM::LLVMFuncOp getOrInsertFunction(PatternRewriter &rewriter, ModuleOp module,
                                     StringRef name,
                                     LLVM::LLVMFunctionType type) {
  if (auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(name))
    return func;

  PatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  return rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), name, type);
}

Value extractMemRefBasePtr(Location loc, Value memrefDescriptor,
                           ConversionPatternRewriter &rewriter) {

  auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

  Value alignedPtr = rewriter.create<LLVM::ExtractValueOp>(
      loc, memrefDescriptor, ArrayRef<int64_t>{1});

  return alignedPtr;
}

// Conversion patterns
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

    Value src1Ptr = extractMemRefBasePtr(loc, adaptor.getSrc1(), rewriter);
    Value src2Ptr = extractMemRefBasePtr(loc, adaptor.getSrc2(), rewriter);

    Value n = rewriter.create<LLVM::ConstantOp>(loc, i64Type, op.getNAttr());

    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, resultType, funcName, ValueRange{n, src1Ptr, src2Ptr});

    rewriter.replaceOp(op, callOp.getResult());
    return success();
  }
};

// Pass definition
struct LowerHPCToLLVMPass
    : public PassWrapper<LowerHPCToLLVMPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerHPCToLLVMPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();

    LLVMTypeConverter typeConverter(context);

    ConversionTarget target(*context);
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addIllegalDialect<hpc::HPCDialect>();

    RewritePatternSet patterns(context);
    patterns.add<DotOpLowering>(typeConverter, context);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
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
