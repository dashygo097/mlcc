#include "../Dialect.hpp"
#include "../Passes.hpp"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace mlir {
namespace hpc {

void populateDotToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                         RewritePatternSet &patterns);

namespace {

struct LowerHPCToLLVMPass
    : public PassWrapper<LowerHPCToLLVMPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerHPCToLLVMPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, func::FuncDialect>();
  }

  StringRef getArgument() const final { return "lower-hpc-to-llvm"; }

  StringRef getDescription() const final {
    return "Lower HPC dialect to LLVM dialect with runtime calls";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();

    LLVMTypeConverter typeConverter(context);

    RewritePatternSet patterns(context);

    populateDotToLLVMConversionPatterns(typeConverter, patterns);

    populateFuncToLLVMConversionPatterns(typeConverter, patterns);
    populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);

    LLVMConversionTarget target(*context);
    target.addLegalOp<ModuleOp>();
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addIllegalDialect<hpc::HPCDialect>();

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createLowerHPCToLLVMPass() {
  return std::make_unique<LowerHPCToLLVMPass>();
}

} // namespace hpc
} // namespace mlir

#define GEN_PASS_DEF_LOWERHPCTOLLVM
#include "hpc/Passes.h.inc"

namespace mlir {
namespace hpc {

void registerPasses() { PassRegistration<LowerHPCToLLVMPass>(); }

} // namespace hpc
} // namespace mlir
