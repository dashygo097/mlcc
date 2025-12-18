#include "../Dialect.hpp"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::hpc;

namespace {

static LLVM::LLVMFuncOp
getOrInsertRuntimeFunction(OpBuilder &builder, ModuleOp module, StringRef name,
                           Type resultType, ArrayRef<Type> argTypes) {

  if (auto funcOp = module.lookupSymbol<LLVM::LLVMFuncOp>(name)) {
    return funcOp;
  }

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());

  auto funcType = LLVM::LLVMFunctionType::get(resultType, argTypes);
  return LLVM::LLVMFuncOp::create(builder, module.getLoc(), name, funcType);
}

// DOT
struct DotOpLowering : public ConvertOpToLLVMPattern<hpc::DotOp> {
  using ConvertOpToLLVMPattern<hpc::DotOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(hpc::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    auto ctx = rewriter.getContext();

    auto src1Type = llvm::cast<MemRefType>(op.getSrc1().getType());
    auto elementType = src1Type.getElementType();

    bool isF32 = elementType.isF32();
    bool isF64 = elementType.isF64();

    if (!isF32 && !isF64) {
      return rewriter.notifyMatchFailure(op, "only f32 and f64 supported");
    }

    auto llvmI64Type = IntegerType::get(ctx, 64);
    auto llvmPtrType = LLVM::LLVMPointerType::get(ctx);

    Type llvmFloatType;
    if (isF32) {
      llvmFloatType = Float32Type::get(ctx);
    } else {
      llvmFloatType = Float64Type::get(ctx);
    }

    StringRef funcName = isF32 ? "hpc_dot_f32" : "hpc_dot_f64";

    auto runtimeFunc =
        getOrInsertRuntimeFunction(rewriter, module, funcName, llvmPtrType,
                                   {llvmI64Type, llvmPtrType, llvmPtrType});

    MemRefDescriptor src1Desc(adaptor.getSrc1());
    MemRefDescriptor src2Desc(adaptor.getSrc2());

    Value src1Ptr = src1Desc.alignedPtr(rewriter, loc);
    Value src2Ptr = src2Desc.alignedPtr(rewriter, loc);

    Value sizeVal;
    if (src1Type.hasStaticShape()) {
      int64_t size = src1Type.getDimSize(0);
      sizeVal = LLVM::ConstantOp::create(rewriter, loc, llvmI64Type,
                                         rewriter.getI64IntegerAttr(size));
    } else {
      sizeVal = src1Desc.size(rewriter, loc, 0);
    }

    auto callOp = LLVM::CallOp::create(rewriter, loc, runtimeFunc,
                                       ValueRange{sizeVal, src1Ptr, src2Ptr});

    rewriter.replaceOp(op, callOp.getResult());

    return success();
  }
};

} // anonymous namespace

namespace mlir {
namespace hpc {

void populateDotToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                         RewritePatternSet &patterns) {
  patterns.add<DotOpLowering>(converter);
}

} // namespace hpc
} // namespace mlir
