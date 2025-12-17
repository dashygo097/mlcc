#include "../lib/Dialect.hpp"
#include "../lib/Passes.hpp"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace llvm;
using namespace mlir;

static cl::opt<std::string>
    inputFilename(cl::Positional, cl::desc("<input file>"), cl::init("-"));

static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"));

static cl::opt<bool> emitMLIR("emit-mlir", cl::desc("Emit MLIR output"),
                              cl::init(false));

static cl::opt<bool> emitLLVMIR("emit-llvm", cl::desc("Emit LLVM IR output"),
                                cl::init(false));

static cl::opt<std::string>
    optimizationLevel("O", cl::desc("Optimization level (0, 1, 2, 3)"),
                      cl::init("2"));

static OwningOpRef<ModuleOp> loadMLIR(MLIRContext &context,
                                      const std::string &filename) {
  context.loadDialect<hpc::HPCDialect>();

  std::string errorMessage;
  auto file = openInputFile(filename, &errorMessage);
  if (!file) {
    llvm::errs() << "Error: " << errorMessage << "\n";
    return nullptr;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());

  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(sourceMgr, &context);

  if (!module) {
    llvm::errs() << "Error: Failed to parse input file\n";
    return nullptr;
  }

  return module;
}

static LogicalResult applyOptimizations(ModuleOp module, int optLevel) {
  mlir::PassManager pm(module.getContext());

  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  if (optLevel >= 2) {
    pm.addPass(mlir::affine::createLoopFusionPass());
  }

  if (optLevel >= 3) {
    pm.addPass(mlir::affine::createAffineScalarReplacementPass());
  }

  if (failed(pm.run(module))) {
    llvm::errs() << "Error: Optimization pipeline failed\n";
    return failure();
  }

  return success();
}

static LogicalResult lowerToLLVM(ModuleOp module) {
  mlir::PassManager pm(module.getContext());

  pm.addPass(hpc::createLowerHPCToLLVMPass());

  pm.addPass(mlir::createConvertSCFToCFPass());
  pm.addPass(mlir::createConvertFuncToLLVMPass());
  pm.addPass(mlir::createArithToLLVMConversionPass());
  pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());

  if (failed(pm.run(module))) {
    llvm::errs() << "Error: Lowering to LLVM failed\n";
    return failure();
  }

  return success();
}

static std::unique_ptr<llvm::Module>
exportToLLVMIR(ModuleOp module, llvm::LLVMContext &llvmContext) {
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);

  if (!llvmModule) {
    llvm::errs() << "Error: Failed to translate to LLVM IR\n";
    return nullptr;
  }

  int optLevel = std::stoi(optimizationLevel);
  if (optLevel > 0) {
    auto optPipeline = mlir::makeOptimizingTransformer(optLevel, 0, nullptr);
    if (optPipeline) {
      if (auto err = optPipeline(llvmModule.get())) {
        llvm::errs() << "Warning:  LLVM optimization had issues\n";
      }
    }
  }

  return llvmModule;
}

static LogicalResult writeOutput(ModuleOp module, llvm::Module *llvmModule,
                                 const std::string &filename) {
  std::string errorMessage;
  auto output = openOutputFile(filename, &errorMessage);
  if (!output) {
    llvm::errs() << "Error: " << errorMessage << "\n";
    return failure();
  }

  if (emitMLIR) {
    module.print(output->os());
  } else if (emitLLVMIR) {
    if (!llvmModule) {
      llvm::errs() << "Error: No LLVM module to emit\n";
      return failure();
    }
    llvmModule->print(output->os(), nullptr);
  } else {
    module.print(output->os());
  }

  output->keep();
  return success();
}

int main(int argc, char **argv) {
  llvm::InitLLVM initLLVM(argc, argv);
  cl::ParseCommandLineOptions(argc, argv, "MLIR HPC Compiler\n");

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  mlir::DialectRegistry registry;
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);

  MLIRContext context(registry);

  llvm::LLVMContext llvmContext;

  context.loadDialect<hpc::HPCDialect, func::FuncDialect, arith::ArithDialect,
                      affine::AffineDialect, scf::SCFDialect,
                      memref::MemRefDialect, LLVM::LLVMDialect>();

  mlir::registerAllPasses();
  hpc::registerPasses();

  llvm::outs() << "╔════════════════════════════════════════╗\n";
  llvm::outs() << "║     MLCC HPC++ Compiler v1.0          ║\n";
  llvm::outs() << "╚════════════════════════════════════════╝\n\n";

  auto module = loadMLIR(context, inputFilename);
  if (!module) {
    return 1;
  }

  if (failed(mlir::verify(*module))) {
    llvm::errs() << "Error: Input module verification failed\n";
    module->dump();
    return 1;
  }

  llvm::outs() << "[1/4] Module loaded and verified\n";

  int optLevel = std::stoi(optimizationLevel);
  if (failed(applyOptimizations(*module, optLevel))) {
    return 1;
  }

  llvm::outs() << "[2/4] Optimizations applied (level " << optLevel << ")\n";

  if (emitMLIR) {
    if (failed(writeOutput(*module, nullptr, outputFilename))) {
      return 1;
    }
    llvm::outs() << "[3/4] MLIR emitted\n";
    llvm::outs() << "\n✓ Output written to:  " << outputFilename << "\n";
    return 0;
  }

  if (failed(lowerToLLVM(*module))) {
    llvm::errs() << "Lowered module:\n";
    module->dump();
    return 1;
  }

  llvm::outs() << "[3/4] Lowered to LLVM dialect\n";

  auto llvmModule = exportToLLVMIR(*module, llvmContext);
  if (!llvmModule) {
    return 1;
  }

  llvm::outs() << "[4/4] LLVM IR generated\n";

  if (failed(writeOutput(*module, llvmModule.get(), outputFilename))) {
    return 1;
  }

  llvm::outs() << "\n✓ Compilation successful!\n";
  llvm::outs() << "✓ Output written to: " << outputFilename << "\n\n";

  return 0;
}
