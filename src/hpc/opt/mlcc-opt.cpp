#include "lib/Dialect.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace llvm;

int main(int argc, char **argv) {
  DialectRegistry registry;

  registry.insert<hpc::HPCDialect>();
  registry.insert<func::FuncDialect>();
  registry.insert<arith::ArithDialect>();
  registry.insert<memref::MemRefDialect>();

  registerCSEPass();
  registerCanonicalizerPass();

  return asMainReturnCode(MlirOptMain(argc, argv, "hpc-mlcc-opt", registry));
}
