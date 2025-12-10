#include "hpc/Dialect.hpp"
#include "hpc/Passes.hpp"
#include <mlcc.hh>

#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  // Register all MLIR passes
  mlir::registerAllPasses();

  // Register HPC passes
  mlir::hpc::registerPasses();

  // Register all dialects
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::hpc::HPCDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "MLCC HPC Optimizer\n", registry));
}
