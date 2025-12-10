#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include <mlcc.hh>

namespace mlcc {

void initialize(mlir::MLIRContext &context) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();

  mlir::registerAllPasses();
}

void registerPasses() { mlir::registerAllPasses(); }

} // namespace mlcc
