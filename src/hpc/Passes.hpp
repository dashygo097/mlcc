#pragma once

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class ModuleOp;

namespace hpc {

/// Create pass to lower HPC dialect to LLVM
std::unique_ptr<OperationPass<ModuleOp>> createLowerHPCToLLVMPass();

/// Register all HPC passes
void registerPasses();

} // namespace hpc
} // namespace mlir
