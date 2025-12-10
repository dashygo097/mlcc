#pragma once

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class ModuleOp;

namespace hpc {

std::unique_ptr<OperationPass<ModuleOp>> createLowerHPCToLLVMPass();

void registerPasses();

} // namespace hpc
} // namespace mlir
