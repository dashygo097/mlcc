#pragma once

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class ModuleOp;

namespace hpc {

std::unique_ptr<OperationPass<ModuleOp>> createLowerHPCToLLVMPass();

void registerPasses();

#define GEN_PASS_REGISTRATION
#include "hpc/Passes.h.inc"

} // namespace hpc
} // namespace mlir
