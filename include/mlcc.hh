#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlcc {
void initialize(mlir::MLIRContext &context);
void registerPasses();
} // namespace mlcc
