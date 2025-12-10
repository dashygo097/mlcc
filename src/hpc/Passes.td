#ifndef HPC_PASSES
#define HPC_PASSES

include "mlir/Pass/PassBase.td"

def LowerHPCToLLVM : Pass<"lower-hpc-to-llvm", "ModuleOp"> {
  let summary = "Lower HPC dialect to LLVM";
  let description = [{
    Lowers HPC BLAS operations to LLVM function calls to libhpc.a.
  }];
  let constructor = "mlir::hpc::createLowerHPCToLLVMPass()";
}

#endif // HPC_PASSES
