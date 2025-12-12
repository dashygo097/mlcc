#ifndef HPC_OPS
#define HPC_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def HPC_Dialect : Dialect {
  let name = "hpc";
  let summary = "HPC dialect";
  let description = [{
    High-performance operations that lower to libhpc.a calls.
  }];
  let cppNamespace = "::mlir::hpc";
}

class HPC_Op<string mnemonic, list<Trait> traits = []> : 
    Op<HPC_Dialect, mnemonic, traits>;

// dot
def HPC_DotOp :  HPC_Op<"dot", [Pure]> {
  let summary = "result = X · Y";
  let description = [{
    Computes the dot product of vectors X and Y.
  }];
  
  let arguments = (ins
    I64Attr:$n,
    AnyMemRef:$src1,
    AnyMemRef: $src2
  );
  
  let results = (outs AnyFloat: $result);
  
  let assemblyFormat = [{
    $n `,` $src1 `,` $src2 attr-dict `:` type($src1) `,` type($src2) `->` type($result)
  }];
  
  let hasVerifier = 1;
}

#endif
