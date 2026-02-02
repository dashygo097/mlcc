#ifndef HPC_OPS
#define HPC_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def HPC_Dialect : Dialect {
  let name = "hpc";
  let summary = "HPC dialect";
  let cppNamespace = "::mlir::hpc";
}

class HPC_Op<string mnemonic, list<Trait> traits = []> : 
    Op<HPC_Dialect, mnemonic, traits>;

// DOT 
def HPC_DotOp :  HPC_Op<"dot", [Pure]> {
  let summary = "dot production";
  
  let arguments = (ins
    AnyMemRef: $src1,
    AnyMemRef: $src2
  );

  let assemblyFormat = [{
    $src1 `,` $src2 attr-dict `:` type($src1) `,` type($src2) `->` type($result)
  }];
  
  let results = (outs AnyFloat: $result);
}

#endif
