#ifndef HPC_OPS
#define HPC_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def HPC_Dialect : Dialect {
  let name = "hpc";
  let summary = "HPC BLAS dialect";
  let description = [{
    High-performance BLAS operations that lower to libhpc. a calls.
  }];
  let cppNamespace = "::mlir::hpc";
}

class HPC_Op<string mnemonic, list<Trait> traits = []> : 
    Op<HPC_Dialect, mnemonic, traits>;

def HPC_AxpyOp : HPC_Op<"axpy", [MemoryEffects<[MemWrite]>]> {
  let summary = "Y = alpha * X + Y";
  let arguments = (ins
    I64Attr:$n,
    AnyFloat: $alpha,
    AnyMemRef:$src,
    AnyMemRef: $dst
  );
  let assemblyFormat = [{
    $n `,` $alpha `,` $src `,` $dst attr-dict `:` type($alpha) `,` type($src) `,` type($dst)
  }];
}

def HPC_CopyOp : HPC_Op<"copy", [MemoryEffects<[MemWrite]>]> {
  let summary = "Y = X";
  let arguments = (ins
    I64Attr:$n,
    AnyMemRef:$src,
    AnyMemRef:$dst
  );
  let assemblyFormat = [{
    $n `,` $src `,` $dst attr-dict `:` type($src) `,` type($dst)
  }];
}

def HPC_ScalOp : HPC_Op<"scal", [MemoryEffects<[MemWrite]>]> {
  let summary = "X = alpha * X";
  let arguments = (ins
    I64Attr:$n,
    AnyFloat:$alpha,
    AnyMemRef: $dst
  );
  let assemblyFormat = [{
    $n `,` $alpha `,` $dst attr-dict `:` type($alpha) `,` type($dst)
  }];
}

def HPC_DotOp :  HPC_Op<"dot", [Pure]> {
  let summary = "result = X · Y";
  let arguments = (ins
    I64Attr: $n,
    AnyMemRef:$src1,
    AnyMemRef:$src2
  );
  let results = (outs AnyFloat: $result);
  let assemblyFormat = [{
    $n `,` $src1 `,` $src2 attr-dict `:` type($src1) `,` type($src2) `->` type($result)
  }];
}

#endif // HPC_OPS
