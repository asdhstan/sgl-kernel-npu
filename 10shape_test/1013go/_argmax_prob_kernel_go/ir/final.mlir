// -----// IR Dump After AppendUsePrintDebugData (append-use-print-debug-data) //----- //
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9579">, hivm.aic_bitcode = #hivm.aic_bitcode<"/home/y00953696/AscendNPU-IR/build/lib/meta_op.aic.c310.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/home/y00953696/AscendNPU-IR/build/lib/meta_op.aiv.c310.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/home/y00953696/AscendNPU-IR/build/lib/host.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/home/y00953696/AscendNPU-IR/build/lib/meta_op.mix.aic.c310.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/home/y00953696/AscendNPU-IR/build/lib/meta_op.mix.aiv.c310.bc">, hivm.module_core_type = #hivm.module_core_type<AIV>, ssbuffer.insertionOptimization} {
  llvm.func @_argmax_prob_kernel_fused_1_outlined_vf_0(%arg0: !llvm.ptr<6>, %arg1: !llvm.ptr<6>, %arg2: !llvm.ptr<6>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %0 = llvm.mlir.constant(2147483647 : i32) : i32
    %1 = llvm.mlir.constant(-2139095040 : i32) : i32
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.mlir.constant(0 : i32) : i32
    %4 = llvm.mlir.constant(0xFF800000 : f32) : f32
    %5 = llvm.mlir.constant(3 : i32) : i32
    %6 = llvm.mlir.constant(5 : i32) : i32
    %7 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%3, %3) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %8 = "hivm_regbaseintrins.intr.hivm.vdups.z"(%4, %7, %2) : (f32, vector<256xi1>, i32) -> vector<64xf32>
    %9 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%3, %3) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %10 = "hivm_regbaseintrins.intr.hivm.vdups.z"(%3, %9, %2) : (i32, vector<256xi1>, i32) -> vector<64xi32>
    %11 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%3, %3) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %12 = "hivm_regbaseintrins.intr.hivm.vdups.z"(%0, %11, %2) : (i32, vector<256xi1>, i32) -> vector<64xi32>
    %13 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%2, %3) {mask_bit_width = 32 : i32, mask_op_idx = 0 : i32} : (i32, i32) -> vector<256xi1>
    %14 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%arg0, %3, %5, %3) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %15 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%arg1, %3, %5, %3) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %16 = llvm.bitcast %15 : vector<64xf32> to vector<64xi32>
    %17 = "hivm_regbaseintrins.intr.hivm.vand.x"(%16, %12, %13) : (vector<64xi32>, vector<64xi32>, vector<256xi1>) -> vector<64xi32>
    %18 = "hivm_regbaseintrins.intr.hivm.vadds.s.x"(%17, %1, %13) : (vector<64xi32>, i32, vector<256xi1>) -> vector<64xi32>
    %19 = "hivm_regbaseintrins.intr.hivm.vmins.s.x"(%18, %2, %13) : (vector<64xi32>, i32, vector<256xi1>) -> vector<64xi32>
    %20 = "hivm_regbaseintrins.intr.hivm.vmaxs.s.x"(%19, %3, %13) : (vector<64xi32>, i32, vector<256xi1>) -> vector<64xi32>
    %21 = "hivm_regbaseintrins.intr.hivm.vcmp.ne.s.z"(%20, %10, %13) : (vector<64xi32>, vector<64xi32>, vector<256xi1>) -> vector<256xi1>
    %22 = "hivm_regbaseintrins.intr.hivm.vsel"(%8, %15, %21) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %23 = llvm.bitcast %14 : vector<64xf32> to vector<64xi32>
    %24 = "hivm_regbaseintrins.intr.hivm.vand.x"(%23, %12, %13) : (vector<64xi32>, vector<64xi32>, vector<256xi1>) -> vector<64xi32>
    %25 = "hivm_regbaseintrins.intr.hivm.vadds.s.x"(%24, %1, %13) : (vector<64xi32>, i32, vector<256xi1>) -> vector<64xi32>
    %26 = "hivm_regbaseintrins.intr.hivm.vmins.s.x"(%25, %2, %13) : (vector<64xi32>, i32, vector<256xi1>) -> vector<64xi32>
    %27 = "hivm_regbaseintrins.intr.hivm.vmaxs.s.x"(%26, %3, %13) : (vector<64xi32>, i32, vector<256xi1>) -> vector<64xi32>
    %28 = "hivm_regbaseintrins.intr.hivm.vcmp.ne.s.z"(%27, %10, %13) : (vector<64xi32>, vector<64xi32>, vector<256xi1>) -> vector<256xi1>
    %29 = "hivm_regbaseintrins.intr.hivm.vsel"(%8, %14, %28) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %30 = "hivm_regbaseintrins.intr.hivm.vmax.s.x"(%29, %22, %13) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%30, %arg2, %3, %6, %3, %13) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    llvm.return
  }
  llvm.func @_argmax_prob_kernel_fused_2_outlined_vf_0(%arg0: !llvm.ptr<6>, %arg1: !llvm.ptr<6>, %arg2: !llvm.ptr<6>, %arg3: !llvm.ptr<6>, %arg4: !llvm.ptr<6>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(3 : i32) : i32
    %3 = llvm.mlir.constant(5 : i32) : i32
    %4 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%0, %1) {mask_bit_width = 32 : i32, mask_op_idx = 0 : i32} : (i32, i32) -> vector<256xi1>
    %5 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%arg0, %1, %2, %1) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %6 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%arg1, %1, %2, %1) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %7 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%arg2, %1, %2, %1) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %8 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%arg3, %1, %2, %1) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %9 = "hivm_regbaseintrins.intr.hivm.vexpdif"(%6, %7, %4, %1) : (vector<64xf32>, vector<64xf32>, vector<256xi1>, i32) -> vector<64xf32>
    %10 = "hivm_regbaseintrins.intr.hivm.vmul.s.x"(%5, %9, %4) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %11 = "hivm_regbaseintrins.intr.hivm.vadd.s.x"(%10, %8, %4) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%11, %arg4, %1, %3, %1, %4) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    llvm.return
  }
  llvm.func @_argmax_prob_kernel_fused_3_outlined_vf_0(%arg0: !llvm.ptr<6>, %arg1: f32, %arg2: !llvm.ptr<6>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %0 = llvm.mlir.constant(64 : i32) : i32
    %1 = llvm.mlir.constant(16384 : i32) : i32
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(3 : i32) : i32
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.mlir.constant(5 : i32) : i32
    %6 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%arg2, %2, %3, %2) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %7 = "hivm_regbaseintrins.intr.hivm.vbr"(%arg1) : (f32) -> vector<64xf32>
    llvm.br ^bb1(%2, %6 : i32, vector<64xf32>)
  ^bb1(%8: i32, %9: vector<64xf32>):  // 2 preds: ^bb0, ^bb2
    %10 = llvm.icmp "slt" %8, %1 : i32
    llvm.cond_br %10, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %11 = llvm.sext %8 : i32 to i64
    %12 = llvm.getelementptr %arg0[%11] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %13 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%12, %2, %2, %2) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %14 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%2, %2) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %15 = "hivm_regbaseintrins.intr.hivm.vexpdif"(%13, %7, %14, %2) : (vector<64xf32>, vector<64xf32>, vector<256xi1>, i32) -> vector<64xf32>
    %16 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%2, %2) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %17 = "hivm_regbaseintrins.intr.hivm.vcadd.s.x"(%15, %16) : (vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %18 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%2, %2) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %19 = "hivm_regbaseintrins.intr.hivm.vadd.s.x"(%9, %17, %18) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %20 = llvm.add %8, %0 overflow<nsw> : i32
    llvm.br ^bb1(%20, %19 : i32, vector<64xf32>)
  ^bb3:  // pred: ^bb1
    %21 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%4, %2) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%9, %arg2, %2, %5, %2, %21) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    llvm.return
  }
  llvm.func @_argmax_prob_kernel_outlined_vf_0(%arg0: !llvm.ptr<6>) attributes {hfusion.has_fill, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %0 = llvm.mlir.constant(0xFC00 : f16) : f16
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(16384 : i32) : i32
    %3 = llvm.mlir.constant(128 : i32) : i32
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = "hivm_regbaseintrins.intr.hivm.pge.b16"(%1, %1) {mask_bit_width = 16 : i32} : (i32, i32) -> vector<256xi1>
    %6 = "hivm_regbaseintrins.intr.hivm.vdups.z"(%0, %5, %4) : (f16, vector<256xi1>, i32) -> vector<128xf16>
    llvm.br ^bb1(%1 : i32)
  ^bb1(%7: i32):  // 2 preds: ^bb0, ^bb2
    %8 = llvm.icmp "slt" %7, %2 : i32
    llvm.cond_br %8, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %9 = llvm.sext %7 : i32 to i64
    %10 = "hivm_regbaseintrins.intr.hivm.pge.b16"(%1, %1) {mask_bit_width = 16 : i32} : (i32, i32) -> vector<256xi1>
    %11 = llvm.getelementptr %arg0[%9] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f16
    "hivm_regbaseintrins.intr.hivm.vstsx1.v128f16"(%6, %11, %1, %4, %1, %10) : (vector<128xf16>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %12 = llvm.add %7, %3 overflow<nsw> : i32
    llvm.br ^bb1(%12 : i32)
  ^bb3:  // pred: ^bb1
    llvm.return
  }
  llvm.func @_argmax_prob_kernel_outlined_vf_1(%arg0: !llvm.ptr<6>, %arg1: !llvm.ptr<6>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %0 = llvm.mlir.constant(128 : i32) : i32
    %1 = llvm.mlir.constant(16384 : i32) : i32
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(64 : index) : i64
    %4 = llvm.mlir.constant(14 : i32) : i32
    %5 = llvm.mlir.constant(2 : i32) : i32
    llvm.br ^bb1(%2 : i32)
  ^bb1(%6: i32):  // 2 preds: ^bb0, ^bb2
    %7 = llvm.icmp "slt" %6, %1 : i32
    llvm.cond_br %7, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %8 = llvm.sext %6 : i32 to i64
    %9 = llvm.getelementptr %arg0[%8] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f16
    %10 = "hivm_regbaseintrins.intr.hivm.vldsx1.v128f16"(%9, %2, %4, %2) : (!llvm.ptr<6>, i32, i32, i32) -> vector<128xf16>
    %11 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%2, %2) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %12 = "hivm_regbaseintrins.intr.hivm.vcvtff.f162f32.x"(%10, %11, %2) : (vector<128xf16>, vector<256xi1>, i32) -> vector<64xf32>
    %13 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%2, %2) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %14 = llvm.getelementptr %arg1[%8] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%12, %14, %2, %5, %2, %13) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %15 = llvm.add %8, %3 : i64
    %16 = llvm.getelementptr %arg0[%15] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f16
    %17 = "hivm_regbaseintrins.intr.hivm.vldsx1.v128f16"(%16, %2, %4, %2) : (!llvm.ptr<6>, i32, i32, i32) -> vector<128xf16>
    %18 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%2, %2) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %19 = "hivm_regbaseintrins.intr.hivm.vcvtff.f162f32.x"(%17, %18, %2) : (vector<128xf16>, vector<256xi1>, i32) -> vector<64xf32>
    %20 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%2, %2) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %21 = llvm.getelementptr %arg1[%15] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%19, %21, %2, %5, %2, %20) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %22 = llvm.add %6, %0 overflow<nsw> : i32
    llvm.br ^bb1(%22 : i32)
  ^bb3:  // pred: ^bb1
    llvm.return
  }
  llvm.func @_argmax_prob_kernel_outlined_vf_2(%arg0: !llvm.ptr<6>, %arg1: !llvm.ptr<6>, %arg2: !llvm.ptr<6>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %0 = llvm.mlir.constant(64 : i32) : i32
    %1 = llvm.mlir.constant(16384 : i32) : i32
    %2 = llvm.mlir.constant(2147483647 : i32) : i32
    %3 = llvm.mlir.constant(-2139095040 : i32) : i32
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.mlir.constant(0 : i32) : i32
    %6 = llvm.mlir.constant(0xFF800000 : f32) : f32
    %7 = llvm.mlir.constant(2 : i32) : i32
    %8 = llvm.mlir.constant(3 : i32) : i32
    %9 = llvm.mlir.constant(5 : i32) : i32
    %10 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%5, %5) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %11 = "hivm_regbaseintrins.intr.hivm.vdups.z"(%6, %10, %4) : (f32, vector<256xi1>, i32) -> vector<64xf32>
    %12 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%5, %5) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %13 = "hivm_regbaseintrins.intr.hivm.vdups.z"(%5, %12, %4) : (i32, vector<256xi1>, i32) -> vector<64xi32>
    %14 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%5, %5) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %15 = "hivm_regbaseintrins.intr.hivm.vdups.z"(%2, %14, %4) : (i32, vector<256xi1>, i32) -> vector<64xi32>
    llvm.br ^bb1(%5 : i32)
  ^bb1(%16: i32):  // 2 preds: ^bb0, ^bb2
    %17 = llvm.icmp "slt" %16, %1 : i32
    llvm.cond_br %17, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %18 = llvm.sext %16 : i32 to i64
    %19 = llvm.getelementptr %arg0[%18] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %20 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%19, %5, %5, %5) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %21 = llvm.bitcast %20 : vector<64xf32> to vector<64xi32>
    %22 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%5, %5) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %23 = "hivm_regbaseintrins.intr.hivm.vand.x"(%21, %15, %22) : (vector<64xi32>, vector<64xi32>, vector<256xi1>) -> vector<64xi32>
    %24 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%5, %5) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %25 = "hivm_regbaseintrins.intr.hivm.vadds.s.x"(%23, %3, %24) : (vector<64xi32>, i32, vector<256xi1>) -> vector<64xi32>
    %26 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%5, %5) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %27 = "hivm_regbaseintrins.intr.hivm.vmins.s.x"(%25, %4, %26) : (vector<64xi32>, i32, vector<256xi1>) -> vector<64xi32>
    %28 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%5, %5) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %29 = "hivm_regbaseintrins.intr.hivm.vmaxs.s.x"(%27, %5, %28) : (vector<64xi32>, i32, vector<256xi1>) -> vector<64xi32>
    %30 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%5, %5) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %31 = "hivm_regbaseintrins.intr.hivm.vcmp.ne.s.z"(%29, %13, %30) : (vector<64xi32>, vector<64xi32>, vector<256xi1>) -> vector<256xi1>
    %32 = "hivm_regbaseintrins.intr.hivm.vsel"(%11, %20, %31) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %33 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%5, %5) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %34 = llvm.getelementptr %arg1[%18] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%32, %34, %5, %7, %5, %33) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %35 = llvm.add %16, %0 overflow<nsw> : i32
    llvm.br ^bb1(%35 : i32)
  ^bb3:  // pred: ^bb1
    %36 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%arg2, %5, %8, %5) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    llvm.br ^bb4(%5, %36 : i32, vector<64xf32>)
  ^bb4(%37: i32, %38: vector<64xf32>):  // 2 preds: ^bb3, ^bb5
    %39 = llvm.icmp "slt" %37, %1 : i32
    llvm.cond_br %39, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %40 = llvm.sext %37 : i32 to i64
    %41 = llvm.getelementptr %arg1[%40] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %42 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%41, %5, %5, %5) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %43 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%5, %5) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %44 = "hivm_regbaseintrins.intr.hivm.vcmax.s.x"(%42, %43) : (vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %45 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%5, %5) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %46 = "hivm_regbaseintrins.intr.hivm.vmax.s.x"(%38, %44, %45) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %47 = llvm.add %37, %0 overflow<nsw> : i32
    llvm.br ^bb4(%47, %46 : i32, vector<64xf32>)
  ^bb6:  // pred: ^bb4
    %48 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%4, %5) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%38, %arg2, %5, %9, %5, %48) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    llvm.return
  }
  llvm.func private @load_gm_to_ubuf_1d_half(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: !llvm.ptr<6>, %arg6: !llvm.ptr<6>, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i32, %arg11: f16, %arg12: i64, %arg13: i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface, sym_visibility = "private"} {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %1 = llvm.mlir.constant(1 : index) : i64
    %2 = llvm.mlir.undef : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %3 = llvm.insertvalue %arg0, %2[0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %4 = llvm.insertvalue %arg1, %3[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %5 = llvm.insertvalue %arg2, %4[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %6 = llvm.insertvalue %arg3, %5[3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %7 = llvm.insertvalue %arg4, %6[4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %8 = llvm.alloca %1 x !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %7, %8 : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %9 = llvm.insertvalue %arg5, %0[0] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %10 = llvm.insertvalue %arg6, %9[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %11 = llvm.insertvalue %arg7, %10[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = llvm.insertvalue %arg8, %11[3, 0] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %13 = llvm.insertvalue %arg9, %12[4, 0] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %14 = llvm.alloca %1 x !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %13, %14 : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    llvm.call @_mlir_ciface_load_gm_to_ubuf_1d_half(%8, %14, %arg10, %arg11, %arg12, %arg13) : (!llvm.ptr, !llvm.ptr, i32, f16, i64, i32) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_load_gm_to_ubuf_1d_half(!llvm.ptr, !llvm.ptr, i32, f16, i64, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface, sym_visibility = "private"}
  llvm.func private @reduce_max_with_index_left_r_float(%arg0: !llvm.ptr<6>, %arg1: !llvm.ptr<6>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: !llvm.ptr<6>, %arg6: !llvm.ptr<6>, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: !llvm.ptr<6>, %arg11: !llvm.ptr<6>, %arg12: i64, %arg13: i64, %arg14: i64, %arg15: f32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface, sym_visibility = "private"} {
    %0 = llvm.mlir.constant(1 : index) : i64
    %1 = llvm.mlir.undef : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %2 = llvm.insertvalue %arg0, %1[0] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %3 = llvm.insertvalue %arg1, %2[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %4 = llvm.insertvalue %arg2, %3[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %5 = llvm.insertvalue %arg3, %4[3, 0] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %6 = llvm.insertvalue %arg4, %5[4, 0] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %7 = llvm.alloca %0 x !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %6, %7 : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %8 = llvm.insertvalue %arg5, %1[0] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %9 = llvm.insertvalue %arg6, %8[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %10 = llvm.insertvalue %arg7, %9[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %11 = llvm.insertvalue %arg8, %10[3, 0] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = llvm.insertvalue %arg9, %11[4, 0] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %13 = llvm.alloca %0 x !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %12, %13 : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %14 = llvm.insertvalue %arg10, %1[0] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %15 = llvm.insertvalue %arg11, %14[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %16 = llvm.insertvalue %arg12, %15[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %17 = llvm.insertvalue %arg13, %16[3, 0] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %18 = llvm.insertvalue %arg14, %17[4, 0] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %19 = llvm.alloca %0 x !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %18, %19 : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    llvm.call @_mlir_ciface_reduce_max_with_index_left_r_float(%7, %13, %19, %arg15) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, f32) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_reduce_max_with_index_left_r_float(!llvm.ptr, !llvm.ptr, !llvm.ptr, f32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface, sym_visibility = "private"}
  llvm.func private @store_ubuf_to_gm_1d_int64_t(%arg0: !llvm.ptr<6>, %arg1: !llvm.ptr<6>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: !llvm.ptr<1>, %arg6: !llvm.ptr<1>, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface, sym_visibility = "private"} {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %1 = llvm.mlir.constant(1 : index) : i64
    %2 = llvm.mlir.undef : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %3 = llvm.insertvalue %arg0, %2[0] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %4 = llvm.insertvalue %arg1, %3[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %5 = llvm.insertvalue %arg2, %4[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %6 = llvm.insertvalue %arg3, %5[3, 0] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %7 = llvm.insertvalue %arg4, %6[4, 0] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %8 = llvm.alloca %1 x !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %7, %8 : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %9 = llvm.insertvalue %arg5, %0[0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %10 = llvm.insertvalue %arg6, %9[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %11 = llvm.insertvalue %arg7, %10[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = llvm.insertvalue %arg8, %11[3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %13 = llvm.insertvalue %arg9, %12[4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %14 = llvm.alloca %1 x !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %13, %14 : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    llvm.call @_mlir_ciface_store_ubuf_to_gm_1d_int64_t(%8, %14, %arg10) : (!llvm.ptr, !llvm.ptr, i32) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_store_ubuf_to_gm_1d_int64_t(!llvm.ptr, !llvm.ptr, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface, sym_visibility = "private"}
  llvm.func private @store_ubuf_to_gm_1d_float(%arg0: !llvm.ptr<6>, %arg1: !llvm.ptr<6>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: !llvm.ptr<1>, %arg6: !llvm.ptr<1>, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface, sym_visibility = "private"} {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %1 = llvm.mlir.constant(1 : index) : i64
    %2 = llvm.mlir.undef : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %3 = llvm.insertvalue %arg0, %2[0] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %4 = llvm.insertvalue %arg1, %3[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %5 = llvm.insertvalue %arg2, %4[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %6 = llvm.insertvalue %arg3, %5[3, 0] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %7 = llvm.insertvalue %arg4, %6[4, 0] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %8 = llvm.alloca %1 x !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %7, %8 : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %9 = llvm.insertvalue %arg5, %0[0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %10 = llvm.insertvalue %arg6, %9[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %11 = llvm.insertvalue %arg7, %10[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = llvm.insertvalue %arg8, %11[3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %13 = llvm.insertvalue %arg9, %12[4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %14 = llvm.alloca %1 x !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %13, %14 : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    llvm.call @_mlir_ciface_store_ubuf_to_gm_1d_float(%8, %14, %arg10) : (!llvm.ptr, !llvm.ptr, i32) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_store_ubuf_to_gm_1d_float(!llvm.ptr, !llvm.ptr, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface, sym_visibility = "private"}
  llvm.func @_argmax_prob_kernel(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: !llvm.ptr<1>, %arg4: !llvm.ptr<1>, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vf_mode = #hivm.vf_mode<SIMD>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, mix_mode = "aiv", parallel_mode = "simd"} {
    %0 = llvm.mlir.constant(-1 : index) : i64
    %1 = llvm.mlir.constant(48 : i64) : i64
    %2 = llvm.mlir.constant(60 : i64) : i64
    %3 = llvm.mlir.undef : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %4 = llvm.mlir.constant(1 : index) : i64
    %5 = llvm.mlir.constant(0 : index) : i64
    %6 = llvm.mlir.constant(56 : i32) : i32
    %7 = llvm.mlir.constant(0xFC00 : f16) : f16
    %8 = llvm.mlir.constant(16384 : index) : i64
    %9 = llvm.mlir.constant(0xFF800000 : f32) : f32
    %10 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %11 = llvm.mlir.constant(0 : i32) : i32
    %12 = llvm.mlir.constant(16384 : i32) : i32
    %13 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %14 = llvm.mlir.constant(98304 : i64) : i64
    %15 = llvm.mlir.constant(32768 : i64) : i64
    %16 = llvm.mlir.constant(98336 : i64) : i64
    %17 = llvm.mlir.constant(163872 : i64) : i64
    %18 = llvm.mlir.constant(163904 : i64) : i64
    %19 = llvm.mlir.constant(163936 : i64) : i64
    %20 = llvm.mlir.constant(163968 : i64) : i64
    %21 = llvm.mlir.constant(164000 : i64) : i64
    %22 = llvm.mlir.constant(164032 : i64) : i64
    %23 = llvm.mlir.constant(2 : i64) : i64
    %24 = llvm.mlir.constant(3 : i64) : i64
    %25 = llvm.mlir.constant(1 : i64) : i64
    %26 = llvm.mlir.constant(164128 : i64) : i64
    %27 = llvm.mlir.constant(196896 : i64) : i64
    %28 = llvm.mlir.constant(164064 : i64) : i64
    %29 = llvm.mlir.constant(164096 : i64) : i64
    %30 = llvm.mlir.constant(196928 : i64) : i64
    %31 = llvm.mlir.constant(true) : i1
    %32 = llvm.mlir.constant(2 : i32) : i32
    %33 = llvm.mlir.constant(0 : i64) : i64
    %34 = llvm.alloca %4 x i64 : (i64) -> !llvm.ptr
    llvm.store %33, %34 : i64, !llvm.ptr
    %35 = llvm.alloca %4 x i64 : (i64) -> !llvm.ptr
    llvm.store %33, %35 : i64, !llvm.ptr
    %36 = llvm.inttoptr %29 : i64 to !llvm.ptr<6>
    %37 = llvm.insertvalue %36, %3[0] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %38 = llvm.insertvalue %36, %37[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %39 = llvm.insertvalue %5, %38[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %40 = llvm.insertvalue %4, %39[3, 0] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %41 = llvm.insertvalue %4, %40[4, 0] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %42 = llvm.inttoptr %30 : i64 to !llvm.ptr<6>
    %43 = llvm.insertvalue %42, %3[0] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %44 = llvm.insertvalue %42, %43[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %45 = llvm.insertvalue %5, %44[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %46 = llvm.insertvalue %4, %45[3, 0] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %47 = llvm.insertvalue %4, %46[4, 0] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %48 = llvm.inttoptr %28 : i64 to !llvm.ptr<6>
    %49 = llvm.insertvalue %48, %3[0] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %50 = llvm.insertvalue %48, %49[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %51 = llvm.insertvalue %5, %50[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %52 = llvm.insertvalue %4, %51[3, 0] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %53 = llvm.insertvalue %4, %52[4, 0] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %54 = llvm.inttoptr %27 : i64 to !llvm.ptr<6>
    %55 = llvm.insertvalue %54, %3[0] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %56 = llvm.insertvalue %54, %55[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %57 = llvm.insertvalue %5, %56[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %58 = llvm.insertvalue %4, %57[3, 0] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %59 = llvm.insertvalue %4, %58[4, 0] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %60 = llvm.inttoptr %33 : i64 to !llvm.ptr<6>
    %61 = llvm.insertvalue %60, %3[0] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %62 = llvm.insertvalue %60, %61[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %63 = llvm.insertvalue %5, %62[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %64 = llvm.insertvalue %8, %63[3, 0] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %65 = llvm.insertvalue %4, %64[4, 0] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %66 = llvm.inttoptr %26 : i64 to !llvm.ptr<6>
    %67 = llvm.insertvalue %66, %3[0] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %68 = llvm.insertvalue %66, %67[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %69 = llvm.insertvalue %5, %68[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %70 = llvm.insertvalue %8, %69[3, 0] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %71 = llvm.insertvalue %4, %70[4, 0] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %72 = llvm.mul %arg8, %arg9 : i32
    %73 = llvm.mul %72, %arg10 : i32
    %74 = "hivm.intr.hivm.GET.BLOCK.IDX"() : () -> i64
    %75 = llvm.trunc %74 : i64 to i32
    %76 = llvm.sext %arg6 : i32 to i64
    "hivm.intr.hivm.SET.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 5 : i64, wait_pipe = 0 : i64}> : () -> ()
    "hivm.intr.hivm.SET.FLAG.IMM"() <{event_id = 1 : i64, set_pipe = 5 : i64, wait_pipe = 0 : i64}> : () -> ()
    "hivm.intr.hivm.SET.FLAG.IMM"() <{event_id = 2 : i64, set_pipe = 5 : i64, wait_pipe = 0 : i64}> : () -> ()
    "hivm.intr.hivm.SET.FLAG.IMM"() <{event_id = 3 : i64, set_pipe = 5 : i64, wait_pipe = 0 : i64}> : () -> ()
    llvm.br ^bb1(%75 : i32)
  ^bb1(%77: i32):  // 2 preds: ^bb0, ^bb10
    %78 = llvm.icmp "slt" %77, %73 : i32
    llvm.cond_br %78, ^bb2, ^bb11
  ^bb2:  // pred: ^bb1
    %79 = "hivm.intr.hivm.GET.CTRL"() : () -> i64
    %80 = "hivm.intr.hivm.SBITSET0"(%79, %2) : (i64, i64) -> i64
    "hivm.intr.hivm.SET.CTRL"(%80) : (i64) -> ()
    %81 = "hivm.intr.hivm.GET.CTRL"() : () -> i64
    %82 = "hivm.intr.hivm.SBITSET1"(%81, %1) : (i64, i64) -> i64
    "hivm.intr.hivm.SET.CTRL"(%82) : (i64) -> ()
    %83 = llvm.srem %77, %arg8  : i32
    llvm.br ^bb3(%83 : i32)
  ^bb3(%84: i32):  // 2 preds: ^bb2, ^bb9
    %85 = llvm.icmp "slt" %84, %arg5 : i32
    llvm.cond_br %85, ^bb4, ^bb10
  ^bb4:  // pred: ^bb3
    %86 = llvm.load %35 : !llvm.ptr -> i64
    %87 = llvm.urem %86, %23  : i64
    %88 = llvm.icmp "eq" %87, %25 : i64
    %89 = llvm.select %88, %47, %41 : i1, !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %90 = llvm.select %88, %59, %53 : i1, !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %91 = llvm.trunc %87 : i64 to i1
    %92 = llvm.select %91, %23, %24 : i1, i64
    %93 = llvm.xor %91, %31  : i1
    %94 = llvm.zext %93 : i1 to i64
    %95 = llvm.mul %84, %arg7 : i32
    %96 = llvm.sext %95 : i32 to i64
    llvm.br ^bb5(%11, %9, %33, %13 : i32, f32, i64, f32)
  ^bb5(%97: i32, %98: f32, %99: i64, %100: f32):  // 2 preds: ^bb4, ^bb8
    %101 = llvm.icmp "slt" %97, %arg6 : i32
    llvm.cond_br %101, ^bb6, ^bb9
  ^bb6:  // pred: ^bb5
    %102 = llvm.load %34 : !llvm.ptr -> i64
    %103 = llvm.urem %102, %23  : i64
    %104 = llvm.icmp "eq" %103, %25 : i64
    %105 = llvm.select %104, %71, %65 : i1, !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %106 = llvm.sext %97 : i32 to i64
    %107 = llvm.add %96, %106 : i64
    %108 = llvm.add %106, %8 : i64
    %109 = llvm.intr.smax(%106, %76)  : (i64, i64) -> i64
    %110 = llvm.intr.smin(%108, %109)  : (i64, i64) -> i64
    %111 = llvm.mul %106, %0 : i64
    %112 = llvm.add %110, %111 : i64
    %113 = llvm.icmp "slt" %112, %8 : i64
    llvm.cond_br %113, ^bb7, ^bb8
  ^bb7:  // pred: ^bb6
    %114 = llvm.extractvalue %105[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.call @_argmax_prob_kernel_outlined_vf_0(%114) {hivm.vector_function, no_inline} : (!llvm.ptr<6>) -> ()
    "hivm.intr.hivm.SET.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 1 : i64, wait_pipe = 4 : i64}> : () -> ()
    "hivm.intr.hivm.WAIT.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 1 : i64, wait_pipe = 4 : i64}> : () -> ()
    llvm.br ^bb8
  ^bb8:  // 2 preds: ^bb6, ^bb7
    %115 = llvm.extractvalue %105[0] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %116 = llvm.extractvalue %105[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.call @load_gm_to_ubuf_1d_half(%arg2, %arg2, %107, %112, %4, %115, %116, %5, %112, %4, %32, %7, %5, %11) : (!llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, !llvm.ptr<6>, !llvm.ptr<6>, i64, i64, i64, i32, f16, i64, i32) -> ()
    "hivm.intr.hivm.SET.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 4 : i64, wait_pipe = 1 : i64}> : () -> ()
    %117 = llvm.inttoptr %14 : i64 to !llvm.ptr<6>
    %118 = llvm.inttoptr %15 : i64 to !llvm.ptr<6>
    "hivm.intr.hivm.WAIT.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 4 : i64, wait_pipe = 1 : i64}> : () -> ()
    %119 = llvm.extractvalue %105[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.call @_argmax_prob_kernel_outlined_vf_1(%119, %118) {hivm.vector_function, no_inline} : (!llvm.ptr<6>, !llvm.ptr<6>) -> ()
    llvm.store %9, %117 : f32, !llvm.ptr<6>
    "hivm.intr.hivm.SET.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 0 : i64, wait_pipe = 1 : i64}> : () -> ()
    %120 = llvm.inttoptr %16 : i64 to !llvm.ptr<6>
    "hivm.intr.hivm.WAIT.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 0 : i64, wait_pipe = 1 : i64}> : () -> ()
    llvm.call @_argmax_prob_kernel_outlined_vf_2(%118, %120, %117) {hivm.vector_function, no_inline} : (!llvm.ptr<6>, !llvm.ptr<6>, !llvm.ptr<6>) -> ()
    "hivm.intr.hivm.SET.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 1 : i64, wait_pipe = 0 : i64}> : () -> ()
    "hivm.intr.hivm.WAIT.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 1 : i64, wait_pipe = 0 : i64}> : () -> ()
    %121 = llvm.load %117 : !llvm.ptr<6> -> f32
    %122 = llvm.inttoptr %17 : i64 to !llvm.ptr<6>
    llvm.store %98, %122 : f32, !llvm.ptr<6>
    %123 = llvm.inttoptr %18 : i64 to !llvm.ptr<6>
    llvm.store %121, %123 : f32, !llvm.ptr<6>
    "hivm.intr.hivm.SET.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 0 : i64, wait_pipe = 1 : i64}> : () -> ()
    %124 = llvm.inttoptr %18 : i64 to !llvm.ptr<6>
    "hivm.intr.hivm.WAIT.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 0 : i64, wait_pipe = 1 : i64}> : () -> ()
    llvm.call @_argmax_prob_kernel_fused_1_outlined_vf_0(%122, %123, %124) {hivm.vector_function, no_inline} : (!llvm.ptr<6>, !llvm.ptr<6>, !llvm.ptr<6>) -> ()
    "hivm.intr.hivm.SET.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 1 : i64, wait_pipe = 0 : i64}> : () -> ()
    "hivm.intr.hivm.WAIT.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 1 : i64, wait_pipe = 0 : i64}> : () -> ()
    %125 = llvm.load %124 : !llvm.ptr<6> -> f32
    %126 = llvm.inttoptr %19 : i64 to !llvm.ptr<6>
    llvm.store %100, %126 : f32, !llvm.ptr<6>
    %127 = llvm.inttoptr %20 : i64 to !llvm.ptr<6>
    llvm.store %13, %127 : f32, !llvm.ptr<6>
    "hivm.intr.hivm.SET.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 0 : i64, wait_pipe = 1 : i64}> : () -> ()
    "hivm.intr.hivm.WAIT.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 0 : i64, wait_pipe = 1 : i64}> : () -> ()
    llvm.call @_argmax_prob_kernel_fused_3_outlined_vf_0(%118, %125, %127) {hivm.vector_function, no_inline} : (!llvm.ptr<6>, f32, !llvm.ptr<6>) -> ()
    %128 = llvm.inttoptr %17 : i64 to !llvm.ptr<6>
    llvm.call @_argmax_prob_kernel_fused_2_outlined_vf_0(%126, %122, %124, %127, %128) {hivm.vector_function, no_inline} : (!llvm.ptr<6>, !llvm.ptr<6>, !llvm.ptr<6>, !llvm.ptr<6>, !llvm.ptr<6>) -> ()
    "hivm.intr.hivm.SET.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 1 : i64, wait_pipe = 0 : i64}> : () -> ()
    "hivm.intr.hivm.WAIT.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 1 : i64, wait_pipe = 0 : i64}> : () -> ()
    %129 = llvm.load %128 : !llvm.ptr<6> -> f32
    %130 = llvm.inttoptr %21 : i64 to !llvm.ptr<6>
    %131 = llvm.inttoptr %22 : i64 to !llvm.ptr<6>
    llvm.call @reduce_max_with_index_left_r_float(%120, %120, %5, %8, %4, %130, %130, %5, %4, %4, %131, %131, %5, %4, %4, %9) : (!llvm.ptr<6>, !llvm.ptr<6>, i64, i64, i64, !llvm.ptr<6>, !llvm.ptr<6>, i64, i64, i64, !llvm.ptr<6>, !llvm.ptr<6>, i64, i64, i64, f32) -> ()
    "hivm.intr.hivm.SET.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 1 : i64, wait_pipe = 0 : i64}> : () -> ()
    "hivm.intr.hivm.WAIT.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 1 : i64, wait_pipe = 0 : i64}> : () -> ()
    %132 = llvm.load %131 : !llvm.ptr<6> -> i32
    %133 = llvm.sext %132 : i32 to i64
    %134 = llvm.sext %97 : i32 to i64
    %135 = llvm.add %133, %134 : i64
    %136 = llvm.fcmp "ogt" %121, %98 : f32
    %137 = llvm.select %136, %135, %99 : i1, i64
    %138 = llvm.add %102, %25 : i64
    llvm.store %138, %34 : i64, !llvm.ptr
    %139 = llvm.add %97, %12 overflow<nsw> : i32
    llvm.br ^bb5(%139, %125, %137, %129 : i32, f32, i64, f32)
  ^bb9:  // pred: ^bb5
    "hivm.intr.hivm.WAIT.FLAG.REG"(%94) <{set_pipe = 5 : i64, wait_pipe = 0 : i64}> : (i64) -> ()
    %140 = llvm.extractvalue %90[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.store %99, %140 : i64, !llvm.ptr<6>
    "hivm.intr.hivm.SET.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 0 : i64, wait_pipe = 5 : i64}> : () -> ()
    %141 = llvm.sext %84 : i32 to i64
    "hivm.intr.hivm.WAIT.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 0 : i64, wait_pipe = 5 : i64}> : () -> ()
    "hivm.intr.hivm.BARRIER"() <{pipe = 5 : i64}> : () -> ()
    %142 = llvm.extractvalue %90[0] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %143 = llvm.extractvalue %90[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %144 = llvm.extractvalue %90[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %145 = llvm.extractvalue %90[3, 0] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %146 = llvm.extractvalue %90[4, 0] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.call @store_ubuf_to_gm_1d_int64_t(%142, %143, %144, %145, %146, %arg3, %arg3, %141, %4, %4, %11) : (!llvm.ptr<6>, !llvm.ptr<6>, i64, i64, i64, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i32) -> ()
    "hivm.intr.hivm.SET.FLAG.REG"(%94) <{set_pipe = 5 : i64, wait_pipe = 0 : i64}> : (i64) -> ()
    %147 = llvm.fdiv %10, %100  : f32
    "hivm.intr.hivm.WAIT.FLAG.REG"(%92) <{set_pipe = 5 : i64, wait_pipe = 0 : i64}> : (i64) -> ()
    %148 = llvm.extractvalue %89[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.store %147, %148 : f32, !llvm.ptr<6>
    "hivm.intr.hivm.SET.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 0 : i64, wait_pipe = 5 : i64}> : () -> ()
    "hivm.intr.hivm.WAIT.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 0 : i64, wait_pipe = 5 : i64}> : () -> ()
    %149 = llvm.extractvalue %89[0] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %150 = llvm.extractvalue %89[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %151 = llvm.extractvalue %89[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %152 = llvm.extractvalue %89[3, 0] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %153 = llvm.extractvalue %89[4, 0] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.call @store_ubuf_to_gm_1d_float(%149, %150, %151, %152, %153, %arg4, %arg4, %141, %4, %4, %11) : (!llvm.ptr<6>, !llvm.ptr<6>, i64, i64, i64, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i32) -> ()
    "hivm.intr.hivm.SET.FLAG.REG"(%92) <{set_pipe = 5 : i64, wait_pipe = 0 : i64}> : (i64) -> ()
    %154 = llvm.add %86, %25 : i64
    llvm.store %154, %35 : i64, !llvm.ptr
    %155 = llvm.add %84, %arg8 overflow<nsw> : i32
    llvm.br ^bb3(%155 : i32)
  ^bb10:  // pred: ^bb3
    %156 = "hivm.intr.hivm.GET.CTRL"() : () -> i64
    %157 = "hivm.intr.hivm.SBITSET1"(%156, %2) : (i64, i64) -> i64
    "hivm.intr.hivm.SET.CTRL"(%157) : (i64) -> ()
    %158 = llvm.add %77, %6 overflow<nsw> : i32
    llvm.br ^bb1(%158 : i32)
  ^bb11:  // pred: ^bb1
    "hivm.intr.hivm.WAIT.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 5 : i64, wait_pipe = 0 : i64}> : () -> ()
    "hivm.intr.hivm.WAIT.FLAG.IMM"() <{event_id = 1 : i64, set_pipe = 5 : i64, wait_pipe = 0 : i64}> : () -> ()
    "hivm.intr.hivm.WAIT.FLAG.IMM"() <{event_id = 2 : i64, set_pipe = 5 : i64, wait_pipe = 0 : i64}> : () -> ()
    "hivm.intr.hivm.WAIT.FLAG.IMM"() <{event_id = 3 : i64, set_pipe = 5 : i64, wait_pipe = 0 : i64}> : () -> ()
    "hivm.intr.hivm.BARRIER"() <{pipe = 6 : i64}> : () -> ()
    llvm.return
  }
}


bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
