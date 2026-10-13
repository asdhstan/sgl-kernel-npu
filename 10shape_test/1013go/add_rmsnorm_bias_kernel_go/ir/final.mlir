// -----// IR Dump After AppendUsePrintDebugData (append-use-print-debug-data) //----- //
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9579">, hivm.aic_bitcode = #hivm.aic_bitcode<"/home/y00953696/AscendNPU-IR/build/lib/meta_op.aic.c310.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/home/y00953696/AscendNPU-IR/build/lib/meta_op.aiv.c310.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/home/y00953696/AscendNPU-IR/build/lib/host.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/home/y00953696/AscendNPU-IR/build/lib/meta_op.mix.aic.c310.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/home/y00953696/AscendNPU-IR/build/lib/meta_op.mix.aiv.c310.bc">, hivm.module_core_type = #hivm.module_core_type<AIV>, ssbuffer.insertionOptimization} {
  llvm.func @add_rmsnorm_bias_kernel_fused_0_outlined_vf_0(%arg0: i64, %arg1: !llvm.ptr<6>, %arg2: !llvm.ptr<6>, %arg3: !llvm.ptr<6>, %arg4: !llvm.ptr<6>, %arg5: !llvm.ptr<6>, %arg6: f32, %arg7: !llvm.ptr<6>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %0 = llvm.mlir.constant(128 : i32) : i32
    %1 = llvm.mlir.constant(4096 : i32) : i32
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
    %9 = llvm.add %8, %arg0 : i64
    %10 = llvm.add %8, %arg0 : i64
    %11 = llvm.add %8, %arg0 : i64
    %12 = llvm.add %8, %arg0 : i64
    %13 = llvm.getelementptr %arg1[%9] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f16
    %14 = "hivm_regbaseintrins.intr.hivm.vldas"(%13) : (!llvm.ptr<6>) -> vector<32xi8>
    %15 = "hivm_regbaseintrins.intr.hivm.vldus.post.f16"(%13, %14, %2) : (!llvm.ptr<6>, vector<32xi8>, i32) -> !llvm.struct<(vector<128xf16>, vector<32xi8>, ptr<6>)>
    %16 = llvm.extractvalue %15[0] : !llvm.struct<(vector<128xf16>, vector<32xi8>, ptr<6>)> 
    %17 = "hivm_regbaseintrins.intr.hivm.vintlv"(%16, %16) : (vector<128xf16>, vector<128xf16>) -> !llvm.struct<(vector<128xf16>, vector<128xf16>)>
    %18 = llvm.extractvalue %17[0] : !llvm.struct<(vector<128xf16>, vector<128xf16>)> 
    %19 = llvm.getelementptr %arg2[%10] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f16
    %20 = "hivm_regbaseintrins.intr.hivm.vldas"(%19) : (!llvm.ptr<6>) -> vector<32xi8>
    %21 = "hivm_regbaseintrins.intr.hivm.vldus.post.f16"(%19, %20, %2) : (!llvm.ptr<6>, vector<32xi8>, i32) -> !llvm.struct<(vector<128xf16>, vector<32xi8>, ptr<6>)>
    %22 = llvm.extractvalue %21[0] : !llvm.struct<(vector<128xf16>, vector<32xi8>, ptr<6>)> 
    %23 = "hivm_regbaseintrins.intr.hivm.vintlv"(%22, %22) : (vector<128xf16>, vector<128xf16>) -> !llvm.struct<(vector<128xf16>, vector<128xf16>)>
    %24 = llvm.extractvalue %23[0] : !llvm.struct<(vector<128xf16>, vector<128xf16>)> 
    %25 = llvm.getelementptr %arg3[%11] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f16
    %26 = "hivm_regbaseintrins.intr.hivm.vldas"(%25) : (!llvm.ptr<6>) -> vector<32xi8>
    %27 = "hivm_regbaseintrins.intr.hivm.vldus.post.f16"(%25, %26, %2) : (!llvm.ptr<6>, vector<32xi8>, i32) -> !llvm.struct<(vector<128xf16>, vector<32xi8>, ptr<6>)>
    %28 = llvm.extractvalue %27[0] : !llvm.struct<(vector<128xf16>, vector<32xi8>, ptr<6>)> 
    %29 = "hivm_regbaseintrins.intr.hivm.vintlv"(%28, %28) : (vector<128xf16>, vector<128xf16>) -> !llvm.struct<(vector<128xf16>, vector<128xf16>)>
    %30 = llvm.extractvalue %29[0] : !llvm.struct<(vector<128xf16>, vector<128xf16>)> 
    %31 = llvm.getelementptr %arg4[%12] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f16
    %32 = "hivm_regbaseintrins.intr.hivm.vldas"(%31) : (!llvm.ptr<6>) -> vector<32xi8>
    %33 = "hivm_regbaseintrins.intr.hivm.vldus.post.f16"(%31, %32, %2) : (!llvm.ptr<6>, vector<32xi8>, i32) -> !llvm.struct<(vector<128xf16>, vector<32xi8>, ptr<6>)>
    %34 = llvm.extractvalue %33[0] : !llvm.struct<(vector<128xf16>, vector<32xi8>, ptr<6>)> 
    %35 = "hivm_regbaseintrins.intr.hivm.vintlv"(%34, %34) : (vector<128xf16>, vector<128xf16>) -> !llvm.struct<(vector<128xf16>, vector<128xf16>)>
    %36 = llvm.extractvalue %35[0] : !llvm.struct<(vector<128xf16>, vector<128xf16>)> 
    %37 = llvm.getelementptr %arg5[%8] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f16
    %38 = "hivm_regbaseintrins.intr.hivm.vldsx1.v128f16"(%37, %2, %4, %2) : (!llvm.ptr<6>, i32, i32, i32) -> vector<128xf16>
    %39 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%2, %2) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %40 = "hivm_regbaseintrins.intr.hivm.vcvtff.f162f32.x"(%38, %39, %2) : (vector<128xf16>, vector<256xi1>, i32) -> vector<64xf32>
    %41 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%2, %2) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %42 = "hivm_regbaseintrins.intr.hivm.vcvtff.f162f32.x"(%36, %41, %2) : (vector<128xf16>, vector<256xi1>, i32) -> vector<64xf32>
    %43 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%2, %2) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %44 = "hivm_regbaseintrins.intr.hivm.vcvtff.f162f32.x"(%30, %43, %2) : (vector<128xf16>, vector<256xi1>, i32) -> vector<64xf32>
    %45 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%2, %2) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %46 = "hivm_regbaseintrins.intr.hivm.vadd.s.x"(%18, %24, %45) : (vector<128xf16>, vector<128xf16>, vector<256xi1>) -> vector<128xf16>
    %47 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%2, %2) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %48 = "hivm_regbaseintrins.intr.hivm.vcvtff.f162f32.x"(%46, %47, %2) : (vector<128xf16>, vector<256xi1>, i32) -> vector<64xf32>
    %49 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%2, %2) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %50 = "hivm_regbaseintrins.intr.hivm.vmuls.s.x"(%48, %arg6, %49) : (vector<64xf32>, f32, vector<256xi1>) -> vector<64xf32>
    %51 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%2, %2) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %52 = "hivm_regbaseintrins.intr.hivm.vmul.s.x"(%50, %44, %51) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %53 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%2, %2) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %54 = "hivm_regbaseintrins.intr.hivm.vadd.s.x"(%52, %42, %53) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %55 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%2, %2) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %56 = "hivm_regbaseintrins.intr.hivm.vmul.s.x"(%54, %40, %55) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %57 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%2, %2) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %58 = llvm.getelementptr %arg7[%8] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%56, %58, %2, %5, %2, %57) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %59 = llvm.add %8, %3 : i64
    %60 = llvm.add %59, %arg0 : i64
    %61 = llvm.add %59, %arg0 : i64
    %62 = llvm.add %59, %arg0 : i64
    %63 = llvm.add %59, %arg0 : i64
    %64 = llvm.getelementptr %arg1[%60] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f16
    %65 = "hivm_regbaseintrins.intr.hivm.vldas"(%64) : (!llvm.ptr<6>) -> vector<32xi8>
    %66 = "hivm_regbaseintrins.intr.hivm.vldus.post.f16"(%64, %65, %2) : (!llvm.ptr<6>, vector<32xi8>, i32) -> !llvm.struct<(vector<128xf16>, vector<32xi8>, ptr<6>)>
    %67 = llvm.extractvalue %66[0] : !llvm.struct<(vector<128xf16>, vector<32xi8>, ptr<6>)> 
    %68 = "hivm_regbaseintrins.intr.hivm.vintlv"(%67, %67) : (vector<128xf16>, vector<128xf16>) -> !llvm.struct<(vector<128xf16>, vector<128xf16>)>
    %69 = llvm.extractvalue %68[0] : !llvm.struct<(vector<128xf16>, vector<128xf16>)> 
    %70 = llvm.getelementptr %arg2[%61] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f16
    %71 = "hivm_regbaseintrins.intr.hivm.vldas"(%70) : (!llvm.ptr<6>) -> vector<32xi8>
    %72 = "hivm_regbaseintrins.intr.hivm.vldus.post.f16"(%70, %71, %2) : (!llvm.ptr<6>, vector<32xi8>, i32) -> !llvm.struct<(vector<128xf16>, vector<32xi8>, ptr<6>)>
    %73 = llvm.extractvalue %72[0] : !llvm.struct<(vector<128xf16>, vector<32xi8>, ptr<6>)> 
    %74 = "hivm_regbaseintrins.intr.hivm.vintlv"(%73, %73) : (vector<128xf16>, vector<128xf16>) -> !llvm.struct<(vector<128xf16>, vector<128xf16>)>
    %75 = llvm.extractvalue %74[0] : !llvm.struct<(vector<128xf16>, vector<128xf16>)> 
    %76 = llvm.getelementptr %arg3[%62] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f16
    %77 = "hivm_regbaseintrins.intr.hivm.vldas"(%76) : (!llvm.ptr<6>) -> vector<32xi8>
    %78 = "hivm_regbaseintrins.intr.hivm.vldus.post.f16"(%76, %77, %2) : (!llvm.ptr<6>, vector<32xi8>, i32) -> !llvm.struct<(vector<128xf16>, vector<32xi8>, ptr<6>)>
    %79 = llvm.extractvalue %78[0] : !llvm.struct<(vector<128xf16>, vector<32xi8>, ptr<6>)> 
    %80 = "hivm_regbaseintrins.intr.hivm.vintlv"(%79, %79) : (vector<128xf16>, vector<128xf16>) -> !llvm.struct<(vector<128xf16>, vector<128xf16>)>
    %81 = llvm.extractvalue %80[0] : !llvm.struct<(vector<128xf16>, vector<128xf16>)> 
    %82 = llvm.getelementptr %arg4[%63] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f16
    %83 = "hivm_regbaseintrins.intr.hivm.vldas"(%82) : (!llvm.ptr<6>) -> vector<32xi8>
    %84 = "hivm_regbaseintrins.intr.hivm.vldus.post.f16"(%82, %83, %2) : (!llvm.ptr<6>, vector<32xi8>, i32) -> !llvm.struct<(vector<128xf16>, vector<32xi8>, ptr<6>)>
    %85 = llvm.extractvalue %84[0] : !llvm.struct<(vector<128xf16>, vector<32xi8>, ptr<6>)> 
    %86 = "hivm_regbaseintrins.intr.hivm.vintlv"(%85, %85) : (vector<128xf16>, vector<128xf16>) -> !llvm.struct<(vector<128xf16>, vector<128xf16>)>
    %87 = llvm.extractvalue %86[0] : !llvm.struct<(vector<128xf16>, vector<128xf16>)> 
    %88 = llvm.getelementptr %arg5[%59] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f16
    %89 = "hivm_regbaseintrins.intr.hivm.vldsx1.v128f16"(%88, %2, %4, %2) : (!llvm.ptr<6>, i32, i32, i32) -> vector<128xf16>
    %90 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%2, %2) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %91 = "hivm_regbaseintrins.intr.hivm.vcvtff.f162f32.x"(%89, %90, %2) : (vector<128xf16>, vector<256xi1>, i32) -> vector<64xf32>
    %92 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%2, %2) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %93 = "hivm_regbaseintrins.intr.hivm.vcvtff.f162f32.x"(%87, %92, %2) : (vector<128xf16>, vector<256xi1>, i32) -> vector<64xf32>
    %94 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%2, %2) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %95 = "hivm_regbaseintrins.intr.hivm.vcvtff.f162f32.x"(%81, %94, %2) : (vector<128xf16>, vector<256xi1>, i32) -> vector<64xf32>
    %96 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%2, %2) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %97 = "hivm_regbaseintrins.intr.hivm.vadd.s.x"(%69, %75, %96) : (vector<128xf16>, vector<128xf16>, vector<256xi1>) -> vector<128xf16>
    %98 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%2, %2) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %99 = "hivm_regbaseintrins.intr.hivm.vcvtff.f162f32.x"(%97, %98, %2) : (vector<128xf16>, vector<256xi1>, i32) -> vector<64xf32>
    %100 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%2, %2) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %101 = "hivm_regbaseintrins.intr.hivm.vmuls.s.x"(%99, %arg6, %100) : (vector<64xf32>, f32, vector<256xi1>) -> vector<64xf32>
    %102 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%2, %2) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %103 = "hivm_regbaseintrins.intr.hivm.vmul.s.x"(%101, %95, %102) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %104 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%2, %2) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %105 = "hivm_regbaseintrins.intr.hivm.vadd.s.x"(%103, %93, %104) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %106 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%2, %2) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %107 = "hivm_regbaseintrins.intr.hivm.vmul.s.x"(%105, %91, %106) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %108 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%2, %2) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %109 = llvm.getelementptr %arg7[%59] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%107, %109, %2, %5, %2, %108) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %110 = llvm.add %6, %0 overflow<nsw> : i32
    llvm.br ^bb1(%110 : i32)
  ^bb3:  // pred: ^bb1
    llvm.return
  }
  llvm.func @add_rmsnorm_bias_kernel_fused_1_outlined_vf_0(%arg0: !llvm.ptr<6>, %arg1: !llvm.ptr<6>, %arg2: !llvm.ptr<6>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %0 = llvm.mlir.constant(128 : i32) : i32
    %1 = llvm.mlir.constant(4096 : i32) : i32
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(64 : index) : i64
    %4 = llvm.mlir.constant(14 : i32) : i32
    %5 = llvm.mlir.constant(4 : i32) : i32
    %6 = llvm.mlir.constant(1 : i32) : i32
    %7 = llvm.mlir.constant(12 : i32) : i32
    llvm.br ^bb1(%2 : i32)
  ^bb1(%8: i32):  // 2 preds: ^bb0, ^bb2
    %9 = llvm.icmp "slt" %8, %1 : i32
    llvm.cond_br %9, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %10 = llvm.sext %8 : i32 to i64
    %11 = llvm.getelementptr %arg0[%10] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %12 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%11, %2, %2, %2) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %13 = llvm.getelementptr %arg1[%10] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f16
    %14 = "hivm_regbaseintrins.intr.hivm.vldsx1.v128f16"(%13, %2, %4, %2) : (!llvm.ptr<6>, i32, i32, i32) -> vector<128xf16>
    %15 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%2, %2) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %16 = "hivm_regbaseintrins.intr.hivm.vcvtff.f162f32.x"(%14, %15, %2) : (vector<128xf16>, vector<256xi1>, i32) -> vector<64xf32>
    %17 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%2, %2) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %18 = "hivm_regbaseintrins.intr.hivm.vadd.s.x"(%12, %16, %17) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %19 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%2, %2) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %20 = "hivm_regbaseintrins.intr.hivm.vcvtff.f322f16.x"(%18, %19, %5, %2, %2) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<128xf16>
    %21 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%2, %2) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %22 = "hivm_regbaseintrins.intr.hivm.vcvtfi.f162s8.x"(%20, %21, %5, %6, %2) : (vector<128xf16>, vector<256xi1>, i32, i32, i32) -> vector<256xi8>
    %23 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%2, %2) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %24 = llvm.bitcast %22 : vector<256xi8> to vector<64xi32>
    %25 = llvm.getelementptr %arg2[%10] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, i8
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64s32"(%24, %25, %2, %7, %2, %23) : (vector<64xi32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %26 = llvm.add %10, %3 : i64
    %27 = llvm.getelementptr %arg0[%26] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
    %28 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%27, %2, %2, %2) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %29 = llvm.getelementptr %arg1[%26] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f16
    %30 = "hivm_regbaseintrins.intr.hivm.vldsx1.v128f16"(%29, %2, %4, %2) : (!llvm.ptr<6>, i32, i32, i32) -> vector<128xf16>
    %31 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%2, %2) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %32 = "hivm_regbaseintrins.intr.hivm.vcvtff.f162f32.x"(%30, %31, %2) : (vector<128xf16>, vector<256xi1>, i32) -> vector<64xf32>
    %33 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%2, %2) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %34 = "hivm_regbaseintrins.intr.hivm.vadd.s.x"(%28, %32, %33) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %35 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%2, %2) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %36 = "hivm_regbaseintrins.intr.hivm.vcvtff.f322f16.x"(%34, %35, %5, %2, %2) : (vector<64xf32>, vector<256xi1>, i32, i32, i32) -> vector<128xf16>
    %37 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%2, %2) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %38 = "hivm_regbaseintrins.intr.hivm.vcvtfi.f162s8.x"(%36, %37, %5, %6, %2) : (vector<128xf16>, vector<256xi1>, i32, i32, i32) -> vector<256xi8>
    %39 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%2, %2) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %40 = llvm.bitcast %38 : vector<256xi8> to vector<64xi32>
    %41 = llvm.getelementptr %arg2[%26] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, i8
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64s32"(%40, %41, %2, %7, %2, %39) : (vector<64xi32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %42 = llvm.add %8, %0 overflow<nsw> : i32
    llvm.br ^bb1(%42 : i32)
  ^bb3:  // pred: ^bb1
    llvm.return
  }
  llvm.func @add_rmsnorm_bias_kernel_fused_2_outlined_vf_0(%arg0: !llvm.ptr<6>, %arg1: !llvm.ptr<6>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %0 = llvm.mlir.constant(128 : i32) : i32
    %1 = llvm.mlir.constant(16384 : i32) : i32
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(64 : index) : i64
    %4 = llvm.mlir.constant(3 : i32) : i32
    %5 = llvm.mlir.constant(14 : i32) : i32
    %6 = llvm.mlir.constant(1 : i32) : i32
    %7 = llvm.mlir.constant(5 : i32) : i32
    %8 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%arg1, %2, %4, %2) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    llvm.br ^bb1(%2, %8 : i32, vector<64xf32>)
  ^bb1(%9: i32, %10: vector<64xf32>):  // 2 preds: ^bb0, ^bb2
    %11 = llvm.icmp "slt" %9, %1 : i32
    llvm.cond_br %11, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %12 = llvm.sext %9 : i32 to i64
    %13 = llvm.getelementptr %arg0[%12] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f16
    %14 = "hivm_regbaseintrins.intr.hivm.vldsx1.v128f16"(%13, %2, %5, %2) : (!llvm.ptr<6>, i32, i32, i32) -> vector<128xf16>
    %15 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%2, %2) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %16 = "hivm_regbaseintrins.intr.hivm.vcvtff.f162f32.x"(%14, %15, %2) : (vector<128xf16>, vector<256xi1>, i32) -> vector<64xf32>
    %17 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%2, %2) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %18 = "hivm_regbaseintrins.intr.hivm.vmul.s.x"(%16, %16, %17) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %19 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%2, %2) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %20 = "hivm_regbaseintrins.intr.hivm.vcadd.s.x"(%18, %19) : (vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %21 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%2, %2) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %22 = "hivm_regbaseintrins.intr.hivm.vadd.s.x"(%10, %20, %21) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %23 = llvm.add %12, %3 : i64
    %24 = llvm.getelementptr %arg0[%23] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f16
    %25 = "hivm_regbaseintrins.intr.hivm.vldsx1.v128f16"(%24, %2, %5, %2) : (!llvm.ptr<6>, i32, i32, i32) -> vector<128xf16>
    %26 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%2, %2) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %27 = "hivm_regbaseintrins.intr.hivm.vcvtff.f162f32.x"(%25, %26, %2) : (vector<128xf16>, vector<256xi1>, i32) -> vector<64xf32>
    %28 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%2, %2) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %29 = "hivm_regbaseintrins.intr.hivm.vmul.s.x"(%27, %27, %28) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %30 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%2, %2) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %31 = "hivm_regbaseintrins.intr.hivm.vcadd.s.x"(%29, %30) : (vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %32 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%2, %2) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    %33 = "hivm_regbaseintrins.intr.hivm.vadd.s.x"(%22, %31, %32) : (vector<64xf32>, vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    %34 = llvm.add %9, %0 overflow<nsw> : i32
    llvm.br ^bb1(%34, %33 : i32, vector<64xf32>)
  ^bb3:  // pred: ^bb1
    %35 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%6, %2) {mask_bit_width = 32 : i32} : (i32, i32) -> vector<256xi1>
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%10, %arg1, %2, %7, %2, %35) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    llvm.return
  }
  llvm.func @add_rmsnorm_bias_kernel_fused_3_outlined_vf_0(%arg0: !llvm.ptr<6>, %arg1: !llvm.ptr<6>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %0 = llvm.mlir.constant(9.99999997E-7 : f32) : f32
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(3 : i32) : i32
    %4 = llvm.mlir.constant(5 : i32) : i32
    %5 = "hivm_regbaseintrins.intr.hivm.pge.b32"(%1, %2) {mask_bit_width = 32 : i32, mask_op_idx = 0 : i32} : (i32, i32) -> vector<256xi1>
    %6 = "hivm_regbaseintrins.intr.hivm.vldsx1.v64f32"(%arg0, %2, %3, %2) : (!llvm.ptr<6>, i32, i32, i32) -> vector<64xf32>
    %7 = "hivm_regbaseintrins.intr.hivm.vadds.s.x"(%6, %0, %5) : (vector<64xf32>, f32, vector<256xi1>) -> vector<64xf32>
    %8 = "hivm_regbaseintrins.intr.hivm.vsqrt.x"(%7, %5) : (vector<64xf32>, vector<256xi1>) -> vector<64xf32>
    "hivm_regbaseintrins.intr.hivm.vstsx1.v64f32"(%8, %arg1, %2, %4, %2, %5) : (vector<64xf32>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    llvm.return
  }
  llvm.func @add_rmsnorm_bias_kernel_outlined_vf_0(%arg0: !llvm.ptr<6>) attributes {hfusion.has_fill, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %0 = llvm.mlir.constant(0.000000e+00 : f16) : f16
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(4096 : i32) : i32
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
  llvm.func @add_rmsnorm_bias_kernel_outlined_vf_1(%arg0: !llvm.ptr<6>) attributes {hfusion.has_fill, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %0 = llvm.mlir.constant(0.000000e+00 : f16) : f16
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(4096 : i32) : i32
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
  llvm.func @add_rmsnorm_bias_kernel_outlined_vf_2(%arg0: !llvm.ptr<6>, %arg1: !llvm.ptr<6>, %arg2: !llvm.ptr<6>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %0 = llvm.mlir.constant(128 : i32) : i32
    %1 = llvm.mlir.constant(16384 : i32) : i32
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(1 : i32) : i32
    llvm.br ^bb1(%2 : i32)
  ^bb1(%4: i32):  // 2 preds: ^bb0, ^bb2
    %5 = llvm.icmp "slt" %4, %1 : i32
    llvm.cond_br %5, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %6 = llvm.sext %4 : i32 to i64
    %7 = llvm.getelementptr %arg0[%6] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f16
    %8 = "hivm_regbaseintrins.intr.hivm.vldsx1.v128f16"(%7, %2, %2, %2) : (!llvm.ptr<6>, i32, i32, i32) -> vector<128xf16>
    %9 = llvm.getelementptr %arg1[%6] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f16
    %10 = "hivm_regbaseintrins.intr.hivm.vldsx1.v128f16"(%9, %2, %2, %2) : (!llvm.ptr<6>, i32, i32, i32) -> vector<128xf16>
    %11 = "hivm_regbaseintrins.intr.hivm.pge.b16"(%2, %2) {mask_bit_width = 16 : i32} : (i32, i32) -> vector<256xi1>
    %12 = "hivm_regbaseintrins.intr.hivm.vadd.s.x"(%8, %10, %11) : (vector<128xf16>, vector<128xf16>, vector<256xi1>) -> vector<128xf16>
    %13 = "hivm_regbaseintrins.intr.hivm.pge.b16"(%2, %2) {mask_bit_width = 16 : i32} : (i32, i32) -> vector<256xi1>
    %14 = llvm.getelementptr %arg2[%6] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f16
    "hivm_regbaseintrins.intr.hivm.vstsx1.v128f16"(%12, %14, %2, %3, %2, %13) : (vector<128xf16>, !llvm.ptr<6>, i32, i32, i32, vector<256xi1>) -> ()
    %15 = llvm.add %4, %0 overflow<nsw> : i32
    llvm.br ^bb1(%15 : i32)
  ^bb3:  // pred: ^bb1
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
  llvm.func private @store_ubuf_to_gm_1d_half(%arg0: !llvm.ptr<6>, %arg1: !llvm.ptr<6>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: !llvm.ptr<1>, %arg6: !llvm.ptr<1>, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface, sym_visibility = "private"} {
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
    llvm.call @_mlir_ciface_store_ubuf_to_gm_1d_half(%8, %14, %arg10) : (!llvm.ptr, !llvm.ptr, i32) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_store_ubuf_to_gm_1d_half(!llvm.ptr, !llvm.ptr, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface, sym_visibility = "private"}
  llvm.func private @store_ubuf_to_gm_1d_int8_t(%arg0: !llvm.ptr<6>, %arg1: !llvm.ptr<6>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: !llvm.ptr<1>, %arg6: !llvm.ptr<1>, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface, sym_visibility = "private"} {
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
    llvm.call @_mlir_ciface_store_ubuf_to_gm_1d_int8_t(%8, %14, %arg10) : (!llvm.ptr, !llvm.ptr, i32) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_store_ubuf_to_gm_1d_int8_t(!llvm.ptr, !llvm.ptr, i32) attributes {hacc.always_inline, hivm.func_core_type = #hivm.func_core_type<AIV>, llvm.emit_c_interface, sym_visibility = "private"}
  llvm.func @add_rmsnorm_bias_kernel(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: !llvm.ptr<1>, %arg4: !llvm.ptr<1>, %arg5: !llvm.ptr<1>, %arg6: !llvm.ptr<1>, %arg7: !llvm.ptr<1>, %arg8: !llvm.ptr<1>, %arg9: !llvm.ptr<1>, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vf_mode = #hivm.vf_mode<SIMD>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, mix_mode = "aiv", parallel_mode = "simd"} {
    %0 = llvm.mlir.constant(-1 : index) : i64
    %1 = llvm.mlir.constant(48 : i64) : i64
    %2 = llvm.mlir.constant(60 : i64) : i64
    %3 = llvm.mlir.constant(1 : index) : i64
    %4 = llvm.mlir.constant(0 : index) : i64
    %5 = llvm.mlir.constant(0 : i64) : i64
    %6 = llvm.mlir.constant(32768 : i64) : i64
    %7 = llvm.mlir.constant(65536 : i64) : i64
    %8 = llvm.mlir.constant(98304 : i64) : i64
    %9 = llvm.mlir.constant(131072 : i64) : i64
    %10 = llvm.mlir.constant(131104 : i64) : i64
    %11 = llvm.mlir.constant(131136 : i64) : i64
    %12 = llvm.mlir.constant(163904 : i64) : i64
    %13 = llvm.mlir.constant(172096 : i64) : i64
    %14 = llvm.mlir.constant(180288 : i64) : i64
    %15 = llvm.mlir.constant(2 : i32) : i32
    %16 = llvm.mlir.constant(16384 : i32) : i32
    %17 = llvm.mlir.constant(1.638400e+04 : f32) : f32
    %18 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %19 = llvm.mlir.constant(0 : i32) : i32
    %20 = llvm.mlir.constant(4096 : i32) : i32
    %21 = llvm.mlir.constant(0.000000e+00 : f16) : f16
    %22 = llvm.mlir.constant(16384 : index) : i64
    %23 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %24 = llvm.mlir.constant(4096 : index) : i64
    %25 = llvm.mlir.constant(56 : i32) : i32
    %26 = llvm.mul %arg11, %arg12 : i32
    %27 = llvm.mul %26, %arg13 : i32
    %28 = "hivm.intr.hivm.GET.BLOCK.IDX"() : () -> i64
    %29 = llvm.trunc %28 : i64 to i32
    %30 = llvm.mul %arg11, %16 : i32
    %31 = llvm.sext %30 : i32 to i64
    "hivm.intr.hivm.SET.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 5 : i64, wait_pipe = 1 : i64}> : () -> ()
    "hivm.intr.hivm.SET.FLAG.IMM"() <{event_id = 1 : i64, set_pipe = 5 : i64, wait_pipe = 1 : i64}> : () -> ()
    "hivm.intr.hivm.SET.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 1 : i64, wait_pipe = 4 : i64}> : () -> ()
    "hivm.intr.hivm.SET.FLAG.IMM"() <{event_id = 1 : i64, set_pipe = 1 : i64, wait_pipe = 4 : i64}> : () -> ()
    "hivm.intr.hivm.SET.FLAG.IMM"() <{event_id = 2 : i64, set_pipe = 1 : i64, wait_pipe = 4 : i64}> : () -> ()
    llvm.br ^bb1(%29 : i32)
  ^bb1(%32: i32):  // 2 preds: ^bb0, ^bb12
    %33 = llvm.icmp "slt" %32, %27 : i32
    llvm.cond_br %33, ^bb2, ^bb13
  ^bb2:  // pred: ^bb1
    %34 = "hivm.intr.hivm.GET.CTRL"() : () -> i64
    %35 = "hivm.intr.hivm.SBITSET0"(%34, %2) : (i64, i64) -> i64
    "hivm.intr.hivm.SET.CTRL"(%35) : (i64) -> ()
    %36 = "hivm.intr.hivm.GET.CTRL"() : () -> i64
    %37 = "hivm.intr.hivm.SBITSET1"(%36, %1) : (i64, i64) -> i64
    "hivm.intr.hivm.SET.CTRL"(%37) : (i64) -> ()
    %38 = llvm.srem %32, %arg11  : i32
    %39 = llvm.inttoptr %5 : i64 to !llvm.ptr<6>
    "hivm.intr.hivm.WAIT.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 1 : i64, wait_pipe = 4 : i64}> : () -> ()
    "hivm.intr.hivm.BARRIER"() <{pipe = 4 : i64}> : () -> ()
    llvm.call @load_gm_to_ubuf_1d_half(%arg4, %arg4, %4, %22, %3, %39, %39, %4, %22, %3, %19, %21, %4, %19) : (!llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, !llvm.ptr<6>, !llvm.ptr<6>, i64, i64, i64, i32, f16, i64, i32) -> ()
    %40 = llvm.mul %38, %16 : i32
    %41 = llvm.sext %40 : i32 to i64
    "hivm.intr.hivm.SET.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 1 : i64, wait_pipe = 4 : i64}> : () -> ()
    llvm.br ^bb3(%38, %41 : i32, i64)
  ^bb3(%42: i32, %43: i64):  // 2 preds: ^bb2, ^bb11
    %44 = llvm.icmp "slt" %42, %arg10 : i32
    llvm.cond_br %44, ^bb4, ^bb12
  ^bb4:  // pred: ^bb3
    %45 = llvm.inttoptr %6 : i64 to !llvm.ptr<6>
    "hivm.intr.hivm.WAIT.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 1 : i64, wait_pipe = 4 : i64}> : () -> ()
    llvm.call @load_gm_to_ubuf_1d_half(%arg2, %arg2, %43, %22, %3, %45, %45, %4, %22, %3, %19, %21, %4, %19) : (!llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, !llvm.ptr<6>, !llvm.ptr<6>, i64, i64, i64, i32, f16, i64, i32) -> ()
    %46 = llvm.inttoptr %7 : i64 to !llvm.ptr<6>
    llvm.call @load_gm_to_ubuf_1d_half(%arg3, %arg3, %43, %22, %3, %46, %46, %4, %22, %3, %19, %21, %4, %19) : (!llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, !llvm.ptr<6>, !llvm.ptr<6>, i64, i64, i64, i32, f16, i64, i32) -> ()
    "hivm.intr.hivm.SET.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 4 : i64, wait_pipe = 1 : i64}> : () -> ()
    %47 = llvm.inttoptr %8 : i64 to !llvm.ptr<6>
    "hivm.intr.hivm.WAIT.FLAG.IMM"() <{event_id = 1 : i64, set_pipe = 5 : i64, wait_pipe = 1 : i64}> : () -> ()
    "hivm.intr.hivm.WAIT.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 4 : i64, wait_pipe = 1 : i64}> : () -> ()
    llvm.call @add_rmsnorm_bias_kernel_outlined_vf_2(%45, %46, %47) {hivm.vector_function, no_inline} : (!llvm.ptr<6>, !llvm.ptr<6>, !llvm.ptr<6>) -> ()
    "hivm.intr.hivm.SET.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 1 : i64, wait_pipe = 5 : i64}> : () -> ()
    "hivm.intr.hivm.WAIT.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 1 : i64, wait_pipe = 5 : i64}> : () -> ()
    llvm.call @store_ubuf_to_gm_1d_half(%47, %47, %4, %22, %3, %arg9, %arg9, %43, %22, %3, %19) : (!llvm.ptr<6>, !llvm.ptr<6>, i64, i64, i64, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i32) -> ()
    "hivm.intr.hivm.SET.FLAG.IMM"() <{event_id = 1 : i64, set_pipe = 5 : i64, wait_pipe = 1 : i64}> : () -> ()
    %48 = llvm.inttoptr %9 : i64 to !llvm.ptr<6>
    llvm.store %23, %48 : f32, !llvm.ptr<6>
    "hivm.intr.hivm.SET.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 0 : i64, wait_pipe = 1 : i64}> : () -> ()
    "hivm.intr.hivm.WAIT.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 0 : i64, wait_pipe = 1 : i64}> : () -> ()
    llvm.call @add_rmsnorm_bias_kernel_fused_2_outlined_vf_0(%47, %48) {hivm.vector_function, no_inline} : (!llvm.ptr<6>, !llvm.ptr<6>) -> ()
    "hivm.intr.hivm.SET.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 1 : i64, wait_pipe = 0 : i64}> : () -> ()
    "hivm.intr.hivm.WAIT.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 1 : i64, wait_pipe = 0 : i64}> : () -> ()
    %49 = llvm.load %48 : !llvm.ptr<6> -> f32
    %50 = llvm.fdiv %49, %17  : f32
    %51 = llvm.inttoptr %10 : i64 to !llvm.ptr<6>
    llvm.store %50, %51 : f32, !llvm.ptr<6>
    "hivm.intr.hivm.SET.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 0 : i64, wait_pipe = 1 : i64}> : () -> ()
    %52 = llvm.inttoptr %10 : i64 to !llvm.ptr<6>
    "hivm.intr.hivm.WAIT.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 0 : i64, wait_pipe = 1 : i64}> : () -> ()
    llvm.call @add_rmsnorm_bias_kernel_fused_3_outlined_vf_0(%51, %52) {hivm.vector_function, no_inline} : (!llvm.ptr<6>, !llvm.ptr<6>) -> ()
    "hivm.intr.hivm.SET.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 1 : i64, wait_pipe = 0 : i64}> : () -> ()
    "hivm.intr.hivm.WAIT.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 1 : i64, wait_pipe = 0 : i64}> : () -> ()
    %53 = llvm.load %52 : !llvm.ptr<6> -> f32
    %54 = llvm.fdiv %18, %53  : f32
    %55 = llvm.inttoptr %11 : i64 to !llvm.ptr<6>
    llvm.call @load_gm_to_ubuf_1d_half(%arg5, %arg5, %4, %22, %3, %55, %55, %4, %22, %3, %19, %21, %4, %19) : (!llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, !llvm.ptr<6>, !llvm.ptr<6>, i64, i64, i64, i32, f16, i64, i32) -> ()
    %56 = llvm.mul %42, %16 : i32
    %57 = llvm.sext %56 : i32 to i64
    llvm.br ^bb5(%19 : i32)
  ^bb5(%58: i32):  // 2 preds: ^bb4, ^bb10
    %59 = llvm.icmp "slt" %58, %16 : i32
    llvm.cond_br %59, ^bb6, ^bb11
  ^bb6:  // pred: ^bb5
    %60 = llvm.sext %58 : i32 to i64
    %61 = llvm.inttoptr %12 : i64 to !llvm.ptr<6>
    %62 = llvm.add %60, %24 : i64
    %63 = llvm.intr.smax(%60, %22)  : (i64, i64) -> i64
    %64 = llvm.intr.smin(%62, %63)  : (i64, i64) -> i64
    %65 = llvm.mul %60, %0 : i64
    %66 = llvm.add %64, %65 : i64
    %67 = llvm.icmp "slt" %66, %24 : i64
    llvm.cond_br %67, ^bb7, ^bb8
  ^bb7:  // pred: ^bb6
    llvm.call @add_rmsnorm_bias_kernel_outlined_vf_0(%61) {hivm.vector_function, no_inline} : (!llvm.ptr<6>) -> ()
    "hivm.intr.hivm.SET.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 1 : i64, wait_pipe = 4 : i64}> : () -> ()
    "hivm.intr.hivm.WAIT.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 1 : i64, wait_pipe = 4 : i64}> : () -> ()
    llvm.br ^bb8
  ^bb8:  // 2 preds: ^bb6, ^bb7
    "hivm.intr.hivm.WAIT.FLAG.IMM"() <{event_id = 1 : i64, set_pipe = 1 : i64, wait_pipe = 4 : i64}> : () -> ()
    llvm.call @load_gm_to_ubuf_1d_half(%arg6, %arg6, %60, %66, %3, %61, %61, %4, %66, %3, %15, %21, %4, %19) : (!llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, !llvm.ptr<6>, !llvm.ptr<6>, i64, i64, i64, i32, f16, i64, i32) -> ()
    "hivm.intr.hivm.SET.FLAG.IMM"() <{event_id = 1 : i64, set_pipe = 4 : i64, wait_pipe = 1 : i64}> : () -> ()
    %68 = llvm.inttoptr %13 : i64 to !llvm.ptr<6>
    llvm.cond_br %67, ^bb9, ^bb10
  ^bb9:  // pred: ^bb8
    llvm.call @add_rmsnorm_bias_kernel_outlined_vf_1(%68) {hivm.vector_function, no_inline} : (!llvm.ptr<6>) -> ()
    "hivm.intr.hivm.SET.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 1 : i64, wait_pipe = 4 : i64}> : () -> ()
    "hivm.intr.hivm.WAIT.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 1 : i64, wait_pipe = 4 : i64}> : () -> ()
    llvm.br ^bb10
  ^bb10:  // 2 preds: ^bb8, ^bb9
    "hivm.intr.hivm.WAIT.FLAG.IMM"() <{event_id = 2 : i64, set_pipe = 1 : i64, wait_pipe = 4 : i64}> : () -> ()
    llvm.call @load_gm_to_ubuf_1d_half(%arg7, %arg7, %60, %66, %3, %68, %68, %4, %66, %3, %15, %21, %4, %19) : (!llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, !llvm.ptr<6>, !llvm.ptr<6>, i64, i64, i64, i32, f16, i64, i32) -> ()
    "hivm.intr.hivm.SET.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 4 : i64, wait_pipe = 1 : i64}> : () -> ()
    %69 = llvm.inttoptr %14 : i64 to !llvm.ptr<6>
    "hivm.intr.hivm.WAIT.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 5 : i64, wait_pipe = 1 : i64}> : () -> ()
    "hivm.intr.hivm.WAIT.FLAG.IMM"() <{event_id = 1 : i64, set_pipe = 4 : i64, wait_pipe = 1 : i64}> : () -> ()
    llvm.call @add_rmsnorm_bias_kernel_fused_0_outlined_vf_0(%60, %45, %46, %39, %55, %61, %54, %69) {hivm.vector_function, no_inline} : (i64, !llvm.ptr<6>, !llvm.ptr<6>, !llvm.ptr<6>, !llvm.ptr<6>, !llvm.ptr<6>, f32, !llvm.ptr<6>) -> ()
    "hivm.intr.hivm.SET.FLAG.IMM"() <{event_id = 1 : i64, set_pipe = 1 : i64, wait_pipe = 4 : i64}> : () -> ()
    %70 = llvm.inttoptr %14 : i64 to !llvm.ptr<6>
    "hivm.intr.hivm.WAIT.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 4 : i64, wait_pipe = 1 : i64}> : () -> ()
    llvm.call @add_rmsnorm_bias_kernel_fused_1_outlined_vf_0(%69, %68, %70) {hivm.vector_function, no_inline} : (!llvm.ptr<6>, !llvm.ptr<6>, !llvm.ptr<6>) -> ()
    "hivm.intr.hivm.SET.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 1 : i64, wait_pipe = 5 : i64}> : () -> ()
    "hivm.intr.hivm.SET.FLAG.IMM"() <{event_id = 2 : i64, set_pipe = 1 : i64, wait_pipe = 4 : i64}> : () -> ()
    %71 = llvm.add %57, %60 : i64
    "hivm.intr.hivm.WAIT.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 1 : i64, wait_pipe = 5 : i64}> : () -> ()
    llvm.call @store_ubuf_to_gm_1d_int8_t(%70, %70, %4, %66, %3, %arg8, %arg8, %71, %66, %3, %19) : (!llvm.ptr<6>, !llvm.ptr<6>, i64, i64, i64, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i32) -> ()
    "hivm.intr.hivm.SET.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 5 : i64, wait_pipe = 1 : i64}> : () -> ()
    %72 = llvm.add %58, %20 overflow<nsw> : i32
    llvm.br ^bb5(%72 : i32)
  ^bb11:  // pred: ^bb5
    "hivm.intr.hivm.SET.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 1 : i64, wait_pipe = 4 : i64}> : () -> ()
    %73 = llvm.add %43, %31 : i64
    %74 = llvm.add %42, %arg11 overflow<nsw> : i32
    llvm.br ^bb3(%74, %73 : i32, i64)
  ^bb12:  // pred: ^bb3
    "hivm.intr.hivm.WAIT.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 1 : i64, wait_pipe = 4 : i64}> : () -> ()
    "hivm.intr.hivm.SET.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 1 : i64, wait_pipe = 4 : i64}> : () -> ()
    %75 = "hivm.intr.hivm.GET.CTRL"() : () -> i64
    %76 = "hivm.intr.hivm.SBITSET1"(%75, %2) : (i64, i64) -> i64
    "hivm.intr.hivm.SET.CTRL"(%76) : (i64) -> ()
    %77 = llvm.add %32, %25 overflow<nsw> : i32
    llvm.br ^bb1(%77 : i32)
  ^bb13:  // pred: ^bb1
    "hivm.intr.hivm.WAIT.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 1 : i64, wait_pipe = 4 : i64}> : () -> ()
    "hivm.intr.hivm.WAIT.FLAG.IMM"() <{event_id = 1 : i64, set_pipe = 1 : i64, wait_pipe = 4 : i64}> : () -> ()
    "hivm.intr.hivm.WAIT.FLAG.IMM"() <{event_id = 2 : i64, set_pipe = 1 : i64, wait_pipe = 4 : i64}> : () -> ()
    "hivm.intr.hivm.WAIT.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 5 : i64, wait_pipe = 1 : i64}> : () -> ()
    "hivm.intr.hivm.WAIT.FLAG.IMM"() <{event_id = 1 : i64, set_pipe = 5 : i64, wait_pipe = 1 : i64}> : () -> ()
    "hivm.intr.hivm.BARRIER"() <{pipe = 6 : i64}> : () -> ()
    llvm.return
  }
}


bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
