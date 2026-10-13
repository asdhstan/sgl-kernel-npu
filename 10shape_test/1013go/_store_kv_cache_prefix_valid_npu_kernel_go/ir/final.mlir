// -----// IR Dump After AppendUsePrintDebugData (append-use-print-debug-data) //----- //
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9579">, hivm.aic_bitcode = #hivm.aic_bitcode<"/home/y00953696/AscendNPU-IR/build/lib/meta_op.aic.c310.bc">, hivm.aiv_bitcode = #hivm.aiv_bitcode<"/home/y00953696/AscendNPU-IR/build/lib/meta_op.aiv.c310.bc">, hivm.host_bitcode = #hivm.host_bitcode<"/home/y00953696/AscendNPU-IR/build/lib/host.bc">, hivm.mix_aic_bitcode = #hivm.mix_aic_bitcode<"/home/y00953696/AscendNPU-IR/build/lib/meta_op.mix.aic.c310.bc">, hivm.mix_aiv_bitcode = #hivm.mix_aiv_bitcode<"/home/y00953696/AscendNPU-IR/build/lib/meta_op.mix.aiv.c310.bc">, hivm.module_core_type = #hivm.module_core_type<AIV>, ssbuffer.insertionOptimization} {
  llvm.func @_store_kv_cache_prefix_valid_npu_kernel_outlined_vf_0(%arg0: !llvm.ptr<6>) attributes {hfusion.has_fill, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
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
  llvm.func @_store_kv_cache_prefix_valid_npu_kernel_outlined_vf_1(%arg0: !llvm.ptr<6>) attributes {hfusion.has_fill, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
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
  llvm.func @_store_kv_cache_prefix_valid_npu_kernel(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: !llvm.ptr<1>, %arg4: !llvm.ptr<1>, %arg5: !llvm.ptr<1>, %arg6: !llvm.ptr<1>, %arg7: !llvm.ptr<1>, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vf_mode = #hivm.vf_mode<SIMD>, hivm_regbaseintrins.target = #hivm_regbaseintrins.target<"dav-c310">, mix_mode = "aiv", parallel_mode = "simd"} {
    %0 = llvm.mlir.constant(-1 : index) : i64
    %1 = llvm.mlir.constant(48 : i64) : i64
    %2 = llvm.mlir.constant(60 : i64) : i64
    %3 = llvm.mlir.undef : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %4 = llvm.mlir.constant(1 : index) : i64
    %5 = llvm.mlir.constant(0 : index) : i64
    %6 = llvm.mlir.constant(56 : i32) : i32
    %7 = llvm.mlir.constant(0.000000e+00 : f16) : f16
    %8 = llvm.mlir.constant(1024 : index) : i64
    %9 = llvm.mlir.constant(4096 : index) : i64
    %10 = llvm.mlir.constant(64 : i32) : i32
    %11 = llvm.mlir.constant(0 : i32) : i32
    %12 = llvm.mlir.constant(4096 : i32) : i32
    %13 = llvm.mlir.constant(2 : i64) : i64
    %14 = llvm.mlir.constant(3 : i64) : i64
    %15 = llvm.mlir.constant(1 : i64) : i64
    %16 = llvm.mlir.constant(16384 : i64) : i64
    %17 = llvm.mlir.constant(8192 : i64) : i64
    %18 = llvm.mlir.constant(24576 : i64) : i64
    %19 = llvm.mlir.constant(true) : i1
    %20 = llvm.mlir.constant(2 : i32) : i32
    %21 = llvm.mlir.constant(0 : i64) : i64
    %22 = llvm.alloca %4 x i64 : (i64) -> !llvm.ptr
    llvm.store %21, %22 : i64, !llvm.ptr
    %23 = llvm.inttoptr %17 : i64 to !llvm.ptr<6>
    %24 = llvm.insertvalue %23, %3[0] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %25 = llvm.insertvalue %23, %24[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %26 = llvm.insertvalue %5, %25[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %27 = llvm.insertvalue %9, %26[3, 0] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %28 = llvm.insertvalue %4, %27[4, 0] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %29 = llvm.inttoptr %18 : i64 to !llvm.ptr<6>
    %30 = llvm.insertvalue %29, %3[0] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %31 = llvm.insertvalue %29, %30[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %32 = llvm.insertvalue %5, %31[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %33 = llvm.insertvalue %9, %32[3, 0] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %34 = llvm.insertvalue %4, %33[4, 0] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %35 = llvm.inttoptr %21 : i64 to !llvm.ptr<6>
    %36 = llvm.insertvalue %35, %3[0] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %37 = llvm.insertvalue %35, %36[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %38 = llvm.insertvalue %5, %37[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %39 = llvm.insertvalue %9, %38[3, 0] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %40 = llvm.insertvalue %4, %39[4, 0] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %41 = llvm.inttoptr %16 : i64 to !llvm.ptr<6>
    %42 = llvm.insertvalue %41, %3[0] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %43 = llvm.insertvalue %41, %42[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %44 = llvm.insertvalue %5, %43[2] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %45 = llvm.insertvalue %9, %44[3, 0] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %46 = llvm.insertvalue %4, %45[4, 0] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %47 = llvm.mul %arg12, %arg13 : i32
    %48 = llvm.mul %47, %arg14 : i32
    %49 = "hivm.intr.hivm.GET.BLOCK.IDX"() : () -> i64
    %50 = llvm.trunc %49 : i64 to i32
    "hivm.intr.hivm.SET.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 5 : i64, wait_pipe = 4 : i64}> : () -> ()
    "hivm.intr.hivm.SET.FLAG.IMM"() <{event_id = 1 : i64, set_pipe = 5 : i64, wait_pipe = 4 : i64}> : () -> ()
    "hivm.intr.hivm.SET.FLAG.IMM"() <{event_id = 2 : i64, set_pipe = 5 : i64, wait_pipe = 4 : i64}> : () -> ()
    "hivm.intr.hivm.SET.FLAG.IMM"() <{event_id = 3 : i64, set_pipe = 5 : i64, wait_pipe = 4 : i64}> : () -> ()
    llvm.br ^bb1(%50 : i32)
  ^bb1(%51: i32):  // 2 preds: ^bb0, ^bb12
    %52 = llvm.icmp "slt" %51, %48 : i32
    llvm.cond_br %52, ^bb2, ^bb13
  ^bb2:  // pred: ^bb1
    %53 = llvm.load %22 : !llvm.ptr -> i64
    %54 = llvm.urem %53, %13  : i64
    %55 = llvm.icmp "eq" %54, %15 : i64
    %56 = llvm.select %55, %34, %28 : i1, !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %57 = llvm.select %55, %46, %40 : i1, !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>
    %58 = llvm.trunc %54 : i64 to i1
    %59 = llvm.select %58, %13, %14 : i1, i64
    %60 = llvm.xor %58, %19  : i1
    %61 = llvm.zext %60 : i1 to i64
    %62 = "hivm.intr.hivm.GET.CTRL"() : () -> i64
    %63 = "hivm.intr.hivm.SBITSET0"(%62, %2) : (i64, i64) -> i64
    "hivm.intr.hivm.SET.CTRL"(%63) : (i64) -> ()
    %64 = "hivm.intr.hivm.GET.CTRL"() : () -> i64
    %65 = "hivm.intr.hivm.SBITSET1"(%64, %1) : (i64, i64) -> i64
    "hivm.intr.hivm.SET.CTRL"(%65) : (i64) -> ()
    %66 = llvm.srem %51, %arg12  : i32
    %67 = llvm.sdiv %51, %arg12  : i32
    %68 = llvm.srem %67, %arg13  : i32
    %69 = llvm.sdiv %51, %47  : i32
    %70 = llvm.srem %69, %arg14  : i32
    %71 = llvm.sdiv %66, %10  : i32
    %72 = llvm.srem %66, %10  : i32
    %73 = llvm.sext %71 : i32 to i64
    %74 = llvm.getelementptr %arg7[%73] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
    %75 = llvm.load %74 : !llvm.ptr<1> -> i32
    %76 = llvm.icmp "slt" %72, %75 : i32
    %77 = llvm.sext %66 : i32 to i64
    llvm.cond_br %76, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    %78 = llvm.getelementptr %arg6[%77] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
    %79 = llvm.load %78 : !llvm.ptr<1> -> i32
    llvm.br ^bb5(%79 : i32)
  ^bb4:  // pred: ^bb2
    llvm.br ^bb5(%11 : i32)
  ^bb5(%80: i32):  // 2 preds: ^bb3, ^bb4
    llvm.br ^bb6
  ^bb6:  // pred: ^bb5
    %81 = llvm.sext %80 : i32 to i64
    %82 = llvm.mul %68, %12 : i32
    %83 = llvm.icmp "eq" %70, %11 : i32
    llvm.cond_br %83, ^bb7, ^bb10
  ^bb7:  // pred: ^bb6
    %84 = llvm.mul %66, %arg10 : i32
    %85 = llvm.sext %84 : i32 to i64
    %86 = llvm.sext %82 : i32 to i64
    %87 = llvm.add %85, %86 : i64
    %88 = llvm.zext %76 : i1 to i64
    %89 = llvm.mul %88, %9 : i64
    %90 = llvm.add %86, %9 : i64
    %91 = llvm.intr.smax(%86, %8)  : (i64, i64) -> i64
    %92 = llvm.intr.smin(%90, %91)  : (i64, i64) -> i64
    %93 = llvm.mul %86, %0 : i64
    %94 = llvm.add %92, %93 : i64
    %95 = llvm.intr.smin(%89, %94)  : (i64, i64) -> i64
    %96 = llvm.intr.smax(%95, %5)  : (i64, i64) -> i64
    %97 = llvm.icmp "slt" %96, %9 : i64
    llvm.cond_br %97, ^bb8, ^bb9(%57, %61, %arg4, %87, %96, %arg8, %86, %arg2 : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>, i64, !llvm.ptr<1>, i64, i64, i32, i64, !llvm.ptr<1>)
  ^bb8:  // pred: ^bb7
    "hivm.intr.hivm.SET.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 5 : i64, wait_pipe = 1 : i64}> : () -> ()
    "hivm.intr.hivm.WAIT.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 5 : i64, wait_pipe = 1 : i64}> : () -> ()
    %98 = llvm.extractvalue %57[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.call @_store_kv_cache_prefix_valid_npu_kernel_outlined_vf_0(%98) {hivm.vector_function, no_inline} : (!llvm.ptr<6>) -> ()
    "hivm.intr.hivm.SET.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 1 : i64, wait_pipe = 4 : i64}> : () -> ()
    "hivm.intr.hivm.WAIT.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 1 : i64, wait_pipe = 4 : i64}> : () -> ()
    llvm.br ^bb9(%57, %61, %arg4, %87, %96, %arg8, %86, %arg2 : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>, i64, !llvm.ptr<1>, i64, i64, i32, i64, !llvm.ptr<1>)
  ^bb9(%99: !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>, %100: i64, %101: !llvm.ptr<1>, %102: i64, %103: i64, %104: i32, %105: i64, %106: !llvm.ptr<1>):  // 4 preds: ^bb7, ^bb8, ^bb10, ^bb11
    %107 = llvm.extractvalue %99[0] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %108 = llvm.extractvalue %99[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    "hivm.intr.hivm.WAIT.FLAG.REG"(%100) <{set_pipe = 5 : i64, wait_pipe = 4 : i64}> : (i64) -> ()
    llvm.call @load_gm_to_ubuf_1d_half(%101, %101, %102, %103, %4, %107, %108, %5, %103, %4, %20, %7, %5, %11) : (!llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, !llvm.ptr<6>, !llvm.ptr<6>, i64, i64, i64, i32, f16, i64, i32) -> ()
    "hivm.intr.hivm.SET.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 4 : i64, wait_pipe = 5 : i64}> : () -> ()
    %109 = llvm.sext %104 : i32 to i64
    %110 = llvm.mul %81, %109 : i64
    %111 = llvm.add %110, %105 : i64
    %112 = llvm.extractvalue %99[0] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    %113 = llvm.extractvalue %99[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    "hivm.intr.hivm.WAIT.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 4 : i64, wait_pipe = 5 : i64}> : () -> ()
    "hivm.intr.hivm.BARRIER"() <{pipe = 5 : i64}> : () -> ()
    llvm.call @store_ubuf_to_gm_1d_half(%112, %113, %5, %103, %4, %106, %106, %111, %103, %4, %11) : (!llvm.ptr<6>, !llvm.ptr<6>, i64, i64, i64, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i32) -> ()
    "hivm.intr.hivm.SET.FLAG.REG"(%100) <{set_pipe = 5 : i64, wait_pipe = 4 : i64}> : (i64) -> ()
    llvm.br ^bb12
  ^bb10:  // pred: ^bb6
    %114 = llvm.mul %66, %arg11 : i32
    %115 = llvm.sext %114 : i32 to i64
    %116 = llvm.sext %82 : i32 to i64
    %117 = llvm.add %115, %116 : i64
    %118 = llvm.zext %76 : i1 to i64
    %119 = llvm.mul %118, %9 : i64
    %120 = llvm.add %116, %9 : i64
    %121 = llvm.intr.smax(%116, %8)  : (i64, i64) -> i64
    %122 = llvm.intr.smin(%120, %121)  : (i64, i64) -> i64
    %123 = llvm.mul %116, %0 : i64
    %124 = llvm.add %122, %123 : i64
    %125 = llvm.intr.smin(%119, %124)  : (i64, i64) -> i64
    %126 = llvm.intr.smax(%125, %5)  : (i64, i64) -> i64
    %127 = llvm.icmp "slt" %126, %9 : i64
    llvm.cond_br %127, ^bb11, ^bb9(%56, %59, %arg5, %117, %126, %arg9, %116, %arg3 : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>, i64, !llvm.ptr<1>, i64, i64, i32, i64, !llvm.ptr<1>)
  ^bb11:  // pred: ^bb10
    "hivm.intr.hivm.SET.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 5 : i64, wait_pipe = 1 : i64}> : () -> ()
    "hivm.intr.hivm.WAIT.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 5 : i64, wait_pipe = 1 : i64}> : () -> ()
    %128 = llvm.extractvalue %56[1] : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.call @_store_kv_cache_prefix_valid_npu_kernel_outlined_vf_1(%128) {hivm.vector_function, no_inline} : (!llvm.ptr<6>) -> ()
    "hivm.intr.hivm.SET.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 1 : i64, wait_pipe = 4 : i64}> : () -> ()
    "hivm.intr.hivm.WAIT.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 1 : i64, wait_pipe = 4 : i64}> : () -> ()
    llvm.br ^bb9(%56, %59, %arg5, %117, %126, %arg9, %116, %arg3 : !llvm.struct<(ptr<6>, ptr<6>, i64, array<1 x i64>, array<1 x i64>)>, i64, !llvm.ptr<1>, i64, i64, i32, i64, !llvm.ptr<1>)
  ^bb12:  // pred: ^bb9
    %129 = "hivm.intr.hivm.GET.CTRL"() : () -> i64
    %130 = "hivm.intr.hivm.SBITSET1"(%129, %2) : (i64, i64) -> i64
    "hivm.intr.hivm.SET.CTRL"(%130) : (i64) -> ()
    %131 = llvm.add %53, %15 : i64
    llvm.store %131, %22 : i64, !llvm.ptr
    %132 = llvm.add %51, %6 overflow<nsw> : i32
    llvm.br ^bb1(%132 : i32)
  ^bb13:  // pred: ^bb1
    "hivm.intr.hivm.WAIT.FLAG.IMM"() <{event_id = 0 : i64, set_pipe = 5 : i64, wait_pipe = 4 : i64}> : () -> ()
    "hivm.intr.hivm.WAIT.FLAG.IMM"() <{event_id = 1 : i64, set_pipe = 5 : i64, wait_pipe = 4 : i64}> : () -> ()
    "hivm.intr.hivm.WAIT.FLAG.IMM"() <{event_id = 2 : i64, set_pipe = 5 : i64, wait_pipe = 4 : i64}> : () -> ()
    "hivm.intr.hivm.WAIT.FLAG.IMM"() <{event_id = 3 : i64, set_pipe = 5 : i64, wait_pipe = 4 : i64}> : () -> ()
    "hivm.intr.hivm.BARRIER"() <{pipe = 6 : i64}> : () -> ()
    llvm.return
  }
}


bisheng: warning: the flag '--cce-aicore-input-parameter-size=1536' has been deprecated and will be ignored [-Wunused-command-line-argument]
