// -----// IR Dump After RemoveVectorLayoutAttr (remove-vector-layout-attr) //----- //
func.func @add_rmsnorm_bias_kernel_fused_1_outlined_vf_0(%arg0: memref<4096xf32, #hivm.address_space<ub>>, %arg1: memref<4096xf16, #hivm.address_space<ub>>, %arg2: memref<4096xi8, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
  %c64 = arith.constant 64 : index
  %c4096 = arith.constant 4096 : index
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  scf.for %arg3 = %c0 to %c4096 step %c128 {
    %subview = memref.subview %arg0[%arg3] [64] [1] : memref<4096xf32, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_0 = memref.subview %arg1[%arg3] [64] [1] : memref<4096xf16, #hivm.address_space<ub>> to memref<64xf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_1 = memref.subview %arg2[%arg3] [64] [1] : memref<4096xi8, #hivm.address_space<ub>> to memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %res = ave.hir.vload <NORM> %subview[%c0] {functionType = #ave.func_dist_type<norm>} : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xf32>
    %res_2 = ave.hir.vload <UNPK_B16> %subview_0[%c0] {functionType = #ave.func_dist_type<pk>} : memref<64xf16, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xf16>
    %0 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
    %1 = ave.hir.vextf %res_2, <part_even>, %0 {functionType = #ave.func_dist_type<even>} : vector<64xf16>, vector<64xf32>, vector<64xi1>
    %2 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
    %3 = ave.hir.vadd %res, %1, %2 : vector<64xf32>, vector<64xi1>
    %4 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
    %5 = ave.hir.vtruncf %3, <trunc>, false, <part_even>, %4 {functionType = #ave.func_dist_type<even>} : vector<64xf32>, vector<64xf16>, vector<64xi1>
    %6 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
    %7 = ave.hir.vfptosi %5, <trunc>, %6 {functionType = #ave.func_dist_type<even>, part = #ave.vcvt_part_type<part_even>, sat = true} : vector<64xf16>, vector<64xi1>, vector<64xi8>
    %8 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
    %9 = builtin.unrealized_conversion_cast %7 : vector<64xi8> to vector<256xi8>
    %10 = llvm.bitcast %9 : vector<256xi8> to vector<64xi32>
    ave.hir.masked_store <PK4_B32> %subview_1[%c0], %8, %10 {functionType = #ave.func_dist_type<pk4>, hivm.is_continuous} : memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi32>
    %11 = arith.addi %arg3, %c64 : index
    %subview_3 = memref.subview %arg0[%11] [64] [1] : memref<4096xf32, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_4 = memref.subview %arg1[%11] [64] [1] : memref<4096xf16, #hivm.address_space<ub>> to memref<64xf16, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %subview_5 = memref.subview %arg2[%11] [64] [1] : memref<4096xi8, #hivm.address_space<ub>> to memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %res_6 = ave.hir.vload <NORM> %subview_3[%c0] {functionType = #ave.func_dist_type<norm>} : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xf32>
    %res_7 = ave.hir.vload <UNPK_B16> %subview_4[%c0] {functionType = #ave.func_dist_type<pk>} : memref<64xf16, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xf16>
    %12 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
    %13 = ave.hir.vextf %res_7, <part_even>, %12 {functionType = #ave.func_dist_type<even>} : vector<64xf16>, vector<64xf32>, vector<64xi1>
    %14 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
    %15 = ave.hir.vadd %res_6, %13, %14 : vector<64xf32>, vector<64xi1>
    %16 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
    %17 = ave.hir.vtruncf %15, <trunc>, false, <part_even>, %16 {functionType = #ave.func_dist_type<even>} : vector<64xf32>, vector<64xf16>, vector<64xi1>
    %18 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
    %19 = ave.hir.vfptosi %17, <trunc>, %18 {functionType = #ave.func_dist_type<even>, part = #ave.vcvt_part_type<part_even>, sat = true} : vector<64xf16>, vector<64xi1>, vector<64xi8>
    %20 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
    %21 = builtin.unrealized_conversion_cast %19 : vector<64xi8> to vector<256xi8>
    %22 = llvm.bitcast %21 : vector<256xi8> to vector<64xi32>
    ave.hir.masked_store <PK4_B32> %subview_5[%c0], %20, %22 {functionType = #ave.func_dist_type<pk4>, hivm.is_continuous} : memref<64xi8, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<64xi1>, vector<64xi32>
  }
  return
}

loc("/home/y00953696/1012sgl/sgl-kernel-npu/10shape_test/add_rmsnorm_bias_kernel.py":65:0): error: Failed to run BiShengSIMD pipeline

loc("/home/y00953696/1012sgl/sgl-kernel-npu/10shape_test/add_rmsnorm_bias_kernel.py":65:0): error: Failed to run BiShengSIMD pipeline

[ERROR] Failed to run BiShengIR regbase pipeline
