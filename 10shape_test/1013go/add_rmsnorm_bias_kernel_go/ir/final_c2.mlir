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

 #0 0x0000564a62c28f57 llvm::sys::PrintStackTrace(llvm::raw_ostream&, int) Signals.cpp:0:0
 #1 0x0000564a62c26b3e llvm::sys::RunSignalHandlers() Signals.cpp:0:0
 #2 0x0000564a62c295ef SignalHandler(int) Signals.cpp:0:0
 #3 0x00007f4c61ecf420 __restore_rt (/lib/x86_64-linux-gnu/libpthread.so.0+0x14420)
 #4 0x00007f4c6197b00b raise (/lib/x86_64-linux-gnu/libc.so.6+0x4300b)
 #5 0x00007f4c6195a859 abort (/lib/x86_64-linux-gnu/libc.so.6+0x22859)
 #6 0x00007f4c6195a729 (/lib/x86_64-linux-gnu/libc.so.6+0x22729)
 #7 0x00007f4c6196bfd6 (/lib/x86_64-linux-gnu/libc.so.6+0x33fd6)
 #8 0x0000564a5cf0d852 (anonymous namespace)::TreeSolve::getState(mlir::Value) AnalyzeVectorLayout.cpp:0:0
 #9 0x0000564a5ceefab6 (anonymous namespace)::TreeSolve::solveProblem(mlir::Operation*) AnalyzeVectorLayout.cpp:0:0
#10 0x0000564a5ceead2a (anonymous namespace)::AnalyzeVectorLayoutPass::solveIt() AnalyzeVectorLayout.cpp:0:0
#11 0x0000564a5cee6524 (anonymous namespace)::AnalyzeVectorLayoutPass::runOnOperation() AnalyzeVectorLayout.cpp:0:0
#12 0x0000564a61b84040 void llvm::function_ref<void ()>::callback_fn<mlir::detail::OpToOpPassAdaptor::run(mlir::Pass*, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int)::$_7>(long) Pass.cpp:0:0
#13 0x0000564a5c931277 std::_Function_handler<void (llvm::function_ref<void ()>, mlir::tracing::Action const&), bishengir::BiShengIRPassManager::initializeActionHandler(mlir::MLIRContext*)::$_0>::_M_invoke(std::_Any_data const&, llvm::function_ref<void ()>&&, mlir::tracing::Action const&) PassManager.cpp:0:0
#14 0x0000564a62b37ea8 mlir::MLIRContext::executeActionInternal(llvm::function_ref<void ()>, mlir::tracing::Action const&) MLIRContext.cpp:0:0
#15 0x0000564a61b84005 void mlir::MLIRContext::executeActionInternal<mlir::PassExecutionAction, mlir::Pass&>(llvm::function_ref<void ()>, llvm::ArrayRef<mlir::IRUnit>, mlir::Pass&) Pass.cpp:0:0
#16 0x0000564a61b7f6e0 mlir::detail::OpToOpPassAdaptor::run(mlir::Pass*, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int) Pass.cpp:0:0
#17 0x0000564a61b7f9c1 mlir::detail::OpToOpPassAdaptor::runPipeline(mlir::OpPassManager&, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int, mlir::PassInstrumentor*, mlir::PassInstrumentation::PipelineParentInfo const*) Pass.cpp:0:0
#18 0x0000564a61b84e72 mlir::detail::OpToOpPassAdaptor::runOnOperationAsyncImpl(bool)::$_15::operator()(mlir::detail::OpToOpPassAdaptor::runOnOperationAsyncImpl(bool)::OpPMInfo&) const Pass.cpp:0:0
#19 0x0000564a61b84f6f std::_Function_handler<void (), llvm::LogicalResult mlir::failableParallelForEach<__gnu_cxx::__normal_iterator<mlir::detail::OpToOpPassAdaptor::runOnOperationAsyncImpl(bool)::OpPMInfo*, std::vector<mlir::detail::OpToOpPassAdaptor::runOnOperationAsyncImpl(bool)::OpPMInfo, std::allocator<mlir::detail::OpToOpPassAdaptor::runOnOperationAsyncImpl(bool)::OpPMInfo>>>, mlir::detail::OpToOpPassAdaptor::runOnOperationAsyncImpl(bool)::$_15&>(mlir::MLIRContext*, __gnu_cxx::__normal_iterator<mlir::detail::OpToOpPassAdaptor::runOnOperationAsyncImpl(bool)::OpPMInfo*, std::vector<mlir::detail::OpToOpPassAdaptor::runOnOperationAsyncImpl(bool)::OpPMInfo, std::allocator<mlir::detail::OpToOpPassAdaptor::runOnOperationAsyncImpl(bool)::OpPMInfo>>>, __gnu_cxx::__normal_iterator<mlir::detail::OpToOpPassAdaptor::runOnOperationAsyncImpl(bool)::OpPMInfo*, std::vector<mlir::detail::OpToOpPassAdaptor::runOnOperationAsyncImpl(bool)::OpPMInfo, std::allocator<mlir::detail::OpToOpPassAdaptor::runOnOperationAsyncImpl(bool)::OpPMInfo>>>, mlir::detail::OpToOpPassAdaptor::runOnOperationAsyncImpl(bool)::$_15&)::'lambda'()>::_M_invoke(std::_Any_data const&) Pass.cpp:0:0
#20 0x0000564a61a76568 std::_Function_handler<std::unique_ptr<std::__future_base::_Result_base, std::__future_base::_Result_base::_Deleter> (), std::__future_base::_Task_setter<std::unique_ptr<std::__future_base::_Result<void>, std::__future_base::_Result_base::_Deleter>, std::thread::_Invoker<std::tuple<std::function<void ()>>>, void>>::_M_invoke(std::_Any_data const&) Inliner.cpp:0:0
#21 0x0000564a61a764c7 std::__future_base::_State_baseV2::_M_do_set(std::function<std::unique_ptr<std::__future_base::_Result_base, std::__future_base::_Result_base::_Deleter> ()>*, bool*) Inliner.cpp:0:0
#22 0x00007f4c61ecc4df __pthread_once_slow (/lib/x86_64-linux-gnu/libpthread.so.0+0x114df)
#23 0x0000564a61a7684b std::__future_base::_Deferred_state<std::thread::_Invoker<std::tuple<std::function<void ()>>>, void>::_M_complete_async() Inliner.cpp:0:0
#24 0x0000564a61a769d5 std::__basic_future<void>::wait() const Inliner.cpp:0:0
#25 0x0000564a62bf5e22 llvm::StdThreadPool::processTasks(llvm::ThreadPoolTaskGroup*) ThreadPool.cpp:0:0
#26 0x0000564a62bf7141 void* llvm::thread::ThreadProxy<std::tuple<llvm::StdThreadPool::grow(int)::$_0>>(void*) ThreadPool.cpp:0:0
#27 0x00007f4c61ec3609 start_thread (/lib/x86_64-linux-gnu/libpthread.so.0+0x8609)
#28 0x00007f4c61a57353 __clone (/lib/x86_64-linux-gnu/libc.so.6+0x11f353)
timeout: the monitored command dumped core
