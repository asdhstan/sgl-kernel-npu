# `_argmax_prob_kernel` 编译器侧优化（1013go）

工作目录：`/home/y00953696/1012sgl/sgl-kernel-npu/10shape_test/1013go/_argmax_prob_kernel_go`
编译器：`/home/y00953696/AscendNPU-IR`，分支 **`opt/argmax-prob-vector`**（从 `fix/autovectorize-v2-cross-block-fuse` 切出）
目标 kernel：`python/sgl_kernel_npu/sgl_kernel_npu/sample/argmax_softmax_prob.py::_argmax_prob_kernel`

---

## 1. 环境与构建（已验证）

`source /home/y00953696/set_env.sh` 后 `which bishengir-compile` =
`/home/y00953696/AscendNPU-IR/build/bin/bishengir-compile`（用户已改回，非 d00959235 那个）。

按 `/home/y00953696/0918/Q3TritonKernel/AGENTS.md` 的配方构建：

```bash
export PATH=/home/d00959235/projects/triton-ascend/toolchain/clang15/usr/bin:$PATH
export LD_LIBRARY_PATH=/home/d00959235/projects/triton-ascend/toolchain/clang15/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export BISHENG_INSTALL_PATH=/home/d00959235/projects/cann-9.2.0-beta.2/install/cann-9.2.0-beta.2/bin
cd /home/y00953696/AscendNPU-IR
./build-tools/build.sh --c-compiler clang-15 --cxx-compiler clang++-15 \
  '--add-cmake-options=-DLLVM_ENABLE_LLD=ON' --build-type Release -j 96 --enable-assertion \
  --disable-werror --disable-mlir-werror --disable-bishengir-werror \
  --build-triton -t --bisheng-compiler ${BISHENG_INSTALL_PATH} --build ./build
```

**实测：构建目录是增量的，未改代码时整条命令 5 秒返回（`Build Done!!!`，全部 `Up-to-date`）**，
所以「改一个 pass → 重编 → 测」的迭代成本很低。
日志：`/tmp/10shape_logs/build_baseline.log`；脚本：`/tmp/build_bishengir.sh`。

## 2. 基线（autotune 版，avg_time）

数据见 [baseline_pipes.md](baseline_pipes.md)（取自 `result_dir/argmax_prob_*/ASCEND_PROFILER_OUTPUT/kernel_details.csv`）。

| shape | dur(us) | aiv(us) | vec(us) | **vec%** | mte2(load) | scalar | blocks |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1024_V128000 | 538.75 | 538.48 | 428.24 | **79.5%** | 89.67 | 16.10 | 56 |
| B512_V157184 | 343.59 | 343.32 | 277.81 | **80.9%** | 51.84 | 10.76 | 56 |
| B128_V157184 | 98.85 | 98.58 | 84.03 | **85.2%** | 8.69 | 4.39 | 56 |
| B32_V256000 | 55.76 | 55.50 | 46.40 | 83.6% | 5.07 | 2.36 | 32 |

**结论：vector pipe 打满（≈80%），mte2 只占 ~17%，不是带宽受限也不是并行度问题**（block 数已经是 56 = 全部 vector core）。
所以优化方向只有一个：**减少每个元素的 vector 指令数**。

## 3. IR 分析

`ir/` 下：`_argmax_prob_kernel.bcmlir`（bishengir 输入）、`.ttir`（Triton IR）、
`all_ir.mlir`（`--mlir-print-ir-after-all` 全量，97090 行 / 1465 个 pass dump）、`final.mlir`（最后一段）。

复现：

```bash
cd 1013go/_argmax_prob_kernel_go
TRITON_CACHE_DIR=$PWD/tmp_cache python scripts/dump_ir.py     # 只跑 native 一种配置，得到 .bcmlir
cd ir && /home/y00953696/AscendNPU-IR/build/bin/bishengir-compile _argmax_prob_kernel.bcmlir \
  --target=Ascend950PR_9579 --enable-auto-multi-buffer=True --enable-auto-bind-sub-block=True \
  --disable-ffts --limit-auto-multi-buffer-of-local-buffer=no-limit --enable-auto-blockify-loop \
  --enable-hfusion-compile=true --enable-triton-kernel-compile=true --enable-lib-call-no-inline=false \
  --append-bisheng-options="-cce-link-aicore-ll-module .../libdevice.10.bc" \
  --enable-vf-merge-level=1 -o kernel --mlir-print-ir-after-all > all_ir.mlir 2>&1
```

### 3.1 向量宽度已到顶（排除"加宽向量"这条路）

最终 IR 用的是 `vldsx1.v64f32` / `vexpdif` / `vstsx1.v64f32`，全部是 `vector<64xf32>`。
ISA 声明（`include/bishengir/Dialect/HIVMRegbaseIntrins/IR/HIVMRegbaseIntrins.td`）里 load 只有：
`v64f32 / v128f16 / v64i32 / v32i64 / v256i8 …` —— **没有 `v128f32`**。64×f32 = 256 B = 向量寄存器宽度，已经拉满。
`exp` 也已经被融成 `hivm.vexpdif`（直接算 `exp(x - m)`），没有多余步骤。

### 3.2 每个 V-tile（16384 宽）的指令清单（来自 VectorizeOps 之后的 tensor 级 IR）

```
memref.copy(GM→UB) + scf.if{fill -inf}      # other=-inf 的掩码处理：只在"尾块"才 fill，很便宜
cast  f16 → f32                              # 1 条
--- NaN 归一化链（见 3.3）---                 # 8 条全宽 vector op  ← 主要靶子
linalg.reduce  max      over 归一化结果       # 全宽 reduce #1
sub + exp                                    # 2 条全宽（exp 已是 vexpdif）
linalg.reduce  sum      over exp             # 全宽 reduce #2
hfusion.reduce_with_index max over 归一化结果 # 全宽 reduce #3
其余是 1 宽（scalar）的同类运算
```

即每个 tile 有 **3 次全宽 reduce + ~8 条全宽 elementwise**，其中 8 条里有 8 条是为 NaN 掩码服务的。

### 3.3 找到的靶子：`is_nan` 的降级用了 8 条 vector op（可以用 4 条）

链路来源（已逐层定位）：

1. `lib/Dialect/HFusion/Transforms/regbase/Normalize/NormalizeMath.cpp:372` `HFusionNormalizeMaxNumFTraits`
   + `NormalizeMathTemplate.h:1113 NormalizeReduceMinMaxNumFOpTemplate`：
   把 `reduce { maxnumf } src` 改写成 `reduce { maximumf } (maskNanWithInf(src))`
   （因为硬件 `vmax` 是 NaN-propagating，而 Triton 的 `tl.max` 是 maxnum 语义）。
2. `NormalizeMathTemplate.h:1088/1129` → `Utils/MathTemplateHelpers.h:32 maskNanWithInf`
   → 产生 `hfusion.is_nan(src)` + `hfusion.select(isnan, -inf, src)`（1 条 select）。
3. `NormalizeComparisonTemplate.h:473 NormalizeIsNanOpTemplate` 把 `is_nan` 降级成：

```
maskedSign = bitcast(x) & 0x7FFFFFFF          # (1) bitcast  (2) vand
t          = maskedSign - 0x7F800000          # (3) vadds
t          = min_signed(t, 1)                 # (4) vmins
t          = max_signed(t, 0)                 # (5) vmaxs
result     = lowerIntMaskToBoolResult(t)      # (6) cast i32→f32  (7) vcmp.ne  ... 最终 IR 里是 vcmp.ne.s.z
```

半整数语义**是对的**（`|x|=inf` → t=0 → false；只有 NaN → t≥1 → true），
但代价是 **8 条全宽 vector op**（bitcast/vand/vadds/vmins/vmaxs/cast/vcmp.ne + select）。

**可用的更短等价写法**：`isnan(x) ⟺ (bits(x) & 0x7FFFFFFF) > 0x7F800000`（无符号）。

```
maskedSign = bitcast(x) & 0x7FFFFFFF
result     = vcmps.gt.u(maskedSign, 0x7F800000)     # 与标量比较
```

优化前：用"减法 + 夹到 0/1 + 判非零"来间接得到大于比较

代码块


y    = bits(x) & 0x7FFFFFFF      // 清符号位，得到 |x| 的位模式
t    = y - 0x7F800000            // 减去 +inf 的位模式
t    = min_signed(t, 1)          // 夹到 [0,1]
t    = max_signed(t, 0)
cond = ((float)t != 0.0)         // 非零 ⇒ NaN
优化后：直接做那一次大于比较

代码块


y    = bits(x) & 0x7FFFFFFF      // 清符号位，得到 |x| 的位模式
cond = y > 0x7F800000            // 一次整数"大于"比较 ⇒ NaN

ISA 里 `hivm.vcmps.gt.u.z` / `hivm.vcmp.gt.u.z` 都存在，而现有链条最后本来也是
`vcmp.ne.s.z → vsel`，说明"compare 产生 mask → select"这条通路是通的。

**预期：8 条 → 4 条**（bitcast/vand/vcmps/vsel），并且：
- 该 chain 每个 tile 都作用在**整个 reduce 输入**上，是 `vec` pipe 里占比最大的一块可裁剪指令；
- 同一份 `is_nan` 降级还被 `NormalizeIsInfOpTemplate`、`maskNanWithInf`（min/max、minnum/maxnum 归约）共用，
  **所有走 maxnumf/minnumf 归约的 kernel 都会一起受益**。

## 4. 候选优化清单（按优先级）

| # | 候选 | 位置 | 预期 | 风险 |
|---|---|---|---|---|
| **C1** | `is_nan` 降级用 `vcmps.gt.u(|x|, inf)` 替代 sub+clamp+cast+ne：8 op → 4 op | `NormalizeComparisonTemplate.h:473` `NormalizeIsNanOpTemplate`（及 HFusion 层对应实现 `Normalize.cpp` 的 `maskSignBit`/`minusInfConstValue` 使用处） | 该 kernel 每 tile 少 4 条全宽 op；vec 占 80% → 理论 **~15–25%** 端到端 | 中：影响面广，需要跑其它 kernel 回归 |
| **C2** | 把同一输入上的 `reduce(max)` 与 `reduce_with_index(max)` 融合（后者已给出同样的值，只是当前没人用） | `HFusionOps.cpp` 新增 canonicalize pattern | 每 tile 少 1 次全宽 reduce | 中 | ✅ **有效** |
| C3 | `RemoveMaskFromUnalignedReductionLoop` 在本 kernel 上是否可命中（尾块才需要 mask） | 已有 pass | 未知 | 低 |

## 5. 尝试记录

| # | 方案 | 改动 | 结果（avg_time sum，baseline 1173.43us） | 结论 |
|---|---|---|---|---|
| — | 基线（未改编译器） | — | 1173.43 | — |
| **C1** | `is_nan` 降级改成一次整数 GT 比较：`(|x|) > bits(+inf)` | `NormalizeComparisonTemplate.h` +12/−19 行 | **1030.55（−12.2%）** | ✅ **有效** |

### C1 结果明细（3 次运行，取每 shape 最小值）

| shape | baseline | patched | Δ |
|---|---:|---:|---:|
| B1024_V128000 | 545.65 | 481.70 | **−11.7%** |
| B512_V157184 | 346.83 | 301.74 | **−13.0%** |
| B128_V157184 | 99.47 | 86.95 | −12.6% |
| B32_V256000 | 55.77 | 48.97 | −12.2% |
| B32_V157184 | 36.49 | 32.22 | −11.7% |
| B8_V157184 | 34.61 | 30.17 | −12.8% |
| B1_V157184 | 33.10 | 28.82 | −12.9% |
| B32_V32000 | 11.13 | 9.87 | −11.4% |
| B32_V4096 | 5.45 | 5.15 | −5.4% |
| B32_V1024 | 4.92 | 4.95 | +0.6% |
| **SUM** | **1173.43** | **1030.55** | **−12.2%** |

3 次运行 sum = 1031.58 / 1031.90 / 1034.48（非常稳定，非噪声）。原始数据 `patched_measure.json`。

**IR 侧验证**（`ir_new/final_new.mlir` vs `ir/final.mlir`，最终 LLVM 级）：
`vmins.s.x` 3→0、`vmaxs.s.x` 3→0、`vadds.s.x` 3→0、`vcmp.ne.s.z` 3→0，
替换为 `vcmp.gt.s.z` 3 条；mask 生成指令 `pge.b32` 25→22，最终 IR 537→525 行。
即每个 NaN 掩码点从 6 条 vector op 降到 2 条（`vand` + `vcmp.gt`）。

**正确性回归**（全部用同一个 fresh Triton cache 重编，确保走新编译器）：

| 测试 | 结果 |
|---|---|
| `argmax_prob_kernel.py` | 26 passed |
| `argmax_prob_kernel_ori.py` | 1 passed |
| `fused_rope_qk_mqa_kernel_opt.py` | 31 passed, 1 skipped |
| `chain_rejection_sample_kernel.py` | 24 passed |
| `test_mix_fused_kernel.py` | 24 passed |

### C1 的 diff 位置

`bishengir/include/bishengir/Transforms/regbase/Normalize/NormalizeComparisonTemplate.h`
（`NormalizeIsNanOpTemplate::matchAndRewrite`，约 495–514 行）：把
`-inf_bits 减法 → min_signed(1) → max_signed(0) → cast → compare(!=0)`
换成 `Traits::createCmpOp(rewriter, loc, maskedSignValue, bits(+inf), CompareKind::GT)`。

## 6. 备注

- 实测过程中一度出现 **`set_env.sh` 被改成 `d00959235/.../bishengir-compile`**（10:41），
  导致 `split_qkv_rmsnorm_rope` 等出现编译失败；用户已改回 `/home/y00953696/AscendNPU-IR/build/bin`。
- 共享卡干扰很大：单次 `avg_time` 会偶发 2×–5× 离群，对比一律用**多次运行取每 shape 最小值**。


---

## 8. C1 的影响面（回归检查）

用同一个 patched 编译器跑其它 kernel 的 performance（各 1 次）：

| kernel | baseline | patched | Δ |
|---|---:|---:|---:|
| `_mix_fused_kernel` | 192.23 | 193.60 | +0.7%（噪声内） |
| `_chain_rejection_block_sum_kernel` | 263.98 | 263.20 | −0.3%（噪声内） |

即 **C1 的收益集中在 argmax**（它是唯一在大 tile 上做 maxnumf 归约的 kernel），对其它 kernel 基本中性。

## 9. C2 候选（已定位，待实现）

C1 之后每个 V-tile 仍然是 vector-bound（vec 占 76–84%），剩下的全宽操作是
`cast f16→f32` → NaN 掩码（`bitcast/vand/vcmp/vsel`）→ `sub` → `exp` → **3 次全宽归约**。

其中 **2 次归约是重复的**（证据，`ir_new/all_ir_new.mlir` 的 `VectorizeOps` dump，行 8596 起）：

```
%reduced = linalg.reduce ins(%25 : tensor<16384xf32>) outs(%19 : tensor<f32>)   // max over %25，init = fill(-inf)
%extracted = tensor.extract %reduced[]                                          // → chunk_max，被使用
...
%48:2 = hfusion.reduce_with_index {already_initialize_init, tie_break_left} <max>
        ins(%25, %20 : tensor<16384xf32>, tensor<16384xi32>) outs(%46, %47)     // 同一个 %25！
%extracted_13 = tensor.extract %48#1[]                                          // 只用 index
```

- `%48#0`（reduce_with_index 的**值**结果）在整个函数里**出现 0 次** —— 值被算出来但丢掉；
- 两个 op 的源都是 `%25`、维度都是 `[0]`、都是 `max`；`%19` 是 `fill(-inf)`，而 `reduce_with_index`
  带 `already_initialize_init`（init 就是 max 的单位元）→ **两者结果等价**。

**做法**：在 HFusion normalize pass（`hfusion-normalize-ops`，dump 里在 VectorizeOps 之前，行 7514）
加一个 pattern（root 在 `linalg::ReduceOp`）：若
1. combiner 是 `arith.maximumf`/`maxnumf`（或 min 对应），
2. 它的 init 是单位元（`linalg.fill` 常量 −inf / +inf），
3. 它的源同时也是某个 `hfusion.reduce_with_index` 的源（kind/dimensions 一致），

则把所有用到该 reduce 结果的地方换成 `reduce_with_index` 的 `result#0`，并 erase 掉这个 plain reduce。
两个 op 从 `hfusion-normalize-ops`（行 7514）起就一直并存，所以该 pass 是合适的落点。

预期：每 tile 少一次全宽归约（3 → 2），按归约占 vec 时间的比例估计 **~8–12%** 端到端。


---

## 10. C2 结果（已实现并验证）

| | baseline | C1 | **C1+C2** |
|---|---:|---:|---:|
| 10 shapes 合计 avg_time | 1173.43 | 1030.55 (−12.2%) | **860.67（−26.7%）** |

3 次运行 sum = 861.94 / 862.74 / 866.14（稳定）。C2 在 C1 基础上再 **−16.5%**。

| shape | baseline | C1+C2 | Δ |
|---|---:|---:|---:|
| B1024_V128000 | 545.65 | 403.27 | **−26.1%** |
| B512_V157184 | 346.83 | 251.79 | **−27.4%** |
| B128_V157184 | 99.47 | 70.95 | −28.7% |
| B32_V256000 | 55.77 | 40.55 | −27.3% |
| B32_V157184 | 36.49 | 26.98 | −26.0% |
| B8_V157184 | 34.61 | 24.97 | −27.8% |
| B1_V157184 | 33.10 | 23.61 | −28.7% |
| B32_V32000 | 11.13 | 8.80 | −21.0% |
| B32_V4096 | 5.45 | 4.93 | −9.4% |
| B32_V1024 | 4.92 | 4.82 | −2.1% |

### 改动

`bishengir/lib/Dialect/HFusion/IR/HFusionOps.cpp`：新增 canonicalize pattern
`FoldPlainReduceIntoReduceWithIndex`，注册在 `ReduceWithIndexOp::getCanonicalizationPatterns`
（与已有的 `ReplaceWithLinalgReduce` 并列 —— 后者处理"index 没人用 → 退化成 plain reduce"，
本 pattern 处理互补的情形"value 已经有人算了 → 用它的 value，删掉重复的 plain reduce"）。

触发条件（全部满足才动手，保守）：
1. `reduce_with_index` 有 tensor 语义、≥2 个结果、带 `already_initialize_init`（init 即归约单位元）；
2. 源的某个 user 是 `linalg.reduce`，且 1 input / 1 init / 1 result；
3. `dimensions` 相同、结果类型相同、**combiner 名相同**（`arith.maximumf` ↔ `arith.maximumf`）；
4. 该 plain reduce 的 init 是 `linalg.fill` 填的**单位元常量**（max → −inf，min → +inf）。

### 实现中踩的两个坑（都已在 IR 上验证修好）

1. **第一次实现产生非法 IR**：`error: operand #0 does not dominate this use`
   —— Triton 先发射 plain reduce、后发射 `reduce_with_index`，直接替换用途会让后面的值支配前面的使用。
   修法：把 `reduce_with_index` **上提**到 plain reduce 之前。
2. **上提被自己的前置条件挡住**：`reduce_with_index` 的 `outs` 是紧挨着它创建的 `tensor.empty()`，
   位在 plain reduce **之后**，所以"所有操作数都已可用"不成立。
   修法：把这些纯操作（`isMemoryEffectFree`，典型就是 `tensor.empty`）一起上提。
   （用 `llvm::errs()` 打点定位，确认 6 个候选全部数据条件通过、但卡在上提；修好后 0 个 "hoist blocked"。）

### IR 侧验证

`VectorizeOps` 之前每个 V-tile 有 **3 次全宽归约**（max / sum / max-with-index），
现在 plain max `linalg.reduce` 消失，只剩 `reduce_with_index`（被上提到原位置）+ sum：

```
%27:2 = hfusion.reduce_with_index {already_initialize_init, ...} <max> ins(%24, %19 ...)   ← 上提后
%reduced = linalg.reduce ins(%44 : tensor<16384xf32>) ... { arith.addf }                    ← 只剩 sum
```

### 正确性回归（同一个 fresh Triton cache，走 C1+C2 编译器）

| 测试 | 结果 |
|---|---|
| `argmax_prob_kernel.py` | 26 passed |
| `argmax_prob_kernel_ori.py` | 1 passed |
| `fused_rope_qk_mqa_kernel_opt.py` | 32 passed |
| `chain_rejection_sample_kernel.py` | 24 passed |
| `chain_rejection_block_sum_kernel.py` | 23 passed |
| `test_mix_fused_kernel.py` | 24 passed |
| `split_qkv_rmsnorm_rope_kernel.py` | 37 passed |
| `split_qkvgate_gemma_rmsnorm_rope_kernel.py` | 8 passed, 24 skipped |

### 影响面

| kernel | baseline | C1+C2 | Δ |
|---|---:|---:|---:|
| `_mix_fused_kernel` | 192.23 | 193.60 | +0.7% |
| `_chain_rejection_sample_kernel` | 49.22 | 48.97 | −0.5% |
| `_chain_rejection_block_sum_kernel` | 263.98 | 263.03 | −0.4% |

两个改动叠加后对其它 kernel 仍是中性，收益集中在 argmax。


---

## 11. 第三轮：全量回归 + 多缓冲实验（尝试记录）

### 11.1 C1+C2 之后的瓶颈（profiler）

| shape | dur | vec | **vec%** | mte2 | scalar |
|---|---:|---:|---:|---:|---:|
| B1024_V128000 | 403.32 | 291.45 | **72.3%** | 94.48 | 15.45 |
| B512_V157184 | 250.77 | 187.88 | 75.0% | 51.30 | 10.27 |
| B128_V157184 | 71.22 | 57.19 | 80.6% | 8.54 | 4.22 |
| B32_V256000 | 40.63 | 31.86 | 78.9% | 4.92 | 2.23 |

仍然是 vector-bound（72–82%）。注意 **vec + mte2 + scalar ≈ dur**，即各 pipe 的忙碌时间**几乎完全相加**、没有重叠。

### 11.2 每个 V-tile（16384）剩下来的全宽操作

```
cast f16→f32                     1 条
NaN 掩码: bitcast/vand/vcmp/select  4 条   ← 仍占 elementwise 的一半
sub + exp (已融合成 vexpdif)      1 条
reduce_with_index <max>          1 次全宽归约
linalg.reduce {addf} (sum)       1 次全宽归约
```
另有 2 条 1 宽的重复 NaN 掩码链（element 数极小，可忽略）。

### 11.3 尝试：关掉 auto multi-buffering（无效）

因为 vec 与 mte2 的时间完全相加，怀疑 tile 的 GM→UB 拷贝没有和计算重叠。
用 pytest 插件（`/tmp/nomb_plugin.py`，不改算子文件）给 `_argmax_prob_kernel` 的每次发射注入
`multibuffer=False`，用全新 Triton cache 重编后测量：

| 配置 | 10 shapes 合计 |
|---|---:|
| `multibuffer=True`（默认，C1+C2） | 860.67 |
| `multibuffer=False` | **861.95（+0.1%）** |

**结论：多缓冲对这个 kernel 完全没有影响** → "vec/mte2 不重叠"不是多缓冲开关能解决的，
这条线（靠开关拿重叠收益）**判为无效，放弃**。

### 11.4 全量回归（C1+C2 编译器，同一个 fresh Triton cache）

10 个 kernel 全部跑通，**0 failed**：

| 测试 | 结果 |
|---|---|
| `argmax_prob_kernel.py` | 26 passed |
| `causal_conv1d_update_kernel_npu_tiled.py` | 39 passed |
| `store_kv_cache_prefix_valid_npu_kernel.py` | 24 passed |
| `add_rmsnorm_bias_kernel.py` | 32 passed |
| `fused_rope_qk_mqa_kernel_opt.py` | 32 passed |
| `split_qkv_rmsnorm_rope_kernel.py` | 37 passed |
| `split_qkvgate_gemma_rmsnorm_rope_kernel.py` | 8 passed, 24 skipped |
| `test_mix_fused_kernel.py` | 24 passed |
| `chain_rejection_block_sum_kernel.py` | 23 passed |
| `chain_rejection_sample_kernel.py` | 24 passed |
| **合计** | **269 passed / 24 skipped / 0 failed** |

### 11.5 下一个候选（C3，未实现）

**把 NaN 掩码下推到 f16→f32 cast 之前。** 依据：f16 的向量指令是 128 lane（`v128f16`），
f32 是 64 lane（`v64f32`），同样 4 条 op 在 f16 域上只要一半的指令数；而
`isnan(cast_f32(x16)) ⟺ isnan_f16(x16)`、`cast(-inf16) == -inf32`，所以语义等价：

```
现在:  cast(f16→f32) → bitcast → vand(0x7FFFFFFF) → vcmp.gt(0x7F800000) → select(-inf32)
改成:  bitcast(f16→i16) → vand(0x7FFF) → vcmp.gt(0x7C00) → select(-inf16) → cast(f16→f32)
```
按 f32-op 当量算：4 条 → 2 条 + 1 条额外 cast（因为原 cast 还有别的用户）= 净省 1 条约 11% 的 vec 时间（约 8% 端到端）。
落点：`NormalizeMathTemplate.h` 的 `maskNanWithInf`（`hfusion`/`hivm` 两套 Traits）或
`NormalizeReduceMinMaxNumFOpTemplate` 里，识别 `src` 由 f16/bf16 → f32 的 widening cast 定义。


---

## 12. 第四轮：C3（NaN 掩码下推到 f16 域）→ **失败，已回退**

### 12.1 想法与依据

每个 tile 里 NaN 掩码占 4 条全宽 op（bitcast/vand/vcmp/vsel），是 elementwise 的一半。
ISA 里 f16/i16 的向量宽度是 **128 lane**（`vldsx1.v128f16` / `vldsx1.v128s16`），而 f32 只有 **64 lane**（`vldsx1.v64f32`），
所以把掩码挪到 f16 域理论上只要一半指令。语义上严格等价：

```
isnan(cast_f32(x16)) == isnan_f16(x16)      # f16→f32 是精确、保序、保 NaN 的
cast_f32(-inf16)     == -inf32
```

落点：`MathTemplateHelpers.h` 的 `maskNanWithInf`，识别 `src` 由 16bit→f32 的 widening 转换产生时，
改为在窄类型上做 `isnan + select`，再 `clone` 原转换把结果抬回 f32（原转换保留给其它用户）。

### 12.2 IR 侧确认生效

VectorizeOps 之后的 IR 确实变成了窄类型（这正是想要的形态）：

```
%21 = hfusion.bitcast ins(%15 : tensor<16384xf16>) -> tensor<16384xi16>
%22 = hfusion.elemwise_binary {vand} ins(%21, 32767_i16)
%24 = hfusion.compare {vgt} ins(%22, 31744_i16)        # 0x7C00 = f16 +inf
%26 = hfusion.select ins(%24, -inf_f16, %15) -> tensor<16384xf16>
%27 = hfusion.cast ins(%26 : tensor<16384xf16>) -> tensor<16384xf32>   # 新增的 cast
%17 = hfusion.cast ins(%15 : tensor<16384xf16>) -> tensor<16384xf32>   # 原 cast 保留
```

### 12.3 但性能 **大幅倒退**

| 配置 | 10 shapes 合计 |
|---|---:|
| C1+C2 | 860.67 µs |
| C1+C2+**C3** | **1099.55 µs（+27.8%）** |
| 回退 C3 后 | 859.98 µs（恢复） |

### 12.4 根因：后端没有真的用 128 宽的 i16 算术，反而膨胀了

最终 LLVM 级 IR 的指令直方图对比：

| intrinsic | C1+C2 | C1+C2+C3 |
|---|---:|---:|
| `pge.b32`（谓词/mask 建立） | 19 | **33** |
| `vand.x` | 3 | 5 |
| `vcmp.gt.s.z` | 3 | 5 |
| `vsel` | 3 | 5 |
| `vcvtff.f162f32.x`（f16→f32 转换） | 2 | **4** |
| `vdups.z` | 7 | 10 |
| 最终 IR 行数 | 495 | **567** |

也就是说：**掩码 op 数不降反升**，谓词建立（`pge.b32`）几乎翻倍，还多了一整套 f16→f32 转换。
ISA 里虽然存在 128 宽的 f16/i16 **搬运**形式，但这一层的窄类型**算术/比较/mask**并没有被后端按 128 宽执行，
而是被 legalize/展开成更多指令。

### 12.5 结论

**判为无效（负向 27.8%），已 `git checkout` 回退该文件**；编译器改动仍然只有 C1+C2 两个文件。
负向结论本身有价值：**不能假设"窄类型 = 半价"** —— 在这条路径上必须以最终 IR 的指令数（尤其 `pge.b32`）为准，
而不是以 ISA 表里的向量宽度为准。

（记录：C3 的中间产物保留在 `ir_c3/`，其中 `all_ir_c3.mlir` 是失败版本的完整 IR。）


---

## 13. 第五轮：把每个 tile 的剩余开销逐条拆开（定位到最大的一块）

C1+C2 之后 `vec` 仍是 72–82% 且是主要项。这一轮把最终 LLVM IR 的内层循环逐条拆开。

### 13.1 每个 V-tile 到底调用了什么

内层循环（`^bb3..^bb10`）里正好 **7 个 `llvm.call`**：

| 调用 | 内容 |
|---|---|
| `load_gm_to_ubuf_1d_half` | tile 的 GM→UB 拷贝（mte2，94us） |
| `reduce_max_with_index_left_r_float` | max+argmax |
| `_argmax_prob_kernel_fused_1/2/3_outlined_vf_0` | 融合后的 elementwise |
| `_argmax_prob_kernel_outlined_vf_0/1` | f16→f32 cast + 掩码的一部分 |

### 13.2 max+argmax 是**一条裁剪过的硬件归约**（很便宜）

模板库 `ReduceR_With_Index.cpp::reduce_r_with_index_on_vector`：

```cpp
INTRINSIC(set_vector_mask, 0x0, vector_element_num);   // 掩码覆盖整个 16384
reduce_intrin<OP, T, false, Order_t::VALUE_INDEX>(...1 /*repeat*/ ...);
*(dst_value_ptr) = GET_MAX_MIN_CNT();
*(dst_index_ptr) = GET_MAX_MIN_CNT() >> 32;
```

即 **count 模式下一条 `reduce_intrin` 就把 16384 个元素的最大值和索引都算出来了** ✓
（这也解释了为什么 C2 去掉重复的那次 max 归约收益那么大：留下的这个本来就是一条指令级的东西）。

### 13.3 **sum + exp 才是最大的那块**：被降成一个 64 元素/次的串行循环

`fused_3` 的实际机械码结构：

```llvm
^bb1(%i, %acc):  cond_br (%i < 16384)                       // 256 次迭代
^bb2:
  %13 = vldsx1.v64f32(src + i)        // 载入 64 个 f32
  %14 = pge.b32(0,0)                  // 谓词建立
  %15 = vexpdif(%13, %7)              // exp(x - m)     ← exp 被融进这个循环
  %16 = pge.b32(0,0)                  // 谓词建立
  %17 = vcadd.s.x(%15)                // 64 lane 横向求和
  %18 = pge.b32(0,0)                  // 谓词建立
  %19 = vadd.s.x(%9, %17)             // 串行累加（循环携带依赖）
  %20 = add %i, 64
  br ^bb1(%20, %19)
```

每个 tile = **256 次迭代 × 约 9 条指令 ≈ 2300 条指令**，而且：
- 每次迭代有 **3 条循环不变的 `pge.b32`**（本该提到循环外）；
- 累加器 `%19 = vadd(%9, %17)` 是**串行依赖**，没有多累加器/展开来暴露并行度；
- 与 exp 融合虽然省掉了 exp 中间的 UB 往返，但把整块代价绑在了这个串行循环上。

### 13.4 同一个 kernel 里 max 走的是快路径，sum 走的却是慢路径

`ReduceR.cpp::reduce_r_core` 对两类归约的处置完全不同：

- `REDUCE_MAX/MIN`：直接用 per-repeat 内建（`size0 > num_per_repeat` 时先分行再合并）→ 快；
- `REDUCE_SUM`：走 `reduce_ar<OP,T>(...)`，即 `ReduceARImpl.h::reduce_ar_core`，
  其中 `num_per_repeat < size1` 的分支应该用 **`reduceARxToARyByDichotomy`（二分树归约）**：
  每轮把长度减半，总工作量 ≈ 2×数据量（约 512 条向量指令/tile），
  比现在这个 64 元素/次的串行循环（约 2300 条）**少 3–4 倍**。

（对照：`hivm.hir.vreduce` 在 IR 里出现 268 次，最终却落成串行循环；而 max 那条从模板库直接出内建。）

### 13.5 结论 / 下一个候选（C4）

**C4：让 sum 归约走二分树（或至少：把 3 条 `pge` 提出循环 + 多累加器展开）。**
预期收益：这一块约占 vec 时间的一大半，若能降到 1/3，端到端有 **~20–30%** 的空间。

落点候选（按优先级）：
1. `lib/Template/include/Vector/Reduction/ReduceARImpl.h::reduce_ar_core`（`num_per_repeat < size1` 分支）
   —— 但模板会编进 `meta_op.aiv.c310.bc`，影响面覆盖所有 sum 归约，需谨慎；
2. 编译器侧 `hivm.hir.vreduce` 的分解逻辑（哪一步把二分树变成了 64/次的串行循环）。

**注意**：这一轮只做了定位，没有改代码 —— 因为 C3 的教训是"必须先看最终 IR 的指令数再下结论"，
而 sum 归约的改动影响面很大，动手前需要先确认分解到串行循环的确切位置。


---

## 14. 第六轮：把 sum 串行循环的产生位置钉死

上一轮只知道"sum+exp 是个 64 元素/次的串行循环"。这一轮逐 pass 回溯，定位到**确切的产生者**。

### 14.1 定位过程（按 IR pass 顺序倒推）

| 观察 | 结论 |
|---|---|
| 最终 IR 里 `vcadd` 首次出现在 `ConvertHIVMAVEToAVEIntrin`（行 75103）之后 | 循环结构在这之前就有了 |
| 再往前：`ExpandStridedMetadata`（74600）时已是 `ave.hir.*` + 循环 | 循环更早 |
| `ave.hir.*` 首次出现在 `ConvertVectorToHIVMAVE`（64186）之后 | 再往前 |
| 上一 pass `InjectIR`（63890）里**已经是 `scf.for %arg3 = %c0 to %c16384 step %c64`** | 循环在 `InjectIR` 之前就产生了 |
| 全局搜 `scf.for ... %c16384 step %c64`（归约用）**首次出现在行 10785**，位于 pass **`AutoVectorizeV2 (hfusion-auto-vectorize-v2)`**（行 10701） | ✅ **就是它** |

### 14.2 AutoVectorizeV2 生成的形态

```
%1 = scf.for %arg5 = %c0 to %c16384 step %c64 iter_args(%arg6 = %0) -> (tensor<f32>) {
       %6  = ...load 64 个 f32...
       %7  = math.exp %6 : vector<64xf32>
       ...
       %11 = arith.addf %10, %7 : vector<64xf32>      # 64 宽累加器，串行迭代
     }
```

即：**vectorizer 把 16384 的 sum 归约展开成"每 64 个元素一步、64 宽累加器"的串行循环（256 次迭代，无展开）**，
累加器是循环携带依赖。之后 `ConvertVectorToHIVMAVE` / 内建形成阶段又在循环体里加了
**每条迭代 3 个循环不变的 `pge.b32`（谓词建立）**，并把每步的向量累加落成 `vcadd + vadd`。

### 14.3 两个可下手的具体子问题（都有 IR 证据）

| 子问题 | 证据 | 位置 |
|---|---|---|
| **① 循环无展开、单一累加器**，256 次迭代全靠一条循环携带依赖串起来 | `InjectIR` 里的 `scf.for ... step %c64 iter_args(%arg6)` + `arith.addf` | **`AutoVectorizeV2`**（`hfusion-auto-vectorize-v2`）—— 正是本仓库当前分支 `fix/autovectorize-v2-cross-block-fuse` 关注的 pass |
| **② 每次迭代 3 条循环不变的 `pge.b32`** | 最终 IR `fused_3`：`%14/%16/%18 = pge.b32(0,0)`，且 `AVE_VFPgeOp` 本来就标了 `[Pure]` | AVE/内建形成阶段（`ConvertVectorToHIVMAVE` 之后）；理论上 LICM 可提出循环，但该阶段之后似乎没有再跑 |

### 14.4 对照：max+argmax 走的是完全不同的快路径

`reduce_max_with_index_left_r_float` 在模板库里是 `set_vector_mask(0, N)` + `reduce_intrin(repeat=1, counting mode)`
+ `GET_MAX_MIN_CNT()` —— **一条指令搞定整个 16384**。
即同一个 kernel 里，max 是"一条内建"，sum 是"256 次迭代的串行循环"，两者代价差一个数量级。

### 14.5 结论

C4 的靶子从"sum 归约"精确到 **`AutoVectorizeV2` 生成的 64 宽串行累加循环 + 其上的 3 条冗余 `pge`**。
两个子问题都能在不改变数值语义的前提下优化（展开+多累加器；把纯谓词提出循环），
预期这一块（约占 vec 时间的一半）有 **~2× 的压缩空间**，即端到端 **~15–25%**。

**本轮仍只做定位、未改代码**：`AutoVectorizeV2` 是核心 pass，改动需要先构造一个只跑该 pass 的
bishengir-opt 复现用例来快速迭代（而不是每次都走完整 Triton→编译链路），这是下一步的第一件事。


---

## 15. 第六轮补充：C4 的快速迭代环境 + 一次状态复核

### 15.1 AutoVectorizeV2 单 pass 复现（已基本搭好）

`bishengir-opt` 支持单独跑这个 pass：

```
bishengir-opt <in.mlir> --hfusion-auto-vectorize-v2 -o <out.mlir>
```

输入 IR 从全量 dump 里截取（`ir_c2/all_ir_c2b.mlir` 中 `module` 起始行 10389 → AV2 banner 前一行 10700），
已存为 `/tmp/av2_mod.mlir`（312 行）。目前还差一个小问题：`#map = affine_map<...>` 别名定义在截取范围之外，
需要把别名行一并带上，之后就能在不重编、不走 Triton 链路的情况下秒级验证 C4 的改动 ✓。

另外 `bishengir-opt --help` 里发现一个现成旋钮：
`--hfusion-max-fused-ops-in-auto-vectorize-v2=<int>`（"Maximum number of ops to fuse in AutoVectorizeV2"），
下轮可以先扫这个参数看它对归约循环形态/性能的影响。

### 15.2 一次状态复核（发现并修掉一个陈旧对象文件）

用 `bishengir-opt` 时意外看到 `[C2DBG] ...` 输出 —— 说明**当时那个二进制里还留着第二轮 C2 调试打点**：
源码里已经没有了（`grep C2DBG bishengir/` 为空、`git diff | grep -c C2DBG` = 0），
但对象文件没有被重新编译。执行 `touch HFusionOps.cpp` + 重编后，`[C2DBG]` 输出消失 ✓。

**影响评估**：这些打点只在**编译期**写 stderr，不进入生成的内核（而且测量时 pytest 的 stderr 已重定向），
所以对 `avg_time` 无影响；但为了干净，还是在**重新构建后的二进制**上复测了一遍 argmax：

| 运行 | 10 shapes 合计 |
|---|---:|
| 复核 run 1 | 861.14 |
| 复核 run 2 | 860.45 |
| 之前记录 | 860.67 / 859.98 |
| baseline | 1173.43 |

即 **−26.7% 完全复现** ✓，说明 C1+C2 的结果不依赖那次陈旧构建。

### 15.3 当前交付状态

- 编译器分支 `opt/argmax-prob-vector`，**2 个文件 / +127 −19 行**（C1: `NormalizeComparisonTemplate.h`；C2: `HFusionOps.cpp`），源码无调试残留。
- **累计 −26.7%**（1173.43 → 860.5 µs 左右）。
- 全量回归 269 passed / 24 skipped / 0 failed。
- C3（失败，已回退）、C4（已定位到 `AutoVectorizeV2`，未动手）均完整记录。


---

## 16. 第七轮：AV2 单 pass 环境打通；C4 的两条路都试出了结论

### 16.1 单 pass 迭代环境已可用 ✅

```
bishengir-opt /tmp/av2_mod.mlir --hfusion-auto-vectorize-v2 -o /tmp/av2_out.mlir
```

输入从 `ir_c2/all_ir_c2b.mlir` 行 10385–10700 截取（含 `#map` 别名 + `module`，299 行，已滤掉 banner 和旧的
`[C2DBG]` 残留）。输出里能复现出 `scf.for %arg5 = %c0 to %c16384 step %c64` —— 与真实编译链路一致 ✓。
**今后改 AV2 可以秒级验证，不必走 Triton→完整编译。**

### 16.2 结论一：大树归约是**设计上不做的**（不是 bug）

`HFusion/Utils/Utils.cpp::shouldUseRegisterTreeReduction` 里有明确的边界：

```cpp
// A single register tree of at most 64 leaves maps to one vector load per
// row and one balanced add tree.  Larger trees need a hierarchical
// implementation and are left to the regular fused lowering.
constexpr int64_t maxRegisterTreeReductionSize = 64;
if (reductionSize > maxRegisterTreeReductionSize) return false;
```

另有 `maxMaterializedTreeReductionSize = 16`。我们的归约长度是 **16384**，远超 64，
所以被**有意**交给"regular fused lowering"（也就是那个 64 宽串行累加循环）。
→ C4 那条"把它变成树"的路 = 需要**新写一套分层树实现**，不是小改。

### 16.3 结论二：那个串行循环是**延迟受限，不是吞吐受限**（C5 实验）

**C5（尝试）**：AVE 转换里每个带掩码的向量 op 都会新建一个 `ave.hir.pge`（`Pure`、内容完全一样），
于是循环体里塞了 3 条循环不变的 `pge`。在 `hivmave::createMaskByPGE` 里加了**块内复用**
（插入点之前已有同类型同 pattern 的 `pge` 就直接复用），重编后：

| | 整个文件 `pge.b32` | sum 循环体内 `pge` |
|---|---:|---:|
| C1+C2 | 19 | 4 |
| C1+C2+C5 | **17** | **3** |

**但性能没有变化**：

| 配置 | 10 shapes 合计 |
|---|---:|
| C1+C2 | 860.67 / 859.98 |
| C1+C2+**C5** | **861.19** |
| 回退 C5 后 | 862.12 |

→ **减指令数没用 ⇒ 这个循环是延迟受限（累加器的循环携带依赖串起来），不是吞吐受限。**
因此 C5 判为**无效，已 `git checkout` 回退**（无收益就不该多改一个核心文件）。

### 16.4 剩下的唯一杠杆

要让 sum 循环变快，必须**打断累加器的依赖链**，也就是：

- **多累加器展开**：把循环按 4 展开、用 4 个独立的 64 宽累加器，循环结束再合并
  （等价于树归约的第一层，但作为**循环变换**实现，不需要 16384 的分层树）；
- 或直接实现 16.2 里说的分层树。

落点候选：
- `lib/Dialect/HIVMAVE/Transforms/AveLoopOptimize.cpp` —— 已有循环展开 pattern（`LegacyWideVextfUnrollPattern`、`PeelEpiloguePattern`），
  加一个"向量归约循环多累加器展开"的 pattern 是最自然的；
- 或 `AutoVectorizeV2.cpp` 里生成该循环的地方。

### 16.5 本轮交付状态（未变）

- 分支 `opt/argmax-prob-vector`，**仍是 C1+C2 两个文件 / +127 −19 行**（C5 已回退，源码干净）。
- **累计 −26.7%**（1173.43 → ≈860.5–862 µs）。
- 已记录的尝试：C1 ✅ −12.2%、C2 ✅ −16.5%、C3 ❌ 倒退 27.8%（已回退）、C5 ❌ 无变化（已回退）；
  C4 已定位并判定"需新功能"。


---

## 17. 第八轮：sum 循环"为什么慢"的根因彻底查清（并试了最后一个旋钮）

### 17.1 机制：每 64 个元素就做一次**跨 lane 水平归约**，且累加器是**标量**

AV2 输出（`bishengir-opt --hfusion-auto-vectorize-v2`）里那个 sum 循环的原文：

```mlir
%1 = scf.for %arg5 = %c0 to %c16384 step %c64 iter_args(%arg6 = %0) -> (tensor<f32>) {
  %2 = vector.transfer_read ... : vector<64xf32>
  %3 = vector.transfer_read %arg6[] ... : vector<f32>
  %4 = vector.extractelement %3[]                       # 取回标量累加器
  %5 = vector.broadcast %arg1 : f32 to vector<64xf32>
  %6 = arith.subf %2, %5
  %7 = math.exp %6 : vector<64xf32>
  %8 = vector.multi_reduction <add>, %7, %4 [0] : vector<64xf32> to f32   # ← 64→1 水平归约
  %9 = vector.broadcast %8 : f32 to vector<f32>
  %10 = vector.transfer_write %9, %arg6[]
  scf.yield %10 : tensor<f32>
}
```

- 累加器是 **`tensor<f32>`（标量）**，不是 64 宽向量；
- 因此**每一次迭代**都要做一次完整的 `vector.multi_reduction`（64→1，log2(64)=6 级依赖），
  **一个 tile 做 256 次**；
- 循环携带依赖 = `extract → multi_reduction(含 6 级) → broadcast → write → 下一轮 read`，
  这是很长的一条链。

**这就解释了 C5 的结果**：循环是**延迟受限**（被跨 lane 归约串起来），
所以删掉 1 条 `pge` 之类的吞吐优化完全看不到效果。

### 17.2 它由谁生成：标准 transform 的 `TileReductionUsingForOp`

`AutoVectorizeV2.cpp::tileReductionOp`：

```cpp
for (auto i : reductionDims)
  reductionAxisTileSize[i] = tileSize[i];                    // 归约轴按 1 个向量长度(64)切片
builder.create<transform::TileReductionUsingForOp>(           // MLIR transform dialect
    loc, parallelAxisTilingResult.getTiledLinalgOp(), reductionAxisTileSize);
// 生成 split / fill / combining 三个 op，combining 把每片的部分和**逐个**累加
```

即"每片算完就合并进一个标量"是上游 `TileReductionUsingForOp` 的固有形态。
`tileSize` 来自 `AutoVectorize.cpp:951 computeTileSize`：`vectorLengthInBytes / elemWidthInBytes = 256/4 = 64` ✓。

### 17.3 C6（尝试）：把归约轴的切片放大 4 倍 → **无效**

在 `tileReductionOp` 里把 `reductionAxisTileSize` 乘以 4（上限为静态长度），
重编后 IR 里确实出现了 `scf.for ... step %c128` 的循环，
**但含 `multi_reduction` 的那个 sum 循环仍然是 `step %c64`＋标量累加器**（改到的是别的归约）。
性能：**860.37 vs 860.67 —— 无变化** → 判为**无效，已回退**。

### 17.4 结论：剩下的唯一改法及其代价

要让这个循环真正变快，必须**换掉"每片一次跨 lane 归约 + 标量累加"的形态**：

- **循环后置累加**：改成 64 宽向量累加器 `acc_vec += exp(x-m)`，循环结束再做**一次** `multi_reduction`
  （256 次水平归约 → 1 次）。这是标准做法，但要新增一个 **scf.for 重写 pattern**；
- 或实现 16.2 提到的**分层树**。

**代价/风险**：会改变浮点求和顺序（编译器里对"确定性归约顺序"有明确策略，见 `Utils.h` 注释），
需要作为一项**新功能**来设计和验证，而不是一次调参。

---

## 18. 最终交付状态（本轮已全量复验）

### 18.1 编译器改动（分支 `opt/argmax-prob-vector`，未提交）

```
bishengir/include/bishengir/Transforms/regbase/Normalize/NormalizeComparisonTemplate.h   C1  (+12 −19)
bishengir/lib/Dialect/HFusion/IR/HFusionOps.cpp                                          C2  (+115)
2 files changed, 127 insertions(+), 19 deletions(-)
```

### 18.2 性能（profiler avg_time，10 个 shape 合计）

| | 合计 | Δ |
|---|---:|---:|
| baseline | 1173.43 µs | — |
| + C1（`is_nan` 降级 8 op → 4 op） | 1030.55 µs | −12.2% |
| **+ C2（融合重复的 max 归约）** | **≈860.5 µs** | **−26.7%** |

多次复测：860.67 / 859.98 / 861.14 / 860.45 / 860.37 / 862.12（同一区间，稳定）。

### 18.3 全量回归（C1+C2 编译器，fresh Triton cache）

| 测试 | 结果 |
|---|---|
| `argmax_prob_kernel.py` | 26 passed |
| `causal_conv1d_update_kernel_npu_tiled.py` | 39 passed |
| `store_kv_cache_prefix_valid_npu_kernel.py` | 24 passed |
| `add_rmsnorm_bias_kernel.py` | 32 passed |
| `fused_rope_qk_mqa_kernel_opt.py` | 32 passed |
| `split_qkv_rmsnorm_rope_kernel.py` | 37 passed |
| `split_qkvgate_gemma_rmsnorm_rope_kernel.py` | 8 passed, 24 skipped |
| `test_mix_fused_kernel.py` | 24 passed |
| `chain_rejection_block_sum_kernel.py` | 23 passed |
| `chain_rejection_sample_kernel.py` | 24 passed |
| `argmax_prob_kernel_ori.py` | 1 passed |
| **合计** | **270 passed / 24 skipped / 0 failed** |

### 18.4 尝试记录汇总

| # | 方案 | 结果 |
|---|---|---|
| C1 | `is_nan` 降级改为一次整数 GT 比较（8 op → 4 op） | ✅ **−12.2%** |
| C2 | 融合与 `reduce_with_index` 重复的 plain max 归约（3 次全宽归约 → 2 次） | ✅ **−16.5%** |
| C3 | NaN 掩码下推到 f16 域（窄类型=半价） | ❌ 倒退 27.8%，已回退（后端把窄 op 展开成更多指令） |
| C5 | AVE 掩码 `pge` 块内复用 | ❌ 无变化，已回退（⇒ 循环延迟受限） |
| C6 | 归约轴切片放大 4 倍 | ❌ 无变化，已回退（没命中该循环） |
| C4 | sum 循环改多累加器/分层树 | 已定位根因，判定需**新功能**，未实现 |

### 18.5 产物清单（`1013go/_argmax_prob_kernel_go/`）

- `SUMMARY.md`（本文，18 节）
- `baseline_pipes.md`、`patched_measure.json`
- `scripts/dump_ir.py`、`scripts/measure.py`
- `ir/`（C1 前）、`ir_new/`（C1）、`ir_c2/`（C1+C2，含 generic dump）、`ir_c3/`（C3 失败版）、`ir_c3/all_ir_c6.mlir`（C6）
- `tmp_cache*/`、`triton_cache/`（Triton 前端 IR 与当前编译器 cache）
- `/tmp/av2_mod.mlir` + `bishengir-opt --hfusion-auto-vectorize-v2`（AV2 单 pass 迭代环境）


---

## 19. 复核时发现并修掉 C2 里的一个缺陷（守卫被误删）

在回答"当前改动是否就是 C1+C2 的叠加"时逐行复核 diff，发现 C2 的 pattern 里：

```cpp
bool operandsAvailable = llvm::all_of(op->getOperands(), [&](Value operand) { ... });
for (Operation *def : hoistable) def->moveBefore(reduceOp);
op->moveBefore(reduceOp);          // ← 少了 if (!operandsAvailable) continue;
```

`operandsAvailable` **算出来了却没有用** —— 第二轮删 `[C2DBG]` 打点时，把整个
`if (!operandsAvailable) { 打点; continue; }` 块一起删掉了。
后果：当 `reduce_with_index` 有**无法提前**的操作数时，仍会强行把它上提到 plain reduce 之前，
从而产生支配性违规（正是第二轮遇到过的 `error: operand #0 does not dominate this use`）。
对本 kernel 不触发（它的操作数都能提前），但对别的图是**潜在编译失败**。

**已修复**：把 `if (!operandsAvailable) continue;` 补回。

**验证（修复是惰性的）**：
- 最终 LLVM IR 与修复前 **逐字节相同**（`diff -q` 一致）；
- 性能复测 861.28 / 859.57 µs（参考 860.67）；
- 正确性复测：`argmax_prob_kernel` 26 passed、`chain_rejection_sample` 24 passed、`test_mix_fused` 24 passed。

（期间出现过一次 `1 failed` 和一次 `1 skipped`，对应耗时分别为 266s / 228s（正常约 40–65s），
是**共享卡争抢导致的偶发**，重跑即恢复全绿。）

修复后 diffstat：**2 files changed, 129 insertions(+), 19 deletions(-)**。
