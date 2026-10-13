# `add_rmsnorm_bias_kernel` 编译器侧优化（1013go）

工作目录：`/home/y00953696/1012sgl/sgl-kernel-npu/10shape_test/1013go/add_rmsnorm_bias_kernel_go`
编译器：`/home/y00953696/AscendNPU-IR`，分支 **`opt/add-rmsnorm-bias`**（从已提交 C1+C2 的 `f8c4f0a7e` 切出）
目标 kernel：`python/sgl_kernel_npu/sgl_kernel_npu/norm/add_rmsnorm_bias.py::add_rmsnorm_bias_kernel`
测试文件：`10shape_test/add_rmsnorm_bias_kernel.py`（自带 kernel 拷贝 + COL_BLOCK_SIZE×num_stages autotune）

kernel 结构（每个 program 处理若干行，行按 `row_start += row_step` 跨步）：
```
bv = load(input) + load(residual);  store(output2, bv)      # add
bv = bv.to(f32);  squares = bv*bv
variance = tl.sum(squares) / hidden_size                     # ← 整行求和归约
reciprocal_std = 1/sqrt(variance + eps)
bv = bv * reciprocal_std * weight + bias
if SCALE: 按 COL_BLOCK_SIZE=2048 分块量化成 i8 → store；else: store(output, bv)
```

---

## 1. 环境与构建（已验证）

同 argmax 任务：`source /home/y00953696/set_env.sh` 后 `which bishengir-compile` =
`/home/y00953696/AscendNPU-IR/build/bin/bishengir-compile`；构建脚本 `/tmp/build_bishengir.sh`（AGENTS.md 的配方），
未改代码时整条命令 ~5 秒 `Build Done!!!`（增量）。

## 2. 基线（profiler avg_time）

当前编译器（含已提交的 C1+C2）多次运行：sum = **42.46 / 46.26 / 42.78 µs**（共享卡噪声较大）。
`result_baseline/_baseline.json` 里记录的原编译器值为 **44.71 µs**。

per-shape 与各 pipe（取一次代表性运行，sum=42.78）：

| shape | dur | vec | vec% | mte2 | mte3 | scalar | blocks |
|---|---:|---:|---:|---:|---:|---:|---:|
| B512_H1024_scale0 | 10.72 | 2.56 | 23.8% | **5.40** | 1.39 | 2.40 | 56 |
| B32_H16384_scale1 | 7.88 | 3.95 | 51.7% | 2.44 | 0.57 | 1.24 | 32 |
| B32_H4096_scale1 | 3.31 | 1.63 | 34.7% | 1.44 | 0.47 | 1.67 | 32 |
| B128_H1024_scale0 | 4.59 | 0.80 | 18.6% | 1.39 | 0.49 | **2.13** | 56 |
| B32_H8192_scale1 | 4.45 | 1.57 | 38.9% | 1.37 | 0.34 | 1.26 | 32 |
| B32_H8192_scale0 | 3.71 | 1.24 | 37.3% | 1.04 | 0.40 | 1.13 | 32 |
| B32_H1024_scale0 | 2.46 | 0.39 | 16.7% | 0.52 | 0.27 | **1.44** | 32 |
| B32_H256_scale0 | 2.40 | 0.30 | 13.6% | 0.44 | 0.22 | **1.55** | 32 |
| B8_H1024_scale0 | 1.79 | 0.35 | 23.4% | 0.38 | 0.25 | 0.68 | 8 |
| B1_H1024_scale0 | 1.46 | 0.32 | 26.3% | 0.30 | 0.24 | 0.53 | 1 |

**关键结论：这个 kernel 不是 vector-bound。**
各 pipe 占比（合计）：**scalar ≈ 37%**、**mte2 ≈ 34%**、vec ≈ 28%、mte3 ≈ 10%，
而且 **vec + mte2 + mte3 + scalar ≈ dur**（例如 B512：2.56+5.40+1.39+2.40 = 11.75 ≈ 11.00），
即各 pipe 几乎**完全串行**、没有重叠。

- 最大的单 shape **B512_H1024_scale0 占总量 25%**，它的瓶颈是 **mte2（5.40µs，占该 shape 50%）**——纯数据搬运。
- 小 hidden 的 shape 里 **scalar 占主导**（B32_H256：1.55/2.40 = 65%；B128_H1024：2.13/4.59 = 47%），像是固定开销。

## 3. 尝试 1（已否决）：把恒真掩码消掉 → **无效**

所有 perf shape 的 `hidden ∈ {256,1024,4096,8192,16384}` **都是 2 的幂**，而 `BLOCK_SIZE = next_power_of_2(hidden)`，
所以 kernel 里的

```python
valid_mask = cols < hidden_size          # cols = arange(0, BLOCK_SIZE)
tl.load(p, mask=valid_mask, other=0.0)
tl.store(p, v, mask=valid_mask)
```

**掩码恒为真**。IR 里也能看到 `%valid_mask = dense<16384>` 与 `%cols = make_range(0,16384)` 直接比较。
但编译器没有折叠它：VectorizeOps 阶段有 **10 个 `memref.copy`、4 个 `linalg.fill`、3 个 `scf.if`**（掩码的 fill+部分拷贝 emulation），
最终 IR 里 **`pge.b32` 40 + `pge.b16` 6 = 46 条谓词建立**。

**验证方式（诊断用例，不进仓库）**：用脚本从测试文件生成一份"去掉所有 mask/other"的变体
（`scripts/ar_nomask_variant.py`，AST 删除其它 test，trace 名前缀 `ar_nomask_`），跑同一批 shape 对比。

| 运行 | masked | no-mask |
|---|---:|---:|
| 第一次全量 | 46.26 | 42.66（−7.8%） |
| **A/B 复测 B32_H4096_scale1** | **3.25 / 3.31** | **3.48 / 3.33** |

第一次看到的 −7.8%（其中 B32_H4096_scale1 从 6.62 → 3.29，看着像 −50%）**是共享卡争抢的假象**：
A/B 复测后 masked 3.25/3.31、no-mask 3.33/3.48，**没有差别**。

→ **结论：掩码消元对本 kernel 没有实际收益，判为无效，不做。**
（记录在案，避免以后有人再去实现一个没用的优化。）

## 4. 定位到真正的热点：整行求和归约 `tl.sum(squares)`

`add_rmsnorm_bias_kernel_fused_2_outlined_vf_0`（最终 IR，H=16384）的循环体：

```llvm
^bb1(%i, %acc):
  %14 = vldsx1.v128f16(src + i)          # 载入 128 个 f16
  %16 = vcvtff.f162f32.x(%14)            # → 64 f32
  %18 = vmul.s.x(%16, %16)               # 平方
  %20 = vcadd.s.x(%18, %19)              # ★ 跨 lane 水平归约（每 64 个元素一次！）
  %22 = vadd.s.x(%10, %20, %21)          # 累加
  ...（循环按 2 展开，第二份同样再一条 vcadd）
```

即 **每 64 个元素就做一次完整的跨 lane 归约 `vcadd`**，外加 **每次迭代 5 条循环不变的 `pge.b32`** ——
和 argmax 的 `tl.sum` 是**同一个反模式**（来源同样是 `TileReductionUsingForOp` 的"每片算完再合并"）。
大 hidden 时一个 tile 要做 `hidden/64` 次 `vcadd`。

**收益上界（诊断用例）**：把归约整条删掉（`variance = 0.5`，连带平方与 sqrt 一起被折掉），
生成 `scripts/ar_nosum_variant.py`（trace 名 `ar_nosum_`）实测：

| shape | masked | no-sum | Δ |
|---|---:|---:|---:|
| B32_H16384_scale1 | 7.88 | 6.97 | −11.6% |
| B32_H8192_scale1 | 4.45 | 3.85 | −13.5% |
| B32_H8192_scale0 | 3.71 | 3.24 | −12.7% |
| B32_H4096_scale1 | 3.31 | 2.88 | −13.0% |
| B8_H1024_scale0 | 1.79 | 1.53 | −14.7% |
| B1_H1024_scale0 | 1.46 | 1.28 | −12.0% |
| B512_H1024_scale0 | 10.72 | 10.25 | −4.3% |
| B32_H256_scale0 | 2.40 | 2.36 | −1.7% |
| **SUM** | **42.78** | **38.90** | **−9.1%** |

即"归约+平方+sqrt"整条链最多值 **~9%**（对 H≥1024 的 shape 是 12–15%）。
单纯把归约改成"向量累加器 + 结尾一次水平归约"能拿到其中一部分（估 ~5–9%）。

## 5. 下一个候选（C-R1，未实现）

**把大归约改成"向量累加器 + 循环结束只做一次跨 lane 归约"**：
现在每 64 元素一条 `vcadd`（跨 lane，长延迟）→ 改成 `acc_vec += squares`（纯 elementwise，无跨 lane），
循环结束后做**一次** `vcadd`。每个元素的指令数从 ~7 降到 ~4，且消除每 64 元素一次的跨 lane 延迟。

**注意**：这条路和 argmax 的 C4 是**同一个根因、同一个改法**，而且编译器里已经有现成边界：

```cpp
// HFusion/Utils/Utils.cpp::shouldUseRegisterTreeReduction
// A single register tree of at most 64 leaves ... Larger trees need a
// hierarchical implementation and are left to the regular fused lowering.
constexpr int64_t maxRegisterTreeReductionSize = 64;
```

即"大树归约"是**设计上没做**的功能。落点候选：
1. 后置重写：识别 `scf.for` 里的"载入 → elementwise → `vector.multi_reduction` → 累加"形态，
   改成向量累加器 + 结尾一次归约（改浮点求和顺序，需确认策略）；
2. `AutoVectorizeV2.cpp::tileReductionOp`（喂给 `TileReductionUsingForOp` 的归约轴切片）。
   *注：在 argmax 上把该切片 ×4 没有命中那个循环，需要先用单 pass 环境确认命中点。*

## 6. 另一个方向（记录，未深挖）

**B512_H1024_scale0 占总量 25%，是 mte2 瓶颈**：2MB 读 / 5.40µs ≈ 370 GB/s，
而 B32_H16384 的 32KB 级传输能到 ~855 GB/s。原因是该 shape 每行只有 1024×2B = **2KB**，
且行循环是 `row_start += num_programs(56)` 的**跨步访问**（相邻两次访问间隔 112KB），DRAM 局部性差。
这更像 kernel 的网格设计问题，编译器侧不易动，先记录。

## 7. 产物

```
1013go/add_rmsnorm_bias_kernel_go/
├── SUMMARY.md（本文）
├── scripts/dump_ir.py               # 跑两个代表 shape，把 Triton IR 落进 tmp_cache
├── scripts/ar_nomask_variant.py     # 诊断：去掉全部 mask（不进仓库）
├── scripts/ar_nosum_variant.py      # 诊断：删掉整行归约（不进仓库）
├── ir/                              # H=16384(scale1) 的 .bcmlir/.ttir + 全量 IR
│                                    #   all_ir_nomb.mlir（multibuffer=False，103076 行 / 1636 个 pass dump）
│                                    #   final.mlir（最后一段，543 行）
└── triton_cache*/                   # Triton 前端 IR（含各 shape 的 bcmlir）
```


---

## 8. 尝试 2（C-R1，未落地已回退）：把 sum 改成"向量累加器 + 结尾一次归约"

### 8.1 目标形态（已用单 pass 环境固定下来）

用和 argmax 相同的方法做了**单 pass 复现环境**：从 `ir/all_ir_nomb.mlir` 截出 AV2 的输入
（行 11670–12006，含 `#map` 别名，331 行 → `/tmp/ar_av2_in.mlir`），
`bishengir-opt /tmp/ar_av2_in.mlir --hfusion-auto-vectorize-v2` 能复现归约循环 ✓。

要改写的循环（AV2 输出的原文）：

```mlir
%1 = scf.for %arg4 = %c0 to %c16384 step %c64 iter_args(%arg5 = %0) -> (tensor<f32>) {
  %2 = vector.transfer_read %extracted_slice[...], 0.0 : tensor<64xf16>, vector<64xf16>
  %3 = vector.transfer_read %arg5[], 0.0 : tensor<f32>, vector<f32>      ← UB 往返（读）
  %4 = vector.extractelement %3[] : vector<f32>
  %5 = arith.extf %2 : vector<64xf16> to vector<64xf32>
  %6 = arith.mulf %5, %5 : vector<64xf32>
  %7 = vector.multi_reduction <add>, %6, %4 [0] : vector<64xf32> to f32  ← 每 64 个元素一次跨 lane
  %8 = vector.broadcast %7 : f32 to vector<f32>
  %9 = vector.transfer_write %8, %arg5[] : vector<f32>, tensor<f32>      ← UB 往返（写）
  scf.yield %9 : tensor<f32>
}
```

目标：

```mlir
%splat = vector.splat(<载入初始累加器>) : vector<64xf32>
%accv = scf.for %arg4 = ... iter_args(%a = %splat) -> (vector<64xf32>) {
  %6 = ...（原样）
  %a2 = arith.addf %a, %6 : vector<64xf32>
  scf.yield %a2
}
%r = vector.multi_reduction <add>, %accv, 0.0 [0]      ← 整个归约只做这一次
...broadcast + transfer_write 回原 tensor...
```

### 8.2 实现（写在 `AutoVectorizeV2.cpp`，在 transform 之后做后处理）

在 `AutoVectorizeV2::runOnOperation` 里、对每个函数 `runAttempt` 成功之后调用
`rewriteScalarAccumulatorReductions(func)`；匹配条件（全部满足才动手）：
1 个 iter_arg、且是**单元素 f32 tensor**；body 的 `scf.yield` 是
`transfer_write(broadcast(multi_reduction<add>(tile, extractelement(transfer_read(iter_arg)))))`；
tile 是 f32 一维向量。改写用 **clone** 原 op（而不是手工构造 transfer_read/write），
只把累加器从 tensor 换成向量寄存器。

### 8.3 过程中修掉的问题（每一轮都编译+单 pass 验证）

| # | 问题 | 修法 |
|---|---|---|
| 1 | `scf::ForOp::getNumIterOperands` 不存在 | 改用 `getInitArgs().size()` |
| 2 | `vector::MultiReductionOp` 不存在 | 正确名字是 `vector::MultiDimReductionOp` |
| 3 | 改写**没被触发** | 目标循环在 transform **新 outline 出来的函数**里，而 `runOnOperation` 只遍历入口时可融合的函数 → 改成在整个 module 上 walk |
| 4 | 命中后仍不触发（"no tensor-accumulator write"） | `transfer_write` 的 tensor 操作数下标与假设不符 → 改成用 `operandPosition()` 查找，不写死下标 |
| 5 | `Block::getTerminator()` 断言崩溃 | `scf::ForOp` builder 建出的 body **没有终结符**，需自己 `create<scf::YieldOp>` |
| 6 | `'arith.constant' destroyed but still has uses` | 从 body 里 clone 到循环外的 op 仍引用 body 内的常量/下标 → 加 `liftOperands()` 重建这些操作数 |
| 7 | `Operation*` → `Value` 编译错 | `rewriter.clone(*def)->getResult(0)` |

到第 7 轮，**匹配已经成功**（打点显示 `[RR] REWRITING loop in add_rmsnorm_bias_kernel_fused_2`），
但运行仍然触发 MLIR 的
`Assertion 'use_empty() && "Cannot destroy a value that still has uses!"' failed`
—— 崩在 `rewriter.eraseOp(forOp)` 的 `Region::erase`。
我加的诊断（遍历 body 内所有 result、检查是否有 body 之外的 user）**没有报出任何悬挂使用**，
所以剩余那一刻时**没能在本轮预算内定位到根因**。

### 8.4 结论与决定

**判为：本轮未落地，已 `git checkout` 回退 `AutoVectorizeV2.cpp`。**
回退后复核：`bishengir-opt` 在同一个 repro 上的输出与改动前 **逐字节相同** ✓，编译器工作区干净。

下一步（下轮）两条路：
- **A**：继续定位那个悬挂使用 —— 把"循环外新建的 op"全部改成**用 builder 显式构造**
  （而不是 clone 原 op），这样从根上不会引用 body 内的值；
- **B**：换一种不删旧循环的写法（例如就地替换累加器链、保留原循环骨架）。

### 8.5 本轮另一个已做的实验（无效）

**强制 `multibuffer=True`**（pytest 插件注入，不改算子文件）：

| 配置 | 合计 |
|---|---:|
| autotune 默认 | ≈42.5–46.3 |
| 强制 `multibuffer=True` | **42.90** |

→ 与 argmax 的结论一致：**多缓冲开关对这个 kernel 也没有影响**，DMA 与计算的重叠不是开关级问题。


---

## 9. C-R1 第二轮（本轮）：改写**在语义上做成了**，但撞上 outline 的结构约束 → 仍回退

### 9.1 修掉的根因（关键发现）

上一轮那个 `Cannot destroy a value that still has uses`，本轮用**覆盖 block argument 的诊断**定位到了：
悬挂的是**旧循环体的 induction variable（index 类型的 block argument）**，被 `tensor.extract_slice` 使用 ——
因为 `IRMapping` 里**没有**把旧的 induction var / iter arg 映射到新的：

```cpp
mapping.map(forOp.getInductionVar(), newLoop.getInductionVar());
mapping.map(iterArg, newLoop.getRegionIterArg(0));
```

（上一轮的诊断只查了 op 的 *result*，没查 block argument，所以什么都没报。）

随后又修掉两个真实 bug：
- **`OpBuilder` 不会自动推进 insertion point** → 我建的 `initRead / extractelement / splat / finalReduce / broadcast / finalWrite`
  落成了**逆序**，导致 `operand #1 does not dominate this use`；逐个 `setInsertionPointAfter(...)` 修正。
- **lifting 出的常量被插在使用者之后** → `fresh->moveBefore(cloned)` 修正。

### 9.2 单 pass 验证：改写形态**完全正确**

`bishengir-opt ar_av2_in.mlir --hfusion-auto-vectorize-v2` 输出：

```mlir
%1 = vector.transfer_read %0[], %cst_0 : tensor<f32>, vector<f32>
%2 = vector.extractelement %1[]
%3 = vector.broadcast %2 : f32 to vector<64xf32>            ← 向量累加器初值
%4 = scf.for %arg4 = %c0 to %c16384 step %c64 iter_args(%arg5 = %3) -> (vector<64xf32>) {
  %8  = vector.transfer_read %extracted_slice[...] : tensor<64xf16>, vector<64xf16>
  %9  = arith.extf %8 : vector<64xf16> to vector<64xf32>
  %10 = arith.mulf %9, %9
  %11 = arith.addf %arg5, %10 : vector<64xf32>              ← 每 tile 只做一次向量加
  scf.yield %11
}
%5 = vector.multi_reduction <add>, %4, %cst_1 [0] : vector<64xf32> to f32   ← 整个归约只此一次
```

即**每个元素的 `vcadd`（256 次）变成了 1 次**，这正是目标形态 ✓。

### 9.3 但全量管线拒绝它：**outline 的结构约束**

```
error: unexpected vector operation outside vector function
```

追到判据（`AutoVectorize/Verify.cpp`）：

```cpp
bool isNestedInOutlinedLoopTarget(Operation *op) {   // 向量 op 必须"嵌套在带注解的循环里"
  for (Operation *cur = op; cur; cur = cur->getParentOp())
    if (hasOutlinedLoopTargetAttr(cur)) return true;
  return false;
}
```

而 outliner（`OutlineVectorFunction.cpp`）是按 **`outlined-loop-target-*` 注解的循环**为单位
`ExtendedLoopOutlineOp` 抽取成 VF 函数的。原实现之所以合法，是因为**所有向量 op 都在那个带注解的循环体内部**；
我的改写把"结尾那一次跨 lane 归约 + broadcast + 写回"放到了**循环之后**，于是它们不在任何带注解的区域内 → 报错。

> 试过把循环的 `outlined-loop-target-*` 等属性复制到新循环（`newLoop->setAttr(...)`），不解决 ——
> 因为约束是"op 要嵌套在带注解的 op **内部**"，而不是"兄弟节点带注解"。

### 9.4 结论与下一步

**本轮仍未落地，已 `git checkout` 回退**（复核：AV2 输出与改动前**逐字节一致**，工作区干净）。

要在语义正确的前提下满足 outline 约束，可行方向：
- **A**：把结尾的归约 epilogue 也放进一个带同样注解的区域（例如把 epilogue 包进一个
  `outlined-loop-target-N` 的单趟 `scf.for`），让 outliner 一起抽走；
- **B**：把改写**挂到 outlining 之后**的 pass（此时循环已在 VF 函数里，新建的向量 op 天然"在 vector function 内"）；
- **C**：换个不改循环结构的做法——保留 tensor 累加器，但把"每 tile 一次跨 lane 归约"换成
  "向量加 + 把向量写回 tensor<64xf32> 累加器"，让唯一那次跨 lane 归约仍在循环内完成。

**本轮的净收获**：把上一轮那个断言彻底定位并修掉了（block-argument 映射缺失），
拿到了语义正确的改写形态，并查清了它被拒绝的**确切原因**（outline 的嵌套约束）——这三点都是下一轮的直接输入。


---

## 10. C-R1 第三轮（本轮）：路径 B（挂到 outlining 之后）—— 过了 verifier，但**后端不支持**

### 10.1 做法

上一轮查明约束是"向量 op 必须**嵌套在带 `outlined-loop-target-*` 注解的循环内部**"。
于是本轮把改写**从 AV2 挪到 outliner**（`OutlineVectorFunction.cpp`）的 `runOnOperation` 末尾——
此时循环已经在 `*_outlined_vf_*` 向量函数里，新建的向量 op 天然"在 vector function 内"：

```cpp
module.walk([&](scf::ForOp forOp) { removeLabelsForOutlinedLoops(forOp); });
module.walk([&](func::FuncOp func) {
  if (hivm::isVF(func))                       // 只在向量函数里改写
    rewriteScalarAccumulatorReductions(func);
});
```

（改写函数与上一轮相同：向量累加器 + 循环外一次 `multi_reduction`；上一轮修的 4 个 bug 全部带着。）

### 10.2 结果：**verifier 过了**，但卡在后面的 SIMD/AVE 降级

- `unexpected vector operation outside vector function` **消失** ✓（说明"挂到 outlining 之后"这条路走对了）；
- 但换成新的失败：

```
error: Failed to run BiShengSIMD pipeline
[ERROR] Failed to run BiShengIR regbase pipeline
```

对比两条编译链最后成功跑过的 pass：

| | 最后成功阶段 |
|---|---|
| 无改进（基线） | … → `ConvertIndexToLLVM` → `ArithToLLVMConversion` → `ReconcileUnrealizedCasts` → 完成 |
| 带改进 | … → **`AVENormalizeOps` / `RemoveVectorLayoutAttr` 之后就断了** |

即失败发生在 **AVE（SIMD/regbase）降级**阶段，而不是向量化/outline 层，而且**没有更具体的诊断信息**
（日志里只有那句通用的 pipeline 失败）。

### 10.3 判断

新 IR 与旧 IR 的关键结构差异只有一个：**循环携带值从 `tensor<f32>` 变成了 `vector<64xf32>`**
（把累加器放在向量寄存器里而不是 UB/tensor 里）。AVE 降级在这一点上失败，
最合理的解释是：**当前后端不支持"寄存器常驻的循环携带向量累加器"** —— 循环携带值需要是 memref/tensor 形态。

> 注意：这一条我**没能拿到更硬的证据**（没有具体报错，只能从"断在哪个阶段 + 唯一的 IR 结构差异"推断），
> 记录时按推断标注。

### 10.4 结论与下一步

**本轮仍未落地，已 `git checkout` 回退**（复核：回退后同一 bcmlir 编译 `rc=0` ✓，工作区干净）。

三条路的现状：
- **A**（把 epilogue 包进带注解的区域）：未试；
- **B**（挂到 outlining 之后）：**已试，过了 verifier，被后端拦住**；
- **C**（保留 tensor 形态的累加器，只把"每 tile 一次跨 lane 归约"换成"向量加 + 向量写回"）：
  这正好绕开 10.3 里那个后端限制（循环携带值仍是 tensor）——**这是下一步最有希望的方向**：

```
%acc64 = ... tensor<64xf32> 形式的累加器（循环携带）
loop:
  %v = read(acc64) ; %v2 = add(%v, tile) ; write(%v2, acc64)     ← 跨 lane 归约被移除，仍是 tensor 往返
after loop:
  %s = multi_reduction<add>(read(acc64)) → 写回原来的 tensor<f32>
```

**本轮的净收获**：确认了"挂到 outlining 之后"这条集成路径可行（verifier 通过），
并把失败点精确到 AVE 降级 + 唯一的 IR 结构差异（循环携带 `vector<64xf32>`），
据此得出了绕开它的具体方案 C。


---

## 11. C-R1 第四轮（本轮）：方案 C（tensor 形态累加器）—— outliner 阶段正确，**全量管线同样被拒**

### 11.1 动机与实现

上一轮判定失败原因是"寄存器常驻的**循环携带向量**"（`vector<64xf32>` iter_arg）。方案 C 换掉这一点：
**循环携带值仍是 tensor**，只是把"每 tile 一次跨 lane 归约"改成"向量加 + 向量写回"，
唯一那次跨 lane 归约挪到循环之后：

```mlir
%init64 = linalg.fill(%scalar) : tensor<64xf32>            ; 累加器（tensor 形态，循环携带）
%4 = scf.for ... iter_args(%arg6 = %init64) -> (tensor<64xf32>) {
  %9  = vector.transfer_read %extracted_slice[...]         ; tile
  %10 = arith.extf %9 ; %11 = arith.mulf %10, %10
  %12 = vector.transfer_read %arg6[%c0], 0.0 : tensor<64xf32>, vector<64xf32>
  %13 = arith.addf %12, %11 : vector<64xf32>               ; ← 无跨 lane
  %14 = vector.transfer_write %13, %arg6[%c0]
  scf.yield %14 : tensor<64xf32>
}
%5 = vector.transfer_read %4[%c0], 0.0
... multi_reduction<add>(%5, 0) → broadcast → 写回原 tensor<f32>   ; 只此一次
```

落点仍是 outliner 之后（上一轮验证过的路径），并在 `bishengir-opt --outline-vector-function` 上
**单 pass 验证形态正确** ✓（上面的 IR 就是实际输出）。

### 11.2 结果：**仍然被拒，且是同一条无诊断的失败**

```
error: Failed to run BiShengSIMD pipeline
[ERROR] Failed to run BiShengIR regbase pipeline
```

`vcadd` 40→0、`pge.b32` 40→0、最终 IR 543 行→51 行（提前中断），
说明**失败点与上一轮相同**（AVE/SIMD 降级阶段），而不是"循环携带向量"这一个特征。

### 11.3 阻塞条件的总结（跨 3 轮不变）

| 轮次 | 尝试 | 被什么拦住 |
|---|---|---|
| 2 | 向量累加器（写在 AV2） | `unexpected vector operation outside vector function`（outline 的嵌套约束） |
| 3 | 同上，改挂到 outliner 之后 | **BiShengSIMD pipeline 失败**（推断：不支持寄存器携带的循环向量） |
| 4 | **方案 C**：tensor 携带的向量累加器 | **同一条 BiShengSIMD 失败**（无诊断，说明不止"寄存器携带"这一个原因） |

即：**"要消除 `tl.sum` 里每 64 元素一次的跨 lane 归约，唯一办法是重构这个归约循环；
而我能表达的两种重构都被 BiShengSIMD/AVE 降级阶段拒绝（只有一句无细节的 pipeline 失败）"** ——
这个阻塞条件已连续 3 轮（第 2、3、4 轮）成立。

其它已验证无效的方向（见前文）：掩码消元（无效）、强制 multibuffer（无效）；
剩余时间占比的 DMA(34%)/scalar(37%) 未找到可下手的编译器侧杠杆，
且最大的 shape（B512_H1024，占 25%）是 2KB 小传输 + 跨步访问导致的 mte2 受限，属 kernel 网格设计问题。

### 11.4 本轮产物

- 已回退，编译器工作区**干净**；回退后同一 bcmlir 编译 **rc=0** ✓。
- 代码留档（避免像上轮那样被 revert 丢掉）：
  `scripts/attempt_ideaC_OutlineVectorFunction.cpp`（本轮方案 C 的完整实现，含 outliner 挂钩）


---

## 12. 用 `--mlir-print-ir-before-all` 定位失败 pass —— 结论：**不是"差一点点"，是后端不支持**

用户提示可以用全量 pass dump 看出失败的 pass。做法：在编译命令里**加上 `--mlir-print-ir-before-all`**，
然后取"最后一个 `Before X` 却没有对应 `After X`"的 pass；更简单的是**对比基线与失败运行的 pass 序列**。

### 12.1 失败的 pass = `AnnotationLowering`

| | AVE pipeline 结束之后的 pass 序列 |
|---|---|
| 基线（无改写） | `RemoveVectorLayoutAttr` ×5 → **`AnnotationLowering`** → `AllocToAlloca` → `InsertInitAndFinishForDebug` … → 一路到 `ConvertIndexToLLVM`/`ArithToLLVMConversion` |
| 带改写 | … → `AVENormalizeOps` → `RemoveVectorLayoutAttr` → **（到此为止，没有 `AnnotationLowering`）** |

所以 pipeline 断在 `AnnotationLowering`（`buildLowerToLLVMPipeline` 的第一个 pass）。

### 12.2 但**真正的原因在更上游**（这才是关键）

在 dump 里搜 "failed to" 找到了真正的诊断（之前只搜 `error:` 所以漏了）：

```
-> If running end-to-end: an upstream pass failed to lower this op. Op: vector.multi_reduction
Possible causes:
  1. The operation may not have been lowered to the HIVMAVE dialect before VectorLayout analysis.
  2. The operation type may not be handled in solveProblem TypeSwitch.
```

即 **AVE pipeline 里的 `AnalyzeVectorLayout` 报出"有一个 `vector.multi_reduction` 没被降级"**，
后面的 pass 因此失败。

追到源码：
- `lib/Conversion/VectorToHIVMAVE/VectorToHIVMAVE.cpp` 里处理归约的 pattern 是
  **`OpRewritePattern<vector::ReductionOp>`**（`vector.reduction`），**没有 `vector.multi_reduction` 的 pattern**；
- `vector.multi_reduction` 平时是被**更早的一个 normalize pass**（`lib/Dialect/Vector/Transforms/NormalizeVector.cpp`）
  改写成 `vector.reduction` 之后，才被 AVE 转换接住的；
- 而我的改写挂在 **outlining 之后**，那时 normalize 早就跑完了 → 我新造的 `multi_reduction` 没人管 → 报错。

### 12.3 于是改成 `vector.reduction` —— 前进了一步，但**布局求解器崩了**

把循环外那一次归约改成 `vector::ReductionOp`（后端认识的那个 op）：

```cpp
auto finalReduce = rewriter.create<vector::ReductionOp>(
    loc, vector::CombiningKind::ADD, finalRead, finalZero);
```

结果：不再有 "failed to lower" 报错，但变成了**编译器崩溃**（rc=134 / SIGABRT）：

```
bishengir-compile: .../AnalyzeVectorLayout.cpp:276:
  TreeSolve::getState(mlir::Value): Assertion `inputs.contains(v) && "no state for value"' failed.
```

即 **布局求解器在"循环之外的独立归约"上拿不到状态**（它的输入集合是按 AVE 的 op/循环结构建立的）。

### 12.4 结论（回答"差一点点还是根本不支持"）

**不是"差一点点"，而是"后端不支持循环外的独立归约"这条形态**：

1. `vector.multi_reduction` 出循环 → 没人降级（AVE 转换只认 `vector.reduction`，而把它转成 `vector.reduction` 的
   normalize 阶段已经过去）→ `AnalyzeVectorLayout` 报未降级 → 失败；
2. 改成 `vector.reduction` → 降级这一关过了，但**布局求解器**在循环外的归约上 `no state for value` 直接断言崩溃。

要落地就必须**给编译器加能力**（至少两处：让 AVE 转换/布局求解器支持"不在循环里的归约"，
或让我的改写把 epilogue 组织成它们能理解的结构）。这已经是一个**独立的新功能**，
不是本轮能顺带改掉的量级——所以第 4 轮把它判为阻塞是**正确的判断**。

### 12.5 产物

- 本轮两个版本的实现都留档了（避免像上一轮那样被 revert 丢掉）：
  - `scripts/attempt_ideaC_OutlineVectorFunction.cpp` —— `multi_reduction` 版
  - `scripts/attempt_ideaC_reduction_op.cpp` —— 改成 `vector.reduction` 版
- 失败运行的全量 dump（带 before/after）：`ir/all_ir_before.mlir`（339k 行）、`ir/all_ir_c2.mlir`
- 编译器工作区已回退干净。


---

## 13. 第五轮：按用户提示继续深挖 —— `vector.reduction` 之后**崩在布局求解器**，并定位到具体那条 value

用户指出可以用全量 pass dump 看失败 pass（已在 §12 做完：失败 pass = `AnnotationLowering`，
真正原因是 AVE pipeline 里 `AnalyzeVectorLayout` 报"`vector.multi_reduction` 未降级"）。
本轮继续把 `vector.reduction` 版本（`scripts/attempt_ideaC_reduction_op.cpp`）的崩溃点钉到具体值上：
在 `AnalyzeVectorLayout.cpp::TreeSolve::getState()` 里加打印：

```cpp
if (!inputs.contains(v))
  llvm::errs() << "[LAYOUT] no state for value: " << v << " defined by "
               << v.getDefiningOp()->getName() << "\n";
```

结果（rc=134 / SIGABRT）：

```
[LAYOUT] no state for value: %10 = unrealized_conversion_cast %9 : vector<1xf32> to f32
        defined by builtin.unrealized_conversion_cast
bishengir-compile: .../AnalyzeVectorLayout.cpp:276: TreeSolve::getState(mlir::Value):
  Assertion `inputs.contains(v) && "no state for value"' failed.
```

即：**布局求解器的状态集合里没有"循环之后那条归约链"上的中间值**（这里是
`unrealized_conversion_cast vector<1xf32> -> f32`，也就是把循环携带的 tensor 累加器读回来那一步）。
求解器的输入集合是按 AVE 的 op/循环结构建立的，**循环之外的归约链不在其中**。

### 13.1 结论（已从"推断"升级为"证据"）

要落地这个优化，必须**给编译器加能力**，至少两处：
1. AVE 转换要能把"循环外的独立归约"降级（`vector.multi_reduction` 没有 pattern，只有 `vector.reduction` 有，
   而把前者规范化成后者的 pass 在 outlining 之前就跑了）；
2. 布局求解器（`AnalyzeVectorLayout`）要把"循环外的归约链"纳入状态推导（现在直接 `no state for value` 断言崩溃）。

这不是调参或小补丁，而是**编译器功能开发**。至此这条路的可行性已经调查完毕、结论明确。

### 13.2 本轮状态

- 编译器**已完全回退**（`OutlineVectorFunction.cpp`、`AnalyzeVectorLayout.cpp` 的插桩都撤了），
  工作区干净；回退后同一 bcmlir 编译 **rc=0** ✅（验证过两次，中间一次 rc=1 是我把
  `--append-bisheng-options` 敲成了 `--append-bishengir-options`，与代码无关）。
- 留档：`scripts/attempt_ideaC_OutlineVectorFunction.cpp`（`multi_reduction` 版）、
  `scripts/attempt_ideaC_reduction_op.cpp`（`vector.reduction` 版）。
- 至此该 kernel 的编译器侧调查结论：**没有可落地的优化**；唯一被量化的热点（`tl.sum` 的
  每-64-元素跨 lane 归约，收益上界 −9.1%）需要新增后端能力才能改。


---

## 14. 换方向：量化 COL_BLOCK_SIZE 调到"整行一次"（用户建议）—— **有效，−3.1%**

编译器侧先放下后，用户提出：量化 tile `COL_BLOCK_SIZE 2048→8192`（整行一次量化，去掉逐块
`extract_slice` 与重复载入 scale/offset）。评估与实验如下。

### 14.1 先看现状：4096/8192 早就被搜过了，**16384（整行）没有**

测试文件里的 autotune 列表是：

```python
_ARB_CONFIGS = [triton.Config({"COL_BLOCK_SIZE": cb}, num_stages=ns)
                for cb in (2048, 4096, 8192) for ns in (1, 2)]
# prune: COL_BLOCK_SIZE <= BLOCK_SIZE（否则 extract_slice 会越界）
```

而 `BLOCK_SIZE = next_power_of_2(hidden_size)`，所以：
- `hidden=4096/8192` 的"整行"分别是 4096/8192 —— **已经在列表里**；
- `hidden=16384` 的"整行"是 **16384 —— 不在列表里**，只能选到 8192（每行 2 块）✗

即用户说的"整行一次"，**真正新增的只有 hidden=16384 那一档**。

### 14.2 实验（诊断用例，不改跟踪文件）

写了两份变体（`scripts/ar_colbig_variant.py`、`scripts/ar_whole_row_variant.py`，trace 前缀分别
`ar_colbig_` / `ar_whole_`，不进仓库）：

| 方案 | 合计 avg_time | vs 基线 |
|---|---:|---:|
| 基线（列表 2048/4096/8192，autotune） | **42.88** µs | — |
| 列表加入 16384（autotune 自选） | 41.80 µs | −2.5% |
| **强制整行 `COL_BLOCK_SIZE = BLOCK_SIZE`** | **41.58 / 41.77** µs | **−3.1%** |

（基线另有一次 81.77 µs，是共享卡争抢，已丢弃。）

per-shape（取同一次干净对照）：

| shape | 基线 | +16384 | 强制整行 | Δ |
|---|---:|---:|---:|---:|
| **B32_H16384_scale1** | 7.93 | 6.48 | **6.56** | **−17.3%** |
| B512_H1024_scale0 | 10.91 | 10.99 | 10.79 | −1.1% |
| B32_H8192_scale1 | 4.28 | 4.43 | 4.29 | +0.4% |
| B32_H4096_scale1 | 3.34 | 3.32 | 3.30 | −1.0% |
| B128_H1024_scale0 | 4.46 | 4.43 | 4.68 | +5.0%（该 shape SCALE=False，量化路径不参与 → 噪声） |
| 其余 5 个 | — | — | — | ±2%（噪声） |
| **SUM** | **42.88** | **41.80** | **41.58** | **−3.1%** |

**机制确认**：只有 `B32_H16384_scale1` 出现量级变化（−17.3%），正是"每行 2 块 → 1 块"的那一档 ✓；
其余 shape 不受影响（4096/8192 本来就在列表里；SCALE=False 的形状根本不走量化分支）✓。

### 14.3 落地结论与注意事项

**建议**：把算子 wrapper（`python/sgl_kernel_npu/sgl_kernel_npu/norm/add_rmsnorm_bias.py`）
里的 `COL_BLOCK_SIZE = 2048` 改成 **`COL_BLOCK_SIZE = BLOCK_SIZE`**（整行一次量化）——
这是 wrapper 的一行改动，不是 kernel 体，符合"可以调算子参数"。

**两个必须注意的点**（都有本会话的证据）：
1. **必须同时保留 `num_stages=1` 的回退**：`hidden=16384 + SCALE` 在 `multibuffer=True`（num_stages=2）下
   UB 溢出（`requires 2359296 bits while 2031616 bits available`）。测试文件里靠 `num_stages ∈ {1,2}`
   的 autotune 选中 ns=1；算子若只改 `COL_BLOCK_SIZE` 而不带这个回退，那一档会编译不过。
2. **`COL_BLOCK_SIZE` 必须 ≤ `BLOCK_SIZE`**：kernel 里 `extract_slice(buffered_values, (0,), (COL_BLOCK_SIZE,), (1,))`
   作用在 `BLOCK_SIZE` 宽的张量上，取整行正好等于 `BLOCK_SIZE` ✓，再大就越界。

### 14.4 关于"改 grid"

查了一下，**不改 kernel 实现拿不到收益**：
- 发射是 `n_rows = min(batch_size, num_vectorcore)`，kernel 用 `row_start = program_id(0)` +
  `for i in range(row_start, batch_size, row_step)` 跨步行；
- `B512/B128` 这类 `batch > 56` 的形状 **grid 已经是 56（吃满向量核）**（profiler 里 `Block Num = 56`）✓ 无可改；
- `B32*` 这类 `batch < 56` 的形状只有 32 个 program（`Block Num = 32`，24 个核闲着），
  但要让它们吃满核必须**把一行拆到多个 program**（需要跨 program 归约/原子）→ **必须改 kernel 体** ✗。

所以 grid 这条按你说的"不需要同步改算子 kernel 实现"的标准，**不成立**，先不动。


### 14.5 正确性验证（整行量化）

只跑性能是不够的，所以另做了保留全部精度用例、但强制整行的变体
（`scripts/ar_whole_precision_variant.py`）：

```
python -m pytest .../ar_whole_precision_variant.py -k "precision or batch_larger"
-> 32 passed
```

即"整行一次量化"**数值正确** ✓（量化数学没变，只是 `COL_BLOCK_SIZE` 从 2048 变成整行；
`valid_mask2 = col_indices < hidden_size` 的尾部掩码照旧生效）。

### 14.6 算子侧落地的注意点（未直接验证算子本身）

建议改动（wrapper 一行，属"调算子参数"）：

```python
# python/sgl_kernel_npu/sgl_kernel_npu/norm/add_rmsnorm_bias.py
-    COL_BLOCK_SIZE = 2048
+    COL_BLOCK_SIZE = BLOCK_SIZE        # 整行一次量化
```

**两个 caveat**：
1. **大 hidden + SCALE 需要 `num_stages=1` 的回退**：整行量化会加大 UB 占用；
   本会话里 `hidden=16384 + SCALE` 只有关掉多缓冲（`num_stages=1` / `multibuffer=False`）才编得过
   （否则 `ub overflow, requires 2359296 bits while 2031616 available`）。
   测试文件靠 `num_stages ∈ {1,2}` 的 autotune 选到 ns=1 ✓；而算子的 `warmup(...)` 调用
   **没有传 `num_stages`/`multibuffer`**（用默认 ns=2），所以只改 `COL_BLOCK_SIZE` 可能让那一档编不过。
   ⚠️ **这一点我没有直接验证算子**：算子的 `sgl_kernel_npu` 扩展在本环境 import 失败
   （`libsgl_kernel_npu.so: cannot open shared object file`），我只能验证 kernel 本体。
2. `COL_BLOCK_SIZE` 必须 ≤ `BLOCK_SIZE`（kernel 里对 `BLOCK_SIZE` 宽的张量做 `extract_slice`），
   取"整行"正好等于 `BLOCK_SIZE` ✓。


---

## 15. 结论汇总：这个算子现在还剩哪些优化点

### 15.1 一句话

**在"只调算子参数 / 只改编译器"的范围内，只剩一条已经验证但还没落地的收益：
量化整行（`COL_BLOCK_SIZE = BLOCK_SIZE`，−3.1%，见 §14）。**
其余方向要么已实测无效，要么必须改 **kernel 体**，要么需要给**编译器加新功能**。

### 15.2 还在桌上的（唯一一条）

| 项 | 数值（本会话实测） |
|---|---|
| 收益 | 合计 **−3.1%**（42.88 → **41.58 / 41.77** µs）；`B32_H16384_scale1` **−17.3%**（7.93 → 6.56） |
| 正确性 | 强制整行的精度用例 **32 passed** ✓ |
| 改法 | `python/sgl_kernel_npu/sgl_kernel_npu/norm/add_rmsnorm_bias.py` 一行：`COL_BLOCK_SIZE = 2048` → `= BLOCK_SIZE` |
| 前提/caveat | 大 hidden + SCALE 需要 `num_stages=1`（关多缓冲）的回退，否则 UB 溢出编不过；算子当前 `warmup(...)` 没传 `num_stages`。**算子本身我没能直接验证**（`import sgl_kernel_npu` 在本环境报 `libsgl_kernel_npu.so: cannot open shared object file`），只验证了 kernel 本体 |

### 15.3 已封死的方向（每条都有依据，指向本文件对应小节）

| 方向 | 状态 | 依据 |
|---|---|---|
| 可调参数空间 | **基本穷尽** | kernel 的 constexpr 只有 `hidden_size / eps / BLOCK_SIZE / COL_BLOCK_SIZE / SCALE`；只有 `COL_BLOCK_SIZE` 是自由的（`BLOCK_SIZE = next_power_of_2(hidden)` 是推导的）。`COL_BLOCK_SIZE` 已扫 2048/4096/8192/**16384**，`num_stages` 1/2 也扫过（§14） |
| `num_stages` / `multibuffer` | 对性能**无效** | 强制 `multibuffer=True`：42.90 vs 默认 ≈42.8（§3、§8.5）；只在 `H=16384+SCALE` 作为"能编过"的回退有用 |
| `num_warps` | 预计**无效** | 后端源码：`num_warps` 只在 `is_pure_simt` 模式转发给 bishengir，SIMD 路径不生效（**读码结论，未实测**） |
| 恒真掩码消元 | **无效** | 所有 perf shape 的 hidden 都是 2 的幂 ⇒ `cols < hidden` 恒真；A/B 复测 masked **3.25/3.31** vs no-mask **3.33/3.48**（H=4096），先前的 −7.8%/−50% 是共享卡争抢假象（§3） |
| **grid** | **必须改 kernel 体** | `n_rows = min(batch, num_vectorcore)`：`B512/B128` 已是 56 个 program（吃满，profiler `Block Num = 56`）；`B32*` 只有 32 个（`Block Num = 32`，24 核闲着），要吃满必须把一行拆到多个 program → 跨 program 归约，属 kernel 体改动（§14.4） |
| 归约重构（去掉每 64 元素一次跨 lane `vcadd`） | **需要编译器加功能** | 上界实测 **−9.1%**（42.78 → 38.90；H≥1024 的 shape −12~−15%，§7）；两种改写都被后端拦住：`vector.multi_reduction` 无转换 pattern、改 `vector.reduction` 后 `AnalyzeVectorLayout` 断言 `no state for value`（§12、§13） |

### 15.4 剩余收益的归属（哪些属于谁）

- **参数侧（允许）**：整行量化 −3.1% —— **已在桌上，只差落地**。
- **编译器侧（允许，但需新功能）**：归约重构上界 −9.1%（且可与 argmax 的 C4 共用同一套后端能力）；
  需要的两处能力：① AVE 转换支持"循环外的独立归约"；② `AnalyzeVectorLayout` 把循环外的归约链纳入状态推导。
- **算子实现侧（当前不允许）**：真正的大头都在这里 ——
  - `B512_H1024_scale0` 占总量 **25%**，其 **mte2 占该 shape 50%**（2KB 小传输 + `row_start += 56` 跨步访问，
    每传输 ≈9.5 cycle，已接近 DMA 发射率上限）；
  - `B32*` 系列只用到 **32/56** 个向量核（`batch < cores`）；
  - 行循环的 `tl.range` 选项（如 `loop_unroll_factor`）也在 kernel 体内。

### 15.5 建议的下一步（按性价比）

1. **落地整行量化**（−3.1%）：改 wrapper 一行 + 确认 `num_stages=1` 回退；顺带在算子侧跑一遍它自己的测试。
2. **实测 `num_warps`**（成本极低，一次运行）：确认或推翻 15.3 里那条读码结论。
3. **编译器加功能**（收益上界 −9.1%，可与 argmax 共用）：作为**独立目标**开，因为它是功能开发而非调优。
4. 改 kernel 体（吃满 B32 的核 / 行内拆分 / 循环展开）：**当前不允许**，但如果哪天放开，这是最大的一块。
