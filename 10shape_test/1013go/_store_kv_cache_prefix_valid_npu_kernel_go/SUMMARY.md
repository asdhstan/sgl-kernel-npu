# `_store_kv_cache_prefix_valid_npu_kernel` 优化（1013go）

工作目录：`/home/y00953696/1012sgl/sgl-kernel-npu/10shape_test/1013go/_store_kv_cache_prefix_valid_npu_kernel_go`
编译器：`/home/y00953696/AscendNPU-IR`，分支 **`opt/store-kv-prefix-valid`**（**基于 master** `2ac1b376d` 新建，
确认不含之前的 C1+C2：`FoldPlainReduceIntoReduceWithIndex` 计数 = 0）
算子：`python/sgl_kernel_npu/sgl_kernel_npu/mem_cache/kv_cache_store.py::_store_kv_cache_prefix_valid_npu_kernel`
测试文件：`10shape_test/store_kv_cache_prefix_valid_npu_kernel.py`（自带 kernel 拷贝 + `BLOCK` autotune）

kernel 结构（纯 gather/scatter 拷贝）：
```
src_row=pid(0); block_id=pid(1); kv_id=pid(2)
batch_id=src_row // PREFIX_WIDTH ; row_in_batch=src_row % PREFIX_WIDTH
active = row_in_batch < load(commit_lens + batch_id)        # 标量
loc = load(loc_ptr + src_row, mask=active, other=0).to(int64)
offsets = block_id*BLOCK + arange(0,BLOCK)
kv_id==0: mask = active & (offsets < K_ROW_DIM); v=load(src_k + src_row*src_k_row_stride + offsets, mask, 0); store(dst_k + loc*dst_k_row_stride + offsets, v, mask)
else    : 同样对 V
```
算子 wrapper 里 **`block = 128` 写死**，grid = `(num_rows, cdiv(max(K,V)_ROW_DIM, block), 2)`。

---

## 1. 环境与构建

`source /home/y00953696/set_env.sh` 后 `bishengir-compile` = `/home/y00953696/AscendNPU-IR/build/bin/bishengir-compile`；
构建脚本 `/tmp/build_bishengir.sh`（AGENTS.md 的配方）；切到 master 分支后重建，二进制时间 16:02:25 ✓。

## 2. 基线（master 编译器的 profiler avg_time）

| 运行 | 10 个 shape 合计 |
|---|---:|
| 第 1 次（共享卡争抢，耗时 307s） | 55.50（丢弃） |
| **第 2 次（干净，59s）** | **23.28 µs** |

per-shape 与各 pipe（干净那次的头部）：

| shape | dur | vec | mte2 | mte3 | scalar | blocks |
|---|---:|---:|---:|---:|---:|---:|
| B1_W64_H8_D128_C64 | 3.45 | **3.19** | 0.44 | 0.35 | 2.91 | 56 |
| B4_W16_H8_D128_C8 | 3.22 | 2.95 | 0.23 | 0.18 | 2.75 | 56 |
| B1_W32_H8_D128_C32 | 3.17 | 2.90 | 0.23 | 0.17 | 2.71 | 56 |
| B1_W16_H8_D128_C16 | 2.32 | 2.05 | 0.21 | 0.16 | 1.84 | 32 |

**画像要点**：
- 拷贝本身确实走 DMA（最终 IR 里 `llvm.call @load_gm_to_ubuf_1d_half` + `@store_ubuf_to_gm_1d_half` 各一条），
  mte2/mte3 只有 0.44/0.35 µs；
- 但 **vec（3.19µs）与 scalar（2.91µs）才是大头**，而且 **vec+scalar > dur**（6.1 > 3.45）→ 这两者**互相重叠**，
  真正限制 dur 的是其中较大者；
- 换算一下：`B1_W64_H8_D128_C64` 共 64 行 × 1024 元素 × 2 个张量 = 131072 元素 / 3.45µs，
  按 64 lane 向量算 = 2048 条向量指令 / 6210 cycle ≈ **0.33 条/cycle** —— 远离向量单元吞吐上限
  ⇒ 这个 kernel 是**每 program 的延迟/固定开销受限**，不是容量受限。

## 3. 关键发现 ①（参数侧，收益最大）：算子把 `BLOCK` 写死 128，比 autotune 慢 2.4×

测试文件里的 autotune 是 `BLOCK ∈ {256,512,1024,2048,4096}`（本会话之前加的），而**算子 wrapper 里 `block = 128`**。

用诊断变体（`scripts/skv_b128_variant.py`，把配置列表固定成 `(128,)`，trace 前缀 `skv_b128_`，不进仓库）实测：

| | 合计 avg_time |
|---|---:|
| autotune（测试文件现状） | **23.28 µs** |
| **固定 `BLOCK=128`（= 算子现状）** | **36.71 µs**（干净复测） |

per-shape：

| shape | autotune | BLOCK=128 | Δ |
|---|---:|---:|---:|
| B1_W64_H8_D128_C64 | 3.45 | 5.07 | +47% |
| B4_W16_H8_D128_C8 | 3.22 | 5.31 | +65% |
| B1_W32_H8_D128_C32 | 3.17 | 4.37 | +38% |
| B1_W16_H8_D128_C16 | 2.32 | 3.38 | +46% |
| B1_W16_H32_D128_C16 | 2.27 | 4.76 | +110% |
| B1_W16_H8_D64_C16 | 2.24 | 3.55 | +59% |
| B1_W16_H1_D128_C16 | 2.17 | 2.24 | +3% |
| B1_W8_H8_D128_C8 | 1.75 | 3.24 | +85% |
| B1_W4_H8_D128_C4 | 1.46 | 3.15 | +115% |
| B1_W1_H8_D128_C1 | 1.24 | 1.64 | +32% |
| **SUM** | **23.28** | **36.71** | **+58%** |

（第一次测到 56.36 是共享卡争抢；**下面这张是干净复测**。方向与之前会话一致：36.07 → 23.80。）

⇒ **把算子的 `BLOCK` 从 128 改成"按 shape 选/autotune"（典型落到 4096），实测 −36.6%**（36.71 → 23.28）。
这属于"调算子参数"，和之前 add_rmsnorm 的 `COL_BLOCK_SIZE` 同一性质；但注意 `BLOCK` 影响 **grid 的第二维**
（`cdiv(row_dim, BLOCK)`），所以改它同时改了并行度分布，需要完整回归。

## 4. 关键发现 ②（编译器侧候选）：`other=0` 的填充是**死代码**，但它不是瓶颈

最终 IR 里有两个几乎相同的 outline 向量函数，都是**把整个 BLOCK 宽的 UB buffer 循环写 0**：

```llvm
llvm.func @..._outlined_vf_0(%arg0: !llvm.ptr<6>) attributes {hfusion.has_fill, ...} {
  %6 = "hivm_regbaseintrins.intr.hivm.vdups.z"(0.0f16, mask, 1) : -> vector<128xf16>
  ^bb1(%i): cond_br (%i < 4096)
    %11 = getelementptr %arg0[%i]
    "hivm_regbaseintrins.intr.hivm.vstsx1.v128f16"(%6, %11, ...)     // 每次写 128 lane
    %12 = add %i, 128 ; br ^bb1(%12)
}
```

即 `tl.load(..., mask, other=0)` 的 `other` 是用**向量 store 循环**把缓冲填 0（4096/128 = 32 次）。

**为什么说它是死代码**：同一个 kernel 里 `values` 的**唯一下游**是 `tl.store(..., mask=mask)`，**掩码与 load 完全相同**
⇒ 被填 0 的那些 lane 永远不会被读、也不会被写出去 ⇒ 填充无效果。

**但它看起来不是瓶颈**：autotune 偏好**更大**的 BLOCK（填充更大），却明显更快（§3）
⇒ 说明主导因素是**每 program 的固定开销/串行链**（与 §2 的 0.33 ops/cycle 一致），而不是这条填充。
所以编译器侧的候选先记为"可做但预期收益有限"，优先级低于 §3 的参数修正。

## 5. 产物

```
1013go/_store_kv_cache_prefix_valid_npu_kernel_go/
├── SUMMARY.md（本文）
├── scripts/skv_b128_variant.py     # 诊断：BLOCK 固定 128（不进仓库）
├── ir/                             # 一个 4096 宽配置的 .bcmlir/.ttir + 全量 IR
│                                   #   all_ir.mlir（57267 行）、final.mlir（277 行）
└── triton_cache*/                  # Triton 前端 IR（含各 shape/各 BLOCK 的 bcmlir）
```

## 6. 下一步

1. **复测 §3 里那个异常 shape**（`B1_W32_H8_D128_C32` 23.90µs），确认是不是争抢；然后给出**每个 shape 的最优 BLOCK** 表。
2. 把 §3 的结论落成可执行的建议：算子侧改用 autotune（或按 `max(K,V)_ROW_DIM` 选 BLOCK），并跑该算子的精度用例回归。
3. 编译器侧：先量化 §4 那条死填充到底占多少（例如做一个"去掉 other 填充"的诊断变体），再决定是否值得做消除。


---

## 7. 第二轮：两个诊断 —— 死填充 **无效**，参数修正 **几乎吃满全部收益**

### 7.1 诊断 A：去掉 `other=0`（= 消除那条死填充）→ **无效**

变体 `scripts/skv_noother_variant.py`（把 K/V 两个 load 的 `other=0` 去掉；保留 `loc` 那个 1 元素的
标量 load 的 `other=0`，因为它是单元素、不产生宽填充）：

| | 合计 |
|---|---:|
| autotune 基线 | 23.28 µs |
| **去掉 `other=0`** | **23.08 µs（−0.9%，噪声内）** |

⇒ §4 里那个"死填充"**确实是死代码，但删掉它没有收益** ⇒ **判为无效，不值得为它写编译器消除 pass**
（这与 §4 的推断一致：autotune 偏好更大 BLOCK = 更大填充却更快，说明填充不在关键路径上）。

### 7.2 诊断 B：`BLOCK` 固定 4096（不用 autotune）→ **几乎一样快**

变体 `scripts/skv_b4096_variant.py`（配置列表固定成 `(4096,)`）：

| 方案 | 合计 | vs autotune |
|---|---:|---:|
| autotune（per-shape 选） | 23.28 | — |
| **`BLOCK` 固定 4096** | **24.28** | **+4.3%** |
| `BLOCK` 固定 128（= 算子现状） | 36.71 | +57.6% |

per-shape（全表）：

| shape | autotune | 无 other | BLOCK=4096 | BLOCK=128 |
|---|---:|---:|---:|---:|
| B1_W64_H8_D128_C64 | 3.45 | 3.38 | 3.62 | 5.07 |
| B4_W16_H8_D128_C8 | 3.22 | 3.00 | 3.48 | 5.31 |
| B1_W32_H8_D128_C32 | 3.17 | 3.10 | 3.38 | 4.37 |
| B1_W16_H8_D128_C16 | 2.32 | 2.33 | 2.32 | 3.38 |
| B1_W16_H32_D128_C16 | 2.27 | 2.29 | 2.32 | 4.76 |
| B1_W16_H8_D64_C16 | 2.24 | 2.36 | 2.29 | 3.55 |
| B1_W16_H1_D128_C16 | 2.17 | 2.22 | 2.26 | 2.24 |
| B1_W8_H8_D128_C8 | 1.75 | 1.70 | 1.76 | 3.24 |
| B1_W4_H8_D128_C4 | 1.46 | 1.46 | 1.54 | 3.15 |
| B1_W1_H8_D128_C1 | 1.24 | 1.25 | 1.30 | 1.64 |
| **SUM** | **23.28** | **23.08** | **24.28** | **36.71** |
| | | −0.9% | **+4.3%** | +57.6% |

**固定 4096 在每个 shape 上都与 autotune 相差 ±6% 以内** ⇒ 算子不需要引入 autotune，
**一个固定值 4096 就能拿到几乎全部收益（−33.9%：36.71 → 24.28）**；引入 autotune 再多省那 4.3%（→ 23.28，−36.6%）。

### 7.3 结论

- **本 kernel 的收益几乎全在参数侧**：`BLOCK` 128 → 4096/autotune，**−34% ~ −37%**；
  改动只动 wrapper 的 `block` 值（连带 grid 第二维 `cdiv(row_dim, BLOCK)`），**不动 kernel 体**。
- **编译器侧目前没有找到可用杠杆**：唯一候选（删除 `other` 死填充）实测 **−0.9%**，判为无效。
  这也符合画像——kernel 是**每 program 的延迟/固定开销受限**（§2：0.33 向量指令/cycle），
  而 mte2/mte3 只占 0.44/0.35 µs、DMA 管道 79% 时间是空闲的。
- 附带记录：`BLOCK` 更大时 DMA 传输更大（更好）但并发更低，autotune 就是在找这个折中点；
  编译器改不了这个折中。


---

## 8. 第三轮：`BLOCK` 全量扫描 —— 固定 1024 就能拿到 −34.5%，autotune 再省 3.3%

用同一套诊断变体（`scripts/skv_b{128,1024,2048,4096}_variant.py`）把 `BLOCK` 分别固定，与测试文件里的
autotune（`{256..4096}`，由 key `K_ROW_DIM/V_ROW_DIM/PREFIX_WIDTH` 逐 shape 选）对比：

| shape | auto | 128 | 1024 | 2048 | 4096 | 该 shape 最佳 |
|---|---:|---:|---:|---:|---:|---|
| B1_W64_H8_D128_C64 | 3.45 | 5.07 | **3.29** | 3.87 | 3.62 | 1024 |
| B4_W16_H8_D128_C8 | 3.22 | 5.31 | **3.15** | 3.51 | 3.48 | 1024 |
| B1_W32_H8_D128_C32 | 3.17 | 4.37 | 3.19 | **3.17** | 3.38 | 2048 |
| B1_W16_H8_D128_C16 | 2.32 | 3.38 | **2.28** | 2.35 | 2.32 | 1024 |
| B1_W16_H32_D128_C16 | 2.27 | 4.76 | 3.35 | 3.07 | **2.32** | 4096 |
| B1_W16_H8_D64_C16 | 2.24 | 3.55 | 2.29 | 2.34 | **2.29** | 4096 |
| B1_W16_H1_D128_C16 | 2.17 | 2.24 | 2.21 | **2.19** | 2.26 | 2048 |
| B1_W8_H8_D128_C8 | 1.75 | 3.24 | **1.64** | 1.70 | 1.76 | 1024 |
| B1_W4_H8_D128_C4 | 1.46 | 3.15 | **1.42** | 1.49 | 1.54 | 1024 |
| B1_W1_H8_D128_C1 | 1.24 | 1.64 | **1.23** | 1.28 | 1.30 | 1024 |
| **SUM** | **23.28** | **36.71** | **24.05** | **24.97** | **24.28** | |

### 8.1 结论

1. **最佳单一固定值 = 1024**（24.05 µs），比算子现状（128 → 36.71）**快 −34.5%**；
   autotune（23.28）再省 3.3%。
2. **没有干净的"按 row_dim 选"规则**：`row_dim = H*D` 分别是 1024/128/4096/512…
   而各 shape 的最佳 BLOCK 是 1024/2048/4096/4096 → 与 row_dim、行数、grid 都有关 ⇒ 想要拿到最后 3.3% 就得 autotune。
3. **顺带发现 autotune 的选择有噪声**：有 5 个 shape 上"固定 1024"比 autotune **更快**
   （例如 B1_W64：3.29 vs 3.45；B1_W8：1.64 vs 1.75）——因为 autotune 的即时基准在共享卡上不稳。
   但总体上 autotune 仍胜（它给 B1_W16_H32 选到了 4096 = 2.27，而固定 1024 只有 3.35）。
   ⇒ 若算子要用 autotune，建议**加大基准重复次数**或直接用固定 1024。

### 8.2 给算子侧的落地建议（wrapper 一行级）

```python
# python/sgl_kernel_npu/sgl_kernel_npu/mem_cache/kv_cache_store.py
-    block = 128
+    block = 1024          # 或 min(4096, next_power_of_2(max(k_row_dim, v_row_dim)))；最优是 autotune
```
实测：**36.71 → 24.05 µs（−34.5%）**；改用 autotune 则 → 23.28 µs（−36.6%）。
注意 `block` 同时决定 grid 第二维 `cdiv(max(k,v)_row_dim, block)`，**kernel 体不用改**。

### 8.3 编译器侧（本轮复查）

`BLOCK=4096` 配置下的最终 IR，per-program 的串行链是：

```
prologue: GET.BLOCK.IDX + 4×SET.FLAG.IMM(event 0..3, scalar→MTE)
bb2:      2 组 GET.CTRL/SBITSET/SET.CTRL
bb6:      load commit_lens / compare / load loc / 地址计算
bb7/bb8:  SET/WAIT.FLAG → call outlined_vf_0（零填充）→ SET/WAIT.FLAG
bb9:      WAIT.FLAG.REG → call load_gm_to_ubuf → SET/WAIT.FLAG → BARRIER(5) → call store_ubuf_to_gm → SET.FLAG.REG
epilogue: 4×WAIT.FLAG.IMM + BARRIER(6)
```

- 这条链上唯一被测过的候选（零填充）**实测 −0.9% = 无效**（§7.1）；
- 其余（4 事件 prologue/epilogue、2 组 CTRL、BARRIER）属于同步/控制脚手架，
  **我无法证明它们可以安全去掉**（不了解 Ascend 的 event/pipe 语义），因此不动；
- 与 §2 的画像一致：mte2/mte3 只占 0.44/0.35µs（DMA 管道 ~79% 空闲），
  dur 由"每 program 的延迟 × 每核 program 数"决定，而这个是 **`BLOCK` 的折中**，编译器改不了。


---

## 9. 第四轮：**改到算子本体 + 端到端验证** —— 36.61 → 24.51 µs（−33.1%）

### 9.1 先解决了"算子无法验证"的问题

之前 `import sgl_kernel_npu` 报 `libsgl_kernel_npu.so: cannot open shared object file`。
发现站内已有编译好的扩展：`/home/y00953696/sglang-build/lib/libsgl_kernel_npu.so`，
于是把包内的 `lib` 指向它：

```bash
ln -sfn /home/y00953696/sglang-build/lib \
        python/sgl_kernel_npu/sgl_kernel_npu/lib
```

之后 `from sgl_kernel_npu.mem_cache.kv_cache_store import store_kv_cache_prefix_valid_npu_triton` **可以导入** ✓
（这是为了能测**算子本体**而加的一个符号链接，附加性改动，不需要时删掉即可）。

### 9.2 验证脚本（两个，都在 `scripts/`）

- `op_check.py <tag>`：对 10 个 perf shape，用测试文件的 `_make_case` + `store_kv_cache_prefix_valid_ref`
  做**逐元素精确比对**（`torch.equal`），再用 `profiling_test` 采集 profiler 数据（trace 前缀 `skv_op_<tag>_`）。
- `op_check_edge.py`：**dtype（fp16/fp32）× commit_lens 全 0** 的边界用例精确比对。

### 9.3 改前 / 改后（都是**算子本体**，不是测试文件的拷贝）

| | 正确性 | 10 shapes 合计 avg_time |
|---|---|---:|
| 算子现状（`block = 128`） | ALL OK | **36.61 µs** |
| **算子改后（`block = 1024`）** | **ALL OK** | **24.51 µs** |

⇒ **−33.1%**（36.61 → 24.51）。边界用例：`op_check_edge` → **ALL OK**（fp16/fp32 × 正常/零 commit）。
交叉印证：算子 `block=128` 实测 36.61 与测试文件里固定 128 的 36.71 **一致** ✓
（说明"测试文件里的 kernel 拷贝"确实等价于算子本体）。

### 9.4 实际改动（工作区，未提交）

```python
# python/sgl_kernel_npu/sgl_kernel_npu/mem_cache/kv_cache_store.py
-    block = 128
+    block = 1024
```
（附注释说明：本 kernel 是 per-program 延迟/固定开销受限，更宽的列块 = 更少更大的 DMA；
10shape 实测 1024 比 128 快 1.52×，2048/4096 略差，per-shape autotune 只再多 ~3%。）

### 9.5 结论

- **本 kernel 的收益全部来自这个参数**：`−33.1%`（已验证正确性）。
- **编译器侧无可落地杠杆**：唯一候选（删 `other` 死填充）实测 −0.9% = 无效（§7.1）；
  其余是同步/控制脚手架（4 事件 prologue/epilogue、2 组 CTRL、BARRIER），无法证明可安全去除；
  与画像一致——dur 由"每 program 延迟 × 每核 program 数"决定，而这个折中由 `BLOCK` 控制。
- 编译器分支 `opt/store-kv-prefix-valid`（基于 master）**始终无改动**，本任务的产出是算子参数改动。


---

## 10. 最终状态

### 10.1 交付

| | 正确性 | 10 shapes 合计 avg_time |
|---|---|---:|
| 算子改前（`block = 128`） | ALL OK | 36.61 µs |
| **算子改后（`block = 1024`）** | **ALL OK** | **24.51 / 24.48 µs（两次）** |
| Δ | | **−33.1%** |

- 正确性验证：`op_check.py`（10 shape 逐元素 `torch.equal`）+ `op_check_edge.py`（fp16/fp32 × 正常/零 commit）→ 全部 ALL OK。
- 复现：`python op_check.py <tag>`（需要 `TRITON_CACHE_DIR` 与 `PYTHONPATH` 指向源码树 + 装好扩展，见 §9.1）。
- **仓库里没有这个算子的自带测试**（`tests/python/sgl_kernel_npu/` 68 个测试文件里没有 `kv_cache_store`/`prefix_valid` 的用例）
  ⇒ 上面的 `op_check.py` 用的是 10shape 测试文件里的 `_make_case` + 参考实现，是目前能做到的最强验证。

### 10.2 工作区改动（均在 workspace，未提交）

1. `python/sgl_kernel_npu/sgl_kernel_npu/mem_cache/kv_cache_store.py`：`block = 128` → **`block = 1024`**（正文本优化）。
2. `python/sgl_kernel_npu/sgl_kernel_npu/lib` → 符号链接到 `/home/y00953696/sglang-build/lib`（**仅为了能 import 算子本体做验证**，附加性，不需要时删掉）。
3. 编译器分支 `opt/store-kv-prefix-valid`（基于 master）**自始至终没有代码改动**。

### 10.3 尝试记录汇总

| # | 尝试 | 结果 |
|---|---|---|
| 1 | 恒真/死填充：删掉 `other=0` 产生的那条"整缓冲向量写 0" | ❌ **−0.9%（无效）** —— 虽确认是死代码，但不在关键路径上 |
| 2 | `BLOCK` 全量扫描 128/1024/2048/4096 + autotune | ✅ **最佳固定值 1024**：−34.5%（36.71→24.05，测试文件拷贝）；autotune 再多 3.3% |
| 3 | **把 `block = 1024` 落进算子本体** | ✅ **−33.1%（36.61→24.48），正确性 ALL OK（含边界用例）** |
| — | 编译器侧其它候选（4 事件 prologue/epilogue、2 组 CTRL、BARRIER） | 未尝试：同步/控制脚手架，无法证明可安全去除 |

### 10.4 为什么编译器侧没有杠杆（结论）

最终 IR 显示 per-program 的串行链是
`prologue 同步 → CTRL → commit_lens/loc → 零填充 → DMA load → BARRIER → DMA store → epilogue 同步`，
而 profiler 表明 **mte2/mte3 只占 0.44/0.35 µs（DMA 管道约 79% 空闲）**、vec 只有 0.33 条指令/cycle
⇒ dur 由"**每 program 延迟 × 每核 program 数**"决定。这个乘积正是 `BLOCK` 控制的折中
（更大 BLOCK = 更少更大的 DMA 且固定开销摊薄，但并发更低），**编译器无法改变这个折中**，
所以本 kernel 的最优解落在算子参数上。


---

## 11. 用户提供的 `_ori`（原生写法）结果分析 + 一处**更正**

结果目录：`/home/y00953696/1012sgl/sgl-kernel-npu/result_dir`（10 个 `store_kv_cache_prefix_valid_ori_*`，
kernel 名 `_store_kv_cache_prefix_valid_npu_kernel`，每 shape 10 行 profiler 记录）。
合计 **36.78 µs** —— 与我在测试文件里固定 `BLOCK=128` 的实测 **36.71** 吻合 ✓（`_ori` 就是原生 `BLOCK=128` 配置）。

### 11.1 每个 shape 的 bound：**全部是 scalar-pipe bound**

| shape | 行数 | row_dim | blk | dur | aiv | vec% | **scal%** | mte2% | mte3% | 数据(MB) | 实测 GB/s | icache | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| B4_W16_H8_D128_C8 | 64 | 1024 | 56 | 5.29 | 5.03 | 6% | **83%** | 24% | 19% | 0.52 | 99 | 0 | scalar-bound |
| B1_W64_H8_D128_C64 | 64 | 1024 | 56 | 5.24 | 4.98 | 0% | **85%** | 39% | 38% | 0.52 | 100 | 0 | scalar-bound |
| B1_W16_H32_D128_C16 | 16 | 4096 | 56 | 4.87 | 4.61 | 0% | **79%** | 43% | 43% | 0.52 | 108 | 0 | scalar-bound |
| B1_W32_H8_D128_C32 | 32 | 1024 | 56 | 4.24 | 3.99 | 0% | **87%** | 27% | 26% | 0.26 | 62 | 0 | scalar-bound |
| B1_W16_H8_D128_C16 | 16 | 1024 | 56 | 3.47 | 3.21 | 0% | **89%** | 17% | 17% | 0.13 | 38 | 0 | scalar-bound |
| B1_W16_H8_D64_C16 | 16 | 512 | 56 | 3.43 | 3.18 | 0% | **92%** | 11% | 9% | 0.07 | 19 | 0 | scalar-bound |
| B1_W8_H8_D128_C8 | 8 | 1024 | 56 | 3.19 | 2.94 | 0% | **91%** | 12% | 10% | 0.07 | 21 | 0 | scalar-bound |
| B1_W4_H8_D128_C4 | 4 | 1024 | 56 | 3.14 | 2.88 | 0% | **94%** | 6% | 5% | 0.03 | 10 | 0 | scalar-bound |
| B1_W16_H1_D128_C16 | 16 | 128 | 32 | 2.26 | 2.00 | 0% | **91%** | 8% | 6% | 0.02 | 7 | 0 | scalar-bound |
| B1_W1_H8_D128_C1 | 1 | 1024 | 16 | 1.64 | 1.38 | 0% | **87%** | 11% | 8% | 0.01 | 5 | 0 | scalar-bound |

（百分比 = 该 pipe 时间 / `aiv_time`；数据量 = 行数×2(K,V)×row_dim×2B×2(读+写)；`∑pipe/aiv` = 1.06–1.65× ⇒ 各 pipe 有部分重叠。）

**要点**：
- **scalar 占 `aiv_time` 的 79–94%**，`vec` 几乎为 0（只有 B4_W16 有 6%）⇒ **瓶颈是每 program 的标量工作
  （控制/同步/地址运算），不是拷贝本身**；
- **实测带宽只有 5–108 GB/s**，而 mte2/mte3 加起来通常 < 0.5×`aiv` ⇒ **不是带宽受限**
  （对照：本机 32KB 级传输能到 ~855 GB/s、64KB 级 ~1.4 TB/s）；
- `icache_miss_rate = 0` ⇒ 也不是取指问题；
- `dur` 与 `aiv` 几乎相等（差距 <6%）⇒ 也**不是启动/排队开销**，就是 AIV 自己在忙。

⇒ 这解释了为什么"更大 BLOCK"能赢 33%：**program 数变少 ⇒ 总标量开销变少**
（B1_W4 这种只有 4 行却仍要 56 个 block × 每 block 一整套同步/控制，标量占比自然飙到 94%）。

### 11.2 更正：上一轮"死填充 −0.9%"的诊断**无效**

复查方式：把 noother 变体与基线的 bcmlir 分别编一遍，数最终 IR 里填充循环的标志
（`vstsx1.v128f16` 与带 `has_fill` 的 outline 函数）：

| 输入 | 最终 IR `vstsx1.v128f16` | 带 `has_fill` 的函数 |
|---|---:|---:|
| 基线（保留 `other=0`） | 2 | 2 |
| **noother 变体（去掉 `other=0`）** | **2** | **2** |

**两者完全相同** ⇒ 去掉源码里的 `other=0` **并不会让编译器省掉填充**（掩码外 lane 仍要初始化），
所以那次实验只证明了"改源码 `other` 不影响 codegen"，**并没有测到填充的成本** ✗。
§7.1 里"死填充无效"的说法**作废**，正确表述是：**填充的成本仍未测量**。
（不过对本 kernel 影响有限：`_ori`/`BLOCK=128` 时填充只有 1 次 128-lane 写，scalar 占比却仍是 79–94%
⇒ 主导标量开销的不是填充，而是同步/控制脚手架。）
