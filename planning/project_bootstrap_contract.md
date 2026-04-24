# Project Bootstrap Contract (24h Version)

## 1. 文件目的

这份文件是 **24 小时冲刺版启动合同**。  
目标不是完成完整版研究项目，而是在 24 小时内交付一个**面试强度不变的 vertical slice**。

## 2. 固定目标

在 24 小时内，完成一个基于 TensorRT-LLM PyTorch backend 的 **MoE-aware runtime scheduler enhancement** 最小闭环，并在固定主模型

**`Qwen/Qwen1.5-MoE-A2.7B-Chat + TensorRT-LLM INT4 weight-only`**

上完成真实端到端验证。

最终必须证明：

1. 真的改了 TensorRT-LLM 内部 scheduler 路径
2. 真的用了真实 MoE 模型
3. 真的跑了 MoE-specific workload
4. 至少在一个高压 workload 上拿到可解释收益
5. 调度决策不是散落 heuristic，而是消费一个最小 runtime resource model

## 3. 不可变更项

- 主模型：`Qwen/Qwen1.5-MoE-A2.7B-Chat`
- 主量化：TensorRT-LLM `INT4 weight-only`
- 主环境：WSL2 Ubuntu 24.04
- 主实现边界：TensorRT-LLM backend 内部 `MicroBatchScheduler` 为主，辅以最小 `resource_model.py`，`CapacityScheduler` 轻量 guard 为辅
- 主故事线：MoE-first runtime scheduler enhancement

## 4. 24h 版明确非目标

以下内容不进入 24h 主线：

1. 多 GPU / EP / WideEP / DWDP
2. disaggregated serving / KV transfer
3. CUDA extension
4. 大量强 baseline 穷举
5. 大规模 replay-vs-live 深度分析
6. dense 模型正式实验

## 5. 固定交付范围

24h 冲刺只要求完成以下闭环：

1. TensorRT-LLM 环境打通
2. `Qwen/Qwen1.5-MoE-A2.7B-Chat` conversion + INT4 WO build
3. 单请求 generate + 小规模 sanity benchmark
4. 最小 telemetry
5. 三类 workload
   - `Balanced MoE`
   - `Hot-Expert`
   - `Hot-Rank`
6. 一个主 patch
   - `MoE-aware MicroBatchScheduler`
   - 最小 `resource_model.py`
7. 一组最小实验梯度
   - `Balanced MoE` `SMOKE`
   - `Hot-Expert` `PILOT`
   - `Hot-Rank` `PILOT`
   - 一个 winning hot workload 的 `FULL` 对比
8. 一份 final report
9. 一份面试讲稿
10. 一份简历 bullet

## 6. 24h 压缩契约

为了把时间压回 24h 附近，24h 版采用两条强约束：

1. **长跑任务必须和控制面工作并行**
2. **实验必须按 `SMOKE -> PILOT -> FULL` 梯度推进**

不允许一上来就把三个 workload 都跑成完整 benchmark。

## 7. 24h 版固定架构

```text
Workloads
  +-- Balanced MoE
  +-- Hot-Expert
  +-- Hot-Rank
  v

TensorRT-LLM PyTorch Backend
  +-- PyExecutor
  +-- Runtime Resource Model
  |     +-- request profile
  |     +-- pressure budget
  |     +-- prefill quota
  |     +-- step plan
  +-- CapacityScheduler (optional light guard)
  +-- MicroBatchScheduler (main patch consuming step plan)
  +-- Minimal Telemetry
  v

Results
  +-- baseline
  +-- patched
  +-- final report
```

## 8. 24h 版仓库最小结构

```text
trtllm-moe-runtime-exp/
  docs/
    env_fingerprint.md
    install_log.md
    qwen15_build_log.md
    codepath_notes.md
    final_report.md
    interview_talking_points.md
    resume_bullets.md
    run_log.md
    blockers.md
  scripts/
    sanity_backend.py
    generate_workloads.py
    run_baseline.py
    run_patched.py
  scheduler/
    telemetry.py
    moe_pressure.py
    resource_model.py
    moe_microbatch_scheduler.py
    optional_capacity_guard.py
  workloads/
    balanced_moe.jsonl
    hot_expert.jsonl
    hot_rank.jsonl
  artifacts/
    qwen15_moe_int4wo/
  results/
    sanity/
    baseline/
    patched/
```

## 9. 24h 版指标契约

必须保留的核心指标：

- TTFT p50 / p90 / p99
- TPOT p50 / p90 / p99
- step latency variance
- request tail latency

可保留但不是必须：

- KV block usage
- GPU utilization

## 10. 24h 版工作负载契约

只允许 3 个主 workload：

### Balanced MoE

- pressure 分布均衡
- 用于证明 patched 不应明显回退

### Hot-Expert

- 请求集中打到同一组 experts
- 用于观察 batch straggler 和 TPOT tail

### Hot-Rank

- rank 维度偏斜明显
- 用于观察 step variance

## 11. 24h 版实验深度契约

### `SMOKE`

- 路径验证
- 指标采集验证
- 用于 `Balanced MoE`

### `PILOT`

- 短跑一轮 `default vs patched`
- 用于 `Hot-Expert` 与 `Hot-Rank`
- 目标是决定 winning workload

### `FULL`

- 只对一个 winning hot workload 做完整对比
- 作为最终展示证据

## 12. 24h 版并行契约

### P1

- `A02 安装`
- `A03 下载`

### P2

- `A04 build`
- `B00-B04` 控制面代码与脚本

### P3

- `B05` baseline pilot
- `C00` 主 patch

### P4

- `D00` winning workload `FULL`
- `E00-E02` 文档骨架和结果填充

## 13. 24h 版里程碑

### M1

- TensorRT-LLM 安装成功
- 主模型 build 成功
- 单请求 generate 成功

### M2

- 三个 workload 就绪
- `Balanced` `SMOKE`
- `Hot-Expert` / `Hot-Rank` `PILOT`
- winning workload 选定

### M3

- patch 跑通
- winning workload `FULL` 对比结果就绪

### M4

- final report
- interview talking points
- resume bullets

## 14. 24h 版完成定义

只有满足下面全部条件，24h 项目才算完成：

1. `Qwen/Qwen1.5-MoE-A2.7B-Chat + TRT-LLM INT4 weight-only` 路径跑通
2. `Balanced MoE` `SMOKE` 完成
3. `Hot-Expert` 与 `Hot-Rank` `PILOT` 完成
4. 一个 winning hot workload 的 `FULL` 对比完成
5. 至少一个 `Hot-Expert` 或 `Hot-Rank` workload 上出现可解释收益
6. 最终文档三件套完成
