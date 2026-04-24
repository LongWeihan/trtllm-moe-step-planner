# TensorRT-LLM MoE Runtime 架构优化项目方案（24h 版本）

## 1. 项目标题

**MoE-Aware Runtime Scheduling Enhancement for TensorRT-LLM**

中文表达：

**面向 MoE 模型的 TensorRT-LLM Runtime 调度增强：基于 expert/rank pressure 的推理执行优化**

## 2. 24h 版项目定位

24h 版不是完整研究计划，而是一个**架构上自洽、实现上真实、结果上足够支撑面试**的最小闭环。

本项目在 24h 版本中固定为：

- 主模型：`Qwen/Qwen1.5-MoE-A2.7B-Chat`
- 主量化：TensorRT-LLM `INT4 weight-only`
- 主实现：TensorRT-LLM backend 内部 `resource_model.py + MicroBatchScheduler` patch
- 主结果：`default` vs `patched` 在 `Hot-Expert` / `Hot-Rank` workload 上的差异

## 3. 24h 版核心问题

MoE inference runtime 的核心问题在于：

> 默认 scheduler 主要感知 request size 和资源 fit，却不显式感知 expert/rank pressure，因此在高压请求叠加时，容易放大 decode tail 和 step variance。

24h 版只聚焦这一件事，不扩散到更大系统问题。  
但为了不让它显得只是“加了一层策略”，24h 版会把调度输入和调度输出抽象成一个最小 runtime planning contract。

## 4. 24h 版架构核心

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
  +-- Minimal Telemetry
  +-- MoE-aware MicroBatchScheduler (main patch consuming step plan)
  +-- Optional light Capacity Guard
  v

Benchmark Results
```

## 5. 24h 版唯一主创新

### 最小 Runtime Resource Model + MoE-aware MicroBatch Scheduling

核心思路：

1. 用 `request profile` 表达 request 的 runtime 特征
2. 用 `pressure budget` 和 `prefill quota` 表达当前 step 的预算
3. 产出一个 `step plan`
4. 让 `MicroBatchScheduler` 消费这个 step plan 执行 pressure-aware grouping

示意伪代码：

```text
request_profile = build_request_profile(req)
runtime_budget = build_runtime_budget(state)
step_plan = []

for req in remaining_candidates:
    if pressure(step_plan + req) <= runtime_budget.pressure_budget:
        step_plan.add(req)
    else:
        defer req
```

这让 24h 版不只是“散落在 scheduler 里的 heuristic”，而是一个最小但明确的 runtime planning interface。

## 6. 24h 版 pressure 模型

24h 版不追求完整 replay/live 体系，只需要一个够用的 pressure 定义。

可用形式：

```text
pressure_class in {balanced, hot_expert, hot_rank}
pressure_score = simple scalar or small vector
```

它的作用不是精确模拟真实 kernel time，而是：

- 为 scheduler 提供 batch selection 信号
- 构造 MoE-specific workload
- 支撑对比实验

同时，pressure 不直接散落在逻辑里，而是进入 `request profile -> runtime budget -> step plan` 这条链路。

## 7. 24h 版 workload 设计

### Balanced MoE

- 作为无 skew 对照组

### Hot-Expert

- 测试高压 expert 聚集时的 batch straggler

### Hot-Rank

- 测试 rank 维度偏斜时的 step variance

## 8. 24h 版评价标准

只看最关键的指标：

- TTFT p50 / p90 / p99
- TPOT p50 / p90 / p99
- step latency variance
- request tail latency

24h 版不追求“所有维度都漂亮”，只要求：

> 在至少一个高压 workload 上，patched 相比 default 有明确且可解释的改进。

## 9. 24h 版实验矩阵

采用两阶段最小矩阵：

### 第一阶段：信号探测

- `Balanced MoE`：`SMOKE`
- `Hot-Expert`：`PILOT`
- `Hot-Rank`：`PILOT`

目标：

- 找到哪个 hot workload 的信号更强
- 验证 patched 没有明显退化

### 第二阶段：集中火力

- 只保留一个 winning hot workload
- 做 `default vs patched` 的 `FULL` 对比

如果时间富余，再加：

3. 第二个 hot workload 的 `FULL`
4. `patched + capacity guard`

## 10. 24h 版结果表达

最成熟的表达不是：

- “所有场景大幅提升”

而是：

> 在 `Balanced MoE` workload 下，patched 不应明显回退；在 `Hot-Expert` 或 `Hot-Rank` workload 下，patched 应更好地控制 TPOT tail 或 step variance。

24h 版不要求三个 workload 都做完整长跑；真正要展示的是：

> 你知道如何用最小实验梯度快速定位信号，再把时间集中到最有解释力的高压 MoE 场景上。

## 11. 24h 版完成定义

24h 版完成，只需要满足：

1. 主模型真实端到端路径打通
2. `Balanced` `SMOKE` 完成
3. `Hot-Expert` 与 `Hot-Rank` `PILOT` 完成
4. 一个 winning hot workload 的 `FULL` 对比完成
5. 至少一个高压 workload 有收益
6. 有可讲清楚的 final report

## 12. 24h 版面试价值

即使是 24h 版，它仍然能展示：

1. 你理解 TensorRT-LLM runtime 内部路径
2. 你没有在外面做 toy router
3. 你知道 MoE inference 和 dense inference 的差异
4. 你能把 runtime 架构问题压缩成一个最小 `resource model + scheduler patch`
5. 你有真实模型、真实 benchmark 和真实结果
6. 你知道如何设计一个不会把工期拖爆的实验梯度

这已经足够作为一版高强度面试展示。 
