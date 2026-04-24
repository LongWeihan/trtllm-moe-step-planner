# TensorRT-LLM MoE-First Runtime 项目落地实施计划（24h 版本）

## 1. 这份计划的目标

这不是完整版项目计划，而是 **24 小时面试冲刺版**。

目标不是把所有研究细节跑满，而是把下面这条主链路跑通：

> `Qwen/Qwen1.5-MoE-A2.7B-Chat + TensorRT-LLM INT4 weight-only + 最小 runtime resource model + 一个真实 scheduler patch + 三个 MoE-specific workload + 一组最小实验梯度`

只要这条主链路跑通，项目在面试中的强度就够。

## 2. 24h 版为什么仍然足够强

面试强度主要来自：

1. 真实 TensorRT-LLM backend
2. 真实 MoE 模型
3. 真实 scheduler patch
4. 真实 MoE-specific workload
5. 真实 baseline vs patched 结果

而不是来自：

- 跑很多 baseline 变体
- 写很多文档
- 做很多 optional 分支
- 做 replay 深化或 CUDA 扩展

## 3. 固定技术路线

### 主模型

`Qwen/Qwen1.5-MoE-A2.7B-Chat`

### 主量化

TensorRT-LLM `INT4 weight-only`

### 主 patch

`MoE-aware MicroBatchScheduler`

### 最小架构增强

新增一个最小的 `resource_model.py`，不只是给 request 打 pressure label，而是显式定义：

- `request profile`
- `pressure budget`
- `prefill quota`
- `step plan`

这样 24h 版就不是单纯“加一层 heuristic”，而是把调度决策提升成一个最小 runtime resource model。

### 可选补充

轻量 `CapacityScheduler` guard  
只有在主 patch 很顺利的情况下才做。

## 4. 固定 workload

只保留这 3 个：

### Balanced MoE

- 用于证明 patched 不应明显退化

### Hot-Expert

- 用于观察高压 expert 叠加导致的 TPOT tail

### Hot-Rank

- 用于观察 rank hotspot 导致的 step variance

## 5. 最小实验梯度

24h 版不把 3 个 workload 都跑成完整 benchmark，而是做成下面这个梯度：

### `Balanced MoE` -> `SMOKE`

- 只验证 patched 不明显退化

### `Hot-Expert` -> `PILOT`

- 短跑一轮 `default vs patched`

### `Hot-Rank` -> `PILOT`

- 短跑一轮 `default vs patched`

### winning hot workload -> `FULL`

- 在 `Hot-Expert` 和 `Hot-Rank` 里选一个信号更强的
- 只对这个 workload 做完整结果展示

## 6. 24h 时间切分

## 0-4 小时：安装与下载并行

必须完成：

- TensorRT-LLM 安装
- 主模型下载

注意：

- 安装与下载并行启动
- 不允许串行等待一个结束再做另一个

## 4-9 小时：build 与控制面并行

必须完成：

- conversion
- INT4 WO build
- codepath 定位
- 最小 runtime resource model
- 最小 telemetry
- pressure classes
- 3 个 workload 骨架

注意：

- build 跑起来后，马上切去做 `resource_model.py` 和 workload 脚本

## 9-15 小时：pilot 与 patch 并行

必须完成：

- 单请求 generate
- `Balanced` `SMOKE`
- `Hot-Expert` `PILOT`
- `Hot-Rank` `PILOT`
- `MoE-aware MicroBatchScheduler v1`

优先实现：

- decode-first
- pressure-aware grouping
- step-plan consumption

不要一开始就做复杂 admission 系统。

## 15-20 小时：winning workload FULL

必须完成：

- winning workload 的 `FULL` 对比
- `Balanced` 不明显回退证据
- 汇总关键结果

## 20-24 小时：交付包装与补一轮短确认

必须完成：

- final report
- interview talking points
- resume bullets
- 必要时补一轮短确认 run

## 7. 24h 版成功标准

下面这些全部做到，就算成功：

1. 真实主模型路径跑通
2. 真实 patch 跑通
3. `Balanced` `SMOKE` 完成
4. `Hot-Expert` 与 `Hot-Rank` `PILOT` 完成
5. 至少一个 `Hot-Expert` 或 `Hot-Rank` workload 上，patched 有可解释收益
6. 有 final report 和面试材料

## 8. 24h 版必须砍掉的内容

为了守住 24 小时，必须砍掉：

- 多 GPU
- disaggregated serving
- replay 深化
- CUDA extension
- 大量 baseline 穷举
- 大量文档拆分
- dense 正式实验

## 9. 24h 版真实风险

### 最大风险 1：INT4 WO build 卡住

应对：

- 优先确保 conversion / build 路径打通
- 不要在 build 之前开始写大量 patch

### 最大风险 2：patch 写太复杂

应对：

- 先做最小版本
- 优先保 `MicroBatchScheduler`

### 最大风险 3：实验矩阵过大

应对：

- 只跑 `SMOKE -> PILOT -> FULL` 梯度
- 不把 3 个 workload 都扩成 `FULL`

## 10. 面试叙事

24h 版的面试讲法应该是：

> 我在 24 小时冲刺里，没有去做泛化 serving 系统，而是直接把问题限定在 MoE inference runtime 上。我基于 TensorRT-LLM PyTorch backend，先定义了一个最小 runtime resource model，把 request profile、pressure budget 和 prefill quota 显式化，再让 `MicroBatchScheduler` 消费这个 step plan 做 pressure-aware scheduling。实验上我没有盲目把所有 workload 都跑满，而是用 `Balanced SMOKE + Hot-Expert/Hot-Rank PILOT + winning workload FULL` 的梯度，把时间集中到最有信号的高压 MoE 场景上，最终在真实的 `Qwen/Qwen1.5-MoE-A2.7B-Chat` INT4 WO 路径上展示 tail latency 或 step variance 的改善。

## 11. 最后一条原则

24h 版的本质不是“做小”，而是：

> **把最能证明项目价值的那根主梁先立起来。**
