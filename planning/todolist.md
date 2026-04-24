# TODO List (24h Version)

## 1. 使用方式

这是 **24 小时冲刺版**执行清单。  
只保留真正影响面试强度的主链路任务。

状态：

- `TODO`
- `IN_PROGRESS`
- `DONE`
- `BLOCKED`

## 2. 固定约束

- 主模型：`Qwen/Qwen1.5-MoE-A2.7B-Chat`
- 主量化：TensorRT-LLM `INT4 weight-only`
- 主实现：TensorRT-LLM backend 内部 `resource_model.py + MicroBatchScheduler` patch
- 主 workload：`Balanced MoE` / `Hot-Expert` / `Hot-Rank`
- 主目标：24 小时内拿到真实 baseline vs patched 对比

## 3. 实验深度定义

- `SMOKE`：只验证路径、指标采集和明显回退
- `PILOT`：短跑一轮 `default vs patched`，用来判断信号方向
- `FULL`：对一个 winning workload 做完整对比，作为最终展示结果

## 4. 并行窗口

### P1 安装和下载并行

- `A02 TensorRT-LLM 安装`
- `A03 主模型下载`

### P2 build 和控制面代码并行

- `A04 conversion + build`
- `B00-B04 codepath / telemetry / resource model / pressure / workload`

### P3 baseline pilot 和 patch 实现并行

- `B05 最小 baseline 梯度`
- `C00 主 patch`

### P4 full benchmark 和文档骨架并行

- `D00 winning workload full compare`
- `E00-E02` 先写结构和占位

## 5. Phase A - 真实路径打通

### A00 建立工作区与最小目录

- 状态：`TODO`
- 依赖：无
- 完成标准：
  - WSL ext4 工作区建立
  - 最小目录结构建立
- 产物：
  - `docs/run_log.md`
  - `docs/blockers.md`

### A01 采集环境指纹

- 状态：`TODO`
- 依赖：A00
- 完成标准：
  - GPU / CUDA / Python / WSL / `nsys` 记录完成
- 产物：
  - `docs/env_fingerprint.md`

### A02 安装 TensorRT-LLM

- 状态：`TODO`
- 依赖：A01
- 完成标准：
  - `from tensorrt_llm import LLM` 成功
- 产物：
  - `docs/install_log.md`

### A03 下载主模型

- 状态：`TODO`
- 依赖：A01
- 完成标准：
  - `Qwen/Qwen1.5-MoE-A2.7B-Chat` 下载完成
- 产物：
  - `docs/qwen15_download.md`

### A04 完成 conversion + INT4 WO build

- 状态：`TODO`
- 依赖：A02, A03
- 完成标准：
  - conversion 成功
  - INT4 WO 构建成功
- 产物：
  - `docs/qwen15_build_log.md`
  - `artifacts/qwen15_moe_int4wo/`

### A05 完成真实主模型 sanity

- 状态：`TODO`
- 依赖：A04
- 完成标准：
  - 单请求 generate 成功
  - 小 benchmark sanity 成功
- 产物：
  - `results/sanity/`
  - `docs/qwen15_sanity.md`

## 6. Phase B - 最小 benchmark 基础设施

### B00 梳理 scheduler 代码路径

- 状态：`TODO`
- 依赖：A02
- 完成标准：
  - `PyExecutor` / `MicroBatchScheduler` 接入点明确
- 产物：
  - `docs/codepath_notes.md`

### B01 加入最小 telemetry

- 状态：`TODO`
- 依赖：B00
- 完成标准：
  - 能输出 step latency / batch composition / waiting info
- 产物：
  - `scheduler/telemetry.py`
  - `docs/minimal_telemetry.md`

### B02 实现最小 resource model

- 状态：`TODO`
- 依赖：B00
- 完成标准：
  - 至少定义 `request profile` / `pressure budget` / `prefill quota` / `step plan`
- 产物：
  - `scheduler/resource_model.py`
  - `docs/resource_model.md`

### B03 实现 pressure classes

- 状态：`TODO`
- 依赖：B02
- 完成标准：
  - 至少支持 `balanced` / `hot_expert` / `hot_rank`
- 产物：
  - `scheduler/moe_pressure.py`
  - `docs/pressure_classes.md`

### B04 生成 3 个 workload

- 状态：`TODO`
- 依赖：B03
- 完成标准：
  - 三个 JSONL workload 可复现
- 产物：
  - `workloads/balanced_moe.jsonl`
  - `workloads/hot_expert.jsonl`
  - `workloads/hot_rank.jsonl`

### B05 跑最小 baseline 梯度

- 状态：`TODO`
- 依赖：A05, B04
- 完成标准：
  - `Balanced MoE` 完成 `SMOKE`
  - `Hot-Expert` 完成 `PILOT`
  - `Hot-Rank` 完成 `PILOT`
  - 已选出一个 winning workload 进入 `FULL`
- 产物：
  - `results/baseline/`
  - `docs/baseline_gate.md`

## 7. Phase C - 主 patch

### C00 实现 MoE-aware MicroBatchScheduler v1

- 状态：`TODO`
- 依赖：B00, B02, B03
- 完成标准：
  - decode-first + pressure-aware grouping 跑通
  - scheduler 消费 `step plan` 而不是只有散落 heuristic
- 产物：
  - `scheduler/moe_microbatch_scheduler.py`
  - `docs/scheduler_v1.md`

### C01 可选轻量 admission guard

- 状态：`TODO`
- 依赖：C00
- 完成标准：
  - 若实现，则 guard 可运行
- 产物：
  - `scheduler/optional_capacity_guard.py`
  - `docs/capacity_guard.md`

## 8. Phase D - 结果

### D00 跑 winning workload 的 FULL 对比

- 状态：`TODO`
- 依赖：C00, A05, B05
- 完成标准：
  - 对 winning workload 完成 `default vs patched` 的 `FULL` 对比
  - `Balanced MoE` 至少保留 `SMOKE` 不明显回退证据
- 产物：
  - `results/patched/`

### D01 形成 baseline vs patched 对比

- 状态：`TODO`
- 依赖：B05, D00
- 完成标准：
  - 至少一个 `Hot-Expert` 或 `Hot-Rank` workload 出现可解释收益
  - 次优 hot workload 至少保留一轮 `PILOT` 结果
- 产物：
  - `docs/result_summary.md`
  - `results/compare_tables/`

## 9. Phase E - 面试交付

### E00 完成 final report

- 状态：`TODO`
- 依赖：D01
- 完成标准：
  - 方法、workload、结果、局限性写清楚
- 产物：
  - `docs/final_report.md`

### E01 完成 interview talking points

- 状态：`TODO`
- 依赖：E00
- 完成标准：
  - 60 秒 / 3 分钟 / 深挖问答齐全
- 产物：
  - `docs/interview_talking_points.md`

### E02 完成 resume bullets

- 状态：`TODO`
- 依赖：E00
- 完成标准：
  - 3-5 条 bullet，和真实结果一致
- 产物：
  - `docs/resume_bullets.md`

## 10. Stretch Only

以下任务只有在主链路全部完成后才允许做：

- replay provider 深化
- 强 baseline 扩展
- Mixed Burst workload
- Repeated-Prefix under MoE Pressure
- CUDA telemetry extension

## 11. 24h 完成定义

只有下面全部满足，24h 冲刺才算完成：

1. A05 完成
2. B02 完成
3. B05 完成
4. C00 完成
5. D01 完成
6. E00-E02 完成
