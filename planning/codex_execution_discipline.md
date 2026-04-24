# Codex Execution Discipline (24h Version)

## 1. 总原则

24h 版不是“完整研究版自动推进”，而是：

> **在 24 小时内，用最小但真实的 TensorRT-LLM + MoE 模型 + runtime patch + benchmark 闭环，交付一个面试强度不变、并且比“单纯加策略”更像 runtime architecture enhancement 的项目切片。**

## 2. 时间预算纪律

Codex 必须按下面时间盒推进：

### 0-4 小时

- 环境
- TensorRT-LLM 安装
- 主模型下载

说明：

- 安装和下载必须并行启动，不允许串行等待

### 4-9 小时

- conversion / build
- codepath
- 最小 telemetry 骨架
- 最小 `resource_model.py`
- pressure classes
- workload 生成骨架

说明：

- conversion / build 跑起来后，Codex 必须立刻切去做控制面代码，不允许等待 build 结束

### 9-15 小时

- 单请求 generate
- 最小 baseline 梯度
- `MoE-aware MicroBatchScheduler` 主 patch

说明：

- baseline pilot 跑的时候，Codex 必须并行实现 patch
- 不允许先把 baseline 全矩阵跑完再开始写 patch

### 15-20 小时

- winning workload 的 `FULL` 对比
- baseline vs patched 汇总
- 结果汇总

### 20-24 小时

- final report
- interview talking points
- resume bullets
- 如需要，补一轮短确认 run

## 3. 单一事实来源

24h 冲刺时，优先级如下：

1. `24h_version/project_bootstrap_contract.md`
2. `24h_version/todolist.md`
3. `24h_version/trtllm_moe_runtime_4060ti_implementation_plan_moe_first.md`
4. `24h_version/trtllm_moe_runtime_architecture_optimization_plan.md`

## 4. 不可漂移项

Codex 不得擅自：

- 把主模型换成 dense
- 把主线变成 kernel 项目
- 把主线变成外部 router
- 加入多 GPU / disaggregated serving
- 为了“看起来更完整”而把 24h 变成 40h

Codex 也不得把 24h 版做成“只有 if/else heuristic，没有最小资源模型”的纯技巧堆砌版本。

## 5. 24h 版必须优先的事情

优先级永远是：

1. **真实主模型路径**
2. **最小 runtime resource model**
3. **真实 TensorRT-LLM patch**
4. **最小实验梯度**
5. **结果和文档**

优先级低于以上的内容：

- 多 baseline 变体
- replay 深化
- 额外 workload
- 可选扩展

## 6. 阻塞处理纪律

### 环境 / build 问题

同一个问题最多尝试 **90 分钟**。  
超过后必须：

- 记录 blocker
- 收口已有尝试
- 选择次优但仍保强度的 fallback

### benchmark / OOM 问题

优先缩：

- batch
- sequence length
- 并发
- run depth（先 `SMOKE` / `PILOT`，最后才 `FULL`）

不允许因为 OOM 就换掉主模型。

### patch 问题

如果 `MicroBatchScheduler` patch 难度高于预期：

- 先做最小版本
- 保证 `resource_model.py + patched path` 可运行
- 不要贪多做复杂 v2/v3

## 7. 并行执行纪律

只要出现下面任意长跑任务，Codex 就必须切去做可以脱钩的控制面工作：

- 模型下载
- conversion / build
- baseline run
- patched benchmark

优先并行的控制面工作：

1. 代码路径笔记
2. `resource_model.py`
3. workload 生成脚本
4. 文档骨架
5. 结果汇总脚本

禁止行为：

- 开着长跑命令原地等待
- 为了“保持顺序整洁”放弃可并行项

## 8. 实验收缩纪律

24h 版实验必须按下面梯度执行：

1. `Balanced MoE` 只做 `SMOKE`
2. `Hot-Expert` 做 `PILOT`
3. `Hot-Rank` 做 `PILOT`
4. 选择信号更强的 hot workload 做 `FULL`

只有在以下条件同时满足时，才允许扩展第二个 hot workload 的 `FULL`：

- winning workload 已经拿到清晰收益
- 还有剩余时间
- 文档主线已经成型

## 9. 证据纪律

24h 冲刺里，每个关键节点必须留下证据：

### 环境

- 版本
- 安装命令

### 模型

- 下载路径
- conversion/build 命令
- 生成结果

### baseline

- workload
- 原始结果
- `SMOKE / PILOT / FULL` 标记
- 汇总表

### patch

- `resource_model.py`
- 修改文件
- patched 结果

## 10. 完成纪律

以下情况不得宣称“完成”：

- 只完成 synthetic 压力测试
- 没有真实主模型对比
- 只有 patch 没有 benchmark
- 只有 benchmark 没有最终文档

24h 冲刺只有在下面全部满足时才算完成：

1. 主模型真实路径打通
2. `resource_model.py` 完成
3. `Balanced MoE` 的 `SMOKE` 完成
4. `Hot-Expert` 与 `Hot-Rank` 的 `PILOT` 完成
5. 至少一个高压 workload 的 `FULL` 对比完成并出现改进
6. 三份最终文档完成

## 11. 汇报纪律

Codex 对用户汇报时只说 4 件事：

1. 当前在哪个任务
2. 是否完成
3. 关键证据是什么
4. 是否存在 blocker

避免空话。
