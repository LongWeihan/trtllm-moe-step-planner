# Resume Bullets

- Built a **MoE-first TensorRT-LLM runtime optimization** project on `Qwen/Qwen1.5-MoE-A2.7B-Chat`, completing the full **HF download -> TRT-LLM conversion -> INT4 weight-only engine build -> real inference** path on a single `RTX 4060 Ti 16GB`.
- Implemented a minimal **runtime resource model** for MoE scheduling in Python, introducing `RequestProfile`, `RuntimeBudget`, and `StepPlan` so scheduler decisions consumed explicit pressure and token budgets rather than ad hoc heuristics.
- Patched the TRT-LLM 1.2.1 internal scheduler seam around **`BindMicroBatchScheduler`** and request conversion to carry MoE pressure metadata into scheduling decisions.
- Designed three **MoE-specific workloads** (`Balanced`, `Hot-Expert`, `Hot-Rank`) and benchmarked real TensorRT engine runs, showing no material regression on the balanced control while reducing hot-workload tail latency.
- On a 24-request **Hot-Expert** full compare, improved **TTFT p90 from `0.0733s` to `0.0111s`** and **E2E p90 from `1.8348s` to `1.5983s`**, while identifying the next-step tradeoff of throughput loss from conservative pressure isolation.
