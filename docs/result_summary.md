# Result Summary

Machine-readable tables:

- [results/compare_tables/summary.json](C:/26spring/nv项目/trtllm-moe-runtime-exp/results/compare_tables/summary.json)
- [results/compare_tables/summary.md](C:/26spring/nv项目/trtllm-moe-runtime-exp/results/compare_tables/summary.md)

## Headline

The project met the 24h objective:

- real Qwen MoE model path ran end to end
- a minimal TRT-LLM runtime resource model was implemented
- a real internal scheduler patch was written
- at least one hot MoE workload showed a clear, explainable latency win

## Balanced MoE

Control result:

- no obvious regression
- patched path slightly improved latency and step variance
- throughput stayed effectively unchanged

Key deltas:

- TTFT p90: `-0.0077s`
- E2E p90: `-0.0129s`
- TPOT p90: `-0.0001s`
- step latency std: `-46.83ms`
- throughput: `+2.92 tok/s`

## Hot-Expert pilot

This became the winning workload because the latency signal was strong and aligned directly with the MoE pressure story.

12-request pilot deltas:

- TTFT p90: `-0.0635s`
- E2E p90: `-0.2927s`
- TPOT p90: `-0.0017s`
- step latency std: `-46.10ms`
- throughput: `-197.46 tok/s`

Interpretation:

- the pressure budget avoided stacking hot requests in the same step
- tail latency improved
- throughput fell because the planner mostly serialized hot requests

## Hot-Rank pilot

12-request pilot deltas:

- TTFT p90: `-0.0606s`
- E2E p90: `-0.2827s`
- TPOT p90: `-0.0015s`
- step latency std: `-127.98ms`
- throughput: `-177.88 tok/s`

Interpretation:

- the same architecture also works for rank-skew pressure
- but `Hot-Expert` was kept as the final story because it is simpler to explain end to end

## Hot-Expert full compare

24-request `FULL` compare:

- baseline: [results/baseline/hot_expert_full24.json](C:/26spring/nv项目/trtllm-moe-runtime-exp/results/baseline/hot_expert_full24.json)
- patched: [results/patched/hot_expert_full24.json](C:/26spring/nv项目/trtllm-moe-runtime-exp/results/patched/hot_expert_full24.json)
- telemetry: [results/patched/hot_expert_full24_telemetry.jsonl](C:/26spring/nv项目/trtllm-moe-runtime-exp/results/patched/hot_expert_full24_telemetry.jsonl)

Key deltas:

- TTFT p90: `0.0733s -> 0.0111s`
- E2E p90: `1.8348s -> 1.5983s`
- TPOT p90: `0.0114s -> 0.0101s`
- step latency std: `158.69ms -> 157.35ms`
- throughput: `310.44 tok/s -> 99.81 tok/s`

## Bottom line

This version of the planner is a **latency-first MoE pressure isolation policy**.

It succeeds at:

- lowering TTFT and tail latency under hot MoE workloads
- reducing per-step pressure stacking
- preserving non-regression on the balanced control

It does **not** yet solve the throughput tradeoff. That is the next iteration target.
