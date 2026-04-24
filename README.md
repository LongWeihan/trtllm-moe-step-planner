# TRT-LLM MoE Step Planner

Focused MoE-aware runtime step planning for TensorRT-LLM on a fixed `Qwen/Qwen1.5-MoE-A2.7B-Chat` path with real `INT4 weight-only` engine runs.

## What This Project Does

This repo implements a compact MoE runtime optimization slice:

- a minimal runtime resource model
- MoE pressure classes for `balanced`, `hot_expert`, and `hot_rank`
- a scheduler patch path around TensorRT-LLM runtime seams
- real TensorRT engine evaluation on a single `RTX 4060 Ti 16GB`

The planner is intentionally latency-first: it isolates high-pressure MoE requests to reduce tail latency and decode stragglers.

## Fixed Setup

- Model: `Qwen/Qwen1.5-MoE-A2.7B-Chat`
- Quantization: TensorRT-LLM `INT4 weight-only`
- Hardware: `RTX 4060 Ti 16GB`
- Main files:
  - [`scheduler/resource_model.py`](scheduler/resource_model.py)
  - [`scheduler/moe_microbatch_scheduler.py`](scheduler/moe_microbatch_scheduler.py)
  - [`scheduler/moe_pressure.py`](scheduler/moe_pressure.py)
  - [`scripts/run_patched.py`](scripts/run_patched.py)

## Key Results

### Headline metrics

| Workload | Baseline | Patched | Delta |
| --- | --- | --- | --- |
| Balanced control TTFT p90 | `0.0816s` | `0.0739s` | `-0.0077s` |
| Balanced control E2E p90 | `1.4878s` | `1.4748s` | `-0.0129s` |
| Balanced control throughput | `278.55 tok/s` | `281.47 tok/s` | `+2.92 tok/s` |
| Hot-Expert full TTFT p90 | `0.0733s` | `0.0111s` | `-0.0622s` |
| Hot-Expert full E2E p90 | `1.8348s` | `1.5983s` | `-0.2365s` |
| Hot-Expert full throughput | `310.44 tok/s` | `99.81 tok/s` | `-210.63 tok/s` |
| Hot-Rank TTFT p90 | `0.0737s` | `0.0131s` | `-0.0606s` |
| Hot-Rank E2E p90 | `1.9942s` | `1.7115s` | `-0.2827s` |

### What the numbers mean

- The planner preserves non-regression on the balanced control.
- On hot MoE workloads it dramatically lowers TTFT and tail latency.
- The tradeoff is throughput: this variant isolates pressure aggressively and often collapses the batch to size 1.

## Repository Layout

- [`scheduler/`](scheduler) - resource model, pressure model, scheduler patch, telemetry
- [`scripts/`](scripts) - workload generation, baseline runs, patched runs, result summarization
- [`workloads/`](workloads) - fixed MoE workloads
- [`results/`](results) - benchmark outputs and compare tables
- [`docs/`](docs) - build notes, result summaries, final report
- [`planning/`](planning) - execution contract, todolist, and project design docs

## Reproduce

Core flow:

```bash
python scripts/generate_workloads.py
python scripts/run_baseline.py ...
python scripts/run_patched.py ...
python scripts/summarize_results.py ...
```

Detailed notes:

- [`docs/final_report.md`](docs/final_report.md)
- [`docs/result_summary.md`](docs/result_summary.md)
- [`results/compare_tables/summary.md`](results/compare_tables/summary.md)

## Caveat

The quantitative benchmark path uses the real TensorRT engine backend with the project planner applied to batch composition. The runtime patch seam is real, but the final measured comparison is not a pure in-backend PyTorch quantitative benchmark.

## Project Notes

- Final report: [`docs/final_report.md`](docs/final_report.md)
- Result summary: [`docs/result_summary.md`](docs/result_summary.md)
- Planning docs: [`planning/`](planning)
