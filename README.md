# TRT-LLM MoE Step Planner

Pressure-aware step formation for Mixture-of-Experts inference on TensorRT-LLM.

## Overview

`TRT-LLM MoE Step Planner` is a focused runtime optimization project for MoE inference. It addresses a specific systems problem: default batch formation is generally aware of fit, but not of MoE-specific contention structure.

For dense models, token count, prompt length, and memory pressure are often sufficient scheduling signals. For MoE models, those signals are incomplete. Two requests with similar lengths can still have very different runtime impact when routing skew creates:

- expert hotspots
- rank hotspots
- decode tail amplification
- batch stragglers
- larger per-step latency variance

This project introduces a narrow planning layer that makes MoE pressure explicit in the runtime decision path and uses it to form safer execution steps.

## Scope

This repository is intentionally narrow in scope.

It does **not**:

- modify model weights
- replace TensorRT-LLM kernels
- introduce an external serving stack

It **does**:

1. estimate request-level pressure
2. construct an explicit runtime budget
3. build a pressure-aware step plan
4. validate the plan on the real TensorRT engine path

The resulting policy is intentionally latency-oriented. Its purpose is to reduce pressure collisions inside a step, even if that sometimes reduces effective batch size.

## Architecture

```text
request metadata
    |
    v
pressure model
    |
    v
request profile
    |
    v
runtime budget
    |
    v
step plan
    |
    v
TensorRT-LLM engine execution
    |
    +--> telemetry
    +--> result metrics
```

The key engineering decision is that pressure is treated as a schedulable runtime resource rather than as an explanatory label attached after execution.

## Design Principles

### Keep the integration narrow

The project stays close to TensorRT-LLM runtime behavior. The change is about step planning, not about rebuilding the executor or competing with optimized kernels.

### Make scheduling inputs explicit

The planner is structured as:

`request metadata -> runtime model -> planning decision -> execution`

This makes the scheduling logic inspectable and composable.

### Optimize the failure mode that matters

For hotspot-heavy MoE traffic, the main failure mode is unstable step composition and tail inflation. This project is intentionally biased toward step stability and tail control.

### Preserve a real execution path

All reported numbers come from the real TensorRT engine path, not from a simulator.

## Runtime Model

The central systems assumption is that MoE step cost has two parts:

```text
step_latency(batch)
  =
  compute_cost(token_volume)
  +
  contention_cost(pressure, skew)
```

Where:

- `compute_cost` is mostly token-driven
- `contention_cost` increases as expert or rank concentration rises

The planner is effective if two conditions hold:

1. contention increases with aggregate pressure
2. contention grows disproportionately once several hot requests are stacked into the same step

Under those conditions, dispersing hot requests across steps reduces tail risk, even if average batch size falls. That is the exact tradeoff this repository is designed to surface.

## Core Components

### [`scheduler/moe_pressure.py`](scheduler/moe_pressure.py)

Purpose:

- normalize MoE-specific runtime pressure into a stable scheduling signal

Why it was introduced:

- default batching has no first-class representation of MoE pressure

What it adds:

- pressure classes: `balanced`, `hot_expert`, `hot_rank`
- scalar pressure scores that can be aggregated at step level

### [`scheduler/resource_model.py`](scheduler/resource_model.py)

Purpose:

- convert raw request metadata into an explicit runtime planning contract

Why it was introduced:

- planning quality degrades quickly when scheduling assumptions stay implicit

What it adds:

- `RequestProfile`
- `RuntimeBudget`
- `StepPlan`

### [`scheduler/moe_microbatch_scheduler.py`](scheduler/moe_microbatch_scheduler.py)

Purpose:

- turn pressure-aware runtime state into a concrete execution order

Why it was introduced:

- requests that merely “fit” are not necessarily safe to batch together in MoE serving

What it adds:

- decode-first planning
- pressure-aware dispersion
- step-level selection logic aligned with TensorRT-LLM scheduling concepts

### [`scheduler/telemetry.py`](scheduler/telemetry.py)

Purpose:

- make step composition observable

Why it was introduced:

- end metrics alone do not explain whether the planner actually changed batch structure

What it adds:

- JSONL telemetry for scheduled requests, deferred requests, and planned pressure

### [`scripts/run_patched.py`](scripts/run_patched.py)

Purpose:

- execute the planner on the real engine path

Why it was introduced:

- the planner must be measured where its batch decisions actually matter

## Evaluation

### Workloads

- `Balanced MoE`
- `Hot-Expert`
- `Hot-Rank`
- `Hot-Expert` 24-request full run

### Validation environment

The reported measurements were collected on:

- model: `Qwen/Qwen1.5-MoE-A2.7B-Chat`
- quantization: TensorRT-LLM `INT4 weight-only`
- hardware: `RTX 4060 Ti 16GB`

### Results

| Workload | Metric | Baseline | Patched | Delta |
| --- | --- | ---: | ---: | ---: |
| Balanced | TTFT p90 | `0.0816s` | `0.0739s` | `-0.0077s` |
| Balanced | E2E p90 | `1.4878s` | `1.4748s` | `-0.0129s` |
| Balanced | Throughput | `278.55 tok/s` | `281.47 tok/s` | `+2.92 tok/s` |
| Hot-Expert | TTFT p90 | `0.0748s` | `0.0112s` | `-0.0635s` |
| Hot-Expert | E2E p90 | `1.8668s` | `1.5741s` | `-0.2927s` |
| Hot-Expert | TPOT p90 | `0.0116s` | `0.0098s` | `-0.0017s` |
| Hot-Rank | TTFT p90 | `0.0737s` | `0.0131s` | `-0.0606s` |
| Hot-Rank | E2E p90 | `1.9942s` | `1.7115s` | `-0.2827s` |
| Hot-Expert (24 req) | TTFT p90 | `0.0733s` | `0.0111s` | `-0.0622s` |
| Hot-Expert (24 req) | E2E p90 | `1.8348s` | `1.5983s` | `-0.2365s` |
| Hot-Expert (24 req) | Throughput | `310.44 tok/s` | `99.81 tok/s` | `-210.63 tok/s` |

### What the data supports

The data supports three conclusions:

1. The planner does not introduce an obvious regression on the balanced control.
2. On hotspot-heavy workloads, the planner materially reduces TTFT and E2E tail latency.
3. The current policy is deliberately aggressive and trades throughput for tail control.

This repository is therefore best understood as a successful latency-first MoE pressure isolation policy.

## Limitations

The main limitation is explicit:

- the quantitative benchmark runs on the real TensorRT engine backend
- the planning logic is applied through batch composition on that path
- the final measured comparison is therefore not a pure in-backend PyTorch quantitative benchmark

That limitation narrows the scope of the claim, but it does not invalidate the runtime measurements reported here.

## Repository Layout

- [`scheduler/`](scheduler) - runtime model, pressure model, planner, telemetry
- [`scripts/`](scripts) - workload generation, execution drivers, summarization
- [`workloads/`](workloads) - fixed MoE workloads
- [`results/`](results) - raw outputs and compare tables
- [`docs/`](docs) - supporting notes, summary documents, final report

## Reproducibility

Core flow:

```bash
python scripts/generate_workloads.py
python scripts/run_baseline.py ...
python scripts/run_patched.py ...
python scripts/summarize_results.py ...
```

Reference documents:

- [`docs/final_report.md`](docs/final_report.md)
- [`docs/result_summary.md`](docs/result_summary.md)
- [`results/compare_tables/summary.md`](results/compare_tables/summary.md)
