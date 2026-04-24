# TRT-LLM MoE Step Planner

Pressure-aware runtime step planning for Mixture-of-Experts inference on TensorRT-LLM, evaluated on a fixed `Qwen/Qwen1.5-MoE-A2.7B-Chat` `INT4 weight-only` engine path.

## Abstract

This repository studies a narrow but important runtime problem in MoE serving: requests with different routing behavior do not contribute equally to batch cost, and default batching policies do not explicitly model that asymmetry. The project introduces a minimal runtime resource model and a pressure-aware step planner that attempts to reduce decode-tail latency by preventing high-pressure MoE requests from stacking in the same execution step.

The implementation is deliberately small in scope:

- it does not modify model weights
- it does not replace TensorRT-LLM kernels
- it does not introduce an external serving system

Instead, it adds a planner layer around real TensorRT-LLM execution and validates the effect on real engine runs.

## Problem Statement

For dense models, batch cost is often approximated primarily by token count, prompt length, and memory fit. For MoE models, this is incomplete. Routing skew induces additional runtime interference:

- expert hotspots
- rank hotspots
- larger per-step variance
- longer decode tails
- batch stragglers

The practical consequence is that two requests with similar token lengths may behave very differently once their routed experts concentrate contention into the same step.

## Design Objective

The objective of this project is to make MoE pressure explicit in the runtime decision path:

1. estimate per-request pressure
2. aggregate it into a step-level resource view
3. build a step plan that limits pressure stacking
4. validate the effect on real TensorRT engine execution

## Method

### 1. Pressure Representation

Implemented in [`scheduler/moe_pressure.py`](scheduler/moe_pressure.py).

Each request is assigned a pressure class and score:

- `balanced`
- `hot_expert`
- `hot_rank`

This abstraction is intentionally simple. The goal is not to reconstruct the router in detail, but to create a stable scheduling signal that is easy to consume downstream.

### 2. Runtime Resource Model

Implemented in [`scheduler/resource_model.py`](scheduler/resource_model.py).

The resource model adds three explicit objects:

- `RequestProfile`
- `RuntimeBudget`
- `StepPlan`

This changes the scheduler interface from “pick the next few requests” to:

`request metadata -> resource model -> step plan -> execution`

That explicit contract matters because it separates signal extraction from scheduling action.

### 3. Pressure-Aware Step Planning

Implemented in [`scheduler/moe_microbatch_scheduler.py`](scheduler/moe_microbatch_scheduler.py) and executed quantitatively through [`scripts/run_patched.py`](scripts/run_patched.py).

The planner is decode-first and pressure-aware:

- generation requests are considered before context requests
- aggregate pressure is bounded at the step level
- hot requests are dispersed across steps when necessary

In practice, this means the planner is willing to reduce effective batch size if that is what it takes to avoid a high-pressure collision.

## Why These Changes Should Work

The project is based on a simple runtime argument rather than a claim about TensorRT-LLM internals being analytically solved.

Let:

- `p_i` be the pressure score of request `i`
- `P(B) = sum(p_i)` be the total pressure of batch `B`
- `T(B)` be the token-related compute cost of batch `B`

Assume step latency can be decomposed as:

```text
L(B) = L_compute(T(B)) + L_contention(P(B), skew(B))
```

where:

- `L_compute` is mostly token-driven
- `L_contention` captures expert/rank interference

The relevant scheduling assumption is:

1. `L_contention` is increasing with pressure
2. near hotspot regions, `L_contention` is approximately convex

Under these assumptions, stacking several hot requests into one step increases tail risk superlinearly. Dispersing those requests across steps reduces extreme latency, even if it gives up some batching efficiency. This is exactly the tradeoff observed in the experiments:

- lower TTFT / E2E tail latency
- reduced step variance
- lower throughput when isolation becomes too aggressive

This is not a proof of optimality for the engine, but it is a sound scheduler rationale under a standard monotonicity-plus-convexity contention model.

## What Was Changed and Why

### [`scheduler/moe_pressure.py`](scheduler/moe_pressure.py)

Reason for change:

- default scheduling does not have a first-class MoE pressure signal

What it adds:

- stable request-level pressure labels
- scalar pressure scores that can be composed at the batch level

### [`scheduler/resource_model.py`](scheduler/resource_model.py)

Reason for change:

- scheduling logic is easier to reason about when resource assumptions are explicit

What it adds:

- request profiling
- budget construction
- step-plan generation

### [`scheduler/moe_microbatch_scheduler.py`](scheduler/moe_microbatch_scheduler.py)

Reason for change:

- the runtime needs a path that can consume pressure rather than treating all requests as equivalent once they fit

What it adds:

- pressure-aware ordering
- step-level dispersion
- integration seam aligned with TensorRT-LLM scheduling concepts

### [`scheduler/telemetry.py`](scheduler/telemetry.py)

Reason for change:

- the project needs evidence that the planner is changing step composition, not only end metrics

What it adds:

- structured JSONL telemetry for step contents and deferred requests

### [`scripts/run_patched.py`](scripts/run_patched.py)

Reason for change:

- the planner must be evaluated on real engine execution, not on a simulator

What it adds:

- execution of the same planning logic against the TensorRT engine path
- telemetry collection
- reproducible baseline-versus-patched comparison

## Experimental Setup

- Model: `Qwen/Qwen1.5-MoE-A2.7B-Chat`
- Quantization: TensorRT-LLM `INT4 weight-only`
- Hardware: `RTX 4060 Ti 16GB`

Workloads:

- `Balanced MoE`
- `Hot-Expert`
- `Hot-Rank`
- `Hot-Expert` 24-request full run

Baselines:

- default engine batching
- pressure-aware patched path

## Results

### Summary Table

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

### Interpretation

The balanced control shows that the planner does not introduce an obvious regression on non-hot traffic. The hot workloads show the intended behavior clearly:

- TTFT tail improves sharply
- E2E tail improves materially
- TPOT improves modestly
- throughput falls because the policy isolates pressure aggressively

In other words, this repository demonstrates a successful latency-first MoE pressure isolation policy.

## Limitations

The central limitation is explicit:

- the quantitative benchmark uses the real TensorRT engine backend with the planner applied to batch composition
- the internal runtime seam work is real
- the final measured comparison is therefore not a pure in-backend PyTorch quantitative benchmark

This limitation does not invalidate the measured result, but it does define its scope.

## Repository Structure

- [`scheduler/`](scheduler): runtime resource model, pressure model, planner, telemetry
- [`scripts/`](scripts): workload generation, execution drivers, summarization
- [`workloads/`](workloads): fixed MoE workloads used in the study
- [`results/`](results): raw outputs and comparison tables
- [`docs/`](docs): detailed notes, result summary, and final report

## Reproducibility

Core commands:

```bash
python scripts/generate_workloads.py
python scripts/run_baseline.py ...
python scripts/run_patched.py ...
python scripts/summarize_results.py ...
```

Reference material:

- [`docs/final_report.md`](docs/final_report.md)
- [`docs/result_summary.md`](docs/result_summary.md)
- [`results/compare_tables/summary.md`](results/compare_tables/summary.md)
