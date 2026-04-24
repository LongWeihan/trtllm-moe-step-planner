# Scheduler v1

## What changed

The compact implementation adds a minimal planning layer and a MoE-aware scheduler patch:

- [scheduler/resource_model.py](C:/26spring/nv项目/trtllm-moe-runtime-exp/scheduler/resource_model.py)
- [scheduler/moe_microbatch_scheduler.py](C:/26spring/nv项目/trtllm-moe-runtime-exp/scheduler/moe_microbatch_scheduler.py)

## Core idea

The project does not scatter MoE heuristics across the runtime. Instead it makes scheduling consume an explicit step plan:

`request profile -> runtime budget -> step plan -> microbatch decision`

The model is intentionally small:

- `RequestProfile`
- `RuntimeBudget`
- `StepPlan`

## Internal TRT-LLM patch seam

The patch is implemented on the TRT-LLM 1.2.1 PyTorch backend seam:

- patch `executor_request_to_llm_request`
- patch `BindMicroBatchScheduler`

This is the architectural proof that the work targets TRT-LLM internals rather than an external toy scheduler.

## Quantitative evaluation path

Because the PyTorch backend could not run the needed quantized MoE model path directly on this machine, the quantitative evaluation executed the same step-planning logic over the real TensorRT engine backend:

- baseline:
  - fixed contiguous microbatches of 4
- patched:
  - use `build_step_plan(...)`
  - admit only the request set that fits the pressure budget
  - run that batch on the real TensorRT engine

This keeps the runtime idea intact while staying honest about backend limitations.

## Behavior in practice

Current budget settings are intentionally conservative:

- balanced pressure budget permits normal 4-request batches
- hot pressure budget usually admits only 1 request at a time

That is why the patched path improves tail latency strongly on hot workloads, but loses throughput.
