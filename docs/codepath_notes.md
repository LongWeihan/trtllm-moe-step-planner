# Codepath Notes

## Qwen model support

Confirmed from TensorRT-LLM source:

- `examples/models/core/qwen/README.md` lists `Qwen1.5-MoE-A2.7B(-Chat)` as supported on `Ampere+`.
- Weight-only (`WO`) quantization is marked supported for this model.

## Scheduler path

Primary source paths:

- `tensorrt_llm/_torch/pyexecutor/scheduler/scheduler.py`
- `tensorrt_llm/_torch/pyexecutor/_util.py`
- `tensorrt_llm/_torch/pyexecutor/llm_request.py`

Relevant classes and functions:

- `PyExecutor`
- `BindCapacityScheduler`
- `BindMicroBatchScheduler`
- `PyRequestScheduler`
- `executor_request_to_llm_request`

## Chosen patch seam

The least invasive patch seam for this project is:

1. Monkey-patch `executor_request_to_llm_request` so MoE pressure metadata survives conversion into `LlmRequest`.
2. Monkey-patch `BindMicroBatchScheduler` with a subclass that consumes a local runtime resource model.
3. Leave `CapacityScheduler` as-is for the main path; optional guard stays a stretch goal.

## Real experimental split

There are two paths in the final project:

1. **Internal patch path**
   - implemented against the installed TRT-LLM 1.2.1 PyTorch backend
   - proves that the project really modifies the internal scheduler seam

2. **Quantitative benchmark path**
   - runs on the real `Qwen/Qwen1.5-MoE-A2.7B-Chat` INT4 weight-only TensorRT engine
   - applies the same `resource_model -> step_plan` logic externally to real batch composition

This split exists because the 1.2.1 PyTorch backend on this machine could not consume the quantized HF MoE variants needed for direct internal-patch benchmarking.
