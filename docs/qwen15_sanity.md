# Qwen1.5-MoE Sanity

## Sanity goals

Before running any workload experiments, the project needed to prove three things:

1. TRT-LLM imports successfully on this machine.
2. The INT4 weight-only engine loads.
3. A real generation call succeeds on the built Qwen MoE engine.

## Evidence

### Backend import

- [results/sanity/backend_import.json](C:/26spring/nv项目/trtllm-moe-runtime-exp/results/sanity/backend_import.json)

This confirms the working `tensorrt_llm` import path.

### Real engine generation

- [results/sanity/qwen15_run_output.txt](C:/26spring/nv项目/trtllm-moe-runtime-exp/results/sanity/qwen15_run_output.txt)

The engine loaded and produced a text completion for:

`Explain why MoE routing skew can create decode tail latency.`

### Small benchmark path

The earliest small benchmark artifacts live here:

- [results/baseline/balanced_smoke.json](C:/26spring/nv项目/trtllm-moe-runtime-exp/results/baseline/balanced_smoke.json)
- [results/patched/balanced_smoke.json](C:/26spring/nv项目/trtllm-moe-runtime-exp/results/patched/balanced_smoke.json)

Note: the smoke pair was used as a path check, not the final comparison source, because the first smoke run was launched during active bring-up.

## Sanity conclusion

The fixed project path is real and working:

- real Qwen MoE model
- real TRT-LLM conversion/build
- real TensorRT engine inference
- real outputs written to disk
