# Blockers

## Active

### TRT-LLM 1.2.1 PyTorch backend cannot serve as the final quantitative path for this quantized MoE setup

- Symptom:
  - The internal scheduler patch was implemented on the TRT-LLM PyTorch backend.
  - The real project model path is `Qwen/Qwen1.5-MoE-A2.7B-Chat + INT4 weight-only` on the TensorRT engine backend.
  - The PyTorch backend on this machine could not directly serve a matching quantized HF MoE path for apples-to-apples internal patch benchmarking.
  - Attempting `Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4` failed with `Unsupported quantization_config ... gptq`.
- Final mitigation:
  - Keep the internal scheduler patch as the architectural contribution.
  - Execute quantitative runs on the real TensorRT engine by externalizing the same `resource_model -> step_plan` logic into batch composition.
  - State this limitation explicitly in the final report instead of hiding it.

## Resolved

### `pip` missing in base WSL image

- Resolved by user-scoped bootstrap.

### TensorRT-LLM wheel installation is slow / unstable

- Resolved by downloading the pinned `tensorrt_llm==1.2.1` wheel and installing it directly.

### Loader / shared library failures during import

- Symptoms included missing `libpython3.12.so`, MPI libraries, and CUDA loader path issues.
- Resolved through the venv symlink plus [scripts/wsl_env.sh](C:/26spring/nv项目/trtllm-moe-runtime-exp/scripts/wsl_env.sh).
