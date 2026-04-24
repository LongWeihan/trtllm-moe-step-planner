# Qwen1.5-MoE INT4 WO Build Log

## Artifact paths

- Converted checkpoint:
  - `/home/a/trtllm-moe-runtime-exp/artifacts/qwen15_moe_int4wo_checkpoint`
- Built TensorRT engine:
  - `/home/a/trtllm-moe-runtime-exp/artifacts/qwen15_moe_int4wo_engine`
- Raw logs:
  - `/home/a/trtllm-moe-runtime-exp/artifacts/logs/qwen15_convert_int4wo.log`
  - `/home/a/trtllm-moe-runtime-exp/artifacts/logs/qwen15_build_int4wo.log`

Approximate sizes:

- checkpoint: `7.6G`
- engine: `7.6G`

## Conversion

The official TRT-LLM Qwen example path was used to convert the HF checkpoint to TRT-LLM format with INT4 weight-only quantization.

Observed result from the conversion log:

- `Total time of converting checkpoints: 00:03:13`

## Build

The engine was built with a small single-GPU envelope suitable for this machine:

- `--gemm_plugin float16`
- `--max_batch_size 4`
- `--max_input_len 128`
- `--max_seq_len 256`
- `--max_num_tokens 512`

Observed result from the build log:

- `Total Weights Memory: 8134053232 bytes`
- `Engine generation completed in 16.5664 seconds`
- `Total time of building all engines: 00:00:38`
- `Build phase peak memory: 18130.59 MB`

## Important config confirmation

The built engine config confirms the project's fixed quantized path:

- quantization: `W4A16`
- runtime batch target: `max_batch_size = 4`
- single GPU engine

This is the exact engine family used for the benchmark runs in `results/`.
