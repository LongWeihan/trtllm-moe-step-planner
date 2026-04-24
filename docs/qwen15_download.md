# Qwen1.5-MoE Download

## Fixed main model

- Model: `Qwen/Qwen1.5-MoE-A2.7B-Chat`
- Local path: `/home/a/trtllm-moe-runtime-exp/hf/Qwen1.5-MoE-A2.7B-Chat`

## Download result

The model directory was downloaded successfully and includes the full HF checkpoint set:

- 8 `safetensors` shards
- tokenizer files
- model config
- generation config

Example files:

- `config.json`
- `model.safetensors.index.json`
- `model-00001-of-00008.safetensors`
- `model-00008-of-00008.safetensors`
- `tokenizer.json`

## Size

Approximate on-disk size:

- HF model directory: `27G`

Command used to confirm:

```bash
du -sh /home/a/trtllm-moe-runtime-exp/hf/Qwen1.5-MoE-A2.7B-Chat
```

## Why this model stayed fixed

This project intentionally kept the originally chosen fixed model because it satisfied all of the constraints at once:

- real Qwen MoE model
- downloadable on this machine
- explicitly supported by the TRT-LLM Qwen path
- realistically buildable on a `4060 Ti 16GB` with INT4 weight-only
