# Run Log

## 2026-04-24

- Created project workspace at `C:\26spring\nv项目\trtllm-moe-runtime-exp`.
- Created WSL runtime workspace at `/home/a/trtllm-moe-runtime-exp`.
- Bootstrapped user-scoped `pip` and `virtualenv` in WSL because `python3-venv` was unavailable and `sudo` requires a password.
- Installed `torch==2.10.0+cu130` and `torchvision==0.25.0+cu130` inside `/home/a/trtllm-moe-runtime-exp/venv`.
- Cloned `NVIDIA/TensorRT-LLM` into `/home/a/trtllm-moe-runtime-exp/src/TensorRT-LLM`.
- Started `tensorrt_llm` installation attempts and moved in parallel on the control-plane implementation.
- Installed a pinned local `tensorrt_llm==1.2.1` wheel after index-based solves proved unstable.
- Fixed WSL runtime loader issues and verified `from tensorrt_llm import LLM`.
- Downloaded `Qwen/Qwen1.5-MoE-A2.7B-Chat` into `/home/a/trtllm-moe-runtime-exp/hf/Qwen1.5-MoE-A2.7B-Chat`.
- Converted the model to a TRT-LLM INT4 weight-only checkpoint and built a working engine.
- Verified the real engine path with a direct `run.py` generate and saved the output under `results/sanity/`.
- Implemented:
  - `scheduler/telemetry.py`
  - `scheduler/moe_pressure.py`
  - `scheduler/resource_model.py`
  - `scheduler/moe_microbatch_scheduler.py`
- Generated the three fixed workloads:
  - `workloads/balanced_moe.jsonl`
  - `workloads/hot_expert.jsonl`
  - `workloads/hot_rank.jsonl`
- Ran baseline and patched evaluations on:
  - `Balanced MoE` (`SMOKE`/short full set)
  - `Hot-Expert` (`PILOT`, then `FULL`)
  - `Hot-Rank` (`PILOT`)
- Selected `Hot-Expert` as the winning workload and extended it to the 24-request `FULL` compare set.
- Generated machine-readable and markdown comparison tables under `results/compare_tables/`.
