# Install Log

## WSL user-space bootstrap

Because the base WSL image had no usable `pip` / `venv` path and `sudo` was password-protected, the environment was bootstrapped entirely in user space.

```bash
curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /home/a/trtllm-moe-runtime-exp/get-pip.py
python3 /home/a/trtllm-moe-runtime-exp/get-pip.py --user --break-system-packages
~/.local/bin/pip3 install --user --break-system-packages virtualenv
~/.local/bin/virtualenv -p python3 /home/a/trtllm-moe-runtime-exp/venv
```

## PyTorch base

Initial bootstrap:

```bash
/home/a/trtllm-moe-runtime-exp/venv/bin/pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu130
```

During TensorRT-LLM installation, the environment converged to the package-compatible stack that ships with the TRT-LLM wheel. The final working runtime used:

- `torch==2.9.1`
- `torchvision==0.24.1`
- `tensorrt==10.14.1.48.post1`
- `tensorrt-llm==1.2.1`

## TensorRT-LLM install

Direct dependency solving against the NVIDIA index was unstable on this machine, so the installation was completed by downloading and installing the pinned wheel:

```bash
/home/a/trtllm-moe-runtime-exp/venv/bin/pip download tensorrt_llm==1.2.1 --extra-index-url https://pypi.nvidia.com -d /home/a/trtllm-moe-runtime-exp/cache
/home/a/trtllm-moe-runtime-exp/venv/bin/pip install /home/a/trtllm-moe-runtime-exp/cache/tensorrt_llm-1.2.1-cp312-cp312-linux_x86_64.whl
```

## Loader fixes

The installed package required a few runtime loader fixes before `import tensorrt_llm` became stable:

- symlinked `libpython3.12.so` into the venv `lib/` directory
- installed user-space MPI library bindings
- exported a broader `LD_LIBRARY_PATH` via [scripts/wsl_env.sh](C:/26spring/nv项目/trtllm-moe-runtime-exp/scripts/wsl_env.sh) so TRT-LLM can see:
  - `/usr/lib/wsl/lib`
  - `torch/lib`
  - `tensorrt_llm/libs`
  - all pip-installed `nvidia/*/lib` directories

## Final sanity

The final import sanity succeeded through:

```bash
cd /mnt/c/26spring/nv项目/trtllm-moe-runtime-exp
./scripts/wsl_env.sh python scripts/sanity_backend.py
```

Evidence:

- [results/sanity/backend_import.json](C:/26spring/nv项目/trtllm-moe-runtime-exp/results/sanity/backend_import.json)

Confirmed:

- `tensorrt_llm.__version__ == 1.2.1`
- `from tensorrt_llm import LLM` succeeds
