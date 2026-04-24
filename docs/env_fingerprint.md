# Environment Fingerprint

- Host OS: Windows 11 (`10.0.26200.8246`)
- WSL:
  - Version: `2.6.3.0`
  - Kernel: `6.6.87.2-1`
  - Distro: `Ubuntu 24.04.4 LTS`
- GPU:
  - Model: `NVIDIA GeForce RTX 4060 Ti`
  - VRAM: `16380 MiB`
  - Driver: `591.86`
- CUDA:
  - Host-visible CUDA version: `13.1` from `nvidia-smi`
  - WSL `nvcc`: `12.0.140`
- CPU: `AMD Ryzen 9 7950X 16-Core Processor`
- RAM: `136320499712` bytes (`~127 GiB`)
- Python:
  - WSL base: `Python 3.12.3`
  - venv root: `/home/a/trtllm-moe-runtime-exp/venv`
- Nsight Systems: `2024.5.1.113-245134619542v0`

## Notes

- WSL GPU access is healthy: `nvidia-smi` runs successfully inside WSL.
- Base Ubuntu image does not provide `pip` or usable `venv` bootstrap out of the box.
