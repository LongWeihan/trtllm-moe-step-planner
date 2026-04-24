#!/usr/bin/env bash
set -euo pipefail

VENV_ROOT="/home/a/trtllm-moe-runtime-exp/venv"
SITE_PACKAGES="$VENV_ROOT/lib/python3.12/site-packages"

source "$VENV_ROOT/bin/activate"

python_lib_paths=(
  "$VENV_ROOT/lib"
  "/usr/lib/wsl/lib"
  "/usr/lib/x86_64-linux-gnu"
  "$SITE_PACKAGES/torch/lib"
  "$SITE_PACKAGES/tensorrt_libs"
  "$SITE_PACKAGES/tensorrt_llm/libs"
)

while IFS= read -r path; do
  python_lib_paths+=("$path")
done < <(find "$SITE_PACKAGES/nvidia" -maxdepth 2 -type d -name lib | sort)

joined_paths=""
for path in "${python_lib_paths[@]}"; do
  if [[ -d "$path" ]]; then
    if [[ -n "$joined_paths" ]]; then
      joined_paths="${joined_paths}:$path"
    else
      joined_paths="$path"
    fi
  fi
done

export LD_LIBRARY_PATH="${joined_paths}:${LD_LIBRARY_PATH:-}"

exec "$@"
