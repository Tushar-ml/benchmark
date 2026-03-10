#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

# Ensure uv is available
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi

# --- vllm env ---
echo "Creating vllm environment (.venv-vllm)..."
uv venv .venv-vllm
uv pip install --python .venv-vllm/bin/python -r requirements.txt
uv pip install --python .venv-vllm/bin/python aiohttp matplotlib pandas "vllm==0.16.0"

# --- sglang env ---
echo "Creating sglang environment (.venv-sglang)..."
uv venv .venv-sglang
uv pip install --python .venv-sglang/bin/python -r requirements.txt
uv pip install --python .venv-sglang/bin/python aiohttp matplotlib pandas "sglang==0.5.9"

echo "Done."
echo "  vllm:   source .venv-vllm/bin/activate"
echo "  sglang: source .venv-sglang/bin/activate"
