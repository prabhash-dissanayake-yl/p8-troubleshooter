#!/bin/bash

set -euo pipefail
if command -v pyenv >/dev/null 2>&1; then
  uv venv --python "$(pyenv which python)" --allow-existing
else
  uv venv --allow-existing
fi

if [[ ${1-} != "local" ]]; then
  uv sync --all-extras
else
  # For local development of agentkernel, you can force reinstall from local dist
  uv sync --find-links ../../../ak-py/dist --all-extras
  uv pip install --force-reinstall --find-links ../../../ak-py/dist agentkernel[cli,openai,test] || true
fi

KA_REPO="https://github.com/yaala-internal/p8-knowledge-agent.git"
KA_BRANCH="feature/add-unified-retrieval-framework"
KA_TMP_DIR="$(mktemp -d)/p8-knowledge-agent"
if command -v git >/dev/null 2>&1; then
    if git clone --depth=1 --branch "$KA_BRANCH" "$KA_REPO" "$KA_TMP_DIR"; then
        uv pip install --force-reinstall -e "${KA_TMP_DIR}[all]" || true
        rm -rf "$KA_TMP_DIR"
    else
        echo "Warning: failed to clone $KA_REPO (branch: $KA_BRANCH)"
    fi
else
    echo "Warning: git not found; skipping clone/install of p8-knowledge-agent"
fi
