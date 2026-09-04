#!/usr/bin/env bash
# Creates the virtualenv used for the FAISS baseline and the plots.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 -m venv .venv
./.venv/bin/pip install --quiet --upgrade pip
./.venv/bin/pip install --quiet numpy faiss-cpu matplotlib
./.venv/bin/python -c "import faiss, numpy, matplotlib; print('faiss', faiss.__version__)"
echo "Activate with: source .venv/bin/activate"
