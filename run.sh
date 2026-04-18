#!/usr/bin/env bash
# One-shot reproducibility wrapper.
#
#   ./run.sh              # auto-detects device (MPS > CUDA > CPU)
#   ./run.sh --device cpu # force a device
#   ./run.sh --smoke      # one-minute wiring check
#
# Requires Python 3.10+. First run creates .venv and installs deps;
# subsequent runs reuse them.
set -euo pipefail
cd "$(dirname "$0")"

find_python() {
    for cand in python3.12 python3.11 python3.10 python3; do
        if command -v "$cand" >/dev/null 2>&1 && \
           "$cand" -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" 2>/dev/null; then
            echo "$cand"
            return 0
        fi
    done
    return 1
}

if [ ! -x .venv/bin/python ]; then
    PY=$(find_python) || {
        echo "[run.sh] error: need Python 3.10+ on PATH. try: brew install python@3.12"
        exit 1
    }
    echo "[run.sh] creating virtual environment with $($PY --version)..."
    "$PY" -m venv .venv
fi

source .venv/bin/activate

if [ ! -f .venv/.deps_installed ]; then
    echo "[run.sh] installing dependencies..."
    pip install -q -r requirements.txt
    touch .venv/.deps_installed
fi

python run.py "$@"
