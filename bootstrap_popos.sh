#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOOTSTRAP_PYTHON="${PYTHON_BIN:-python3}"
VENV_DIR="$ROOT_DIR/.venv"
PYTHON_BIN="$VENV_DIR/bin/python"
VENV_CFG="$VENV_DIR/pyvenv.cfg"

if [[ -f "$VENV_CFG" ]] && ! grep -Fq "$VENV_DIR" "$VENV_CFG"; then
    echo "[0/3] Recreating virtual environment (.venv) due to stale path in pyvenv.cfg"
    rm -rf "$VENV_DIR"
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "[0/3] Creating virtual environment (.venv)"
    "$BOOTSTRAP_PYTHON" -m venv "$VENV_DIR"
fi

echo "[1/3] Installing runtime/build dependencies"
"$PYTHON_BIN" -m pip install --upgrade pip wheel
"$PYTHON_BIN" -m pip install --upgrade "setuptools<81"
"$PYTHON_BIN" -m pip install --upgrade numpy cython numba pygame torch

echo "[2/3] Building per_cython backend"
"$PYTHON_BIN" "$ROOT_DIR/setup_cython_per.py" build_ext --inplace

chmod +x "$ROOT_DIR/run_train.sh"
ln -sf "$ROOT_DIR/run_train.sh" "$VENV_DIR/bin/run_train.sh"

echo "[3/3] Verifying accelerators"
"$PYTHON_BIN" "$ROOT_DIR/verify_runtime.py"

echo "Done. Linux accelerator setup is complete."
