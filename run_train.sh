#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
while [[ -L "$SCRIPT_PATH" ]]; do
    SCRIPT_DIR="$(cd -P "$(dirname "$SCRIPT_PATH")" && pwd)"
    SCRIPT_PATH="$(readlink "$SCRIPT_PATH")"
    if [[ "$SCRIPT_PATH" != /* ]]; then
        SCRIPT_PATH="$SCRIPT_DIR/$SCRIPT_PATH"
    fi
done
ROOT_DIR="$(cd -P "$(dirname "$SCRIPT_PATH")" && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
VENV_CFG="$ROOT_DIR/.venv/pyvenv.cfg"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Virtual environment not found. Running bootstrap_popos.sh..."
    "$ROOT_DIR/bootstrap_popos.sh"
elif [[ -f "$VENV_CFG" ]] && ! grep -Fq "$ROOT_DIR/.venv" "$VENV_CFG"; then
    echo "Virtual environment path mismatch detected. Rebuilding via bootstrap_popos.sh..."
    "$ROOT_DIR/bootstrap_popos.sh"
fi

exec "$PYTHON_BIN" "$ROOT_DIR/train.py" "$@"
