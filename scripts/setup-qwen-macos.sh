#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PYTHON=${PYTHON:-python3}

if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
    echo "Qwen setup requires an Apple Silicon Mac." >&2
    exit 2
fi

"$PYTHON" -m venv "$ROOT/.venv"
"$ROOT/.venv/bin/python" -m pip install --upgrade pip
"$ROOT/.venv/bin/python" -m pip install --requirement "$ROOT/requirements-macos.txt"
"$ROOT/.venv/bin/python" -m pip install --no-deps --editable "$ROOT"

echo
echo "DwarfStar Qwen is installed."
echo "Next: $ROOT/dwarfstar download"
