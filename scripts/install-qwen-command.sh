#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
TARGET="$ROOT/dwarfstar"
USER_BIN=${XDG_BIN_HOME:-"$HOME/.local/bin"}
DESTINATION="$USER_BIN/dwarfstar"

if [ ! -x "$ROOT/.venv/bin/python" ]; then
    echo "DwarfStar Qwen is not installed. Run: make qwen-setup" >&2
    exit 2
fi

mkdir -p "$USER_BIN"
if [ -L "$DESTINATION" ]; then
    CURRENT=$(readlink "$DESTINATION")
    if [ "$CURRENT" = "$TARGET" ]; then
        echo "dwarfstar is already installed at $DESTINATION"
        exit 0
    fi
    echo "$DESTINATION is a symlink to another command; leaving it unchanged" >&2
    exit 2
fi
if [ -e "$DESTINATION" ]; then
    echo "$DESTINATION already exists; leaving it unchanged" >&2
    exit 2
fi

ln -s "$TARGET" "$DESTINATION"
echo "Installed: $DESTINATION -> $TARGET"
echo "Try: dwarfstar \"Spiegami questo problema\""
