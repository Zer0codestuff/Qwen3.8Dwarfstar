#!/usr/bin/env bash
# Probe whether the local Qwen 3.8 GGUF can generate on this Mac via llama.cpp.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GGUF="${QWEN_GGUF:-$HOME/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/f1bfb127c64f7072bdd2cad55f258b9c8b2910fe/Qwen3.8-27B-Q3_K_M.gguf}"
LLAMA="${LLAMA_CLI:-/tmp/llama.cpp/build/bin/llama-cli}"
OUT="${1:-$ROOT/.audit/baseline-llamacpp.json}"
PROMPT="The capital of France is"
N="${N_TOKENS:-32}"
CTX="${CTX:-512}"

if [[ ! -x "$LLAMA" ]]; then
  echo "missing llama-cli at $LLAMA" >&2
  exit 2
fi
if [[ ! -f "$GGUF" ]]; then
  echo "missing GGUF at $GGUF" >&2
  exit 2
fi

LOG="$(mktemp)"
TIMELOG="$(mktemp)"
set +e
/usr/bin/time -l "$LLAMA" \
  -m "$GGUF" \
  -p "$PROMPT" \
  -n "$N" \
  -c "$CTX" \
  --temp 0 \
  --top-k 1 \
  --seed 1 \
  -ngl 99 \
  --no-display-prompt \
  --log-disable \
  >"$LOG" 2>"$TIMELOG"
RC=$?
set -e

TEXT="$(tr '\n' ' ' < "$LOG" | tr -s ' ')"
PARIS=0
case "$TEXT" in
  *[Pp]aris*) PARIS=1 ;;
esac

PEAK="$(awk '/maximum resident set size/ {print $1}' "$TIMELOG" | tail -1)"
python3 - "$OUT" "$RC" "$PARIS" "$TEXT" "$PEAK" "$GGUF" "$N" "$CTX" <<'PY'
import json, sys
out, rc, paris, text, peak, gguf, n, ctx = sys.argv[1:9]
doc = {
    "ok": int(rc) == 0 and int(paris) == 1,
    "exit_code": int(rc),
    "contains_paris": bool(int(paris)),
    "text": text[:2000],
    "peak_rss_bytes": int(peak or 0),
    "gguf": gguf,
    "n_predict": int(n),
    "ctx": int(ctx),
    "engine": "llama.cpp",
}
open(out, "w").write(json.dumps(doc, indent=2) + "\n")
print(json.dumps({k: doc[k] for k in ("ok", "exit_code", "contains_paris", "peak_rss_bytes")}, indent=2))
if not doc["ok"]:
    sys.exit(1)
PY
