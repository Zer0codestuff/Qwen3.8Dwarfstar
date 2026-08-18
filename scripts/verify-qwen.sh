#!/usr/bin/env bash
# Product gate: DwarfStar greedy-decodes Qwen 3.8 on this Mac.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
GGUF="${QWEN_GGUF:-$HOME/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/f1bfb127c64f7072bdd2cad55f258b9c8b2910fe/Qwen3.8-27B-Q3_K_M.gguf}"
OUT="${1:-$ROOT/.audit/verify-qwen.json}"
PROMPT="The capital of France is"
N="${N_TOKENS:-32}"
CTX="${CTX:-512}"

if [[ ! -x "$ROOT/ds4" ]]; then
  echo "build ./ds4 first (make ds4)" >&2
  exit 2
fi
if [[ ! -f "$GGUF" ]]; then
  echo "missing GGUF at $GGUF" >&2
  exit 2
fi

LOG="$(mktemp)"
TIMELOG="$(mktemp)"
set +e
/usr/bin/time -l "$ROOT/ds4" \
  -m "$GGUF" \
  --ssd-streaming \
  --ctx "$CTX" \
  -n "$N" \
  --temp 0 \
  -p "$PROMPT" \
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
    "text": text[:4000],
    "peak_rss_bytes": int(peak or 0),
    "gguf": gguf,
    "n_predict": int(n),
    "ctx": int(ctx),
    "engine": "ds4",
}
open(out, "w").write(json.dumps(doc, indent=2) + "\n")
print(json.dumps({k: doc[k] for k in ("ok", "exit_code", "contains_paris", "peak_rss_bytes")}, indent=2))
if not doc["ok"]:
    sys.exit(1)
PY
