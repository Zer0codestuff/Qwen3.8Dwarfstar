#!/usr/bin/env python3
"""Run greedy Qwen 3.8 prompts through ./ds4 and record tok/s plus RSS."""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GGUF = os.environ.get(
    "QWEN_GGUF",
    str(
        Path.home()
        / ".cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF"
        / "snapshots/f1bfb127c64f7072bdd2cad55f258b9c8b2910fe"
        / "Qwen3.8-27B-Q3_K_M.gguf"
    ),
)
CTX = int(os.environ.get("CTX", "512"))
DS4 = ROOT / "ds4"
PROMPTS = ROOT / ".audit/qwen-hard-prompts.json"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / ".audit/qwen-hard-run.json"


def dump_tokens(prompt: str) -> list[int]:
    r = subprocess.run(
        [str(DS4), "-m", GGUF, "--dump-tokens", "-p", prompt],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-2000:])
    for line in r.stdout.splitlines():
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            return json.loads(line)
    raise RuntimeError("no token id list in dump-tokens stdout")


def run_one(item: dict) -> dict:
    n = int(item["n_predict"])
    prompt = item["prompt"]
    ids = dump_tokens(prompt)
    stderr_path = ROOT / ".audit" / f"qwen-{item['id']}.stderr"
    stdout_path = ROOT / ".audit" / f"qwen-{item['id']}.stdout"
    cmd = [
        "/usr/bin/time",
        "-l",
        str(DS4),
        "-m",
        GGUF,
        "--ssd-streaming",
        "--ctx",
        str(CTX),
        "-n",
        str(n),
        "--temp",
        "0",
        "-p",
        prompt,
    ]
    with stdout_path.open("w") as out, stderr_path.open("w") as err:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=out, stderr=err, check=False)
    text = stdout_path.read_text()
    err = stderr_path.read_text()
    prefill = gen = None
    m = re.search(r"prefill: ([0-9.]+) t/s, generation: ([0-9.]+) t/s", err)
    if m:
        prefill = float(m.group(1))
        gen = float(m.group(2))
    peak = 0
    for line in err.splitlines():
        if "maximum resident set size" in line:
            peak = int(line.split()[0])
    real = None
    for line in err.splitlines():
        if " real " in line or re.match(r"\s+[0-9.]+ real", line):
            parts = line.split()
            try:
                real = float(parts[0])
            except ValueError:
                pass
    return {
        "id": item["id"],
        "title": item["title"],
        "prompt": prompt,
        "expect": item["expect"],
        "n_predict": n,
        "prompt_tokens": len(ids),
        "prompt_ids": ids,
        "exit_code": proc.returncode,
        "text": text,
        "prefill_tok_s": prefill,
        "generation_tok_s": gen,
        "elapsed_sec": real,
        "peak_rss_bytes": peak,
        "ctx": CTX,
    }


def main() -> int:
    if not DS4.is_file():
        print("build ./ds4 first", file=sys.stderr)
        return 2
    items = json.loads(PROMPTS.read_text())
    runs = []
    for item in items:
        print(f"running {item['id']} n={item['n_predict']}", file=sys.stderr, flush=True)
        rec = run_one(item)
        runs.append(rec)
        print(
            json.dumps(
                {
                    "id": rec["id"],
                    "exit": rec["exit_code"],
                    "prompt_tokens": rec["prompt_tokens"],
                    "prefill_tok_s": rec["prefill_tok_s"],
                    "generation_tok_s": rec["generation_tok_s"],
                    "peak_rss_gb": round(rec["peak_rss_bytes"] / 1e9, 2),
                }
            ),
            file=sys.stderr,
            flush=True,
        )
    OUT.write_text(json.dumps({"gguf": GGUF, "runs": runs}, indent=2) + "\n")
    print(str(OUT))
    return 0 if all(r["exit_code"] == 0 for r in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
