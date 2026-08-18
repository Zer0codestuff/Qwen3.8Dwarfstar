#!/usr/bin/env python3
"""Print qwen35 GGUF metadata, per-layer tensor layout, and byte estimates."""

from __future__ import annotations

import argparse
import collections
import json
import os
import struct
import sys

GGML = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
    16: "IQ2_XXS",
}


def read_str(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", "replace")


def skip_val(f, t):
    if t in (0, 1, 7):
        f.read(1)
    elif t in (2, 3):
        f.read(2)
    elif t in (4, 5, 6):
        f.read(4)
    elif t in (10, 11, 12):
        f.read(8)
    elif t == 8:
        read_str(f)
    elif t == 9:
        et = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            skip_val(f, et)
    else:
        raise SystemExit(f"unknown GGUF value type {t}")


def read_val(f, t):
    if t == 4:
        return struct.unpack("<I", f.read(4))[0]
    if t == 5:
        return struct.unpack("<i", f.read(4))[0]
    if t == 6:
        return struct.unpack("<f", f.read(4))[0]
    if t == 7:
        return bool(f.read(1)[0])
    if t == 8:
        return read_str(f)
    if t == 10:
        return struct.unpack("<Q", f.read(8))[0]
    if t == 11:
        return struct.unpack("<q", f.read(8))[0]
    if t == 12:
        return struct.unpack("<d", f.read(8))[0]
    if t == 9:
        et = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        if n > 64 and et != 8:
            for _ in range(n):
                skip_val(f, et)
            return f"array[{et} x {n}]"
        return [read_val(f, et) for _ in range(n)]
    skip_val(f, t)
    return f"<type {t}>"


def parse(path):
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise SystemExit(f"not GGUF: {magic!r}")
        version = struct.unpack("<I", f.read(4))[0]
        n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
        kv = {}
        for _ in range(n_kv):
            k = read_str(f)
            t = struct.unpack("<I", f.read(4))[0]
            kv[k] = read_val(f, t)
        tensors = []
        for _ in range(n_tensors):
            name = read_str(f)
            nd = struct.unpack("<I", f.read(4))[0]
            dims = struct.unpack("<" + "Q" * nd, f.read(8 * nd))
            typ = struct.unpack("<I", f.read(4))[0]
            off = struct.unpack("<Q", f.read(8))[0]
            tensors.append({"name": name, "dims": list(dims), "type": typ, "offset": off})
    alignment = int(kv.get("general.alignment", 32))
    data_start = None
    for i, t in enumerate(tensors):
        if i + 1 < len(tensors):
            t["nbytes"] = tensors[i + 1]["offset"] - t["offset"]
        else:
            t["nbytes"] = size - t["offset"]
            # GGUF data section is after aligned metadata; nbytes from last
            # offset to EOF is the last tensor plus padding. Good enough.
    return {
        "path": path,
        "bytes": size,
        "version": version,
        "n_tensors": n_tensors,
        "alignment": alignment,
        "kv": {k: v for k, v in kv.items() if "tokenizer.ggml.tokens" not in k and "merges" not in k and "scores" not in k and "token_type" not in k},
        "tensors": tensors,
    }


def layer_kind(i, interval):
    if i % interval == interval - 1:
        return "full_attention"
    return "linear_attention"


def summarize(doc):
    kv = doc["kv"]
    arch = kv.get("general.architecture", "")
    prefix = f"{arch}." if arch else ""
    interval = int(kv.get(prefix + "full_attention_interval", 4))
    n_block = int(kv.get(prefix + "block_count", 0))
    type_counts = collections.Counter(GGML.get(t["type"], t["type"]) for t in doc["tensors"])
    layers = []
    for i in range(n_block):
        items = [t for t in doc["tensors"] if t["name"].startswith(f"blk.{i}.")]
        nbytes = sum(t.get("nbytes", 0) for t in items)
        kinds = sorted({GGML.get(t["type"], t["type"]) for t in items})
        names = [t["name"].split(".", 2)[-1] for t in items]
        kind = "mtp" if any(n.startswith("nextn.") for n in names) else layer_kind(i, interval)
        layers.append(
            {
                "index": i,
                "kind": kind,
                "nbytes": nbytes,
                "quants": kinds,
                "tensors": names,
            }
        )
    other = [t for t in doc["tensors"] if not t["name"].startswith("blk.")]
    return {
        "architecture": arch,
        "name": kv.get("general.name"),
        "file_gb": round(doc["bytes"] / 1e9, 3),
        "block_count": n_block,
        "full_attention_interval": interval,
        "embedding": kv.get(prefix + "embedding_length"),
        "ffn": kv.get(prefix + "feed_forward_length"),
        "heads": kv.get(prefix + "attention.head_count"),
        "kv_heads": kv.get(prefix + "attention.head_count_kv"),
        "head_dim": kv.get(prefix + "attention.key_length"),
        "context": kv.get(prefix + "context_length"),
        "nextn_predict_layers": kv.get(prefix + "nextn_predict_layers"),
        "ssm": {
            "conv_kernel": kv.get(prefix + "ssm.conv_kernel"),
            "state_size": kv.get(prefix + "ssm.state_size"),
            "group_count": kv.get(prefix + "ssm.group_count"),
            "time_step_rank": kv.get(prefix + "ssm.time_step_rank"),
            "inner_size": kv.get(prefix + "ssm.inner_size"),
        },
        "type_counts": dict(type_counts),
        "non_layer_bytes": sum(t.get("nbytes", 0) for t in other),
        "max_layer_bytes": max((L["nbytes"] for L in layers), default=0),
        "layers": layers,
        "other_tensors": [{"name": t["name"], "type": GGML.get(t["type"], t["type"]), "dims": t["dims"], "nbytes": t.get("nbytes", 0)} for t in other],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "gguf",
        nargs="?",
        default=os.path.expanduser(
            "~/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/"
            "snapshots/f1bfb127c64f7072bdd2cad55f258b9c8b2910fe/Qwen3.8-27B-Q3_K_M.gguf"
        ),
    )
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    doc = parse(args.gguf)
    summary = summarize(doc)
    if args.json:
        json.dump(summary, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return
    print(f"{summary['name']}  arch={summary['architecture']}  {summary['file_gb']} GB")
    print(
        f"blocks={summary['block_count']}  interval={summary['full_attention_interval']}  "
        f"d_model={summary['embedding']}  d_ff={summary['ffn']}  "
        f"heads={summary['heads']}/{summary['kv_heads']}  head_dim={summary['head_dim']}"
    )
    print(f"quants {summary['type_counts']}")
    print(f"non-layer {summary['non_layer_bytes']/1e6:.1f} MB  heaviest layer {summary['max_layer_bytes']/1e6:.1f} MB")
    print("ssm", summary["ssm"])
    for t in summary["other_tensors"]:
        print(f"  {t['name']:28} {t['type']:8} {t['nbytes']/1e6:8.1f} MB  {t['dims']}")
    for L in summary["layers"]:
        print(
            f"  blk.{L['index']:<3} {L['kind']:<18} {L['nbytes']/1e6:8.1f} MB  "
            f"{','.join(L['quants'])}  {', '.join(L['tensors'])}"
        )


if __name__ == "__main__":
    main()
