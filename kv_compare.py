#!/usr/bin/env python3
"""
kv_compare.py — one table across every measured engine configuration.

Joins the sweep manifests (`kv-sweep*.json`, written by kv_sweep.py) with the bench JSONs they point
at, and prints a single row per (KV budget, max-model-len, concurrency) level. The columns are the
ones a provider has to reason about at once: what the engine allocated (blocks, KV GiB, its own
concurrency ceiling), what it delivered (aggregate output tok/s, p95 TTFT) and what that costs per
million output tokens at a given GPU-hour price.

    python3 kv_compare.py --sweeps results/kv-sweep.json results/kv-sweep-util045.json \\
        results/kv-sweep-util055.json --usd-per-hour 0.20 --md results/kv-compare.md
"""

from __future__ import annotations

import argparse
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))


def util_of(entry: dict) -> str:
    eng = entry.get("_engine", "")
    for tok in eng.split():
        if tok.replace(".", "").isdigit() and tok.startswith("0."):
            return tok
    return "?"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweeps", nargs="+", required=True)
    ap.add_argument("--usd-per-hour", default="0.20")
    ap.add_argument("--md", default=None)
    args = ap.parse_args()
    price = float(args.usd_per_hour.split(",")[0])

    rows = []
    for path in args.sweeps:
        sweep = json.load(open(os.path.join(ROOT, path)))
        util = "?"
        eng = sweep.get("engine", "")
        # "... --gpu-memory-utilization 0.45" — quoted from the manifest, not guessed
        if "--gpu-memory-utilization" in eng:
            util = eng.split("--gpu-memory-utilization")[-1].strip()
        for run in sweep["runs"]:
            if not run.get("bench_ok"):
                continue
            bench = json.load(open(os.path.join(ROOT, run["result_file"])))
            for lvl in bench["models"][0]["levels"]:
                s = lvl.get("server", {}) or {}
                kv = s.get("gpu_cache_usage_perc")
                rows.append({
                    "util": util, "mml": run["max_model_len"], "conc": lvl["concurrency"],
                    "blocks": run.get("cuda_blocks"), "kv_gib": run.get("kv_cache_gib"),
                    "ceiling": run.get("reported_max_concurrency"),
                    "tok_s": lvl["agg_out_tps"], "ttft_p95": lvl["ttft_p95_ms"],
                    "itl_p50": lvl["itl_p50_ms"],
                    "usd_per_1m": round(price / (lvl["agg_out_tps"] * 3600) * 1e6, 3)
                    if lvl["agg_out_tps"] else None,
                    "kv_used_pct": round(kv * 100) if isinstance(kv, (int, float)) else None,
                    "waiting": s.get("num_requests_waiting"),
                    "failed": lvl["requests_failed"], "source": os.path.basename(path),
                })
    rows.sort(key=lambda r: (r["util"], r["mml"], r["conc"]))

    lines = ["# Every configuration, one table", "",
             f"Output tokens only; `$/1M` at **{price:.2f} $/GPU-h**, 100% duty, one model resident.",
             "`blocks` / `kv GiB` / `ceiling` are quoted from each server's own startup log.",
             "`KV used` and `waiting` come from the server's Prometheus endpoint (peak over the level); "
             "`—` means the endpoint did not answer for that run.", "",
             "| KV util | max-model-len | conc | blocks | KV GiB | ceiling | agg tok/s | p95 TTFT | itl p50 | $/1M | KV used | waiting |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(
            f"| {r['util']} | {r['mml']} | {r['conc']} | {r['blocks']} | {r['kv_gib']} | {r['ceiling']}x | "
            f"{r['tok_s']} | {r['ttft_p95']} ms | {r['itl_p50']} ms | ${r['usd_per_1m']} | "
            f"{r['kv_used_pct'] if r['kv_used_pct'] is not None else '—'}% | "
            f"{r['waiting'] if r['waiting'] is not None else '—'} |")
    md = "\n".join(lines) + "\n"
    if args.md:
        with open(os.path.join(ROOT, args.md), "w") as f:
            f.write(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
