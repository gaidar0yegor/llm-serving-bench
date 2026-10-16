#!/usr/bin/env python3
"""
kv_report.py — turn the KV sweep into the table a provider would actually price from.

Input: results/kv-sweep.json (written by kv_sweep.py) plus the per-config bench JSONs it points at.
Output: results/kv-sweep.md — for each --max-model-len: what the engine allocated (KV GiB, CUDA blocks,
its own concurrency ceiling), and then per concurrency level: aggregate output tok/s, p95 TTFT,
$/1M output tokens, peak GPU KV-cache usage and whether requests started waiting.

Output tokens only, 100% duty cycle, one resident model — the same accounting as cost_model.py.
The ceiling column is the engine's own arithmetic (blocks x block_size / max_model_len), quoted from
the server log, never inferred from the flag.

    python3 kv_report.py --sweep results/kv-sweep.json --md results/kv-sweep.md --usd-per-hour 0.20
"""

from __future__ import annotations

import argparse
import json
import os
import re

ROOT = os.path.dirname(os.path.abspath(__file__))
BLOCK_SIZE = 16          # vLLM default; cross-checked against cuda_blocks x block_size / max_model_len


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="results/kv-sweep.json")
    ap.add_argument("--md", default="results/kv-sweep.md")
    ap.add_argument("--usd-per-hour", default="0.20")
    args = ap.parse_args()

    sweep = json.load(open(os.path.join(ROOT, args.sweep)))
    prices = [float(x) for x in args.usd_per_hour.split(",")]

    lines = ["# KV budget sweep — `--max-model-len` as the provider's real lever", "",
             f"vLLM {sweep.get('engine', '')}", "",
             f"GPU: {sweep['gpu']}; prompts: mode `{sweep['prompt_mode']}`, "
             f"{sweep['max_tokens']} output tokens max, {sweep['requests_per_level']} requests per level.",
             f"Co-tenants: {sweep.get('host_note', '')}", "",
             "## What the engine allocated (quoted from its own log)", "",
             "| max-model-len | CUDA KV blocks | KV budget GiB | engine's concurrency ceiling | weights GiB | activation peak GiB | peak VRAM MiB |",
             "|---:|---:|---:|---:|---:|---:|---:|"]

    for r in sweep["runs"]:
        if not r.get("server_up"):
            lines.append(f"| {r['max_model_len']} | — | — | **server did not start** ({r['ready']}) | — | — | — |")
            continue
        lines.append(
            f"| {r['max_model_len']} | {r.get('cuda_blocks', '—')} | {r.get('kv_cache_gib', '—')} | "
            f"**{r.get('reported_max_concurrency', '—')}x** | {r.get('weights_gib', '—')} | "
            f"{r.get('activation_peak_gib', '—')} | {r.get('vram_peak_mib', '—')} |")

    lines += ["", "## Throughput and cost per level", ""]

    for r in sweep["runs"]:
        mml = r["max_model_len"]
        lines.append(f"### `--max-model-len {mml}` — ceiling {r.get('reported_max_concurrency', '—')}x, "
                     f"KV {r.get('kv_cache_gib', '—')} GiB")
        lines.append("")
        if not r.get("bench_ok"):
            lines.append("_no measurement: server did not come up._")
            lines.append("")
            continue
        d = json.load(open(os.path.join(ROOT, r["result_file"])))
        head = ("| conc | agg tok/s | tok/hour | p50 TTFT | p95 TTFT | itl p50 | decode tok/s | KV cache % | "
                "running | waiting | preemptions | failed | " + " | ".join(f"$/1M @ {p:.2f}" for p in prices) + " |")
        lines.append(head)
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|" + "---:|" * len(prices))
        for lvl in d["models"][0]["levels"]:
            tps = lvl["agg_out_tps"]
            tok_h = (tps or 0) * 3600
            s = lvl.get("server", {}) or {}
            costs = " | ".join(
                f"${p / tok_h * 1e6:.2f}" if tok_h else "—" for p in prices)
            lines.append(
                f"| {lvl['concurrency']} | {tps} | {tok_h:,.0f} | {lvl['ttft_p50_ms']} ms | "
                f"{lvl['ttft_p95_ms']} ms | {lvl['itl_p50_ms']} ms | {lvl['decode_tps_p50']} | "
                f"{s.get('gpu_cache_usage_perc', '—')} | {s.get('num_requests_running', '—')} | "
                f"{s.get('num_requests_waiting', '—')} | {s.get('num_preemptions_total', '—')} | "
                f"{lvl['requests_failed']} | {costs} |")
        lines.append("")
        # where the wall was in THIS config
        wall = [lvl for lvl in d["models"][0]["levels"]
                if (lvl.get("server", {}) or {}).get("gpu_cache_usage_perc") is not None]
        if wall:
            busiest = max(wall, key=lambda l: l["server"]["gpu_cache_usage_perc"])
            lines.append(f"Busiest level: conc {busiest['concurrency']} — GPU KV cache "
                         f"{busiest['server']['gpu_cache_usage_perc']}% used, "
                         f"{busiest['server'].get('num_requests_waiting')} waiting. "
                         f"{'KV is the wall here.' if busiest['server']['gpu_cache_usage_perc'] > 80 else 'KV was not the wall here.'}")
            lines.append("")

    md = "\n".join(lines) + "\n"
    with open(os.path.join(ROOT, args.md), "w") as f:
        f.write(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
