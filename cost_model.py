#!/usr/bin/env python3
"""
Cost per token — the number an inference provider actually sells.

Everything else in this repo measures the machine. This script turns those measurements into the
provider's unit of account: **dollars per million output tokens on a rented GPU**, and the SLO cost
of buying throughput with concurrency.

Method, stated so it can be argued with:
  * tokens/hour = measured aggregate output tokens/s x 3600 (100% duty cycle, one model resident)
  * $/1M tokens = GPU price per hour / tokens per hour x 1e6
  * the SLO column carries the p95 TTFT that concurrency level actually produced, because throughput
    bought with queueing is not throughput a customer will pay for
  * output tokens only. A real invoice also bills input tokens and prefill; those numbers are not in
    this measurement set and are therefore not invented here.

Reads the JSON written by bench.py. GPU price is a parameter, not a claim:
  python3 cost_model.py --usd-per-hour 0.20,0.35,0.50
"""

from __future__ import annotations

import argparse
import glob
import json
import os

# a TTFT p95 budget a real API customer would tolerate, in ms
SLO_BUDGETS_MS = (250, 500, 2000)


def load(paths):
    out = []
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        out.append((os.path.basename(p), d))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/*.json")
    ap.add_argument("--usd-per-hour", default="0.20,0.35,0.50",
                    help="GPU-hour price(s) to model; a parameter, not an estimate of any real invoice")
    ap.add_argument("--md", default="cost_model.md")
    args = ap.parse_args()

    prices = [float(x) for x in args.usd_per_hour.split(",")]
    files = sorted(glob.glob(args.results))

    report = {"usd_per_hour": prices, "slo_budgets_ms": list(SLO_BUDGETS_MS), "runs": []}
    lines = ["# Cost per million output tokens (measured throughput x GPU price)", "",
             "Throughput is what `bench.py` measured; price is a parameter. 100% duty cycle assumed,",
             "one model resident, output tokens only. The p95 TTFT column is what that throughput cost",
             "in queueing.", ""]

    for name, d in load(files):
        model = d["models"][0]["model"]
        run = {"result_file": name, "model": model, "prompt_mode": d.get("prompt_mode"),
               "levels": []}
        lines.append(f"## {model} — `{name}` (prompt mode: {d.get('prompt_mode')})")
        lines.append("")
        hdr = "| conc | aggregate tok/s | tok/hour | p95 TTFT | " + " | ".join(
            f"$/1M @ {p:.2f} $/h" for p in prices) + " |"
        lines.append(hdr)
        lines.append("|---:|---:|---:|---:|" + "---:|" * len(prices))
        for lvl in d["models"][0]["levels"]:
            tps = lvl["agg_out_tps"]
            tok_h = tps * 3600 if tps else None
            costs = [p / tok_h * 1e6 for p in prices] if tok_h else [None] * len(prices)
            lines.append(
                f"| {lvl['concurrency']} | {tps} | {tok_h:,.0f} | {lvl['ttft_p95_ms']} ms | "
                + " | ".join(f"${c:.2f}" for c in costs) + " |")
            run["levels"].append({"concurrency": lvl["concurrency"], "agg_out_tps": tps,
                                  "tok_per_hour": tok_h, "ttft_p95_ms": lvl["ttft_p95_ms"],
                                  "usd_per_1m": {f"{p:.2f}": (round(c, 2) if c else None)
                                                 for p, c in zip(prices, costs)}})

        # SLO frontier: the cheapest level that still fits a latency budget
        lines.append("")
        lines.append("**Cheapest level that still meets the latency budget** (p95 TTFT):")
        lines.append("")
        lines.append("| p95 budget | max concurrency | tok/s there | " +
                     " | ".join(f"$/1M @ {p:.2f}" for p in prices) + " |")
        lines.append("|---:|---:|---:|" + "---:|" * len(prices))
        for budget in SLO_BUDGETS_MS:
            ok = [l for l in run["levels"] if l["ttft_p95_ms"] is not None
                  and l["ttft_p95_ms"] <= budget and l["tok_per_hour"]]
            best = max(ok, key=lambda l: l["agg_out_tps"]) if ok else None
            if best is None:
                lines.append(f"| {budget} ms | — | — | " + " | ".join("—" for _ in prices) + " |")
            else:
                cells = " | ".join(f"${p / best['tok_per_hour'] * 1e6:.2f}" for p in prices)
                lines.append(f"| {budget} ms | {best['concurrency']} | {best['agg_out_tps']} | {cells} |")
                run.setdefault("slo_best", {})[f"{budget}ms"] = {
                    "concurrency": best["concurrency"], "agg_out_tps": best["agg_out_tps"],
                    "usd_per_1m": {f"{p:.2f}": round(p / best["tok_per_hour"] * 1e6, 2)
                                   for p in prices}}
        # what misplacement costs
        worst = min(run["levels"], key=lambda l: l["agg_out_tps"] or 1e9)
        best_tps = max(l["agg_out_tps"] or 0 for l in run["levels"])
        if worst["agg_out_tps"]:
            run["penalty_vs_best"] = round(best_tps / worst["agg_out_tps"], 1)
            lines.append("")
            lines.append(f"Cheapest-to-most-expensive spread inside this run: "
                         f"**{run['penalty_vs_best']}x** "
                         f"({worst['agg_out_tps']} tok/s at conc {worst['concurrency']} vs "
                         f"{best_tps} tok/s at its best).")
        lines.append("")
        report["runs"].append(run)

    md = "\n".join(lines)
    with open(args.md, "w") as f:
        f.write(md + "\n")
    with open("cost_model.json", "w") as f:
        json.dump(report, f, indent=2)
    print(md)


if __name__ == "__main__":
    main()
