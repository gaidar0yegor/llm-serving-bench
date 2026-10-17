#!/usr/bin/env python3
"""
cost_per_answer.py — what one answer costs, measured in tokens first and in money second.

A price per million tokens is not a price anyone experiences. What a customer pays is
**tokens per answer x price**, and on a reasoning model most of those tokens are thinking the customer
never sees. This script takes bench JSONs (local or remote) and a price table and prints, per model and
per reasoning setting:

  * input tokens, billed output tokens and how many of them were reasoning — from the server's own usage
  * cost per answer, and the same cost with reasoning off, where a measurement exists
  * a monthly figure at a stated number of answers

Prices are an INPUT, never invented: --prices market_prices.json, matched by model id. Where a model id
has no price, the token counts are still printed — which is the useful part when the provider's own price
is behind a login.

    python3 cost_per_answer.py --results 'results/munito-*.json' --prices market_prices.json \\
        --answers-per-month 100000
"""

from __future__ import annotations

import argparse
import glob
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/munito-*.json")
    ap.add_argument("--prices", default="market_prices.json")
    ap.add_argument("--concurrency", type=int, default=None,
                    help="which level to read (default: the highest measured level)")
    ap.add_argument("--answers-per-month", type=int, default=100_000)
    ap.add_argument("--md", default="results/cost-per-answer.md")
    args = ap.parse_args()

    prices = json.load(open(os.path.join(ROOT, args.prices)))

    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, args.results))):
        d = json.load(open(path))
        if not d.get("models") or not d["models"][0]["levels"]:
            continue
        model = d["models"][0]["model"]
        levels = d["models"][0]["levels"]
        lvl = (next((l for l in levels if l["concurrency"] == args.concurrency), None)
               if args.concurrency else max(levels, key=lambda l: l["concurrency"]))
        if lvl is None or not lvl["requests_ok"]:
            continue
        ok = lvl["requests_ok"]
        rows.append({
            "file": os.path.basename(path), "model": model,
            "effort": d.get("reasoning_effort"), "concurrency": lvl["concurrency"],
            "in_tokens": lvl.get("prompt_tokens_mean"),
            "billed_out": round(lvl["out_tokens_total"] / ok),
            "reasoning": lvl.get("reasoning_tokens_mean"),
            "ttft_ms": lvl["ttft_p50_ms"], "ttft_answer_ms": lvl.get("ttft_answer_p50_ms"),
            "agg_tps": lvl["agg_out_tps"],
        })

    lines = ["# Cost per answer", "",
             f"Billed output tokens are the server's own `usage.completion_tokens` — on a reasoning model "
             f"they include the thinking the customer never sees. Monthly figures assume "
             f"{args.answers_per_month:,} answers.", "",
             "| model | effort | conc | in tokens | billed out tokens/answer | of which reasoning | TTFT | TTFT to answer | cost @ in/out price | /month |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        price = None
        for _, m in prices.items():
            if isinstance(m, dict) and not _.startswith("_") and r["model"] in (m.get("model_ids") or []):
                price = m
        cost_s = per_month_s = "—"
        if price and r["in_tokens"] is not None:
            cost = (r["in_tokens"] * price["in"] + r["billed_out"] * price["out"]) / 1e6
            cost_s = f"${cost:.6f}"
            per_month_s = f"${cost * args.answers_per_month:,.0f}"
        lines.append(
            f"| `{r['model']}` | {r['effort'] or 'default'} | {r['concurrency']} | {r['in_tokens']} | "
            f"**{r['billed_out']}** | {r['reasoning'] if r['reasoning'] is not None else '—'} | "
            f"{r['ttft_ms']} ms | {r['ttft_answer_ms'] or '—'} ms | {cost_s} | {per_month_s} |")

    lines += ["", "**Read it as: the same three-bullet answer costs N tokens, and N is mostly thinking.** "
                  "Where a model has both a default and a `low` reasoning measurement, the cheaper row is "
                  "the same answer with the thinking turned off — that difference is the largest single "
                  "lever visible from outside a provider, and it does not require knowing their prices.",
              "", "_Prices are an input from the market table; where a model has no price row, the token "
                  "counts stand alone._"]
    md = "\n".join(lines) + "\n"
    open(os.path.join(ROOT, args.md), "w").write(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
