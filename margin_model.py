#!/usr/bin/env python3
"""
margin_model.py — the provider's two computable quantities, in one place.

Alexey's framing, made arithmetic:
  * the latency bar comes from OUTSIDE: what other providers show for the same model (their published
    price and their measured TTFT / tok-s). It is an input here, not something we invent.
  * margin per 1M output tokens = market price per 1M output tokens - our measured cost per 1M,
    where our cost = GPU price per hour / (measured aggregate output tokens/s x 3600) x 1e6.

So this script joins two sources that are both explicit:
  --results   bench.py JSON (measured on this box: throughput and the tail that throughput cost)
  --market    a small JSON of the external bar, one entry per model:
              {"model": {"in": 0.15, "out": 0.50, "cache_in": 0.03, "ttft_ms": 697, "tps": 77,
                         "provider": "Melious (EU)"}}

Honest limits, printed with the table rather than buried:
  * output tokens only. The input side of the bill needs a prefill measurement this harness does not
    take (prompt mode `same` reuses the prefix cache); a blended margin would need `unique` runs.
  * our cost is for OUR model on OUR card. Quoting this margin against a much larger model's market
    price is an apples-to-oranges comparison unless the weight/quant/KV-footprint difference is stated:
    what transfers is the ARITHMETIC, not the number.

    python3 margin_model.py --results results/vllm-gguf-mml4096.json --usd-per-hour 0.20 \\
        --market market_prices.json
"""

from __future__ import annotations

import argparse
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="bench.py JSON to price")
    ap.add_argument("--market", default="market_prices.json")
    ap.add_argument("--usd-per-hour", default="0.20")
    ap.add_argument("--md", default=None)
    args = ap.parse_args()

    d = json.load(open(os.path.join(ROOT, args.results)))
    market = json.load(open(os.path.join(ROOT, args.market)))
    prices = [float(x) for x in args.usd_per_hour.split(",")]
    levels = d["models"][0]["levels"]

    lines = ["# Margin per million output tokens (measured cost vs market price)", "",
             f"Source run: `{args.results}` — {d['models'][0]['model']}", "",
             "Cost side (ours, measured): `$/1M = price_per_hour / (agg_tok_s * 3600) * 1e6`, "
             "100% duty, output tokens only.", ""]

    for name, m in market.items():
        if name.startswith("_") or not isinstance(m, dict):   # "_note": documentation, not a model
            continue
        lines += [f"## vs {name} — {m.get('provider', '')}: "
                  f"${m['in']}/1M in, **${m['out']}/1M out**"
                  + (f" (cached in ${m['cache_in']})" if m.get("cache_in") else ""), ""]
        if m.get("ttft_ms") or m.get("tps"):
            lines.append(f"Provider's own tail: TTFT {m.get('ttft_ms', '—')} ms, "
                         f"{m.get('tps', '—')} tok/s. That is the bar our p95 has to clear.")
            lines.append("")
        head = ("| conc | our agg tok/s | our p95 TTFT | " +
                " | ".join(f"our $/1M @ {p:.2f} $/h" for p in prices) +
                " | market $/1M | " +
                " | ".join(f"margin $/1M @ {p:.2f}" for p in prices) + " | " +
                " | ".join(f"margin × @ {p:.2f}" for p in prices) + " |")
        lines.append(head)
        lines.append("|---:|---:|---:|" + "---:|" * len(prices) + "---:|" + "---:|" * (2 * len(prices)))
        for lvl in levels:
            tps = lvl["agg_out_tps"] or 0
            tok_h = tps * 3600
            costs = [p / tok_h * 1e6 for p in prices] if tok_h else [None] * len(prices)
            cost_cells = " | ".join(f"${c:.3f}" if c else "—" for c in costs)
            # margin against the market output price, using each GPU-hour price in turn
            margins, mults = [], []
            for c in costs:
                if c and c > 0:
                    margins.append(f"${m['out'] - c:+.3f}")
                    mults.append(f"{m['out'] / c:.1f}x")
                else:
                    margins.append("—")
                    mults.append("—")
            lines.append(
                f"| {lvl['concurrency']} | {tps} | {lvl['ttft_p95_ms']} ms | {cost_cells} | "
                f"${m['out']:.2f} | " + " | ".join(margins) + " | " + " | ".join(mults) + " |")
        lines.append("")
        # break-even: which concurrency level is the first to turn a profit at each GPU price
        be = []
        for p in prices:
            ok = [l for l in levels
                  if (l["agg_out_tps"] or 0) * 3600 and
                  p / ((l["agg_out_tps"] or 0) * 3600) * 1e6 < m["out"]]
            be.append(f"{p:.2f} $/h → " + (f"conc {min(l['concurrency'] for l in ok)}"
                                           if ok else "never (cost > market price)"))
        lines.append("Break-even (first concurrency level whose measured cost is below the market "
                     "price): " + "; ".join(be) + ".")
        lines.append("")

    lines.append("_Output tokens only. Input/prefill cost is not in this measurement set, so the "
                 "blended invoice margin is not claimed here._")
    md = "\n".join(lines) + "\n"
    out_md = args.md or (os.path.splitext(args.results)[0] + "-margin.md")
    with open(os.path.join(ROOT, out_md), "w") as f:
        f.write(md)
    print(md)
    print(f"\nwrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
