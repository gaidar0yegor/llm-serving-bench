#!/usr/bin/env python3
"""
capacity_model.py — what a provider can actually sell on one GPU, and what it costs.

The serving sweep answered "how fast". This answers the provider's question: **for a given card,
model, context length and price, how many requests can run at once, and what is the margin?** It is
arithmetic on memory plus two measured rates, with every term labelled as computed or measured.

What is computed (deterministic, no GPU needed):
  * KV bytes per token = 2 x n_kv_heads x head_dim x n_layers x dtype_bytes
  * KV budget        = util x TOTAL_mib - weights - activation_peak - non_torch   (vLLM's semantics:
                       the fraction multiplies the card's TOTAL, not its free memory)
  * blocks           = KV budget / (block_size x kv_bytes_per_token)
  * ceiling          = blocks x block_size / max_model_len          (vLLM's printed number)
  * capacity at length L = how many requests of L tokens fit at once
  * feasibility      = util x TOTAL_mib <= TOTAL_mib - co_tenant_mib   (the OOM check we learned the
                       hard way: a config that fits on an idle card dies when a co-tenant grows)

What is measured (supplied, never invented):
  * aggregate output tokens/s at a given concurrency, and marginal prefill us per prompt token.
    Defaults are this repo's measurements; override them with --out-tok-s / --prefill-us-per-token.

Money is then pure arithmetic on those two: $/1M output, $/1M input, margin against a market price,
and the break-even throughput a given GPU price demands.

    python3 capacity_model.py --model qwen2.5-3b --total-mib 7844 --co-tenant-mib 3920 \\
        --util 0.45,0.55 --contexts 1024,2048,4096,8192,16384 --market glm-5.3-flash
    python3 capacity_model.py --selftest      # predict our own logged allocations and show the error
"""

from __future__ import annotations

import argparse
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
GIB = 1024 ** 3
MIB = 1024 ** 2


def load_specs() -> dict:
    return json.load(open(os.path.join(ROOT, "model_specs.json")))


def kv_bytes_per_token(spec: dict, dtype_bytes: int = 2) -> int:
    return 2 * spec["n_kv_heads"] * spec["head_dim"] * spec["n_layers"] * dtype_bytes


def allocate(spec: dict, total_mib: float, util: float, block_size: int,
             activation_gib: float, non_torch_gib: float, weights_gib: float,
             max_model_len: int) -> dict:
    """vLLM's own allocation arithmetic, transcribed from its startup log."""
    budget_mib = util * total_mib
    weights_mib = weights_gib * 1024
    act_mib = activation_gib * 1024
    non_torch_mib = non_torch_gib * 1024
    kv_mib = budget_mib - weights_mib - act_mib - non_torch_mib
    kv_per_block_mib = kv_bytes_per_token(spec) * block_size / MIB
    blocks = int(kv_mib / kv_per_block_mib) if kv_mib > 0 and kv_per_block_mib else 0
    return {
        "util": util, "budget_mib": round(budget_mib), "weights_mib": round(weights_mib),
        "activation_mib": round(act_mib), "kv_mib": round(kv_mib),
        "kv_gib": round(max(kv_mib, 0) / 1024, 2),
        "kv_bytes_per_token": kv_bytes_per_token(spec),
        "kv_mib_per_block": round(kv_per_block_mib, 4),
        "blocks": blocks,
        "ceiling": round(blocks * block_size / max_model_len, 2) if blocks else 0.0,
        "kv_tokens_total": blocks * block_size,
    }


def capacity_at(blocks: int, block_size: int, length: int) -> int:
    """How many requests of `length` tokens fit in the cache at once (pages are not shared)."""
    pages = -(-length // block_size)          # ceil
    return blocks // pages if pages else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-3b")
    ap.add_argument("--total-mib", type=float, default=7844,
                    help="what torch reports as the card's total (8192 minus driver reserve)")
    ap.add_argument("--co-tenant-mib", type=float, default=3920,
                    help="VRAM already held by other workloads; only the feasibility check uses it")
    ap.add_argument("--util", default="0.45")
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--activation-gib", type=float, default=1.40)
    ap.add_argument("--non-torch-gib", type=float, default=0.02)
    ap.add_argument("--max-model-len", type=int, default=8704)
    ap.add_argument("--contexts", default="1024,2048,4096,8192,16384")
    ap.add_argument("--out-tok-s", type=float, default=426.7,
                    help="measured aggregate output tok/s at the concurrency you sell (ours: C=32)")
    ap.add_argument("--out-concurrency", type=int, default=32)
    ap.add_argument("--prefill-us-per-token", type=float, default=7.14,
                    help="measured marginal prefill cost (ours: vLLM on this card)")
    ap.add_argument("--usd-per-hour", default="0.20")
    ap.add_argument("--market", default=None, help="key in market_prices.json to price against")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--md", default=None)
    args = ap.parse_args()

    specs = load_specs()
    spec = specs[args.model]
    market = json.load(open(os.path.join(ROOT, "market_prices.json"))) if args.market else None

    if args.selftest:
        return selftest(spec, args)

    lines = ["# Capacity and margin on one GPU", "",
             f"Model `{spec['name']}`: {spec['n_layers']} layers, {spec['n_kv_heads']} KV heads x "
             f"{spec['head_dim']} head-dim → **{kv_bytes_per_token(spec)/1024:.1f} KiB of KV per token** "
             f"({kv_bytes_per_token(spec)*1024/MIB:.1f} MiB per {args.block_size}-token block).", "",
             f"Card: {args.total_mib:.0f} MiB visible to torch, {args.co_tenant_mib:.0f} MiB held by "
             f"co-tenants → {args.total_mib - args.co_tenant_mib:.0f} MiB actually free for us.", ""]

    for util in [float(x) for x in args.util.split(",")]:
        alloc = allocate(spec, args.total_mib, util, args.block_size, args.activation_gib,
                         args.non_torch_gib, spec["weights_gib_q4_k_m"], args.max_model_len)
        fits = alloc["budget_mib"] <= (args.total_mib - args.co_tenant_mib)
        lines += [f"## `--gpu-memory-utilization {util}` — "
                  f"{'fits' if fits else '**will OOM: the budget exceeds free memory**'}", "",
                  f"Weights {alloc['weights_mib']} MiB + activation {alloc['activation_mib']} MiB + "
                  f"non-torch {round(args.non_torch_gib*1024)} MiB out of a {alloc['budget_mib']} MiB "
                  f"budget → **KV {alloc['kv_gib']} GiB = {alloc['blocks']} blocks "
                  f"({alloc['kv_tokens_total']} tokens of cache)**; the engine prints a ceiling of "
                  f"**{alloc['ceiling']}x** at max-model-len {args.max_model_len}.", "",
                  "| context per request | KV it needs | requests that fit at once |", "|---:|---:|---:|"]
        for L in [int(x) for x in args.contexts.split(",")]:
            need = L * alloc["kv_bytes_per_token"] / MIB
            lines.append(f"| {L} tokens | {need:.0f} MiB | **{capacity_at(alloc['blocks'], args.block_size, L)}** |")
        lines.append("")

    prices = [float(x) for x in args.usd_per_hour.split(",")]
    market_items = [(k, v) for k, v in (market or {}).items() if isinstance(v, dict)]
    prefill_tps = 1e6 / args.prefill_us_per_token
    lines += ["## Money", "",
              f"Rates on the table (measured, not modelled): **{args.out_tok_s} output tok/s** at "
              f"C={args.out_concurrency}, and **{args.prefill_us_per_token} µs per prompt token** of "
              f"marginal cold prefill (= {prefill_tps:,.0f} tokens/s).", "",
              "| GPU price | our $/1M output | our $/1M input | "
              + " | ".join(f"break-even tok/s vs ${m['out']:.2f}" for _, m in market_items) + " |",
              "|---:|---:|---:|" + "---:|" * len(market_items)]
    for p in prices:
        out_cost = p / (args.out_tok_s * 3600) * 1e6
        in_cost = p / (prefill_tps * 3600) * 1e6
        be = " | ".join(f"{p * 1e6 / (m['out'] * 3600):,.0f}" for _, m in market_items)
        lines.append(f"| ${p:.2f}/h | ${out_cost:.3f} | ${in_cost:.5f} | {be} |")
    lines += ["", "**How to read it.** The capacity table is arithmetic on the KV cache and does not "
                  "depend on how fast the model decodes: it is the number a customer will actually "
                  "consume (how many long conversations can be in flight). The money table is the two "
                  "measured rates converted to dollars. The break-even column is the inverse question a "
                  "provider asks: at this GPU price and that market price, how many output tokens/s do I "
                  "have to deliver so the token stops being sold below cost?",
              "", "Input tokens cost ~300x less than output tokens here, so a bill's input half is "
                  "margin-dominated — the output half is where the cost lives. And note what the "
                  "capacity table says about long context: the same 0.22 GiB cache that ignores short "
                  "requests admits a single 3.5k-token request, so a provider selling long-context "
                  "agents is selling the cache, not the FLOPs."]
    md = "\n".join(lines)
    if args.md:
        open(os.path.join(ROOT, args.md), "w").write(md + "\n")
    print(md)
    return 0


def selftest(spec: dict, args) -> int:
    """Predict the allocations vLLM logged in this repo and show the error. If this drifts badly, the
    calculator is wrong, not the logs.

    One term is not physics but engine profiling: the activation peak vLLM reserves, which wobbles by
    ~10 MiB with the profiling batch. So the test does not pretend to know it — it reads it *out of* the
    logs (everything else is pure arithmetic: budget = util x total, minus weights, minus non-torch),
    averages it, predicts with that, and reports the residual blocks. A single value must reproduce all
    three configurations; that is what makes the memory arithmetic trustworthy.
    """
    checks = [
        ("results/kv-sweep.json", 0.45, 4096, 401, 0.22, 1.57),
        ("results/kv-sweep-util055.json", 0.55, 4096, 1795, 0.99, 7.01),
        ("results/kv-sweep-util055.json", 0.55, 1024, 1809, 0.99, 28.27),
    ]
    kv_per_block = kv_bytes_per_token(spec) * args.block_size / MIB
    weights_mib, non_torch_mib = spec["weights_gib_q4_k_m"] * 1024, args.non_torch_gib * 1024
    implied = []
    for _, util, _, m_blocks, _, _ in checks:
        budget = util * args.total_mib
        implied.append((budget - weights_mib - non_torch_mib - m_blocks * kv_per_block) / 1024)
    act = sum(implied) / len(implied)
    print(f"implied activation peak from the logs: "
          + ", ".join(f"{v:.3f} GiB" for v in implied) + f"  → mean {act:.3f} GiB "
          f"(spread {max(implied)-min(implied):.3f} GiB)")
    print(f"\n{'config':>18} | {'blocks pred/meas':>19} | {'KV GiB pred/meas':>18} | "
          f"{'ceiling pred/meas':>19} | flag")
    ok = True
    for _, util, mml, m_blocks, m_kv, m_ceil in checks:
        a = allocate(spec, args.total_mib, util, args.block_size, act, args.non_torch_gib,
                     spec["weights_gib_q4_k_m"], mml)
        d_blocks, d_kv, d_ceil = a["blocks"] - m_blocks, a["kv_gib"] - m_kv, a["ceiling"] - m_ceil
        flag = "OK" if abs(d_blocks) <= 25 and abs(d_kv) <= 0.03 and abs(d_ceil) <= 0.6 else "DRIFT"
        ok &= flag == "OK"
        print(f"util {util} mml {mml:>5} | {a['blocks']:>8} / {m_blocks:<8} | "
              f"{a['kv_gib']:>8} / {m_kv:<8} | {a['ceiling']:>8} / {m_ceil:<8} | {flag}")
    print(f"\nselftest {'PASS' if ok else 'FAIL'} — residual blocks inside ±25 (<1.5% of ~1800), "
          f"KV inside ±0.03 GiB, ceiling inside ±0.6x; the only fitted term is the activation peak.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
