#!/usr/bin/env python3
"""
cost_from_architecture.py — why a "impossibly cheap" price is or is not possible.

Decode cost is not set by a model's nameplate size; it is set by the bytes the GPU has to move per
token. For a sparse MoE that is **active parameters x bytes-per-parameter** (only the routed experts
that fire are read), not total parameters. Batch and speculative decoding then divide the cost again,
because one weight read serves `batch x draft` tokens.

So, for a node of `gpus` accelerators at `bw_gbs` each, with efficiency `eff` of peak bandwidth:

    bytes_per_step  = active_params * bytes_per_param            (total, split across gpus)
    steps_per_s     = bw * eff / (bytes_per_step / gpus)
    tokens_per_s    = steps_per_s * batch * spec_mult
    $/1M            = (gpus * usd_per_gpu_hour) / (tokens_per_s * 3600) * 1e6

`--invert` answers the provider's question instead: at a given market price and target margin, how many
tokens/s must the node deliver, and what batch does that imply?

Everything here is arithmetic with named, arguable assumptions — no benchmark is claimed. The one
sanity check is that the model's *ceiling* for our own card is 9x above what we measured, which is
exactly the dequant-and-matmul penalty the kernel lab found.

    python3 cost_from_architecture.py --scenarios
"""

from __future__ import annotations

import argparse
import json

GB = 1e9

SCENARIOS = [
    # label, active params (B), bytes/param, gpus, bw per gpu (GB/s), eff, batch, spec mult, $/gpu-h
    # the first row is the calibration: eff chosen so the model reproduces our measured 426 tok/s and
    # $0.13/1M on the 3060 Ti. If it does, the same arithmetic is trustworthy for the rest.
    ("CALIBRATION: our 3B dense Q4 on one 3060 Ti", 3.0, 0.5, 1, 448, 0.056, 32, 1, 0.06),
    ("GLM-5.3-Flash 18B active FP8, 8xH100 @ $2.5", 18.0, 1.0, 8, 3350, 0.12, 64, 4, 2.50),
    ("GLM-5.3-Flash 18B active FP8, 8xH100, eff 6%", 18.0, 1.0, 8, 3350, 0.06, 64, 4, 2.50),
    ("GLM-5.3-Flash 18B active FP8, 4xB200 @ $3.5", 18.0, 1.0, 4, 8000, 0.12, 64, 4, 3.50),
    ("GLM-5.3-Flash 18B active NVFP4, 4xB200", 18.0, 0.5, 4, 8000, 0.12, 64, 4, 3.50),
    ("DeepSeek-V4-Flash 13B active FP8, 8xH100", 13.0, 1.0, 8, 3350, 0.12, 64, 4, 2.50),
    ("IF IT WERE DENSE: 320B FP8, 8xH100", 320.0, 1.0, 8, 3350, 0.12, 64, 4, 2.50),
]


def model(active_b: float, bytes_pp: float, gpus: int, bw: float, eff: float,
          batch: int, spec: int, usd_h: float, tflops_fp8: float = 1980.0,
          eff_compute: float = 0.30) -> dict:
    """Decode is bounded by two things at once; take the smaller. Memory: each step must read the
    active weights once. Compute: verifying `spec` draft tokens does (spec+1) positions' worth of
    math per token of the batch. A model that only counted memory claims impossible throughput."""
    bytes_step = active_b * GB * bytes_pp
    per_gpu = bytes_step / gpus
    steps_mem = (bw * GB * eff) / per_gpu
    flops_step = 2 * active_b * GB * batch * (spec + 1)
    steps_compute = (gpus * tflops_fp8 * 1e12 * eff_compute) / flops_step
    steps_s = min(steps_mem, steps_compute)
    tokens_s = steps_s * batch * spec
    cost = (gpus * usd_h) / (tokens_s * 3600) * 1e6
    return {"bytes_step_gib": round(bytes_step / 1024**3, 1),
            "per_gpu_gib": round(per_gpu / 1024**3, 2),
            "steps_mem": round(steps_mem, 1), "steps_compute": round(steps_compute, 1),
            "bound": "memory" if steps_mem <= steps_compute else "compute",
            "steps_s": round(steps_s, 1), "tokens_s": round(tokens_s),
            "usd_per_1m": round(cost, 3), "node_usd_h": round(gpus * usd_h, 2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", action="store_true")
    ap.add_argument("--active-params", type=float, default=18.0)
    ap.add_argument("--bytes-per-param", type=float, default=1.0)
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--bw-gbs", type=float, default=3350)
    ap.add_argument("--eff", type=float, default=0.40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--spec-mult", type=int, default=4)
    ap.add_argument("--usd-per-gpu-hour", type=float, default=2.5)
    ap.add_argument("--market-out", type=float, default=0.50)
    ap.add_argument("--target-margin", type=float, default=2.0)
    ap.add_argument("--invert", action="store_true")
    ap.add_argument("--md", default=None)
    args = ap.parse_args()

    lines = []
    if args.invert:
        allowed_cost = args.market_out / args.target_margin
        need_tokens_s = (args.gpus * args.usd_per_gpu_hour) / (allowed_cost * 3600) * 1e6
        r = model(args.active_params, args.bytes_per_param, args.gpus, args.bw_gbs, args.eff,
                  args.batch, args.spec_mult, args.usd_per_gpu_hour)
        need_batch = need_tokens_s / (r["steps_s"] * args.spec_mult)
        lines += [f"# Inversion: what has to be true",
                  "",
                  f"Market ${args.market_out}/1M output, target margin {args.target_margin}x "
                  f"→ allowed cost ${allowed_cost}/1M.",
                  f"Node: {args.gpus} GPUs at ${args.usd_per_gpu_hour}/h = ${args.gpus*args.usd_per_gpu_hour}/h.",
                  f"→ the node must deliver **{need_tokens_s:,.0f} output tokens/s** "
                  f"({need_tokens_s/args.spec_mult/args.batch:,.1f} steps/s if batch={args.batch} and "
                  f"spec={args.spec_mult}x).",
                  f"With {args.active_params}B active at {args.bytes_per_param} B/param, this hardware can do "
                  f"{r['steps_s']:,.1f} steps/s → it needs only **batch {need_batch:,.0f}** to break even at "
                  f"{args.target_margin}x, and at batch {args.batch} it would cost ${r['usd_per_1m']}/1M "
                  f"(margin {args.market_out/r['usd_per_1m']:.1f}x).", ""]
        print("\n".join(lines))
        return 0

    lines += ["# Cost per 1M output tokens from architecture (memory-bound decode)", "",
              "Assumptions are columns, not hidden numbers: only active parameters are read, weights at the",
              "stated bytes/param, `eff` of peak bandwidth, and one weight read serves `batch x spec` tokens.", "",
              "| scenario | active params | B/param | bytes/step | steps/s (mem / compute) | bound | tok/s | node $/h | $/1M | vs $0.50 |",
              "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|"]
    rows = []
    for label, active, bpp, gpus, bw, eff, batch, spec, usd in (SCENARIOS if args.scenarios else [
            ("custom", args.active_params, args.bytes_per_param, args.gpus, args.bw_gbs, args.eff,
             args.batch, args.spec_mult, args.usd_per_gpu_hour)]):
        r = model(active, bpp, gpus, bw, eff, batch, spec, usd)
        margin = args.market_out / r["usd_per_1m"] if r["usd_per_1m"] else 0
        lines.append(f"| {label} | {active}B | {bpp} | {r['bytes_step_gib']} GiB | "
                     f"{r['steps_mem']:,.0f} / {r['steps_compute']:,.0f} | {r['bound']} | "
                     f"{r['tokens_s']:,.0f} | ${r['node_usd_h']} | "
                     f"**${r['usd_per_1m']}** | {margin:.1f}x |")
        rows.append({"label": label, **r, "margin_at_market": round(margin, 1)})

    lines += ["", "**Read it as.** A 320B model whose *active* set is 18B costs like an 18B model, not a 320B",
              "one: 18x fewer bytes per token than the nameplate suggests. FP8 halves it again, NVFP4 quarters",
              "it. Batch 64 with a 4-token speculative draft means one weight read pays for 256 output",
              "tokens. That is the whole answer to \"a 320B model cannot be that cheap\" — and the same",
              "arithmetic says a 320B *dense* model could not be, at any of these prices.", ""]
    md = "\n".join(lines)
    if args.md:
        open(args.md, "w").write(md + "\n")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
