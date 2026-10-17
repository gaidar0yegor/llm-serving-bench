#!/usr/bin/env python3
"""
prefill_profile.py — what a long prompt costs before the first token.

The serving sweep priced *output* tokens. Half of a real invoice is the input side: the prefill a
request pays before it can start decoding, and that cost grows with prompt length (attention, in
particular, grows faster than linearly). This script measures exactly that curve on one server:

    one vLLM process, prompts of ~1k / 2k / 4k / 8k tokens, every prompt unique (so the prefix cache
    cannot answer for it), concurrency 1 / 2 / 4, and it reports time-to-first-token, the prefill rate
    the server actually delivered, and the dollars per 1M *input* tokens that implies.

Honest accounting, stated up front:
  * TTFT at concurrency 1 is prefill + one decode step + client/Python overhead. The decode step is
    ~8-9 ms here and is NOT subtracted; the prefill rate below is therefore a slight under-estimate.
    It under-estimates every length the same way, so the curve is comparable.
  * input tokens are the server's own `usage.prompt_tokens`.
  * the price of a GPU-hour is an input parameter, not a claim about anyone's invoice.

    python3 prefill_profile.py --prompt-tokens 1024,2048,4096,8192 --out results/prefill-profile.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
VLLM_BIN = os.path.expanduser("~/venvs/vllm/bin/vllm")
MODEL = os.path.expanduser("~/models/gguf/qwen2.5-3b-instruct-q4_k_m.gguf")
TOKENIZER = "Qwen/Qwen2.5-3B-Instruct"

RE_BLOCKS = re.compile(r"# cuda blocks: (\d+)")
RE_KV = re.compile(r"reserved for KV Cache is ([\d.]+)GiB")
RE_MAXCONC = re.compile(r"Maximum concurrency for (\d+) tokens per request: ([\d.]+)x")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def http_ok(url: str, timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:      # noqa: S310 - local endpoint
            return r.status == 200
    except urllib.error.HTTPError:
        return False
    except Exception:                                                # noqa: BLE001
        return False


def gpu_used_mib() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:                                                # noqa: BLE001
        return None


def kill_servers(wait_s: int = 45) -> None:
    subprocess.run(["pkill", "-f", "[v]llm serve"], check=False)
    t0 = time.time()
    while time.time() - t0 < wait_s:
        time.sleep(2)
        used = gpu_used_mib()
        if used is not None and used < 4200:
            log(f"VRAM released after {time.time()-t0:.0f}s ({used} MiB used by co-tenants)")
            return
    log(f"WARNING: VRAM still at {gpu_used_mib()} MiB after {wait_s}s — continuing anyway")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-tokens", default="1024,2048,4096,8192")
    ap.add_argument("--concurrency", default="1,2,4")
    ap.add_argument("--requests", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--max-model-len", type=int, default=8704)
    ap.add_argument("--gpu-memory-utilization", default="0.55")
    ap.add_argument("--port", type=int, default=8082)
    ap.add_argument("--usd-per-hour", default="0.20")
    ap.add_argument("--ready-timeout", type=float, default=420)
    ap.add_argument("--out", default="results/prefill-profile.json")
    ap.add_argument("--md", default="results/prefill-profile.md")
    args = ap.parse_args()

    lengths = [int(x) for x in args.prompt_tokens.split(",")]
    prices = [float(x) for x in args.usd_per_hour.split(",")]
    logdir = os.path.join(ROOT, "kv-logs")
    os.makedirs(logdir, exist_ok=True)

    kill_servers()
    slog = os.path.join(logdir, "vllm-prefill-profile.log")
    cmd = [VLLM_BIN, "serve", MODEL, "--tokenizer", TOKENIZER, "--quantization", "gguf",
           "--port", str(args.port), "--max-model-len", str(args.max_model_len),
           "--gpu-memory-utilization", args.gpu_memory_utilization,
           "--enable-prefix-caching"]
    log("launch: " + " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=open(slog, "w"), stderr=subprocess.STDOUT,
                            start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < args.ready_timeout:
        if proc.poll() is not None:
            log(f"server exited with code {proc.returncode} — see {slog}")
            return 1
        if http_ok(f"http://127.0.0.1:{args.port}/health"):
            break
        time.sleep(2)
    else:
        log(f"server not ready after {args.ready_timeout}s")
        stop(proc)
        return 1
    log(f"server ready in {time.time()-t0:.0f}s")

    txt = open(slog, errors="ignore").read()
    m_blocks, m_kv, m_ceil = RE_BLOCKS.search(txt), RE_KV.search(txt), RE_MAXCONC.search(txt)
    alloc = {"cuda_blocks": int(m_blocks.group(1)) if m_blocks else None,
             "kv_cache_gib": float(m_kv.group(1)) if m_kv else None,
             "ceiling": float(m_ceil.group(2)) if m_ceil else None}

    profile = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gpu": "NVIDIA GeForce RTX 3060 Ti (8 GB)",
        "engine": f"vLLM 0.8.5.post1, GGUF (V0), --max-model-len {args.max_model_len}, "
                  f"--gpu-memory-utilization {args.gpu_memory_utilization}",
        "prompt_mode": "unique (prefix cache defeated -> cold prefill)",
        "measured": alloc,
        "usd_per_hour": prices,
        "runs": [],
    }

    for n in lengths:
        out_json = os.path.join(ROOT, "results", f"prefill-{n}.json")
        cmd = [sys.executable, os.path.join(ROOT, "bench.py"),
               "--base-url", f"http://127.0.0.1:{args.port}", "--model", MODEL,
               "--concurrency", args.concurrency, "--requests", str(args.requests),
               "--max-tokens", str(args.max_tokens), "--prompt-tokens", str(n),
               "--prompt-mode", "unique", "--json", out_json,
               "--metrics-url", f"http://127.0.0.1:{args.port}/metrics",
               "--host-note", "cold prefill; co-tenants hold ~3.9 GiB of the 8 GiB"]
        log(f"bench: prompt~{n} tokens, concurrency {args.concurrency}")
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        with open(os.path.join(ROOT, "prefill-bench.log"), "a") as f:
            f.write(f"\n===== prompt~{n} (exit {p.returncode}) =====\n{p.stdout}\n{p.stderr}\n")
        if p.returncode != 0:
            profile["runs"].append({"prompt_tokens_target": n, "error": f"bench exited {p.returncode}"})
            continue
        d = json.load(open(out_json))
        levels = d["models"][0]["levels"]
        c1 = next((l for l in levels if l["concurrency"] == 1), None)
        entry = {"prompt_tokens_target": n, "levels": levels,
                 "prompt_tokens_measured": c1.get("prompt_tokens_mean") if c1 else None}
        if c1 and c1.get("ttft_p50_ms") and c1.get("prompt_tokens_mean"):
            prefill_s = c1["ttft_p50_ms"] / 1000.0
            rate = c1["prompt_tokens_mean"] / prefill_s              # tokens/s of prefix ingested
            entry["prefill_tokens_per_s"] = round(rate, 1)
            entry["ms_per_1k_input_tokens"] = round(prefill_s * 1000 / (c1["prompt_tokens_mean"] / 1000), 2)
            entry["usd_per_1m_input"] = {f"{p:.2f}": round(p / (rate * 3600) * 1e6, 3) for p in prices}
        profile["runs"].append(entry)

    stop(proc)
    json.dump(profile, open(os.path.join(ROOT, args.out), "w"), indent=2, ensure_ascii=False)

    # ---- report
    lines = ["# Cold prefill: what a long prompt costs before the first token", "",
             f"vLLM {profile['engine']}", f"{profile['gpu']}; {profile['prompt_mode']}; "
             f"prompt lengths {', '.join(str(n) for n in lengths)} (target tokens).",
             f"Allocated: {alloc['cuda_blocks']} CUDA KV blocks, {alloc['kv_cache_gib']} GiB KV, "
             f"ceiling {alloc['ceiling']}x at max-model-len.", "",
             "| prompt tokens (measured) | C=1 TTFT p50 | C=1 TTFT p95 | C=2 p95 | C=4 p95 | prefill tok/s | ms / 1k input | "
             + " | ".join(f"$/1M input @ {p:.2f}" for p in prices) + " |",
             "|---:|---:|---:|---:|---:|---:|---:|" + "---:|" * len(prices)]
    for r in profile["runs"]:
        if "error" in r:
            lines.append(f"| {r['prompt_tokens_target']} | **{r['error']}** | — | — | — | — | — | "
                         + " | ".join("—" for _ in prices) + " |")
            continue
        by_c = {l["concurrency"]: l for l in r["levels"]}
        c1, c2, c4 = by_c.get(1, {}), by_c.get(2, {}), by_c.get(4, {})
        rate = r.get("prefill_tokens_per_s")
        cells = " | ".join(f"${r['usd_per_1m_input'][f'{p:.2f}']:.3f}" for p in prices) if rate else \
                " | ".join("—" for _ in prices)
        lines.append(
            f"| {r.get('prompt_tokens_measured')} | {c1.get('ttft_p50_ms')} ms | {c1.get('ttft_p95_ms')} ms | "
            f"{c2.get('ttft_p95_ms')} ms | {c4.get('ttft_p95_ms')} ms | {rate} | "
            f"{r.get('ms_per_1k_input_tokens')} | {cells} |")
    lines += ["",
              "`prefill tok/s` = the server's own prompt_tokens / TTFT at concurrency 1, i.e. it includes one "
              "decode step and client overhead — a slight under-estimate, applied equally to every length. "
              "Compare it with the price of a provider that sells input tokens: that gap is the input half of "
              "the margin.", ""]
    md = "\n".join(lines)
    open(os.path.join(ROOT, args.md), "w").write(md + "\n")
    print(md)
    return 0


def stop(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=30)
    except Exception:                                                # noqa: BLE001
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:                                            # noqa: BLE001
            pass


if __name__ == "__main__":
    sys.exit(main())
