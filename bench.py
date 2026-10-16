#!/usr/bin/env python3
"""
llm-serving-bench — honest measurement of an OpenAI-compatible LLM server.

Measures what an inference provider actually sells: time-to-first-token, inter-token
latency, per-request and aggregate output throughput, and peak VRAM — under concurrency.

Design rules (why this file exists):
  * Every number comes from a real HTTP streaming call. Nothing is estimated.
  * Latency is reported as p50/p95, never as a mean: serving SLOs are tail SLOs.
  * Token counts come from the server's own `usage` when it provides it; falling back to
    counting stream deltas is recorded explicitly in the output, because the two are not
    the same number.
  * The prompt mode is recorded: `same` reuses the server's prefix cache (fast TTFT, a real
    production case), `unique` forces a cold prefill on every request.
  * Failures are counted and reported. A benchmark that silently drops erroring requests
    is a benchmark that lies.

Usage:
    python3 bench.py --model qwen2.5:3b --concurrency 1,2,4 --requests 6 --max-tokens 128
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import threading
import time
import uuid

try:
    import aiohttp
except ImportError:  # pragma: no cover
    sys.exit("aiohttp is required: pip install aiohttp")


# ---------------------------------------------------------------- VRAM sampling

class VramSampler(threading.Thread):
    """Poll nvidia-smi so we can report *peak* VRAM, not a value read after the fact."""

    def __init__(self, interval: float = 0.2) -> None:
        super().__init__(daemon=True)
        self.interval = interval
        self.peak_mib = 0
        self._stop = threading.Event()
        self.available = True

    def _read(self) -> int | None:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            return int(out.stdout.strip().splitlines()[0])
        except Exception:
            self.available = False
            return None

    def run(self) -> None:
        while not self._stop.is_set():
            v = self._read()
            if v is None:
                return
            self.peak_mib = max(self.peak_mib, v)
            self._stop.wait(self.interval)

    def stop(self) -> None:
        self._stop.set()
        self.join(timeout=2)


class MetricsSampler(threading.Thread):
    """Poll a Prometheus endpoint so a level can be annotated with what the *server* was doing.

    Client-side latency says a request waited; only the server can say whether it waited behind a
    full queue or behind a single decode slot. `requests_deferred` (queue) and
    `n_busy_slots_per_decode` (sequences actually batched into one decode step) are the two series
    that separate those cases — they are the reason this exists.
    """

    SERIES = ("requests_processing", "requests_deferred", "n_busy_slots_per_decode",
              "prompt_tokens_cached_total", "prompt_tokens_total",
              # vLLM (Prometheus) uses different names for the same ideas
              "num_requests_running", "num_requests_waiting", "gpu_cache_usage_perc",
              "num_preemptions_total", "prefix_cache_hit_rate")

    def __init__(self, url: str, interval: float = 0.15) -> None:
        super().__init__(daemon=True)
        self.url = url
        self.interval = interval
        self.available = True
        self._stop = threading.Event()
        self._max: dict[str, float] = {}
        self._last: dict[str, float] = {}

    def _poll(self) -> None:
        import urllib.request
        with urllib.request.urlopen(self.url, timeout=3) as r:      # noqa: S310 - local endpoint
            text = r.read().decode("utf-8", "ignore")
        for line in text.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            name, _, value = line.rpartition(" ")
            # series may carry labels: vllm:num_requests_running{model_name="..."} 0.0
            short = name.split(":")[-1].split("{")[0]
            if short in self.SERIES:
                try:
                    v = float(value)
                except ValueError:
                    continue
                self._last[short] = v
                self._max[short] = max(self._max.get(short, 0.0), v)

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                self._poll()
            except Exception:                                       # noqa: BLE001
                self.available = False
                return
            self._stop.wait(self.interval)

    def reset(self) -> None:
        self._max = {}

    def snapshot(self) -> dict:
        if not self.available:
            return {"error": "metrics endpoint unavailable"}
        return {k: round(v, 2) for k, v in self._max.items()}

    def stop(self) -> None:
        self._stop.set()
        self.join(timeout=2)


# ---------------------------------------------------------------- one request

async def one_request(session, url, model, prompt, max_tokens, timeout):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,          # determinism: we measure the server, not the sampler
        "seed": 42,
        "stream": True,
        "stream_options": {"include_usage": True},   # exact token counts when supported
    }
    t0 = time.perf_counter()
    ttft = None
    t_first = t_last = None
    deltas = 0
    usage = None
    err = None
    try:
        async with session.post(url, json=payload, timeout=timeout) as r:
            if r.status != 200:
                body = await r.text()
                return {"error": f"HTTP {r.status}: {body[:200]}"}
            async for raw in r.content:            # aiohttp iterates SSE lines
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
                for ch in obj.get("choices") or []:
                    txt = (ch.get("delta") or {}).get("content")
                    if txt:
                        now = time.perf_counter()
                        if ttft is None:
                            ttft = now - t0
                            t_first = now
                        t_last = now
                        deltas += 1
    except Exception as e:                          # noqa: BLE001 - we report, not hide
        err = f"{type(e).__name__}: {e}"
    t_end = time.perf_counter()

    if err:
        return {"error": err}
    if ttft is None:
        return {"error": "no content token received"}

    out_tokens = (usage or {}).get("completion_tokens") or deltas
    prompt_tokens = (usage or {}).get("prompt_tokens")
    return {
        "ttft_ms": ttft * 1000,
        "latency_ms": (t_end - t0) * 1000,
        "out_tokens": out_tokens,
        "token_count_source": "usage" if usage else "stream_deltas",
        "prompt_tokens": prompt_tokens,
        # inter-token latency: gaps *between* decoded tokens, so n-1 gaps
        "itl_ms": ((t_last - t_first) / (out_tokens - 1) * 1000) if out_tokens > 1 else None,
        "decode_tps": ((out_tokens - 1) / (t_last - t_first)) if out_tokens > 1 and t_last > t_first else None,
    }


async def run_level(session, args, model, prompt, concurrency, requests, sampler):
    """Launch `concurrency` requests at a time until `requests` are done; time wall-clock."""
    results = []
    inflight = []
    done = 0
    t_start = time.perf_counter()
    while done < requests:
        batch = min(concurrency, requests - done)
        inflight = [
            asyncio.create_task(one_request(session, args.url, model, prompt, args.max_tokens, args.timeout))
            for _ in range(batch)
        ]
        results.extend(await asyncio.gather(*inflight))
        done += batch
    wall = time.perf_counter() - t_start
    return results, wall


def pct(values, p):
    if not values:
        return None
    values = sorted(values)
    k = max(0, min(len(values) - 1, int(round((p / 100) * (len(values) - 1)))))
    return values[k]


def summarize(results, wall, concurrency):
    ok = [r for r in results if "error" not in r]
    bad = [r for r in results if "error" in r]
    ttfts = [r["ttft_ms"] for r in ok]
    itls = [r["itl_ms"] for r in ok if r["itl_ms"]]
    total_out = sum(r["out_tokens"] for r in ok)
    return {
        "concurrency": concurrency,
        "requests_ok": len(ok),
        "requests_failed": len(bad),
        "errors": sorted({r["error"] for r in bad})[:3],
        "wall_s": round(wall, 3),
        "ttft_p50_ms": round(pct(ttfts, 50), 1) if ttfts else None,
        "ttft_p95_ms": round(pct(ttfts, 95), 1) if ttfts else None,
        "itl_p50_ms": round(pct(itls, 50), 2) if itls else None,
        "itl_p95_ms": round(pct(itls, 95), 2) if itls else None,
        "decode_tps_p50": round(pct([r["decode_tps"] for r in ok if r["decode_tps"]], 50), 1) or None,
        "agg_out_tps": round(total_out / wall, 1) if wall > 0 else None,
        "out_tokens_total": total_out,
        "token_count_source": (ok[0]["token_count_source"] if ok else None),
    }


# ---------------------------------------------------------------- prompt

PARA = (
    "An inference provider sells tokens, and every token has a latency budget: the request "
    "arrives, the model prefills the prompt, and then it decodes one token at a time, over and "
    "over, until it stops. The interesting question is never the average case but the tail: "
    "what happens to the ninth request when eight are already in flight on the same GPU. "
)

def build_prompt(mode: str) -> str:
    base = PARA * 8 + "\n\nSummarize the trade-off described above in exactly three short bullet points."
    if mode == "unique":
        # a unique nonce at the very front defeats any prefix cache
        return f"[session {uuid.uuid4()}] " + base
    return base


async def server_meta(session, base_url):
    meta = {"base_url": base_url}
    try:
        async with session.get(f"{base_url}/api/version", timeout=10) as r:
            if r.status == 200:
                meta["ollama_version"] = (await r.json()).get("version")
    except Exception:
        pass
    return meta


async def main_async(args):
    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    args.url = url
    prompt = build_prompt(args.prompt_mode)
    concurrency = [int(c) for c in args.concurrency.split(",")]
    sampler = VramSampler()
    sampler.start()
    msampler = None
    if args.metrics_url:
        msampler = MetricsSampler(args.metrics_url)
        msampler.start()
    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "prompt_mode": args.prompt_mode,
        "prompt_chars": len(prompt),
        "max_tokens": args.max_tokens,
        "requests_per_level": args.requests,
        "concurrency_levels": concurrency,
        "gpu": os.environ.get("BENCH_GPU", "NVIDIA GeForce RTX 3060 Ti (8 GB)"),
        "host_note": args.host_note,
        "models": [],
    }
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        report.update(await server_meta(session, args.base_url))
        for model in args.model.split(","):
            model = model.strip()
            print(f"\n=== {model} ===", flush=True)
            # warmup: loads weights + allocates KV cache, excluded from the numbers
            t_warm = time.perf_counter()
            warm = await one_request(session, url, model, prompt, 8, args.timeout)
            warm_s = time.perf_counter() - t_warm
            print(f"  warmup: {warm_s:.1f}s {'OK' if 'error' not in warm else warm['error']}", flush=True)
            levels = []
            for c in concurrency:
                print(f"  concurrency {c} ...", flush=True)
                if msampler is not None:
                    msampler.reset()
                res, wall = await run_level(session, args, model, prompt, c, args.requests, sampler)
                row = summarize(res, wall, c)
                if msampler is not None:
                    row["server"] = msampler.snapshot()
                levels.append(row)
                extra = ""
                if msampler is not None and "error" not in row["server"]:
                    s = row["server"]
                    extra = (f" | server: processing<={s.get('requests_processing')} "
                             f"deferred<={s.get('requests_deferred')} "
                             f"busy_slots<={s.get('n_busy_slots_per_decode')}")
                print(f"    ttft p50 {row['ttft_p50_ms']} ms | itl p50 {row['itl_p50_ms']} ms | "
                      f"agg {row['agg_out_tps']} tok/s | fail {row['requests_failed']}{extra}", flush=True)
            report["models"].append({
                "model": model,
                "warmup_s": round(warm_s, 2),
                "levels": levels,
            })
    sampler.stop()
    if msampler is not None:
        msampler.stop()
    report["vram_peak_mib"] = sampler.peak_mib if sampler.available else None
    report["vram_note"] = args.vram_note
    print(f"\npeak VRAM during run: {report['vram_peak_mib']} MiB")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"wrote {args.json}")
    print("\n--- markdown ---")
    print(to_markdown(report))
    return report


def to_markdown(rep):
    lines = []
    for m in rep["models"]:
        lines.append(f"\n**{m['model']}** (warmup {m['warmup_s']} s)\n")
        has_server = any("server" in r and "error" not in r.get("server", {}) for r in m["levels"])
        head = "| conc | ttft p50 | ttft p95 | itl p50 | itl p95 | decode tok/s | aggregate tok/s | failed |"
        sep = "|---:|---:|---:|---:|---:|---:|---:|---:|"
        if has_server:
            head += " processing | deferred | busy slots/decode |"
            sep += "---:|---:|---:|"
        lines.append(head)
        lines.append(sep)
        for r in m["levels"]:
            line = (f"| {r['concurrency']} | {r['ttft_p50_ms']} ms | {r['ttft_p95_ms']} ms | "
                    f"{r['itl_p50_ms']} ms | {r['itl_p95_ms']} ms | {r['decode_tps_p50']} | "
                    f"{r['agg_out_tps']} | {r['requests_failed']} |")
            if has_server:
                s = r.get("server", {})
                line += (f" {s.get('requests_processing','—')} | {s.get('requests_deferred','—')} | "
                         f"{s.get('n_busy_slots_per_decode','—')} |")
            lines.append(line)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:11434")
    ap.add_argument("--model", required=True, help="comma-separated model names")
    ap.add_argument("--concurrency", default="1,2,4")
    ap.add_argument("--requests", type=int, default=6, help="total requests per concurrency level")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--prompt-mode", choices=["same", "unique"], default="same")
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--json", default="results.json")
    ap.add_argument("--metrics-url", default=None,
                    help="Prometheus endpoint (e.g. http://127.0.0.1:8081/metrics) to annotate each "
                         "level with server-side queue and batching state")
    ap.add_argument("--host-note", default="")
    ap.add_argument("--vram-note", default="")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
