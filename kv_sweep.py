#!/usr/bin/env python3
"""
kv_sweep.py — the provider's trade-off, in one table.

Same harness, same prompts, same GPU memory budget; only `--max-model-len` changes. For each value
the sweep records the engine's *own* concurrency ceiling (CUDA KV blocks / max_model_len), the KV
budget it actually reserved, peak GPU KV-cache usage under load, throughput and the resulting
$/1M output tokens. The point is to turn "short context is cheaper per token" from folklore into a
measurement, including its cost: the same flag that buys concurrency also caps the longest request
the box can serve.

Design rules:
  * One GPU, one resident model at a time — servers are strictly serialised (co-tenancy changes both
    VRAM placement and throughput, so a parallel run would not be a measurement).
  * Readiness is a status code, not a body: vLLM's /health answers 200 with an empty body.
  * Every server log is kept, and the ceiling numbers are parsed from it, not recalled from the
    command line: the flag says what we asked for, the log says what was allocated.
  * A server that fails to come up is recorded as a failure row. It is never silently skipped.

Usage:
    python3 kv_sweep.py --max-model-len 1024,2048,4096 --concurrency 1,2,4,8,16,32 --requests 32
"""

from __future__ import annotations

import argparse
import glob
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

RE_MAXCONC = re.compile(r"Maximum concurrency for (\d+) tokens per request: ([\d.]+)x")
RE_BLOCKS = re.compile(r"# cuda blocks: (\d+), # CPU blocks: (\d+)")
RE_KV = re.compile(r"reserved for KV Cache is ([\d.]+)GiB")
RE_WEIGHTS = re.compile(r"model weights take ([\d.]+)GiB")
RE_ACTIVATION = re.compile(r"PyTorch activation peak memory takes ([\d.]+)GiB")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def http_code(url: str, timeout: float = 3.0) -> int | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:      # noqa: S310 - local endpoint
            return r.status
    except urllib.error.HTTPError as e:                              # a real answer, just not 200
        return e.code
    except Exception:                                                # noqa: BLE001
        return None


def gpu_used_mib() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:                                                # noqa: BLE001
        return None


def kill_servers(wait_s: int = 45) -> None:
    """Stop any vLLM/llama-server left over from an earlier run and wait for VRAM to come back."""
    subprocess.run(["pkill", "-f", "[v]llm serve"], check=False)
    subprocess.run(["pkill", "-f", "[l]lama-server"], check=False)
    t0 = time.time()
    while time.time() - t0 < wait_s:
        time.sleep(2)
        # free memory = whatever the co-tenants (TTS etc.) hold, not zero
        used = gpu_used_mib()
        if used is not None and used < 3600:
            log(f"VRAM released after {time.time()-t0:.0f}s ({used} MiB used by co-tenants)")
            return
    log(f"WARNING: VRAM still at {gpu_used_mib()} MiB after {wait_s}s — continuing anyway")


def start_server(mml: int, port: int, logdir: str, gpu_util: str) -> tuple[subprocess.Popen, str]:
    path = os.path.join(logdir, f"vllm-mml{mml}.log")
    f = open(path, "w")                                              # noqa: SIM115 - handed to Popen
    cmd = [VLLM_BIN, "serve", MODEL,
           "--tokenizer", TOKENIZER, "--quantization", "gguf",
           "--port", str(port), "--max-model-len", str(mml),
           "--gpu-memory-utilization", gpu_util,
           "--enable-prefix-caching"]
    log("launch: " + " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    return proc, path


def wait_ready(proc: subprocess.Popen, port: int, timeout: float) -> tuple[bool, str]:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False, f"process exited with code {proc.returncode}"
        if http_code(f"http://127.0.0.1:{port}/health") == 200:
            return True, f"ready in {time.time()-t0:.0f}s"
        time.sleep(2)
    return False, f"not ready after {timeout}s"


def stop_server(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait(timeout=15)


def parse_log(path: str) -> dict:
    txt = open(path, errors="ignore").read()
    out: dict = {"log": os.path.basename(path)}
    if m := RE_MAXCONC.search(txt):
        out["reported_max_concurrency"] = float(m.group(2))
        out["reported_for_tokens"] = int(m.group(1))
    if m := RE_BLOCKS.search(txt):
        out["cuda_blocks"] = int(m.group(1))
    if m := RE_KV.search(txt):
        out["kv_cache_gib"] = float(m.group(1))
    if m := RE_WEIGHTS.search(txt):
        out["weights_gib"] = float(m.group(1))
    if m := RE_ACTIVATION.search(txt):
        out["activation_peak_gib"] = float(m.group(1))
    if "gguf quantization is not fully optimized" in txt:
        out["engine_note"] = "gguf quantization is not fully optimized (V0 engine)"
    return out


def run_bench(port: int, conc: str, requests: int, max_tokens: int, prompt_mode: str,
              json_out: str, host_note: str) -> tuple[bool, str]:
    cmd = [sys.executable, os.path.join(ROOT, "bench.py"),
           "--base-url", f"http://127.0.0.1:{port}",
           "--model", MODEL,
           "--concurrency", conc, "--requests", str(requests),
           "--max-tokens", str(max_tokens), "--prompt-mode", prompt_mode,
           "--metrics-url", f"http://127.0.0.1:{port}/metrics",
           "--json", json_out,
           "--host-note", host_note]
    log("bench: " + " ".join(cmd[2:]))
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=2400)
    with open(os.path.join(ROOT, "kv-sweep-bench.log"), "a") as f:
        f.write(f"\n===== mml run on port {port} (exit {p.returncode}) =====\n{p.stdout}\n{p.stderr}\n")
    return p.returncode == 0, p.stdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-model-len", default="1024,2048,4096")
    ap.add_argument("--concurrency", default="1,2,4,8,16,32")
    ap.add_argument("--requests", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--prompt-mode", default="same", choices=["same", "unique"])
    ap.add_argument("--port", type=int, default=8082)
    ap.add_argument("--gpu-memory-utilization", default="0.45")
    ap.add_argument("--ready-timeout", type=float, default=420)
    ap.add_argument("--out", default="results/kv-sweep.json")
    args = ap.parse_args()

    logdir = os.path.join(ROOT, "kv-logs")
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)

    mmls = [int(x) for x in args.max_model_len.split(",")]
    sweep: dict = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gpu": "NVIDIA GeForce RTX 3060 Ti (8 GB)",
        "engine": "vLLM 0.8.5.post1, GGUF (V0 engine), single GPU, --gpu-memory-utilization "
                  + args.gpu_memory_utilization,
        "prompt_mode": args.prompt_mode,
        "max_tokens": args.max_tokens,
        "requests_per_level": args.requests,
        "concurrency_levels": [int(c) for c in args.concurrency.split(",")],
        "host_note": "co-tenants hold ~2.8 GiB of the 8 GiB (TTS etc.); the KV budget is what is left",
        "runs": [],
    }

    for mml in mmls:
        log(f"===== max-model-len {mml} =====")
        kill_servers()
        proc, path = start_server(mml, args.port, logdir, args.gpu_memory_utilization)
        ok, why = wait_ready(proc, args.port, args.ready_timeout)
        log(f"server for mml={mml}: {why}")
        entry: dict = {"max_model_len": mml, "server_up": ok, "ready": why, "port": args.port,
                       "bench_ok": False, "result_file": None}
        entry.update(parse_log(path))
        if ok:
            json_out = os.path.join(ROOT, "results", f"vllm-gguf-mml{mml}.json")
            bench_ok, _ = run_bench(args.port, args.concurrency, args.requests, args.max_tokens,
                                    args.prompt_mode, json_out, sweep["host_note"])
            entry["bench_ok"] = bench_ok
            entry["result_file"] = os.path.relpath(json_out, ROOT) if bench_ok else None
            if bench_ok:
                d = json.load(open(json_out))
                entry["vram_peak_mib"] = d.get("vram_peak_mib")
        else:
            tail = "".join(open(path, errors="ignore").readlines()[-25:])
            entry["error_tail"] = tail[-1500:]
        stop_server(proc)
        sweep["runs"].append(entry)
        if not ok:
            log(f"mml={mml} FAILED: {why} — row recorded, not skipped")
        json.dump(sweep, open(os.path.join(ROOT, args.out), "w"), indent=2, ensure_ascii=False)

    kill_servers()
    log("sweep done")
    print(json.dumps([{k: v for k, v in r.items() if k not in ("error_tail",)} for r in sweep["runs"]],
                     indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
