# llm-serving-bench

Measure what an inference provider actually sells — **time-to-first-token, inter-token latency,
throughput and peak VRAM under concurrency** — against a real server, and publish the numbers
whether or not they are flattering.

Everything below was produced by `bench.py` on the machine described here. No number is estimated.

```
GPU    NVIDIA GeForce RTX 3060 Ti (8 GB, sm_86), driver 610.57.04
Server ollama 0.32.15 (llama.cpp backend), OpenAI-compatible /v1/chat/completions
Models qwen2.5:3b (Q4_K_M, 1.93 GB) · qwen2.5:7b (Q4_K_M, 4.68 GB)
Client Python + aiohttp, streaming SSE, temperature 0
Co-tenants  two local TTS services hold 2.8–3.9 GB of VRAM for the whole run (see "environment")
```

## Why this exists

Latency and cost in LLM serving are tail properties of a **shared** GPU, not properties of a model.
So this harness reports p50 **and p95** (never the mean), samples peak VRAM during the run instead of
reading it afterwards, takes token counts from the server's own `usage` field rather than guessing
from stream chunks, records how many requests failed, and names the prompt mode — because a
prefix-cache hit and a cold prefill are not the same measurement.

## Results

### qwen2.5:3b — the config that fits (100% GPU, 2.2 GB)

| concurrency | TTFT p50 | TTFT p95 | ITL p50 | ITL p95 | decode tok/s (per req) | aggregate tok/s | failed |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 34 ms | 40 ms | 7.64 ms | 8.19 ms | 131.1 | 125.0 | 0 |
| 2 | 621 ms | 682 ms | 7.05 ms | 7.72 ms | 142.1 | 132.4 | 0 |
| 4 | 1 211 ms | 1 801 ms | 6.97 ms | 7.18 ms | 143.7 | 135.7 | 0 |
| 8 | 2 388 ms | 4 150 ms | 7.01 ms | 7.05 ms | 143.1 | 135.9 | 0 |

Cold prefill (unique prompt per request, prefix cache defeated):

| concurrency | TTFT p50 | TTFT p95 | ITL p50 | aggregate tok/s |
|---:|---:|---:|---:|---:|
| 1 | 39.8 ms | 140 ms | 7.07 ms | 122.0 |
| 4 | 580 ms | 1 729 ms | 7.03 ms | 131.0 |

### qwen2.5:7b — the config that does *not* fit (76% GPU — 23 of 29 layers, 6 on CPU)

| concurrency | TTFT p50 | TTFT p95 | ITL p50 | decode tok/s | aggregate tok/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 268 ms | 1 176 ms | 177.8 ms | 3.8 | 2.6 |
| 2 | 1 457 ms | 12 606 ms | 118.8 ms | 7.6 | 6.4 |
| 4 | 6 430 ms | **60 760 ms** | 122.2 ms | 7.9 | 2.7 |

Peak VRAM during runs: 5 071 MiB (3B) / 6 719 MiB (7B) — both figures *include* ~2.8 GB held by the
co-tenant TTS services.

## Where the compute actually was: the server, not the silicon

Same box, same model (`qwen2.5:3b`, Q4_K_M), same prompts, same GPU price. The only things that
changed are the serving engine and how it was configured:

| engine / config | agg tok/s at C=1 | agg tok/s at C=8 | gain from 8× concurrency | TTFT p95 at C=8 | $/1M @ $0.20/h at C=8 |
|---|---:|---:|---:|---:|---:|
| ollama 0.32.15 (as shipped) | 125.0 | 135.9 | **+9%** | 4 150 ms | $0.41 |
| llama-server 0.1.2-dev, `-np 8 -cb` | 121.0 | 253.7 | **+110%** | 206.8 ms | $0.22 |
| **vLLM 0.8.5.post1, GGUF (V0 engine)** | 104.7 | **324.2** | **+210%** | **145.7 ms** | **$0.17** |

**2.4× cheaper per output token than the same box running its default server, and 28× better tail
latency — from configuration alone.** No kernel change, no quantization change, no new hardware.

Caveats that belong next to the number: vLLM served here through its GGUF path, which falls back to
the **V0 engine** (V1 does not accept GGUF yet), and vLLM itself warns that GGUF quantization is "not
fully optimized yet". A safetensors or AWQ model would very likely beat 324 tok/s. This is the floor
for that engine, not its ceiling.

The mechanism is visible in the servers' own counters (`bench.py --metrics-url`):

**llama.cpp (`-np 8`)** — slot-level counters:

| conc | processing | deferred | busy slots/decode | decode tok/s per request | ITL p50 |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0 | 1.00 | 132.9 | 7.61 ms |
| 8 | 8 | 0 | 1.98 | 39.5 | 25.64 ms |

**vLLM (`/metrics`)** — engine-level counters:

| conc | num_requests_running | num_requests_waiting | gpu_cache_usage_perc | num_preemptions | decode tok/s per request | ITL p50 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0 | 0.10 | 0 | 119.3 | 8.40 ms |
| 8 | 8 | 0 | 0.13 | 0 | 44.5 | 22.48 ms |

- **`requests_deferred` / `num_requests_waiting` never leave zero** on either engine. Nothing was ever
  queued. The 2.4-second TTFT measured on the first engine was **slot starvation**, not saturation — a
  single-slot server serialises work the hardware could have batched.
- **vLLM genuinely overlaps 8 requests** (`num_requests_running` = 8, zero preemptions) and pays for it
  in per-request decode speed (119.3 → 44.5 tok/s) while aggregate triples. That is continuous batching,
  measured rather than asserted.
- **Prefix cache hit rate reached 96.97%** (`Prefix cache hit rate: GPU: 96.97%` in the engine log, after
  0% → 94.12% on the first repetitions). With the same prompt repeated, vLLM answered almost entirely
  from its prefix cache — which is *why* TTFT sits at 39–146 ms instead of paying a full prefill. Honest
  consequence: every TTFT in this table is a **cache-hit** number. The cold-prefill case
  (`--prompt-mode unique`) has been measured on ollama only (≈40 ms at C=1, the cache worth ≈6 ms there)
  and belongs in the next round for vLLM.
- The KV budget was measured next, and the earlier assumption that it is the throughput wall **turned
  out to be wrong** — see "The KV budget" below. It is the wall for *long* requests (the engine's
  reported ceiling is blocks × block_size / `--max-model-len`), not for short-chat throughput.

## The KV budget: 4.5× the cache, none of the throughput

Two levers, same harness, same prompt, same card. All rows: vLLM 0.8.5.post1 on GGUF (V0 engine),
`qwen2.5:3b` Q4_K_M, 128 output tokens max, output tokens only. Manifests in `results/kv-sweep*.json`,
raw per-run JSON next to them; `kv_sweep.py` runs them, `kv_report.py` / `kv_compare.py` render them.

**Lever 1 — `--max-model-len` at a fixed memory budget (`--gpu-memory-utilization 0.45`):**

| max-model-len | CUDA KV blocks | KV budget | engine's concurrency ceiling | agg tok/s C=8 | C=16 | C=32 | $/1M @ $0.20/h (C=32) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 415 | 0.23 GiB | **6.48×** | 325.7 | 364.3 | 427.1 | $0.130 |
| 2048 | 415 | 0.23 GiB | **3.24×** | 328.9 | 364.0 | 426.3 | $0.130 |
| 4096 | 401 | 0.22 GiB | **1.57×** | 327.3 | 362.4 | 426.7 | $0.130 |

The reported ceiling moves by 4×; throughput does not move at all. That ceiling is the engine's worst
case — `blocks × block_size / max_model_len` — and these requests are ~830 tokens long, so the worst case
never binds. It caps the longest *admissible* request, not the throughput of short ones. (The one
side effect: at 1024/2048 with 32 requests in flight, `num_requests_waiting` rose to 8–12 while it stayed
0 at 4096 — consistent with a smaller token budget per scheduler step, not with KV exhaustion.)

**Lever 2 — the KV budget itself** (`--gpu-memory-utilization` 0.45 → 0.55), max-model-len 4096:

| KV budget | blocks | ceiling (4096 tok/req) | agg tok/s C=8 | C=32 | C=64 | KV used (peak) | max waiting |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.22 GiB | 401 | 1.57× | 326.4 | 417.1 | 440.3 | — (endpoint silent that run) | — |
| 0.99 GiB | 1795 | **7.01×** | 307.3 | 420.7 | 408.3 | **6%** | 28 |
| 0.99 GiB, max-model-len 1024 | 1809 | **28.27×** | 326.1 | 412.5 | 432.0 | 6% | 31 |

4.5× the cache and 18× the ceiling buy **no throughput** (differences are inside the ±8% run-to-run
scatter at C=64). Even with 0.99 GiB of KV the scheduler queues at C≥16 — up to 28 requests waiting —
while the cache sits at 3–6% used and preemptions stay at 0. So **the wall is compute, not memory**: at
C=64 the batch decodes ~7 steps/s, i.e. a few GB/s of weight traffic on a card with 448 GB/s. That is the
dequant-and-matmul bottleneck the kernel lab measured from the other side, where a fused W4A16 GEMM
landed 12% *behind* plain fp16 cuBLAS on the same GPU.

**Where concurrency stops paying.** C=32 → C=64 nearly doubles p95 TTFT (469 → 811 ms) and buys nothing
(420.7 → 408.3 tok/s). On this box the SLO frontier ends at **C=32**; anything past it is latency
purchased with a rounding error.

### Margin, not just cost

`margin_model.py` turns measured throughput into the provider's unit of account, with the price/latency
bar taken from **outside** — `market_prices.json`, collected from public provider pages (source URLs in
the file):

| market, output price | our $/1M @ $0.20/h (C=32) | margin | break-even level |
|---|---:|---:|---|
| deepseek-v4-flash $0.28 | $0.130 | +$0.150 (2.2×) | C=4 |
| glm-5.3-flash $0.46 (cheapest EU endpoint) | $0.130 | +$0.330 (3.5×) | C≈2–4 |
| glm-5.3-flash $0.50 (Z.AI / Together) | $0.130 | +$0.370 (3.8×) | C=2 |

The same file carries the providers' own measured tails — 409–697 ms TTFT, 77–129 tok/s per stream — as
the bar our p95 has to clear: C=32 (415–469 ms) clears it, C=64 (787–811 ms) does not.

Caveats, because they are large: output tokens only (prefill is not in this measurement set); our model is
a 3B Q4 on one 8 GB consumer card, so what transfers is the *arithmetic* of margin, not a claim about
serving GLM or DeepSeek-V4-Flash; and the card is shared with TTS co-tenants holding **3.9 GiB**, which is
what caps `--gpu-memory-utilization` at 0.55 in the first place. The KV budget, not the card, is the
honest limit of this experiment.

The conclusion for anyone paying for GPUs: **before optimising a kernel, count the slots and size the KV
cache.** The gap here (2.4× per token, 28× on the tail) is larger than anything a weight-decode kernel
could deliver on this device — our own fused W4A16 attempt landed 12% *behind* plain fp16 cuBLAS.

### Cost model

`cost_model.py` turns any result file into $/1M output tokens at a given GPU price, plus the cheapest
concurrency level that still meets a latency budget:

| engine, conc | aggregate tok/s | tok/hour | p95 TTFT | $/1M @ $0.20/h | $/1M @ $0.35/h |
|---|---:|---:|---:|---:|---:|
| llama.cpp, 8 | 253.7 | 913 320 | 206.8 ms | $0.22 | $0.38 |
| vLLM, 8 | 324.2 | 1 167 120 | 145.7 ms | **$0.17** | $0.30 |
| vLLM, 1 | 104.7 | 376 920 | 49.0 ms | $0.53 | $0.93 |

Price is a parameter, not a claim about anyone's invoice; output tokens only, 100% duty cycle, one
model resident. A real bill also charges prefill, which this measurement set does not contain.

## The three findings

**1. Concurrency is paid in TTFT, not in ITL.** On the healthy config, inter-token latency is flat at
~7 ms from C=1 to C=8 — decoding a stream does not get slower when other streams exist. What grows is
the wait *before* the first token: 34 ms → 621 → 1 211 → 2 388 ms (p50), and 4 150 ms at p95 for C=8.
Aggregate throughput moves from 125 to 136 tok/s: **8× the concurrency buys +9% throughput.**

The shape says the requests queue rather than batch: if the sequences shared a batched decode step,
the weight read (the dominant cost at this size) would be paid once for the whole batch and aggregate
throughput would show a multiple, not a rounding error. That is an inference from client-visible
data only — this backend exposes no queue-depth or running-vs-waiting metrics, so the scheduling policy
itself is *not* proven here. A vLLM run with `/metrics` would settle it. (Next measurement.)

**2. When the config doesn't fit, the failure is a cliff, not a slope.** `ollama ps` reports
`76% GPU / 24% CPU`; the server log says `offloaded 23/29 layers to GPU` and explains why: the GPU had
`4.3 GiB available` because the TTS co-tenants hold ~2.8 GB, so a 4.68 GB Q4 model cannot be placed
whole. Decode falls from 136 tok/s to **3.8 tok/s — a 35× cliff** at C=1, and the p95 TTFT at C=4
reaches **60.8 seconds**, i.e. a request waiting a full minute.

This is *not* a fair 3B-vs-7B comparison: it measures co-tenancy penalty **plus** model size. Reported
that way on purpose — the interesting engineering fact is that throughput is a property of the whole
box (who else holds VRAM), not of the model. A clean 7B baseline needs the GPU to itself; that
measurement is deliberately deferred, see below.

**3. The tail is the product.** The mean TTFT for the 7B config would have looked survivable. The p95
(p95 = 60.8 s) is the number that decides whether anyone keeps using the endpoint.

## What this is not

- **Not yet a vLLM / SGLang comparison.** vLLM 0.8.5.post1 is installed alongside (CUDA verified) and the
  same harness points at it — see "The KV budget" for what it measured. The engine-vs-engine table above
  is still ollama vs `llama-server`; SGLang is untouched.
- **Not a claim that configuration beats kernels always.** It beats kernels *here*, on this card, for
  this workload. The point is the ordering of effort: measure the serving config before rewriting the
  math.
- **Not a model comparison.** See finding 2: the 7B number is a co-tenancy number.
- **Not a statistical study.** One run per level, `n = 8` requests per level — enough to see the shape,
  not enough for confidence intervals. Repeats + CI are the obvious next step.
- **Not a datacenter GPU.** A single 8 GB consumer card, deliberately: the interesting constraints
  appear sooner.

## Deferred on purpose

1. **Clean 7B with the GPU to itself.** Two voice services hold ~3.9 GB; stopping them is the
   operator's call, not mine, so the number waits rather than the GPU getting rebooted at 22:00.
2. **A model served natively rather than through GGUF.** vLLM with GGUF drops to the V0 engine and warns
   that GGUF quantization is "not fully optimized", which is the most likely reason throughput saturates
   at ~420 tok/s on short requests. AWQ / GPTQ / fp16 safetensors would test whether the ceiling moves —
   the one remaining experiment that can distinguish "this card" from "this quant path".
3. **Cold prefill for vLLM** (`--prompt-mode unique`): every TTFT here is a prefix-cache hit, so the
   prefill half of an invoice is still unpriced.
4. ~~Context/KV-cache sweep~~ — **done**, see "The KV budget": the ceiling moves, throughput does not.

## Reproduce

```bash
pip install aiohttp
python3 bench.py --model qwen2.5:3b --concurrency 1,2,4,8 --requests 8 --max-tokens 128
python3 bench.py --model qwen2.5:3b --concurrency 1,4 --requests 6 --prompt-mode unique
python3 bench.py --model qwen2.5:7b --concurrency 1,2,4 --requests 6

# with server-side counters (the run that produced the slot finding):
#   /usr/lib/ollama/llama-server -m <model.gguf> -c 8192 -np 8 -cb --metrics --port 8081
python3 bench.py --base-url http://127.0.0.1:8081 --model qwen25-3b \
  --concurrency 1,2,4,8 --requests 8 --max-tokens 128 \
  --metrics-url http://127.0.0.1:8081/metrics --json results/llamaserver-np8-gpu.json

python3 cost_model.py --usd-per-hour 0.20,0.35,0.50   # -> cost_model.md
```

### Reproducing the KV sweep

```bash
# vLLM is launched and stopped by the sweep itself, one config at a time (never two on one GPU):
python3 kv_sweep.py --max-model-len 1024,2048,4096 --concurrency 1,2,4,8,16,32 --requests 32 \
  --out results/kv-sweep.json                                   # lever 1: the context cap
python3 kv_sweep.py --max-model-len 4096 --gpu-memory-utilization 0.45 \
  --concurrency 8,16,32,64 --requests 64 --tag util045 \
  --out results/kv-sweep-util045.json                           # lever 2: the KV budget
python3 kv_sweep.py --max-model-len 4096,1024 --gpu-memory-utilization 0.55 \
  --concurrency 8,16,32,64 --requests 64 --tag util055 \
  --out results/kv-sweep-util055.json

python3 kv_report.py  --sweep results/kv-sweep.json --usd-per-hour 0.20,0.35,0.50  # -> results/kv-sweep.md
python3 kv_compare.py --sweeps results/kv-sweep.json results/kv-sweep-util045.json \
  results/kv-sweep-util055.json --usd-per-hour 0.20 --md results/kv-compare.md
python3 margin_model.py --results results/vllm-gguf-mml4096.json \
  --market market_prices.json --usd-per-hour 0.20,0.35 --md results/margin-glm-ds.md
```

Server logs for every configuration were kept locally in `kv-logs/` (git-ignored, they run to hundreds of
KB each) — the KV blocks, the reserved KV bytes and the concurrency ceiling in the tables are quoted from
those logs, not recomputed from the flags; the parsed values are preserved in the `results/kv-sweep*.json`
manifests. Each server is killed and its VRAM released before the next one starts, because a second
resident model would change the measurement rather than repeat it.

`--prompt-mode same` reuses the server's prefix cache (a real production case); `unique` puts a nonce
in front of the prompt to force a cold prefill. Both are labelled in the output, because they are not
the same measurement. Raw JSON per run lands in `results/`; `results.md` holds the full tables.
