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
Co-tenants  two local TTS services hold ~2.8 GB of VRAM for the whole run (see "environment")
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
- The KV budget is the next wall, and it is a memory wall, not a compute one: at `--gpu-memory-utilization
  0.45` on this shared 8 GB card, vLLM reported `Maximum concurrency for 4096 tokens per request: 1.57x`
  — 401 GPU blocks, 0.22 GiB of KV. Short requests (≈518 tokens) fit 8 abreast at 13% cache usage;
  full-length 4096-token requests would queue. **What a provider sells is concurrent context, and it runs
  out before the FLOPs do.**

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

- **Not yet a vLLM / SGLang comparison.** vLLM 0.8.5.post1 is now installed alongside (CUDA verified)
  and the same harness points at it; the tables above are ollama vs `llama-server`. The interesting
  question — how far a PagedAttention engine gets past `busy_slots_per_decode ≈ 2` — is still open.
- **Not a claim that configuration beats kernels always.** It beats kernels *here*, on this card, for
  this workload. The point is the ordering of effort: measure the serving config before rewriting the
  math.
- **Not a model comparison.** See finding 2: the 7B number is a co-tenancy number.
- **Not a statistical study.** One run per level, `n = 8` requests per level — enough to see the shape,
  not enough for confidence intervals. Repeats + CI are the obvious next step.
- **Not a datacenter GPU.** A single 8 GB consumer card, deliberately: the interesting constraints
  appear sooner.

## Deferred on purpose

1. **Clean 7B with the GPU to itself.** Two voice services hold ~2.8 GB; stopping them is the
   operator's call, not mine, so the number waits rather than the GPU getting rebooted at 22:00.
2. **vLLM with server-side metrics** — queue depth, running vs waiting, KV-cache utilisation — to
   turn finding 1 from "consistent with queueing" into "measured".
3. **Context/KV-cache sweep** — same model, growing `num_ctx`, to price the context window.

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

`--prompt-mode same` reuses the server's prefix cache (a real production case); `unique` puts a nonce
in front of the prompt to force a cold prefill. Both are labelled in the output, because they are not
the same measurement. Raw JSON per run lands in `results/`; `results.md` holds the full tables.
