# Results — raw tables

Generated 2026-10-16 ~22:00–22:20 CEST by `bench.py` (ollama 0.32.15).

## Environment

```
GPU      NVIDIA GeForce RTX 3060 Ti (8 GB, sm_86), driver 610.57.04
Host     single box; GPU shared with two TTS services for the entire run
         pid 3863139  /opt/kokoro-venv/bin/python    492 MiB
         pid 3863140  /opt/venv-xtts/bin/python     1970 MiB
         pid 211814   ./venv/bin/python              254 MiB
Server   ollama 0.32.15, OLLAMA_MODELS=/opt/from-ai-gpu/ollama/models
Client   python + aiohttp, SSE streaming, temperature 0, seed 42
Prompt   ~390 tokens, same for every request (mode=same) or prefixed with a uuid (mode=unique)
Tokens   taken from the server's `usage.completion_tokens` (exact), verified in every run
```

Server-side placement, from `ollama ps` and the ollama journal:

```
qwen2.5:3b   2.2 GB   100% GPU      4096 ctx
qwen2.5:7b   5.1 GB    24%/76% CPU/GPU   4096 ctx

journal (7b load):
  msg="gpu memory" available="4.3 GiB" free="4.8 GiB" minimum="457.0 MiB"
  common_params_fit_impl: id=0, n_layer=23, n_part=0, overflow_type=4, mem=3708 MiB
  load_tensors: offloaded 23/29 layers to GPU
  llama_kv_cache: size = 224.00 MiB (4096 cells, 28 layers, 1/1 seqs), K (f16), V (f16)
```

## Run 1 — qwen2.5:3b, prompt mode = same (prefix cache warm)

`results/qwen25-3b_same.json` · peak VRAM 5 071 MiB · warmup 0.09 s · 0 failures

| conc | ttft p50 | ttft p95 | itl p50 | itl p95 | decode tok/s p50 | agg tok/s | wall s | out tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 34.2 ms | 40.4 ms | 7.64 ms | 8.19 ms | 131.1 | 125.0 | 9.23 | 1 152 |
| 2 | 621.1 ms | 682.2 ms | 7.05 ms | 7.72 ms | 142.1 | 132.4 | 8.70 | 1 152 |
| 4 | 1 210.5 ms | 1 801.4 ms | 6.97 ms | 7.18 ms | 143.7 | 135.7 | 8.49 | 1 152 |
| 8 | 2 388.0 ms | 4 149.9 ms | 7.01 ms | 7.05 ms | 143.1 | 135.9 | 8.48 | 1 152 |

## Run 2 — qwen2.5:3b, prompt mode = unique (cold prefill, no prefix reuse)

`results/qwen25-3b_unique.json` · peak VRAM 5 071 MiB · warmup 0.2 s · 0 failures

| conc | ttft p50 | ttft p95 | itl p50 | decode tok/s p50 | agg tok/s | out tokens |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 39.8 ms | 140.3 ms | 7.07 ms | 136.3 | 122.0 | 864 |
| 4 | 579.6 ms | 1 728.9 ms | 7.03 ms | 142.2 | 131.0 | 864 |

Prefill of ~390 tokens costs ~40 ms on this GPU, i.e. **prefill is not the bottleneck here — the
queue is.** Removing the prefix cache changed TTFT at C=1 by ~6 ms.

## Run 3 — qwen2.5:7b, prompt mode = same (does not fit: 23/29 layers on GPU)

`results/qwen25-7b_same.json` · peak VRAM 6 719 MiB · warmup 19.82 s · 0 failures

| conc | ttft p50 | ttft p95 | itl p50 | itl p95 | decode tok/s p50 | agg tok/s | out tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 268.0 ms | 1 175.9 ms | 177.83 ms | — | 3.8 | 2.6 | 768 |
| 2 | 1 456.5 ms | 12 605.8 ms | 118.81 ms | — | 7.6 | 6.4 | 768 |
| 4 | 6 429.6 ms | 60 759.6 ms | 122.23 ms | — | 7.9 | 2.7 | 768 |

## Reading notes

- `decode tok/s` is per request: `(completion_tokens - 1) / (t_last_token - t_first_token)`.
- `itl` is the mean gap between decoded tokens within a stream, aggregated across requests as p50/p95.
- `agg tok/s` is `sum(completion_tokens) / wall_clock` for the level — what a provider bills.
- Every run reported `token_count_source = usage`; the delta-counting fallback in `bench.py` was
  never needed here.
- Peak VRAM is sampled at 5 Hz from `nvidia-smi` and therefore includes the co-tenant TTS services.
