# KV budget sweep — `--max-model-len` as the provider's real lever

vLLM vLLM 0.8.5.post1, GGUF (V0 engine), single GPU, --gpu-memory-utilization 0.45

GPU: NVIDIA GeForce RTX 3060 Ti (8 GB); prompts: mode `same`, 128 output tokens max, 32 requests per level.
Co-tenants: co-tenants hold ~2.8 GiB of the 8 GiB (TTS etc.); the KV budget is what is left

## What the engine allocated (quoted from its own log)

| max-model-len | CUDA KV blocks | KV budget GiB | engine's concurrency ceiling | weights GiB | activation peak GiB | peak VRAM MiB |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 415 | 0.23 | **6.48x** | 1.81 | 1.39 | 5757 |
| 2048 | 415 | 0.23 | **3.24x** | 1.81 | 1.39 | 5757 |
| 4096 | 401 | 0.22 | **1.57x** | 1.81 | 1.39 | 5749 |

## Throughput and cost per level

### `--max-model-len 1024` — ceiling 6.48x, KV 0.23 GiB

| conc | agg tok/s | tok/hour | p50 TTFT | p95 TTFT | itl p50 | decode tok/s | KV cache % | running | waiting | preemptions | failed | $/1M @ 0.20 | $/1M @ 0.35 | $/1M @ 0.50 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 91.4 | 329,040 | 59.8 ms | 69.2 ms | 9.17 ms | 109.1 | 10% | 1.0 | 0.0 | 0.0 | 0 | $0.61 | $1.06 | $1.52 |
| 2 | 121.1 | 435,960 | 63.9 ms | 108.2 ms | 14.55 ms | 68.8 | 10% | 2.0 | 0.0 | 0.0 | 0 | $0.46 | $0.80 | $1.15 |
| 4 | 229.4 | 825,840 | 85.1 ms | 92.3 ms | 15.0 ms | 66.7 | 11% | 4.0 | 0.0 | 0.0 | 0 | $0.24 | $0.42 | $0.61 |
| 8 | 325.7 | 1,172,520 | 138.9 ms | 147.6 ms | 22.48 ms | 44.5 | 12% | 8.0 | 0.0 | 0.0 | 0 | $0.17 | $0.30 | $0.43 |
| 16 | 364.3 | 1,311,480 | 231.5 ms | 232.9 ms | 37.33 ms | 26.8 | 14% | 16.0 | 12.0 | 0.0 | 0 | $0.15 | $0.27 | $0.38 |
| 32 | 427.1 | 1,537,560 | 403.4 ms | 404.6 ms | 70.05 ms | 14.3 | 18% | 32.0 | 10.0 | 0.0 | 0 | $0.13 | $0.23 | $0.33 |

Busiest level: conc 32 — peak GPU KV cache 18% used, up to 10.0 request(s) waiting. KV was not the wall here — the ceiling came from compute or from a scheduler limit, not from KV blocks.

### `--max-model-len 2048` — ceiling 3.24x, KV 0.23 GiB

| conc | agg tok/s | tok/hour | p50 TTFT | p95 TTFT | itl p50 | decode tok/s | KV cache % | running | waiting | preemptions | failed | $/1M @ 0.20 | $/1M @ 0.35 | $/1M @ 0.50 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 95.7 | 344,520 | 58.5 ms | 65.3 ms | 8.98 ms | 111.5 | 10% | 1.0 | 0.0 | 0.0 | 0 | $0.58 | $1.02 | $1.45 |
| 2 | 115.4 | 415,440 | 66.5 ms | 74.9 ms | 15.45 ms | 64.8 | 10% | 2.0 | 0.0 | 0.0 | 0 | $0.48 | $0.84 | $1.20 |
| 4 | 213.4 | 768,240 | 115.7 ms | 119.5 ms | 15.84 ms | 63.1 | 11% | 4.0 | 0.0 | 0.0 | 0 | $0.26 | $0.46 | $0.65 |
| 8 | 328.9 | 1,184,040 | 137.1 ms | 140.3 ms | 22.31 ms | 44.8 | 12% | 8.0 | 0.0 | 0.0 | 0 | $0.17 | $0.30 | $0.42 |
| 16 | 364.0 | 1,310,400 | 228.9 ms | 234.5 ms | 37.3 ms | 26.8 | 14% | 16.0 | 2.0 | 0.0 | 0 | $0.15 | $0.27 | $0.38 |
| 32 | 426.3 | 1,534,680 | 356.6 ms | 404.8 ms | 70.78 ms | 14.1 | 18% | 32.0 | 12.0 | 0.0 | 0 | $0.13 | $0.23 | $0.33 |

Busiest level: conc 32 — peak GPU KV cache 18% used, up to 12.0 request(s) waiting. KV was not the wall here — the ceiling came from compute or from a scheduler limit, not from KV blocks.

### `--max-model-len 4096` — ceiling 1.57x, KV 0.22 GiB

| conc | agg tok/s | tok/hour | p50 TTFT | p95 TTFT | itl p50 | decode tok/s | KV cache % | running | waiting | preemptions | failed | $/1M @ 0.20 | $/1M @ 0.35 | $/1M @ 0.50 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 107.2 | 385,920 | 35.2 ms | 37.5 ms | 8.29 ms | 120.7 | 10% | 1.0 | 0.0 | 0.0 | 0 | $0.52 | $0.91 | $1.30 |
| 2 | 114.4 | 411,840 | 64.8 ms | 113.0 ms | 15.47 ms | 64.7 | 11% | 2.0 | 0.0 | 0.0 | 0 | $0.49 | $0.85 | $1.21 |
| 4 | 202.5 | 729,000 | 118.3 ms | 135.5 ms | 16.32 ms | 61.5 | 11% | 4.0 | 0.0 | 0.0 | 0 | $0.27 | $0.48 | $0.69 |
| 8 | 327.3 | 1,178,280 | 140.5 ms | 149.5 ms | 22.27 ms | 44.9 | 13% | 8.0 | 0.0 | 0.0 | 0 | $0.17 | $0.30 | $0.42 |
| 16 | 362.4 | 1,304,640 | 231.2 ms | 253.5 ms | 37.21 ms | 26.9 | 14% | 16.0 | 0.0 | 0.0 | 0 | $0.15 | $0.27 | $0.38 |
| 32 | 426.7 | 1,536,120 | 413.5 ms | 414.8 ms | 70.04 ms | 14.3 | 19% | 32.0 | 8.0 | 0.0 | 0 | $0.13 | $0.23 | $0.33 |

Busiest level: conc 32 — peak GPU KV cache 19% used, up to 8.0 request(s) waiting. KV was not the wall here — the ceiling came from compute or from a scheduler limit, not from KV blocks.

