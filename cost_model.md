# Cost per million output tokens (measured throughput x GPU price)

Throughput is what `bench.py` measured; price is a parameter. 100% duty cycle assumed,
one model resident, output tokens only. The p95 TTFT column is what that throughput cost
in queueing.

## qwen25-3b — `llamaserver-np8-gpu.json` (prompt mode: same)

| conc | aggregate tok/s | tok/hour | p95 TTFT | $/1M @ 0.20 $/h | $/1M @ 0.35 $/h | $/1M @ 0.50 $/h |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 121.0 | 435,600 | 74.8 ms | $0.46 | $0.80 | $1.15 |
| 2 | 180.7 | 650,520 | 207.1 ms | $0.31 | $0.54 | $0.77 |
| 4 | 222.7 | 801,720 | 254.3 ms | $0.25 | $0.44 | $0.62 |
| 8 | 253.7 | 913,320 | 206.8 ms | $0.22 | $0.38 | $0.55 |

**Cheapest level that still meets the latency budget** (p95 TTFT):

| p95 budget | max concurrency | tok/s there | $/1M @ 0.20 | $/1M @ 0.35 | $/1M @ 0.50 |
|---:|---:|---:|---:|---:|---:|
| 250 ms | 8 | 253.7 | $0.22 | $0.38 | $0.55 |
| 500 ms | 8 | 253.7 | $0.22 | $0.38 | $0.55 |
| 2000 ms | 8 | 253.7 | $0.22 | $0.38 | $0.55 |

Cheapest-to-most-expensive spread inside this run: **2.1x** (121.0 tok/s at conc 1 vs 253.7 tok/s at its best).

## qwen2.5:3b — `qwen25-3b_same.json` (prompt mode: same)

| conc | aggregate tok/s | tok/hour | p95 TTFT | $/1M @ 0.20 $/h | $/1M @ 0.35 $/h | $/1M @ 0.50 $/h |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 125.0 | 450,000 | 40.4 ms | $0.44 | $0.78 | $1.11 |
| 2 | 132.4 | 476,640 | 682.2 ms | $0.42 | $0.73 | $1.05 |
| 4 | 135.7 | 488,520 | 1801.4 ms | $0.41 | $0.72 | $1.02 |
| 8 | 135.9 | 489,240 | 4149.9 ms | $0.41 | $0.72 | $1.02 |

**Cheapest level that still meets the latency budget** (p95 TTFT):

| p95 budget | max concurrency | tok/s there | $/1M @ 0.20 | $/1M @ 0.35 | $/1M @ 0.50 |
|---:|---:|---:|---:|---:|---:|
| 250 ms | 1 | 125.0 | $0.44 | $0.78 | $1.11 |
| 500 ms | 1 | 125.0 | $0.44 | $0.78 | $1.11 |
| 2000 ms | 4 | 135.7 | $0.41 | $0.72 | $1.02 |

Cheapest-to-most-expensive spread inside this run: **1.1x** (125.0 tok/s at conc 1 vs 135.9 tok/s at its best).

## qwen2.5:3b — `qwen25-3b_unique.json` (prompt mode: unique)

| conc | aggregate tok/s | tok/hour | p95 TTFT | $/1M @ 0.20 $/h | $/1M @ 0.35 $/h | $/1M @ 0.50 $/h |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 122.0 | 439,200 | 140.3 ms | $0.46 | $0.80 | $1.14 |
| 4 | 131.0 | 471,600 | 1728.9 ms | $0.42 | $0.74 | $1.06 |

**Cheapest level that still meets the latency budget** (p95 TTFT):

| p95 budget | max concurrency | tok/s there | $/1M @ 0.20 | $/1M @ 0.35 | $/1M @ 0.50 |
|---:|---:|---:|---:|---:|---:|
| 250 ms | 1 | 122.0 | $0.46 | $0.80 | $1.14 |
| 500 ms | 1 | 122.0 | $0.46 | $0.80 | $1.14 |
| 2000 ms | 4 | 131.0 | $0.42 | $0.74 | $1.06 |

Cheapest-to-most-expensive spread inside this run: **1.1x** (122.0 tok/s at conc 1 vs 131.0 tok/s at its best).

## qwen2.5:7b — `qwen25-7b_same.json` (prompt mode: same)

| conc | aggregate tok/s | tok/hour | p95 TTFT | $/1M @ 0.20 $/h | $/1M @ 0.35 $/h | $/1M @ 0.50 $/h |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.6 | 9,360 | 1175.9 ms | $21.37 | $37.39 | $53.42 |
| 2 | 6.4 | 23,040 | 12605.8 ms | $8.68 | $15.19 | $21.70 |
| 4 | 2.7 | 9,720 | 60759.6 ms | $20.58 | $36.01 | $51.44 |

**Cheapest level that still meets the latency budget** (p95 TTFT):

| p95 budget | max concurrency | tok/s there | $/1M @ 0.20 | $/1M @ 0.35 | $/1M @ 0.50 |
|---:|---:|---:|---:|---:|---:|
| 250 ms | — | — | — | — | — |
| 500 ms | — | — | — | — | — |
| 2000 ms | 1 | 2.6 | $21.37 | $37.39 | $53.42 |

Cheapest-to-most-expensive spread inside this run: **2.5x** (2.6 tok/s at conc 1 vs 6.4 tok/s at its best).

## /home/yegor/models/gguf/qwen2.5-3b-instruct-q4_k_m.gguf — `vllm-gguf-np-default.json` (prompt mode: same)

| conc | aggregate tok/s | tok/hour | p95 TTFT | $/1M @ 0.20 $/h | $/1M @ 0.35 $/h | $/1M @ 0.50 $/h |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 104.7 | 376,920 | 49.0 ms | $0.53 | $0.93 | $1.33 |
| 2 | 119.8 | 431,280 | 73.4 ms | $0.46 | $0.81 | $1.16 |
| 4 | 226.7 | 816,120 | 92.7 ms | $0.25 | $0.43 | $0.61 |
| 8 | 324.2 | 1,167,120 | 145.7 ms | $0.17 | $0.30 | $0.43 |

**Cheapest level that still meets the latency budget** (p95 TTFT):

| p95 budget | max concurrency | tok/s there | $/1M @ 0.20 | $/1M @ 0.35 | $/1M @ 0.50 |
|---:|---:|---:|---:|---:|---:|
| 250 ms | 8 | 324.2 | $0.17 | $0.30 | $0.43 |
| 500 ms | 8 | 324.2 | $0.17 | $0.30 | $0.43 |
| 2000 ms | 8 | 324.2 | $0.17 | $0.30 | $0.43 |

Cheapest-to-most-expensive spread inside this run: **3.1x** (104.7 tok/s at conc 1 vs 324.2 tok/s at its best).

