# Margin per million output tokens (measured cost vs market price)

Source run: `results/vllm-gguf-mml4096.json` — /home/yegor/models/gguf/qwen2.5-3b-instruct-q4_k_m.gguf

Cost side (ours, measured): `$/1M = price_per_hour / (agg_tok_s * 3600) * 1e6`, 100% duty, output tokens only.

## vs glm-5.3-flash (cheapest EU endpoint) — Melious (EU): $0.12/1M in, **$0.46/1M out** (cached in $0.02)

Provider's own tail: TTFT 697 ms, 77 tok/s. That is the bar our p95 has to clear.

| conc | our agg tok/s | our p95 TTFT | our $/1M @ 0.20 $/h | our $/1M @ 0.35 $/h | market $/1M | margin $/1M @ 0.20 | margin $/1M @ 0.35 | margin × @ 0.20 | margin × @ 0.35 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 107.2 | 37.5 ms | $0.518 | $0.907 | $0.46 | $-0.058 | $-0.447 | 0.9x | 0.5x |
| 2 | 114.4 | 113.0 ms | $0.486 | $0.850 | $0.46 | $-0.026 | $-0.390 | 0.9x | 0.5x |
| 4 | 202.5 | 135.5 ms | $0.274 | $0.480 | $0.46 | $+0.186 | $-0.020 | 1.7x | 1.0x |
| 8 | 327.3 | 149.5 ms | $0.170 | $0.297 | $0.46 | $+0.290 | $+0.163 | 2.7x | 1.5x |
| 16 | 362.4 | 253.5 ms | $0.153 | $0.268 | $0.46 | $+0.307 | $+0.192 | 3.0x | 1.7x |
| 32 | 426.7 | 414.8 ms | $0.130 | $0.228 | $0.46 | $+0.330 | $+0.232 | 3.5x | 2.0x |

Break-even (first concurrency level whose measured cost is below the market price): 0.20 $/h → conc 4; 0.35 $/h → conc 8.

## vs glm-5.3-flash (official / Together) — Z.AI official, also Together AI: $0.15/1M in, **$0.5/1M out** (cached in $0.03)

Provider's own tail: TTFT 409 ms, 129 tok/s. That is the bar our p95 has to clear.

| conc | our agg tok/s | our p95 TTFT | our $/1M @ 0.20 $/h | our $/1M @ 0.35 $/h | market $/1M | margin $/1M @ 0.20 | margin $/1M @ 0.35 | margin × @ 0.20 | margin × @ 0.35 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 107.2 | 37.5 ms | $0.518 | $0.907 | $0.50 | $-0.018 | $-0.407 | 1.0x | 0.6x |
| 2 | 114.4 | 113.0 ms | $0.486 | $0.850 | $0.50 | $+0.014 | $-0.350 | 1.0x | 0.6x |
| 4 | 202.5 | 135.5 ms | $0.274 | $0.480 | $0.50 | $+0.226 | $+0.020 | 1.8x | 1.0x |
| 8 | 327.3 | 149.5 ms | $0.170 | $0.297 | $0.50 | $+0.330 | $+0.203 | 2.9x | 1.7x |
| 16 | 362.4 | 253.5 ms | $0.153 | $0.268 | $0.50 | $+0.347 | $+0.232 | 3.3x | 1.9x |
| 32 | 426.7 | 414.8 ms | $0.130 | $0.228 | $0.50 | $+0.370 | $+0.272 | 3.8x | 2.2x |

Break-even (first concurrency level whose measured cost is below the market price): 0.20 $/h → conc 2; 0.35 $/h → conc 4.

## vs deepseek-v4-flash (official) — DeepSeek: $0.14/1M in, **$0.28/1M out** (cached in $0.07)

| conc | our agg tok/s | our p95 TTFT | our $/1M @ 0.20 $/h | our $/1M @ 0.35 $/h | market $/1M | margin $/1M @ 0.20 | margin $/1M @ 0.35 | margin × @ 0.20 | margin × @ 0.35 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 107.2 | 37.5 ms | $0.518 | $0.907 | $0.28 | $-0.238 | $-0.627 | 0.5x | 0.3x |
| 2 | 114.4 | 113.0 ms | $0.486 | $0.850 | $0.28 | $-0.206 | $-0.570 | 0.6x | 0.3x |
| 4 | 202.5 | 135.5 ms | $0.274 | $0.480 | $0.28 | $+0.006 | $-0.200 | 1.0x | 0.6x |
| 8 | 327.3 | 149.5 ms | $0.170 | $0.297 | $0.28 | $+0.110 | $-0.017 | 1.6x | 0.9x |
| 16 | 362.4 | 253.5 ms | $0.153 | $0.268 | $0.28 | $+0.127 | $+0.012 | 1.8x | 1.0x |
| 32 | 426.7 | 414.8 ms | $0.130 | $0.228 | $0.28 | $+0.150 | $+0.052 | 2.2x | 1.2x |

Break-even (first concurrency level whose measured cost is below the market price): 0.20 $/h → conc 4; 0.35 $/h → conc 16.

_Output tokens only. Input/prefill cost is not in this measurement set, so the blended invoice margin is not claimed here._
