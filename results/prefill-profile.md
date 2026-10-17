# Cold prefill: what a long prompt costs before the first token

vLLM vLLM 0.8.5.post1, GGUF (V0), --max-model-len 4608, --gpu-memory-utilization 0.45
NVIDIA GeForce RTX 3060 Ti (8 GB); unique (prefix cache defeated -> cold prefill); prompt lengths 1024, 2048, 4096 (target tokens).
Allocated: 397 CUDA KV blocks, 0.22 GiB KV, ceiling 1.38x at max-model-len.

| prompt tokens (measured) | C=1 TTFT p50 | C=1 TTFT p95 | C=2 p95 | C=4 p95 | prefill tok/s | ms / 1k input | $/1M input @ 0.20 | $/1M input @ 0.35 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 942 | 46.1 ms | 59.3 ms | 91.1 ms | 144.7 ms | 20433.8 | 48.94 | $0.003 | $0.005 |
| 1822 | 50.0 ms | 53.1 ms | 91.4 ms | 148.7 ms | 36440.0 | 27.44 | $0.002 | $0.003 |
| 3571 | 64.4 ms | 71.6 ms | 664.2 ms | 1868.2 ms | 55450.3 | 18.03 | $0.001 | $0.002 |

`prefill tok/s` = the server's own prompt_tokens / TTFT at concurrency 1, i.e. it includes one decode step and client overhead — a slight under-estimate, applied equally to every length. Compare it with the price of a provider that sells input tokens: that gap is the input half of the margin.

