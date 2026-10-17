# Capacity and margin on one GPU

Model `Qwen2.5-3B-Instruct`: 36 layers, 2 KV heads x 128 head-dim → **36.0 KiB of KV per token** (36.0 MiB per 16-token block).

Card: 7844 MiB visible to torch, 3920 MiB held by co-tenants → 3924 MiB actually free for us.

## `--gpu-memory-utilization 0.45` — fits

Weights 1853 MiB + activation 1434 MiB + non-torch 20 MiB out of a 3530 MiB budget → **KV 0.22 GiB = 395 blocks (6320 tokens of cache)**; the engine prints a ceiling of **0.73x** at max-model-len 8704.

| context per request | KV it needs | requests that fit at once |
|---:|---:|---:|
| 1024 tokens | 36 MiB | **6** |
| 2048 tokens | 72 MiB | **3** |
| 4096 tokens | 144 MiB | **1** |
| 8192 tokens | 288 MiB | **0** |
| 16384 tokens | 576 MiB | **0** |

## `--gpu-memory-utilization 0.55` — **will OOM: the budget exceeds free memory**

Weights 1853 MiB + activation 1434 MiB + non-torch 20 MiB out of a 4314 MiB budget → **KV 0.98 GiB = 1789 blocks (28624 tokens of cache)**; the engine prints a ceiling of **3.29x** at max-model-len 8704.

| context per request | KV it needs | requests that fit at once |
|---:|---:|---:|
| 1024 tokens | 36 MiB | **27** |
| 2048 tokens | 72 MiB | **13** |
| 4096 tokens | 144 MiB | **6** |
| 8192 tokens | 288 MiB | **3** |
| 16384 tokens | 576 MiB | **1** |

## Money

Rates on the table (measured, not modelled): **426.7 output tok/s** at C=32, and **7.14 µs per prompt token** of marginal cold prefill (= 140,056 tokens/s).

| GPU price | our $/1M output | our $/1M input | break-even tok/s vs $0.46 | break-even tok/s vs $0.50 | break-even tok/s vs $0.28 |
|---:|---:|---:|---:|---:|---:|
| $0.20/h | $0.130 | $0.00040 | 121 | 111 | 198 |

**How to read it.** The capacity table is arithmetic on the KV cache and does not depend on how fast the model decodes: it is the number a customer will actually consume (how many long conversations can be in flight). The money table is the two measured rates converted to dollars. The break-even column is the inverse question a provider asks: at this GPU price and that market price, how many output tokens/s do I have to deliver so the token stops being sold below cost?

Input tokens cost ~300x less than output tokens here, so a bill's input half is margin-dominated — the output half is where the cost lives. And note what the capacity table says about long context: the same 0.22 GiB cache that ignores short requests admits a single 3.5k-token request, so a provider selling long-context agents is selling the cache, not the FLOPs.
