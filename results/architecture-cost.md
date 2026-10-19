# Cost per 1M output tokens from architecture (memory-bound decode)

Assumptions are columns, not hidden numbers: only active parameters are read, weights at the
stated bytes/param, `eff` of peak bandwidth, and one weight read serves `batch x spec` tokens.

| scenario | active params | B/param | bytes/step | steps/s (mem / compute) | bound | tok/s | node $/h | $/1M | vs $0.50 |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| CALIBRATION: our 3B dense Q4 on one 3060 Ti | 3.0B | 0.5 | 1.4 GiB | 17 / 1,547 | memory | 535 | $0.06 | **$0.031** | 16.1x |
| GLM-5.3-Flash 18B active FP8, 8xH100 @ $2.5 | 18.0B | 1.0 | 16.8 GiB | 179 / 412 | memory | 45,739 | $20.0 | **$0.121** | 4.1x |
| GLM-5.3-Flash 18B active FP8, 8xH100, eff 6% | 18.0B | 1.0 | 16.8 GiB | 89 / 412 | memory | 22,869 | $20.0 | **$0.243** | 2.1x |
| GLM-5.3-Flash 18B active FP8, 4xB200 @ $3.5 | 18.0B | 1.0 | 16.8 GiB | 213 / 206 | compute | 52,800 | $14.0 | **$0.074** | 6.8x |
| GLM-5.3-Flash 18B active NVFP4, 4xB200 | 18.0B | 0.5 | 8.4 GiB | 427 / 206 | compute | 52,800 | $14.0 | **$0.074** | 6.8x |
| DeepSeek-V4-Flash 13B active FP8, 8xH100 | 13.0B | 1.0 | 12.1 GiB | 247 / 571 | memory | 63,330 | $20.0 | **$0.088** | 5.7x |
| IF IT WERE DENSE: 320B FP8, 8xH100 | 320.0B | 1.0 | 298.0 GiB | 10 / 23 | memory | 2,573 | $20.0 | **$2.159** | 0.2x |

**Read it as.** A 320B model whose *active* set is 18B costs like an 18B model, not a 320B
one: 18x fewer bytes per token than the nameplate suggests. FP8 halves it again, NVFP4 quarters
it. Batch 64 with a 4-token speculative draft means one weight read pays for 256 output
tokens. That is the whole answer to "a 320B model cannot be that cheap" — and the same
arithmetic says a 320B *dense* model could not be, at any of these prices.

