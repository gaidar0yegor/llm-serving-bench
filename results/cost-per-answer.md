# Cost per answer

Billed output tokens are the server's own `usage.completion_tokens` — on a reasoning model they include the thinking the customer never sees. Monthly figures assume 100,000 answers.

| model | effort | conc | in tokens | billed out tokens/answer | of which reasoning | TTFT | TTFT to answer | cost @ in/out price | /month |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `zai-org/glm-5.3-flash` | low | 16 | 640 | **94** | 0 | 402.7 ms | 402.7 ms | $0.000143 | $14 |
| `deepseek-ai/deepseek-v4-flash` | high | 4 | 707 | **152** | 100 | 291.3 ms | 1049.7 ms | $0.000142 | $14 |
| `deepseek-ai/deepseek-v4-flash` | low | 4 | 626 | **340** | 246 | 320.0 ms | 2331.2 ms | $0.000183 | $18 |
| `zai-org/glm-5.3-flash` | default | 4 | 638 | **329** | 228 | 378.5 ms | 3199.5 ms | $0.000260 | $26 |
| `zai-org/glm-5.3-flash` | low | 4 | 640 | **93** | 0 | 713.4 ms | 713.4 ms | $0.000142 | $14 |

**Read it as: the same three-bullet answer costs N tokens, and N is mostly thinking.** Where a model has both a default and a `low` reasoning measurement, the cheaper row is the same answer with the thinking turned off — that difference is the largest single lever visible from outside a provider, and it does not require knowing their prices.

_Prices are an input from the market table; where a model has no price row, the token counts stand alone._
