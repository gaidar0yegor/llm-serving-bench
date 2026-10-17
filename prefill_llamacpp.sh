#!/usr/bin/env bash
# prefill_llamacpp.sh — the same cold-prefill curve, measured on llama.cpp instead of vLLM.
#
# Why a second engine: vLLM could not hold an 8k-token prompt on this box at the time (the TTS
# co-tenants already hold ~3.7 GiB of the 7.66 GiB the card reports, and vLLM's
# --gpu-memory-utilization is a fraction of TOTAL memory, so it asked for more than exists and hit
# CUDA OOM during cudagraph capture). llama.cpp sizes its KV cache explicitly and is much leaner, so
# it can serve the long end of the curve. The two engines are reported separately, never merged.
#
# Slots: -np 2 -c 24576 gives each of the two slots 12 288 tokens, enough for an 8k prompt + output.
# KV for this model: 2 (K,V) x 2 kv-heads x 128 head-dim x 36 layers x 2 bytes ≈ 36.9 KB/token
#   → 24 576 tokens ≈ 0.9 GiB, plus 1.93 GiB of weights.
#
#   bash prefill_llamacpp.sh 1024,2048,4096,8192
set -u
cd "$(dirname "$0")"
LENGTHS=${1:-1024,2048,4096,8192}

export GGML_BACKEND_PATH=/usr/lib/ollama/cuda_v13/libggml-cuda.so
export LD_LIBRARY_PATH=/usr/lib/ollama/cuda_v13:/usr/lib/ollama:${LD_LIBRARY_PATH:-}

mkdir -p kv-logs results
pkill -f "[v]llm serve" || true
sleep 5

/usr/lib/ollama/llama-server -m "$HOME/models/gguf/qwen2.5-3b-instruct-q4_k_m.gguf" -a qwen25-3b \
  --host 127.0.0.1 --port 8081 -ngl 99 -c 24576 -np 2 -cb --metrics --no-webui \
  > kv-logs/llamacpp-prefill.log 2>&1 &
SRV=$!
echo "llama-server pid $SRV; waiting for /health"

for _ in $(seq 1 90); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:8081/health || true)
  [ "$code" = "200" ] && break
  sleep 2
done
echo "health=$code"
if [ "$code" != "200" ]; then
  echo "server did not come up — tail of its log:"; tail -20 kv-logs/llamacpp-prefill.log
  kill $SRV 2>/dev/null; exit 1
fi

grep -iE "n_ctx|n_parallel|CUDA0|model size|KV self" kv-logs/llamacpp-prefill.log | head -8

for N in ${LENGTHS//,/ }; do
  echo "=== prompt ~$N tokens ==="
  python3 bench.py --base-url http://127.0.0.1:8081 --model qwen25-3b \
    --concurrency 1,2 --requests 6 --max-tokens 32 --prompt-tokens "$N" --prompt-mode unique \
    --metrics-url http://127.0.0.1:8081/metrics \
    --json "results/prefill-llamacpp-$N.json" \
    --host-note "cold prefill on llama.cpp; co-tenants hold ~3.7 GiB of the card" \
    || echo "bench for $N failed"
done

kill $SRV 2>/dev/null
wait $SRV 2>/dev/null
echo "llama.cpp prefill profile done"
