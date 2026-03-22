## Benchmark report (gpt-oss-20b_20260322_090403 / all)

**Detected backend(s):** sglang

### Metric directions

- **Lower is better**: TTFT, ITL, request latency, prefill/decode/e2e/queue p95, waiting/retracted/evicted.
- **Higher is better**: throughput (rps, tokens/sec), cache hit ratio, spec accept rate/length.
- **KV/token usage %**: lower = more headroom. SGLang: `token_usage×100`. vLLM: `kv_cache_usage_perc`.
- **decode_p95_ms (SGLang)**: proxy via ITL p95 — SGLang has no per-request decode histogram.
- **gen_tps vs gen_tps_gauge**: counter rate (`generation_tokens`) vs real-time gauge (`gen_throughput`). Both should track; divergence indicates idle periods.
- **new_token_ratio**: closer to 1.0 = all new tokens (no prefix reuse); lower = heavy cache reuse.
- **retracted_avg**: requests retracted due to over-scheduling. Non-zero → increase `--schedule-conservativeness`.
- **spec_*** metrics: only populated when speculative decoding (`--speculative-algorithm`) is enabled.

### Highlights

- **Best TTFT p95**: `concurrency_c1_req48_in256_out160` (c=1) → 108.73 ms
- **Best output throughput**: `concurrency_c512_req512_in256_out160` (c=512) → 1015.67 tok/s
- **Saturation knee (heuristic)**: around c=2 (`concurrency_c2_req48_in256_out160`) where TTFT p95 jumps ≥2× vs previous concurrency.
- **Most KV evictions**: `concurrency_c1024_req1024_in256_out160` → 129548 tokens evicted (KV pressure warning).

### Plots generated

- `ttft_p95_vs_concurrency.png` — TTFT p95
- `ttft_avg_vs_concurrency.png` — TTFT avg
- `itl_p95_vs_concurrency.png` — ITL p95
- `itl_avg_vs_concurrency.png` — ITL avg/p50
- `req_lat_p95_vs_concurrency.png` — Request latency p95 [aiperf]
- `e2e_p95_vs_concurrency.png` — E2E latency p95 [server]
- `prefill_p95_vs_concurrency.png` — Prefill latency p95
- `decode_p95_vs_concurrency.png` — Decode latency p95
- `queue_p95_vs_concurrency.png` — Queue time p95
- `out_tps_vs_concurrency.png` — Output throughput [aiperf]
- `gen_tps_vs_concurrency.png` — Gen token rate (counter + gauge)
- `prompt_tps_vs_concurrency.png` — Prompt token rate
- `kv_usage_p95_vs_concurrency.png` — KV usage p95 %
- `kv_usage_avg_vs_concurrency.png` — KV usage avg %
- `swa_usage_avg_vs_concurrency.png` — SWA usage avg % [SGLang]
- `evicted_tokens_vs_concurrency.png` — KV evictions [SGLang]
- `waiting_avg_vs_concurrency.png` — Avg waiting requests
- `retracted_avg_vs_concurrency.png` — Avg retracted requests [SGLang]
- `decode_inflight_vs_concurrency.png` — Decode tokens in-flight [SGLang]
- `new_token_ratio_vs_concurrency.png` — New token ratio [SGLang]
- `prefix_cache_hit_ratio_vs_concurrency.png` — Prefix cache hit ratio
- `cached_tokens_vs_concurrency.png` — Total cached tokens
- `spec_accept_rate_vs_concurrency.png` — Spec accept rate [SGLang]
- `spec_accept_length_vs_concurrency.png` — Spec accept length [SGLang]
- `littles_law_vs_concurrency.png` — Little's Law L
- `q_pressure_vs_concurrency.png` — Queue pressure

