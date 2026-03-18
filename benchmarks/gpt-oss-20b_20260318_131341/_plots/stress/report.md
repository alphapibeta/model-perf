## Benchmark report (gpt-oss-20b_20260318_131341 / stress)

### Metric directions (how to read ‘better’)

- **Lower is better**: TTFT, request latency, inter-token latency, server-side prefill/decode/e2e p95, waiting/queueing.
- **Higher is better**: throughput (requests/sec, tokens/sec), prompt/gen token rates, cache hit ratio (workload dependent).
- **KV cache usage**: lower means more headroom; high values can limit concurrency or risk OOM.

### Highlights

- **Best latency (TTFT p95)**: `stress_c64_req256_in1200_out180` (c=64) → 10723.56 ms
- **Best throughput (out_tps)**: `stress_c256_req256_in1200_out180` (c=256) → 1678.20 tok/s

### Queueing interpretation

- If **throughput plateaus** but **TTFT/latency rises**, you’re saturated; extra concurrency becomes queueing.
- `little_L ≈ rps * (req_lat_avg_sec)` is a Little’s Law sanity check for in-system work.
- `q_pressure = waiting_avg / concurrency` is a rough queue-pressure indicator.

### Per-request tails (from `profile_export.jsonl`)

- Prefer these when debugging spikes: `pr_ttft_p99/max`, `pr_wait_p99/max`, `pr_req_p99/max`.
- `pr_osl_mismatch` is the fraction of requests with `abs(osl_mismatch_diff_pct) >= threshold`.

### Plots (if generated)

- `ttft_p95_vs_concurrency.png`
- `itl_p95_vs_concurrency.png`
- `out_tps_vs_concurrency.png`
- `kv_usage_p95_vs_concurrency.png`
- `prefill_p95_vs_concurrency.png`, `decode_p95_vs_concurrency.png`, `e2e_p95_vs_concurrency.png`
- `prefix_cache_hit_ratio_vs_concurrency.png`

