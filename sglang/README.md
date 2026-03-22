# SGLang LLM Benchmark Toolchain

A self-contained toolchain for launching an SGLang inference server in Docker, running AIPerf benchmark suites against it, and summarising + plotting the results.

---

## Table of Contents

1. [Repo layout](#repo-layout)
2. [Quick start](#quick-start)
3. [llama-run.sh — Server launcher](#llama-runsh--server-launcher)
   - [How it works](#how-it-works)
   - [All environment variables](#all-environment-variables)
   - [SGLang server arguments explained](#sglang-server-arguments-explained)
   - [Flag arguments](#flag-arguments)
   - [Examples](#llama-runsh-examples)
4. [sglang-gpt.env — Configuration file](#sglang-gptenv--configuration-file)
   - [Full annotated reference](#full-annotated-reference)
5. [sg_lang_bench_report.py — Benchmark summariser](#sg_lang_bench_reportpy--benchmark-summariser)
   - [Directory convention](#directory-convention)
   - [CLI arguments](#cli-arguments)
   - [Output columns explained](#output-columns-explained)
   - [SGLang metric mapping](#sglang-metric-mapping)
   - [Examples](#sg_lang_bench_reportpy-examples)
6. [Benchmark suites](#benchmark-suites)
7. [Tuning guide](#tuning-guide)
8. [Troubleshooting](#troubleshooting)

---

## Repo layout

```
sglang/
├── llama-run.sh            # Launches the SGLang Docker container
├── sglang-gpt.env          # All config lives here — edit this, not the script
├── sg_lang_bench_report.py # Post-run summariser and plotter
└── benchmarks/             # Created automatically by your benchmark runner
    └── <MODEL>_<TIMESTAMP>/
        └── <suite>_c<N>_req<N>_in<N>_out<N>/
            └── aiperf_artifacts/
                ├── profile_export_aiperf.json
                ├── profile_export_aiperf.csv
                ├── profile_export.jsonl
                └── server_metrics_export.csv
```

---

## Quick start

```bash
# 1. Set your HuggingFace token
export HF_TOKEN=hf_xxxxxxxxxxxxxxxx

# 2. Edit sglang-gpt.env with your model path and GPU config, then launch
bash llama-run.sh sglang-gpt.env

# 3. Check the server is healthy
curl http://localhost:30000/health

# 4. Run your AIPerf benchmark suite (produces benchmarks/ tree)
# ... (aiperf profile commands, or your benchmark.sh wrapper)

# 5. Summarise and plot results
python3 sg_lang_bench_report.py \
  --benchmarks-dir benchmarks \
  --model-ts gpt-oss-20b_20260322_080523 \
  --plot --by-suite --report
```

---

## llama-run.sh — Server launcher

### How it works

1. Reads a `.env` file (default: `sglang-gpt.env` next to the script, or pass a path as `$1`).
2. Resolves `HF_TOKEN` — shell environment takes precedence over the env file so you never accidentally commit a token.
3. Builds a `docker run` command array with all SGLang arguments, appending optional flags only when their corresponding variable is set/enabled.
4. Tears down any existing container with the same name, then starts the new one detached (`-d`).
5. Prints the container ID, a `docker logs -f` reminder, and the health/API URLs.

```bash
# Usage
bash llama-run.sh [path/to/custom.env]

# Default env file location
bash llama-run.sh                  # uses ./sglang-gpt.env
bash llama-run.sh /etc/my.env      # uses a custom path
```

---

### All environment variables

Every variable has a built-in default. Override any of them in `sglang-gpt.env` (or export them into the shell before running).

#### Model

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | *(required)* | HuggingFace model ID or local path passed to `--model-path`. **Must be set** — the script exits with an error if missing. |
| `SERVED_MODEL_NAME` | `$MODEL_PATH` | The name the API advertises (e.g. in `/v1/models`). Useful when `MODEL_PATH` is a long HF path but you want a short alias. |

#### GPU & parallelism

| Variable | Default | Description |
|---|---|---|
| `CUDA_DEVICES` | `0` | Comma-separated GPU indices passed to `NVIDIA_VISIBLE_DEVICES` inside the container. E.g. `0,1` for two GPUs. |
| `TP_SIZE` | `1` | Tensor-parallel degree (`--tensor-parallel-size`). Must equal the number of GPUs listed in `CUDA_DEVICES`. |

#### Server networking

| Variable | Default | Description |
|---|---|---|
| `HOST` | `0.0.0.0` | Bind address for the HTTP server (`--host`). Use `127.0.0.1` to restrict to localhost. |
| `PORT` | `30000` | TCP port (`--port`). Must be free on the host. |
| `CONTAINER_NAME` | `sglang-gptoss-server` | Docker container name. Changing this lets you run multiple servers on the same machine. |
| `IMAGE_NAME` | `lmsysorg/sglang:latest` | Docker image. Pin to a specific tag for reproducibility, e.g. `lmsysorg/sglang:v0.4.1-cu124`. |

#### Memory & performance

| Variable | Default | Description |
|---|---|---|
| `MEM_FRACTION_STATIC` | `0.85` | Fraction of GPU memory reserved for the KV cache (`--mem-fraction-static`). Raise toward `0.95` if you have no other GPU workloads. Lower if you hit OOM. |
| `CHUNKED_PREFILL_SIZE` | `2048` | Max tokens processed in a single prefill chunk (`--chunked-prefill-size`). Smaller values reduce peak VRAM at the cost of more prefill passes. |
| `MAX_PREFILL_TOKENS` | `16384` | Maximum *total* tokens across all requests in one prefill batch (`--max-prefill-tokens`). Caps prefill batch size. |
| `STREAM_INTERVAL` | `2` | Flush streaming tokens every N decode steps (`--stream-interval`). `1` = lowest latency; higher values reduce syscall overhead at the cost of first-chunk delay. |
| `MAX_RUNNING_REQUESTS` | `16` | Hard cap on concurrently executing requests (`--max-running-requests`). Requests beyond this go into the queue. Set this to match your GPU's decode capacity. |
| `MAX_QUEUED_REQUESTS` | `128` | Maximum requests held in the waiting queue before the server returns HTTP 503 (`--max-queued-requests`). |
| `SCHEDULE_POLICY` | `fcfs` | Request scheduling policy (`--schedule-policy`). Options: `fcfs` (first-come-first-served), `lpm` (longest-prefix-match for cache efficiency), `dfs-weight`. |
| `NUM_CONTINUOUS_DECODE_STEPS` | `1` | How many decode steps to run between preemption checks (`--num-continuous-decode-steps`). Values > 1 improve decode throughput at the cost of coarser scheduling granularity. |

#### Backends

| Variable | Default | Description |
|---|---|---|
| `ATTENTION_BACKEND` | `triton` | Attention kernel backend (`--attention-backend`). Options: `triton`, `flashinfer`, `torch_native`. `flashinfer` is generally fastest on Ampere/Hopper; `triton` is a safe default. |
| `SAMPLING_BACKEND` | `pytorch` | Token sampling backend (`--sampling-backend`). Options: `pytorch`, `flashinfer`. `flashinfer` is faster for large-vocabulary models. |
| `MOE_RUNNER_BACKEND` | *(empty)* | MoE kernel backend (`--moe-runner-backend`). Set to `flashinfer_cutlass` for MoE/sparse models (e.g. Mixtral, GPT-MoE). Leave blank for dense models. |
| `FP4_GEMM_BACKEND` | `flashinfer_cutlass` | FP4 GEMM kernel (`--fp4-gemm-backend`). Used when the model has FP4 weights. `flashinfer_cutlass` is the recommended option. |

#### HuggingFace cache

| Variable | Default | Description |
|---|---|---|
| `HF_HOME` | `/root/.cache/huggingface` | Path **inside** the container where HF models are cached. |
| `HF_CACHE_HOST_PATH` | `$HOME/.cache/huggingface` | Path **on the host** that is bind-mounted to `HF_HOME`. Reuse across runs to avoid re-downloading. |

#### System / NVIDIA

| Variable | Default | Description |
|---|---|---|
| `SHM_SIZE` | `16g` | Size of `/dev/shm` (`--shm-size`). Multi-GPU tensor-parallel jobs need large shared memory for NCCL. Increase to `32g` for 4+ GPUs. |
| `NVIDIA_DRIVER_CAPABILITIES` | `compute,utility` | NVIDIA Docker capabilities. Almost never needs changing. |
| `NCCL_DEBUG` | `INFO` | NCCL log verbosity. Set to `WARN` or `ERROR` to silence startup chatter in production. |
| `NCCL_DEBUG_SUBSYS` | `INIT,ENV` | Which NCCL subsystems to log. |

#### Logging & flags

| Variable | Default | Description |
|---|---|---|
| `LOG_LEVEL` | `info` | SGLang log level (`--log-level`). Options: `debug`, `info`, `warning`, `error`. |
| `ENABLE_MIXED_CHUNK` | `0` | Set to `1` to enable `--enable-mixed-chunk`: mixes prefill and decode tokens in the same batch, improving GPU utilisation at moderate concurrency. |
| `ENABLE_METRICS` | `0` | Set to `1` to expose a Prometheus `/metrics` endpoint (`--enable-metrics`). **Required for `sg_lang_bench_report.py` to show server-side metrics.** |
| `ENABLE_P2P_CHECK` | `0` | Set to `1` to enable `--enable-p2p-check`: verifies GPU-to-GPU NVLink/P2P before starting. Recommended for multi-GPU setups. |
| `DISABLE_CUDA_GRAPH` | `0` | Set to `1` to add `--disable-cuda-graph`. CUDA graphs accelerate decode but require extra VRAM and can cause issues with some models (e.g. MoE). |
| `SKIP_SERVER_WARMUP` | `0` | Set to `1` to add `--skip-server-warmup`. Skips the initial warmup request. Useful for fast iteration during development. |

---

### SGLang server arguments explained

These are the `python3 -m sglang.launch_server` flags that `llama-run.sh` assembles. Understanding them helps you tune for your workload.

#### `--model-path`
The HuggingFace model ID (e.g. `openai/gpt-oss-20b`) or a local directory path. SGLang downloads weights from HF if not cached.

#### `--served-model-name`
The model name returned by `/v1/models` and expected by API clients. Set this to a short friendly name so your AIPerf `--model` arg matches.

#### `--tensor-parallel-size`
Splits the model across N GPUs using tensor parallelism. Must exactly match the number of GPUs in `CUDA_DEVICES`. For a 20B model on 2×H100s, use `TP_SIZE=2`.

#### `--mem-fraction-static`
Controls what fraction of each GPU's memory is pre-allocated for the KV cache pool. The remaining memory is used for model weights and activations.

- Too low → small KV pool → low max concurrency and frequent evictions.
- Too high → OOM when loading weights.
- Typical range: `0.80` – `0.93`. Start at `0.85`, raise if you see eviction warnings in logs.

#### `--chunked-prefill-size`
Breaks long prompts into chunks of this many tokens for prefill. Prevents a single huge prompt from monopolising the GPU for seconds.

- Lower (e.g. `512`) → smoother decode latency for concurrent short requests.
- Higher (e.g. `8192`) → faster throughput for batch-prefill workloads.

#### `--max-prefill-tokens`
Total token budget across all requests in one prefill step. Setting this too high causes latency spikes when many long requests arrive simultaneously.

Recommended: `2 × CHUNKED_PREFILL_SIZE` to `4 × CHUNKED_PREFILL_SIZE`.

#### `--stream-interval`
Number of decode steps between streaming flushes to the client.

- `1` → minimum TTFT and inter-chunk latency (best for interactive use).
- `4`–`8` → better throughput, slightly higher perceived latency.

#### `--max-running-requests`
The maximum number of requests executing decode simultaneously. Once this limit is hit, new requests wait in the queue.

**This is the single most important tuning knob.** Too low → wasted GPU capacity. Too high → KV cache exhaustion, OOM, or severe latency degradation.

Rule of thumb: start at `MAX_RUNNING_REQUESTS ≈ (KV pool tokens) / (avg sequence length)`.

#### `--max-queued-requests`
Maximum depth of the waiting queue before the server starts rejecting requests with HTTP 503. Set high enough to absorb bursts without dropping requests.

#### `--schedule-policy`
How requests are selected from the queue for execution.

| Policy | Description | Best for |
|---|---|---|
| `fcfs` | First-come-first-served | General purpose, predictable latency |
| `lpm` | Longest-prefix-match — prioritises requests that share a prefix already in the KV cache | Chat/multi-turn workloads with shared system prompts |
| `dfs-weight` | Depth-first with weight | Long context batch workloads |

#### `--attention-backend`
Which attention kernel to use.

| Backend | Notes |
|---|---|
| `triton` | Default. Works on all CUDA GPUs, good compatibility. |
| `flashinfer` | Faster on Ampere (A100) and Hopper (H100). Recommended when available. |
| `torch_native` | PyTorch fallback. Useful for debugging. Slow. |

#### `--sampling-backend`
Which token sampling implementation to use.

| Backend | Notes |
|---|---|
| `pytorch` | Default. Stable and well-tested. |
| `flashinfer` | Lower sampling latency, especially for large vocabularies (100k+ tokens). |

#### `--num-continuous-decode-steps`
How many decode steps to run in a tight loop before yielding to the scheduler.

- `1` (default) → scheduler checks for new requests after every single token. Lowest latency for new arrivals.
- `4`–`10` → higher token throughput at the cost of slightly higher scheduling jitter.

#### `--moe-runner-backend`
Only relevant for Mixture-of-Experts models. `flashinfer_cutlass` is the recommended value for models like Mixtral or GPT-MoE variants. Leave unset for dense models.

#### `--fp4-gemm-backend`
Only relevant for models with FP4-quantised weights. `flashinfer_cutlass` uses the cuTLASS FP4 GEMM kernel, which is significantly faster than the fallback on H100 NVL.

#### `--enable-mixed-chunk`
Allows decode tokens from running requests to be batched together with prefill tokens from new requests in the same forward pass. Improves GPU utilisation at low-to-medium concurrency but can increase prefill latency slightly.

#### `--enable-metrics`
Exposes a Prometheus-compatible `/metrics` endpoint. **You must enable this** for `sg_lang_bench_report.py` to produce server-side charts (KV usage, queue depth, TTFT histograms, etc.). AIPerf scrapes this endpoint during the benchmark run.

#### `--enable-p2p-check`
Runs an NVLink peer-to-peer connectivity check at startup. Recommended on multi-GPU machines to catch misconfigured topology early.

#### `--disable-cuda-graph`
By default SGLang uses CUDA graphs to eliminate CPU-side overhead during decode. Disable if:
- You are running a MoE model that has dynamic shapes incompatible with graphs.
- You are hitting mysterious CUDA errors.
- You want to reduce startup VRAM usage at the cost of decode speed.

#### `--skip-server-warmup`
Normally SGLang sends a warmup request at startup to pre-compile kernels. Skip this for faster iteration during development. **Do not skip in production benchmarks** — first real request will be slower.

#### `--log-level`
SGLang logging verbosity: `debug` | `info` | `warning` | `error`.

---

### Flag arguments

These variables are **boolean** — set to `1` to enable, `0` (or leave unset) to disable:

```bash
ENABLE_MIXED_CHUNK=1    # adds --enable-mixed-chunk
ENABLE_METRICS=1        # adds --enable-metrics
ENABLE_P2P_CHECK=1      # adds --enable-p2p-check
DISABLE_CUDA_GRAPH=1    # adds --disable-cuda-graph
SKIP_SERVER_WARMUP=1    # adds --skip-server-warmup
```

---

### llama-run.sh examples

#### Minimal single-GPU launch

```bash
# sglang-gpt.env
MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct
SERVED_MODEL_NAME=llama-3-8b
CUDA_DEVICES=0
TP_SIZE=1
ENABLE_METRICS=1
```

```bash
export HF_TOKEN=hf_xxxx
bash llama-run.sh sglang-gpt.env
```

#### Two-GPU tensor parallel with FP4

```bash
# sglang-gpt.env
MODEL_PATH=openai/gpt-oss-20b
SERVED_MODEL_NAME=gpt-oss-20b
CUDA_DEVICES=0,1
TP_SIZE=2
MEM_FRACTION_STATIC=0.92
MAX_RUNNING_REQUESTS=32
MAX_QUEUED_REQUESTS=64
ATTENTION_BACKEND=triton
SAMPLING_BACKEND=pytorch
MOE_RUNNER_BACKEND=flashinfer_cutlass
FP4_GEMM_BACKEND=flashinfer_cutlass
ENABLE_MIXED_CHUNK=1
ENABLE_METRICS=1
ENABLE_P2P_CHECK=1
DISABLE_CUDA_GRAPH=1
```

#### High-throughput batch serving (long context)

```bash
MODEL_PATH=mistralai/Mistral-7B-Instruct-v0.3
CUDA_DEVICES=0
TP_SIZE=1
MEM_FRACTION_STATIC=0.90
CHUNKED_PREFILL_SIZE=4096
MAX_PREFILL_TOKENS=32768
STREAM_INTERVAL=4
MAX_RUNNING_REQUESTS=64
MAX_QUEUED_REQUESTS=256
SCHEDULE_POLICY=lpm
NUM_CONTINUOUS_DECODE_STEPS=4
ATTENTION_BACKEND=flashinfer
SAMPLING_BACKEND=flashinfer
ENABLE_METRICS=1
```

#### Interactive low-latency serving

```bash
MODEL_PATH=meta-llama/Llama-3.2-3B-Instruct
CUDA_DEVICES=0
TP_SIZE=1
MEM_FRACTION_STATIC=0.82
CHUNKED_PREFILL_SIZE=512
MAX_PREFILL_TOKENS=4096
STREAM_INTERVAL=1
MAX_RUNNING_REQUESTS=8
SCHEDULE_POLICY=fcfs
NUM_CONTINUOUS_DECODE_STEPS=1
ENABLE_METRICS=1
```

---

## sglang-gpt.env — Configuration file

This is the **only file you should edit** for day-to-day configuration. It is sourced by `llama-run.sh` at startup.

### Full annotated reference

```bash
# =========================
# MODEL
# =========================
MODEL_PATH=openai/gpt-oss-20b       # Required. HF model ID or local path.
SERVED_MODEL_NAME=gpt-oss-20b       # API alias. Match this to aiperf --model.

# =========================
# GPU
# =========================
CUDA_DEVICES=0,1                    # GPUs to use. Comma-separated indices.
TP_SIZE=2                           # Must equal number of devices above.

# =========================
# SERVER
# =========================
HOST=0.0.0.0                        # Bind to all interfaces. Use 127.0.0.1 for local-only.
PORT=30000                          # HTTP port. aiperf uses http://localhost:30000
CONTAINER_NAME=sglang-gptoss-server # Docker container name.
IMAGE_NAME=lmsysorg/sglang:latest   # Pin to a specific version for reproducibility.

# =========================
# MEMORY / PERFORMANCE
# =========================
MEM_FRACTION_STATIC=0.92            # 92% of GPU VRAM for KV cache. Tune vs OOM.
CHUNKED_PREFILL_SIZE=2048           # Max tokens per prefill chunk.
MAX_PREFILL_TOKENS=8192             # Max total prefill tokens per batch step.
STREAM_INTERVAL=2                   # Decode steps between streaming flushes.
MAX_RUNNING_REQUESTS=32             # Max concurrent decode requests. Most critical knob.
MAX_QUEUED_REQUESTS=32              # Queue depth before HTTP 503.
SCHEDULE_POLICY=fcfs                # fcfs | lpm | dfs-weight
NUM_CONTINUOUS_DECODE_STEPS=1       # Decode steps per scheduler tick.

# =========================
# BACKENDS
# =========================
MOE_RUNNER_BACKEND=flashinfer_cutlass  # MoE models only. Leave blank for dense.
ATTENTION_BACKEND=triton               # triton | flashinfer | torch_native
SAMPLING_BACKEND=pytorch               # pytorch | flashinfer
FP4_GEMM_BACKEND=flashinfer_cutlass    # FP4-weight models only.

# =========================
# FLAGS (1=enable, 0=disable)
# =========================
ENABLE_MIXED_CHUNK=1        # Mix prefill+decode in same batch.
ENABLE_METRICS=1            # Expose /metrics for Prometheus / AIPerf.
ENABLE_P2P_CHECK=1          # Verify NVLink at startup (multi-GPU).
DISABLE_CUDA_GRAPH=1        # Disable CUDA graphs (needed for some MoE models).
SKIP_SERVER_WARMUP=0        # Set 1 to skip warmup (dev only, not for benchmarks).

# =========================
# CACHE
# =========================
HF_HOME=/root/.cache/huggingface          # Inside container.
HF_CACHE_HOST_PATH=/root/.cache/huggingface  # Host bind-mount source.

# =========================
# NVIDIA / SYSTEM
# =========================
SHM_SIZE=16g                              # Shared memory for NCCL. Use 32g+ for 4+ GPUs.
NVIDIA_DRIVER_CAPABILITIES=compute,utility
NCCL_DEBUG=INFO                           # WARN/ERROR for quieter logs.
NCCL_DEBUG_SUBSYS=INIT,ENV

# =========================
# LOGGING
# =========================
LOG_LEVEL=info                            # debug | info | warning | error

# =========================
# BENCHMARK CONFIG (used by benchmark.sh, not llama-run.sh)
# =========================
TOKENIZER=openai/gpt-oss-20b      # HF tokenizer ID for AIPerf.
ENDPOINT_TYPE=chat                # chat | completion
STREAMING=true                    # Enable streaming for baseline/concurrency suites.
STREAMING_LONGCTX=false           # Disable streaming for long-context suite.

# Per-suite request counts
REQUEST_COUNT_BASELINE=10
REQUEST_COUNT_CONCURRENCY=48
REQUEST_COUNT_LONGCTX=64
REQUEST_COUNT_STRESS=32

# Concurrency levels swept per suite
CONCURRENCY_LEVELS="1 2 4 6 16 32"
LONGCTX_CONCURRENCY_LEVELS="64 128"
STRESS_CONCURRENCY_LEVELS="1 2 4 8 16 32"

# Input/output token targets per suite
INPUT_TOKENS_BASELINE=256
OUTPUT_TOKENS_BASELINE=128

INPUT_TOKENS_CONCURRENCY=512
OUTPUT_TOKENS_CONCURRENCY=160

INPUT_TOKENS_LONGCTX=4000
OUTPUT_TOKENS_LONGCTX=2000

INPUT_TOKENS_STRESS=1024
OUTPUT_TOKENS_STRESS=512
```

---

## sg_lang_bench_report.py — Benchmark summariser

Reads AIPerf artifact directories, extracts metrics from JSON/CSV exports, and prints a summary table. Optionally generates matplotlib plots and a Markdown report.

Supports both **SGLang** (`sglang:*`) and **vLLM** (`vllm:*`) Prometheus metric namespaces — the backend is auto-detected from the CSV.

### Directory convention

The script walks `--benchmarks-dir` looking for this structure:

```
benchmarks/
└── <MODEL>_<YYYYMMDD>_<HHMMSS>/       ← model_timestamp (--model-ts filter)
    └── <suite>_c<C>_req<R>_in<I>_out<O>/  ← run directory
        └── aiperf_artifacts/
            ├── profile_export_aiperf.json  ← required
            ├── profile_export_aiperf.csv
            ├── profile_export.jsonl        ← used by --per-request
            └── server_metrics_export.csv   ← SGLang/vLLM Prometheus export
```

Run names must match the pattern `<suite>_c<C>_req<R>_in<I>_out<O>` where suite is one of `baseline`, `concurrency`, `longctx`, or `stress`. Directories that don't match are silently skipped.

---

### CLI arguments

```
python3 sg_lang_bench_report.py [OPTIONS]
```

| Argument | Default | Description |
|---|---|---|
| `--benchmarks-dir DIR` | `benchmarks` | Root directory containing the `<MODEL>_<TIMESTAMP>/` subdirectories. |
| `--model-ts STRING` | *(all)* | Filter to a single model+timestamp directory, e.g. `gpt-oss-20b_20260322_080523`. Without this, runs from **all** model timestamps are shown together. |
| `--suite SUITE` | *(all)* | Filter to one benchmark suite: `baseline`, `concurrency`, `longctx`, or `stress`. |
| `--limit N` | `50` | Maximum number of rows to print in the summary table. |
| `--sort KEYS` | `suite,c` | Comma-separated sort keys applied left-to-right. Valid keys: `suite`, `c` / `concurrency`, `in`, `out`, `req`, `ttft_p95`, `out_tps`, `rps`. |
| `--per-request` | off | Parse `profile_export.jsonl` for per-request p99/max tails and OSL mismatch rate. Slower but gives tail-latency visibility. |
| `--osl-mismatch-threshold-pct N` | `5.0` | When `--per-request` is active, count a request as "mismatched" if its output length differs from requested by more than this percentage. |
| `--plot` | off | Generate matplotlib scatter plots (requires `pip install matplotlib`). |
| `--by-suite` | off | When used with `--plot` or `--report`, also generate per-suite outputs under `<plot-out>/<suite>/` in addition to a combined set under `<plot-out>/all/`. |
| `--plot-out DIR` | auto | Directory to write plots into. Defaults to `<benchmarks-dir>/<model-ts>/_plots` when `--model-ts` is set, otherwise `<benchmarks-dir>/_plots`. |
| `--report` | off | Write a `report.md` Markdown summary into the plot output directory. |
| `--report-name NAME` | `report.md` | Filename for the Markdown report. |

---

### Output columns explained

The summary table contains these columns:

| Column | Source | Unit | Direction |
|---|---|---|---|
| `run` | directory path | — | — |
| `suite` | run name | — | — |
| `c` | run name | requests | — |
| `req` | run name | total requests | — |
| `in` | run name | tokens | — |
| `out` | run name | tokens | — |
| `backend` | server_metrics CSV | — | — |
| `rps` | AIPerf profile JSON | req/s | ↑ higher is better |
| `out_tps` | AIPerf profile JSON | tok/s | ↑ higher is better |
| `ttft_avg_ms` | AIPerf JSON or SGLang histogram | ms | ↓ lower is better |
| `ttft_p95_ms` | AIPerf JSON or SGLang histogram | ms | ↓ lower is better |
| `itl_avg_ms` | AIPerf JSON or SGLang histogram p50 | ms | ↓ lower is better |
| `itl_p95_ms` | AIPerf JSON or SGLang histogram | ms | ↓ lower is better |
| `req_lat_avg_ms` | AIPerf profile JSON | ms | ↓ lower is better |
| `req_lat_p95_ms` | AIPerf profile JSON | ms | ↓ lower is better |
| `kv_avg%` | SGLang `token_usage` × 100 | % | ↓ lower = more headroom |
| `kv_p95%` | SGLang `token_usage` p95 × 100 | % | ↓ lower = more headroom |
| `running_avg` | SGLang `num_running_reqs` | requests | — |
| `waiting_avg` | SGLang `num_queue_reqs` | requests | ↓ lower is better |
| `prefill_p95_ms` | SGLang `per_stage_req_latency_seconds` stage=prefill_waiting | ms | ↓ lower is better |
| `decode_p95_ms` | SGLang proxy via ITL p95 | ms | ↓ lower is better |
| `e2e_p95_ms` | SGLang `e2e_request_latency_seconds` | ms | ↓ lower is better |
| `queue_p95_ms` | SGLang `queue_time_seconds` | ms | ↓ lower is better |
| `prompt_tps` | SGLang `prompt_tokens` counter rate | tok/s | ↑ higher is better |
| `gen_tps` | SGLang `generation_tokens` counter rate | tok/s | ↑ higher is better |
| `pcache_hit` | SGLang `cache_hit_rate` gauge | ratio 0–1 | ↑ higher is better |
| `pr_ttft_p99` | per-request JSONL (`--per-request`) | ms | ↓ lower is better |
| `pr_ttft_max` | per-request JSONL | ms | ↓ lower is better |
| `pr_req_p99` | per-request JSONL | ms | ↓ lower is better |
| `pr_req_max` | per-request JSONL | ms | ↓ lower is better |
| `pr_wait_p99` | per-request JSONL | ms | ↓ lower is better |
| `pr_wait_max` | per-request JSONL | ms | ↓ lower is better |
| `pr_osl_mismatch` | per-request JSONL | fraction | ↓ lower is better |
| `little_L` | derived: `rps × (req_lat_avg_sec)` | — | diagnostic |
| `q_pressure` | derived: `waiting_avg / concurrency` | — | ↓ lower is better |

> **Note on `decode_p95_ms` for SGLang:** SGLang does not expose a per-request decode-phase histogram. The value shown is a proxy using the inter-token latency p95, which approximates the worst-case per-step decode time.

---

### SGLang metric mapping

The script auto-detects the backend and maps SGLang's Prometheus metric names to the same logical fields used for vLLM:

| Logical field | SGLang metric | vLLM metric |
|---|---|---|
| KV cache usage | `sglang:token_usage` (×100 → %) | `vllm:kv_cache_usage_perc` |
| Running requests | `sglang:num_running_reqs` | `vllm:num_requests_running` |
| Waiting requests | `sglang:num_queue_reqs` | `vllm:num_requests_waiting` |
| Cache hit ratio | `sglang:cache_hit_rate` (gauge) | `vllm:prefix_cache_hits` / `vllm:prefix_cache_queries` |
| Prompt token rate | `sglang:prompt_tokens` (counter rate) | `vllm:prompt_tokens` (counter rate) |
| Generation token rate | `sglang:generation_tokens` (counter rate) | `vllm:generation_tokens` (counter rate) |
| E2E latency histogram | `sglang:e2e_request_latency_seconds` | `vllm:e2e_request_latency_seconds` |
| TTFT histogram | `sglang:time_to_first_token_seconds` | *(not used; from AIPerf JSON)* |
| ITL histogram | `sglang:inter_token_latency_seconds` | *(not used; from AIPerf JSON)* |
| Prefill latency | `sglang:per_stage_req_latency_seconds` stage=`prefill_waiting` | `vllm:request_prefill_time_seconds` |
| Decode latency | *(proxy via ITL p95)* | `vllm:request_decode_time_seconds` |
| Queue latency | `sglang:queue_time_seconds` | `vllm:request_queue_time_seconds` |

---

### sg_lang_bench_report.py examples

#### Print a summary table for all runs

```bash
python3 sg_lang_bench_report.py --benchmarks-dir benchmarks
```

#### Filter to a specific model + generate all plots

```bash
python3 sg_lang_bench_report.py \
  --benchmarks-dir benchmarks \
  --model-ts gpt-oss-20b_20260322_080523 \
  --plot \
  --by-suite \
  --report
```

Output structure:
```
benchmarks/gpt-oss-20b_20260322_080523/_plots/
├── all/
│   ├── ttft_p95_vs_concurrency.png
│   ├── itl_p95_vs_concurrency.png
│   ├── out_tps_vs_concurrency.png
│   ├── kv_usage_p95_vs_concurrency.png
│   ├── e2e_p95_vs_concurrency.png
│   ├── queue_p95_vs_concurrency.png
│   ├── gen_tps_vs_concurrency.png
│   ├── littles_law_vs_concurrency.png
│   └── report.md
└── longctx/
    ├── ttft_p95_vs_concurrency.png
    └── ...
```

#### Only the longctx suite, sorted by TTFT p95

```bash
python3 sg_lang_bench_report.py \
  --benchmarks-dir benchmarks \
  --model-ts gpt-oss-20b_20260322_080523 \
  --suite longctx \
  --sort ttft_p95
```

#### Include per-request tails (p99 / max)

```bash
python3 sg_lang_bench_report.py \
  --benchmarks-dir benchmarks \
  --model-ts gpt-oss-20b_20260322_080523 \
  --per-request \
  --osl-mismatch-threshold-pct 3.0 \
  --plot
```

#### Custom output directory

```bash
python3 sg_lang_bench_report.py \
  --benchmarks-dir /data/benchmarks \
  --plot \
  --plot-out /reports/gpt-oss-20b \
  --report \
  --report-name summary.md
```

#### Compare multiple model runs side by side

```bash
# Omit --model-ts to include all timestamps
python3 sg_lang_bench_report.py \
  --benchmarks-dir benchmarks \
  --suite concurrency \
  --sort suite,c \
  --plot \
  --plot-out benchmarks/_plots_compare
```

---

## Benchmark suites

The env file defines four suites, each targeting a different operational regime:

| Suite | Concurrency levels | Input tokens | Output tokens | Purpose |
|---|---|---|---|---|
| `baseline` | `1 2 4 6 16 32` | 256 | 128 | Short-context latency baseline; establishes TTFT and ITL floor |
| `concurrency` | `1 2 4 6 16 32` | 512 | 160 | Mid-context throughput sweep; finds saturation point |
| `longctx` | `64 128` | 4000 | 2000 | Long-context stress; tests KV cache pressure and eviction |
| `stress` | `1 2 4 8 16 32` | 1024 | 512 | Sustained mixed load; checks stability under extended load |

---

## Tuning guide

### Finding the right `MAX_RUNNING_REQUESTS`

1. Start at a conservative value (e.g. 16 for a 20B model on 2×H100s).
2. Run the `concurrency` suite.
3. Look at `kv_p95%` in the report. If it stays below ~85%, you have KV headroom — increase `MAX_RUNNING_REQUESTS`.
4. Look at `waiting_avg`. If this grows linearly with concurrency while throughput plateaus, you've found saturation. Back off one step.

### Choosing `SCHEDULE_POLICY`

- Use `lpm` if your workload has shared system prompts (e.g. all requests use the same instruction preamble). The prefix cache hit ratio (`pcache_hit`) will jump from ~0 to 0.5+ and TTFT will drop dramatically.
- Use `fcfs` (default) for unpredictable prompt distributions.

### Mixed chunk (`ENABLE_MIXED_CHUNK`)

Enable this when you see decode throughput drop significantly as concurrency increases. Mixed-chunk batching lets decode tokens "ride along" with prefill batches, improving GPU utilisation. The tradeoff is slightly higher prefill latency per token.

### When to disable CUDA graphs (`DISABLE_CUDA_GRAPH=1`)

- Running a MoE model (irregular expert routing creates variable shapes).
- Hitting cryptic CUDA errors at decode time.
- You need to reduce startup time and VRAM consumption for development.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ERROR: HF_TOKEN is not set` | Token not in env or env file | `export HF_TOKEN=hf_xxxx` before running |
| Container starts but `/health` returns 503 | Model still loading | Wait; watch `docker logs -f sglang-gptoss-server` |
| `kv_p95%` column shows `N/A` in report | `ENABLE_METRICS=0` | Set `ENABLE_METRICS=1` and re-run |
| All server metric columns are `N/A` | Wrong backend detected or no CSV | Verify `/metrics` is reachable and AIPerf is scraping it |
| OOM error in container logs | `MEM_FRACTION_STATIC` too high | Lower by 0.03–0.05 |
| Graphs are empty / "No numeric data" | Only one concurrency level present | Run multiple concurrency levels; plots need ≥2 points |
| `decode_p95_vs_concurrency.png` flat | SGLang proxy limitation | This is expected — it shows ITL p95 as a proxy |
| High `waiting_avg` at all concurrencies | `MAX_RUNNING_REQUESTS` too low | Increase it; watch `kv_p95%` stays under 90% |
| `pr_ttft_p99` always `N/A` | `--per-request` not passed | Add `--per-request` flag to the report command |