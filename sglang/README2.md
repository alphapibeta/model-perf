# SGLang LLM Benchmark Toolchain

A self-contained toolchain for launching an SGLang inference server in Docker, running AIPerf benchmark suites against it, and summarising + plotting the results.

---

## Table of Contents

1. [Repo layout](#repo-layout)
2. [Quick start](#quick-start)
3. [llama-run.sh — Server launcher](#llama-runsh--server-launcher)
4. [sglang-gpt.env — Configuration file](#sglang-gptenv--configuration-file)
5. [SGLang server arguments — complete reference](#sglang-server-arguments--complete-reference)
   - [Model and tokenizer](#model-and-tokenizer)
   - [HTTP server](#http-server)
   - [Quantization and data type](#quantization-and-data-type)
   - [Memory and scheduling](#memory-and-scheduling)
   - [Runtime options and parallelism](#runtime-options-and-parallelism)
   - [Logging and observability](#logging-and-observability)
   - [API and chat templates](#api-and-chat-templates)
   - [Data parallelism](#data-parallelism)
   - [Multi-node distributed serving](#multi-node-distributed-serving)
   - [Kernel backends](#kernel-backends)
   - [Speculative decoding](#speculative-decoding)
   - [Mixture-of-Experts (MoE)](#mixture-of-experts-moe)
   - [LoRA](#lora)
   - [Hierarchical KV cache](#hierarchical-kv-cache)
   - [Mamba cache](#mamba-cache)
   - [Offloading](#offloading)
   - [Multi-modal](#multi-modal)
   - [PD disaggregation](#pd-disaggregation)
   - [Optimization and debug](#optimization-and-debug)
   - [Deprecated arguments](#deprecated-arguments)
   - [Flag quick-reference](#flag-quick-reference)
   - [Launch examples](#launch-examples)
6. [sg_lang_bench_report.py — Benchmark summariser](#sg_lang_bench_reportpy--benchmark-summariser)
7. [Benchmark suites](#benchmark-suites)
8. [Tuning guide](#tuning-guide)
9. [Troubleshooting](#troubleshooting)

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
3. Builds a `docker run` command array with all SGLang arguments, appending optional flags only when their env variable is set to `1`.
4. Tears down any existing container with the same name, then starts the new one detached (`-d`).
5. Prints the container ID, a `docker logs -f` reminder, and the health/API URLs.

```bash
bash llama-run.sh                  # uses ./sglang-gpt.env
bash llama-run.sh /path/to/my.env  # uses a custom path
```

### Environment variables

Every variable has a built-in default. Override in `sglang-gpt.env` or export into the shell.

#### Model

| Variable | Default | SGLang arg | Description |
|---|---|---|---|
| `MODEL_PATH` | *(required)* | `--model-path` | HuggingFace model ID or local path. **Script exits if unset.** |
| `SERVED_MODEL_NAME` | `$MODEL_PATH` | `--served-model-name` | API alias returned by `/v1/models`. Must match `aiperf --model`. |

#### GPU & parallelism

| Variable | Default | SGLang arg | Description |
|---|---|---|---|
| `CUDA_DEVICES` | `0` | `NVIDIA_VISIBLE_DEVICES` | Comma-separated GPU indices, e.g. `0,1`. |
| `TP_SIZE` | `1` | `--tensor-parallel-size` | Must equal the count of `CUDA_DEVICES`. |

#### Server networking

| Variable | Default | SGLang arg | Description |
|---|---|---|---|
| `HOST` | `0.0.0.0` | `--host` | Bind address. `127.0.0.1` for localhost-only. |
| `PORT` | `30000` | `--port` | HTTP port. |
| `CONTAINER_NAME` | `sglang-gptoss-server` | — | Docker container name. Change to run multiple servers. |
| `IMAGE_NAME` | `lmsysorg/sglang:latest` | — | Docker image. Pin a tag for reproducibility. |

#### Memory & performance

| Variable | Default | SGLang arg | Description |
|---|---|---|---|
| `MEM_FRACTION_STATIC` | `0.85` | `--mem-fraction-static` | Fraction of GPU VRAM for KV cache pool. Raise toward `0.93` for max capacity; lower if OOM. |
| `CHUNKED_PREFILL_SIZE` | `2048` | `--chunked-prefill-size` | Max tokens per prefill chunk. Smaller reduces VRAM spikes; larger improves throughput. |
| `MAX_PREFILL_TOKENS` | `16384` | `--max-prefill-tokens` | Total-token budget per prefill step. |
| `STREAM_INTERVAL` | `2` | `--stream-interval` | Decode steps between streaming flushes. `1` = lowest latency; `4–8` = higher throughput. |
| `MAX_RUNNING_REQUESTS` | `16` | `--max-running-requests` | Hard cap on concurrent decode requests. **Most critical tuning knob.** |
| `MAX_QUEUED_REQUESTS` | `128` | `--max-queued-requests` | Queue depth before HTTP 503. |
| `SCHEDULE_POLICY` | `fcfs` | `--schedule-policy` | `fcfs`, `lpm`, `dfs-weight`, `random`, `lof`, `priority`. |
| `NUM_CONTINUOUS_DECODE_STEPS` | `1` | `--num-continuous-decode-steps` | Decode steps per scheduler tick. Values > 1 improve throughput at cost of scheduling jitter. |

#### Backends

| Variable | Default | SGLang arg | Description |
|---|---|---|---|
| `ATTENTION_BACKEND` | `triton` | `--attention-backend` | `triton`, `flashinfer`, `fa3`, `fa4`, `torch_native`, etc. |
| `SAMPLING_BACKEND` | `pytorch` | `--sampling-backend` | `pytorch` or `flashinfer`. |
| `MOE_RUNNER_BACKEND` | *(empty)* | `--moe-runner-backend` | `flashinfer_cutlass`, `deep_gemm`, `triton`, etc. Leave blank for dense models. |
| `FP4_GEMM_BACKEND` | `flashinfer_cutlass` | `--fp4-gemm-backend` | For FP4-weight models. `flashinfer_cutlass` is the recommended default. |

#### HuggingFace cache

| Variable | Default | Description |
|---|---|---|
| `HF_HOME` | `/root/.cache/huggingface` | Path **inside** the container. |
| `HF_CACHE_HOST_PATH` | `$HOME/.cache/huggingface` | Host path bind-mounted to `HF_HOME`. Reuse across runs to avoid re-downloading. |

#### System / NVIDIA

| Variable | Default | Description |
|---|---|---|
| `SHM_SIZE` | `16g` | `/dev/shm` size for NCCL. Use `32g`+ for 4+ GPU TP. |
| `NVIDIA_DRIVER_CAPABILITIES` | `compute,utility` | Docker NVIDIA capabilities. |
| `NCCL_DEBUG` | `INFO` | NCCL verbosity. Use `WARN` to reduce startup noise. |
| `NCCL_DEBUG_SUBSYS` | `INIT,ENV` | NCCL subsystems to log. |

#### Boolean flags (set `=1` to enable)

| Variable | Default | SGLang arg | Effect |
|---|---|---|---|
| `LOG_LEVEL` | `info` | `--log-level` | `debug` / `info` / `warning` / `error` |
| `ENABLE_MIXED_CHUNK` | `0` | `--enable-mixed-chunk` | Mix prefill+decode in same batch for better GPU utilisation |
| `ENABLE_METRICS` | `0` | `--enable-metrics` | Expose Prometheus `/metrics`. **Required for bench report server metrics.** |
| `ENABLE_P2P_CHECK` | `0` | `--enable-p2p-check` | Verify NVLink P2P at startup. Recommended for multi-GPU. |
| `DISABLE_CUDA_GRAPH` | `0` | `--disable-cuda-graph` | Disable CUDA graphs. Use for MoE models or CUDA graph errors. |
| `SKIP_SERVER_WARMUP` | `0` | `--skip-server-warmup` | Skip warmup request. Dev-only — never skip in production benchmarks. |

---

## sglang-gpt.env — Configuration file

This is the **only file you should edit** day-to-day. Full annotated reference:

```bash
# =========================
# MODEL
# =========================
MODEL_PATH=openai/gpt-oss-20b       # Required. HF model ID or local path.
SERVED_MODEL_NAME=gpt-oss-20b       # API alias. Must match aiperf --model arg.

# =========================
# GPU
# =========================
CUDA_DEVICES=0,1                    # GPUs to use. Comma-separated indices.
TP_SIZE=2                           # Must equal number of devices above.

# =========================
# SERVER
# =========================
HOST=0.0.0.0
PORT=30000
CONTAINER_NAME=sglang-gptoss-server
IMAGE_NAME=lmsysorg/sglang:latest   # Pin a version tag for reproducibility.

# =========================
# MEMORY / PERFORMANCE
# =========================
MEM_FRACTION_STATIC=0.92            # 92% GPU VRAM for KV cache.
CHUNKED_PREFILL_SIZE=2048
MAX_PREFILL_TOKENS=8192
STREAM_INTERVAL=2
MAX_RUNNING_REQUESTS=32             # Most critical tuning knob.
MAX_QUEUED_REQUESTS=32
SCHEDULE_POLICY=fcfs
NUM_CONTINUOUS_DECODE_STEPS=1

# =========================
# BACKENDS
# =========================
MOE_RUNNER_BACKEND=flashinfer_cutlass  # MoE models only. Leave blank for dense.
ATTENTION_BACKEND=triton
SAMPLING_BACKEND=pytorch
FP4_GEMM_BACKEND=flashinfer_cutlass    # FP4-weight models only.

# =========================
# FLAGS (1=enable, 0=disable)
# =========================
ENABLE_MIXED_CHUNK=1
ENABLE_METRICS=1        # Needed for sg_lang_bench_report.py server metrics.
ENABLE_P2P_CHECK=1
DISABLE_CUDA_GRAPH=1    # Needed for MoE models.
SKIP_SERVER_WARMUP=0

# =========================
# HF CACHE
# =========================
HF_HOME=/root/.cache/huggingface
HF_CACHE_HOST_PATH=/root/.cache/huggingface
# HF_TOKEN: leave blank here; export in shell

# =========================
# NVIDIA / SYSTEM
# =========================
SHM_SIZE=16g
NVIDIA_DRIVER_CAPABILITIES=compute,utility
NCCL_DEBUG=INFO
NCCL_DEBUG_SUBSYS=INIT,ENV
LOG_LEVEL=info

# =========================
# BENCHMARK CONFIG (used by benchmark.sh, not llama-run.sh)
# =========================
TOKENIZER=openai/gpt-oss-20b
ENDPOINT_TYPE=chat
STREAMING=true
STREAMING_LONGCTX=false

REQUEST_COUNT_BASELINE=10
REQUEST_COUNT_CONCURRENCY=48
REQUEST_COUNT_LONGCTX=64
REQUEST_COUNT_STRESS=32

CONCURRENCY_LEVELS="1 2 4 6 16 32"
LONGCTX_CONCURRENCY_LEVELS="64 128"
STRESS_CONCURRENCY_LEVELS="1 2 4 8 16 32"

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

## SGLang server arguments — complete reference

All arguments are passed to `python3 -m sglang.launch_server`. You can also use a YAML config file — CLI arguments override config file values:

```bash
cat > config.yaml << EOF
model-path: meta-llama/Meta-Llama-3-8B-Instruct
host: 0.0.0.0
port: 30000
tensor-parallel-size: 2
enable-metrics: true
log-requests: true
EOF
python -m sglang.launch_server --config config.yaml
```

---

### Model and tokenizer

| Argument | Default | Description |
|---|---|---|
| `--model-path` / `--model` | *(required)* | Local folder path or HuggingFace repo ID for model weights. |
| `--tokenizer-path` | same as model | Path to tokenizer if stored separately from the model. |
| `--tokenizer-mode` | `auto` | `auto` uses the fast tokenizer when available. `slow` always uses the slow tokenizer. |
| `--tokenizer-worker-num` | `1` | Number of tokenizer manager worker threads. Increase for very high request rates. |
| `--skip-tokenizer-init` | `false` | Skip tokenizer init entirely; callers must pass raw `input_ids`. |
| `--load-format` | `auto` | Weight loading format. `auto` tries safetensors then pytorch. Other options: `pt`, `safetensors`, `npcache`, `dummy` (random weights for profiling), `gguf`, `bitsandbytes`, `layered` (loads one layer at a time to minimise peak memory — useful when quantizing on the fly), `flash_rl`, `fastsafetensors`. |
| `--model-loader-extra-config` | `{}` | JSON string of extra config forwarded to the chosen loader. |
| `--trust-remote-code` | `false` | Allow custom modeling files from HuggingFace Hub. Required for many community models. |
| `--context-length` | from model config | Override the maximum context length. Useful to reduce KV cache pressure by capping at a shorter length. |
| `--is-embedding` | `false` | Treat a CausalLM as an embedding model. |
| `--enable-multimodal` | `false` | Enable multimodal (image/video/audio) inputs. No effect on text-only models. |
| `--revision` | default | Specific model version: branch name, tag, or commit hash. |
| `--model-impl` | `auto` | `auto` prefers SGLang's own implementation; `sglang` forces it; `transformers` forces HF Transformers. |

---

### HTTP server

| Argument | Default | Description |
|---|---|---|
| `--host` | `127.0.0.1` | Bind address. Set to `0.0.0.0` to accept external connections. |
| `--port` | `30000` | HTTP port. |
| `--fastapi-root-path` | `""` | Set when serving behind a path-based reverse proxy (e.g. `/api/v1`). |
| `--grpc-mode` | `false` | Use gRPC instead of HTTP. |
| `--skip-server-warmup` | `false` | Skip the startup warmup request. Dev-only: the first real request will be slower without warmup. |
| `--warmups` | `None` | Comma-separated custom warmup function names to run before the server marks itself ready. |
| `--nccl-port` | random | Explicit port for the NCCL distributed environment setup. Set this if the random port conflicts. |

---

### Quantization and data type

| Argument | Default | Description |
|---|---|---|
| `--dtype` | `auto` | Weight/activation dtype: `auto` (BF16 for BF16 models, FP16 otherwise), `half`/`float16`, `bfloat16`, `float`/`float32`. |
| `--quantization` | `None` | Quantization method: `awq`, `fp8`, `gptq`, `marlin`, `gptq_marlin`, `awq_marlin`, `bitsandbytes`, `gguf`, `modelopt`, `modelopt_fp8`, `modelopt_fp4`, `w8a8_int8`, `w8a8_fp8`, `moe_wna16`, `qoq`, `w4afp8`, `mxfp4`, `mxfp8`, `compressed-tensors`, and others. |
| `--quantization-param-path` | `None` | Path to JSON file with KV cache scaling factors. Provide when `--kv-cache-dtype fp8_*` is used; otherwise scaling defaults to 1.0 and accuracy degrades. |
| `--kv-cache-dtype` | `auto` | KV cache storage dtype: `auto`, `fp8_e5m2`, `fp8_e4m3`, `bf16`/`bfloat16`, `fp4_e2m1` (requires CUDA 12.8+). |
| `--enable-fp32-lm-head` | `false` | Output logits in FP32. Helps when output layer precision matters (e.g. logprob-sensitive applications). |
| `--modelopt-quant` | `None` | NVIDIA ModelOpt quantization: `fp8`, `int4_awq`, `w4a8_awq`, `nvfp4`, `nvfp4_awq`. Requires `pip install nvidia-modelopt`. |
| `--modelopt-checkpoint-restore-path` | `None` | Load a previously saved ModelOpt checkpoint to skip re-quantization at startup. |
| `--modelopt-checkpoint-save-path` | `None` | Save the quantized checkpoint after ModelOpt runs, for future reuse. |
| `--modelopt-export-path` | `None` | Export quantized model in HuggingFace format after quantization. |
| `--quantize-and-serve` | `false` | Quantize with ModelOpt and immediately serve. Good for prototyping; use separate steps for production. |

---

### Memory and scheduling

| Argument | Default | Description |
|---|---|---|
| `--mem-fraction-static` | `0.9` | Fraction of GPU memory pre-allocated for the static KV cache pool. **Lower if OOM.** Typical range: `0.80`–`0.93`. The remaining memory holds model weights and activations. |
| `--max-running-requests` | auto | Hard cap on concurrently executing requests. Requests beyond this are queued. **Most important throughput/latency tradeoff knob.** Too low = wasted GPU; too high = KV cache exhaustion / OOM. |
| `--max-queued-requests` | unlimited | Max queue depth before returning HTTP 503. Ignored in PD disaggregation mode. |
| `--max-total-tokens` | auto | Total token slots in the KV pool. Normally auto-calculated; set only for debugging. |
| `--chunked-prefill-size` | auto | Max tokens per prefill chunk. Set to `-1` to disable chunked prefill. Smaller values = smoother latency for concurrent short requests; larger = better throughput for bulk prefill jobs. |
| `--prefill-max-requests` | unlimited | Cap on how many requests enter a single prefill batch. |
| `--enable-dynamic-chunking` | `false` | Dynamically adjust chunk sizes for pipeline parallelism to keep execution time consistent across stages. |
| `--max-prefill-tokens` | `16384` | Total-token budget per prefill step. Actual bound = `max(this, model_context_length)`. |
| `--schedule-policy` | `fcfs` | Request scheduling algorithm: `fcfs` (first-come-first-served), `lpm` (longest-prefix-match, maximises KV cache reuse), `dfs-weight`, `random`, `lof`, `priority`, `routing-key`. |
| `--enable-priority-scheduling` | `false` | Enable integer priority scheduling. Higher value = higher priority (unless `--schedule-low-priority-values-first` is set). |
| `--priority-scheduling-preemption-threshold` | `10` | Minimum priority difference required for a new request to preempt running requests. |
| `--schedule-conservativeness` | `1.0` | Higher values make scheduling more conservative. Increase (e.g. to `1.5`) if you see frequent request retractions in the logs. |
| `--page-size` | `1` | Tokens per KV cache page. Larger pages reduce overhead but waste memory for short sequences. |
| `--radix-eviction-policy` | `lru` | KV cache eviction: `lru` (Least Recently Used) or `lfu` (Least Frequently Used). `lfu` can improve cache hit rate for repeated-prefix workloads. |
| `--swa-full-tokens-ratio` | `0.8` | Ratio of Sliding Window Attention layer KV tokens to full-layer KV tokens (0–1). |
| `--disable-hybrid-swa-memory` | `false` | Disable hybrid SWA memory management. |

---

### Runtime options and parallelism

| Argument | Default | Description |
|---|---|---|
| `--device` | auto | Target device: `cuda`, `xpu`, `hpu`, `npu`, `cpu`. |
| `--tensor-parallel-size` / `--tp-size` | `1` | Tensor parallelism: split model weights across N GPUs. Must match the number of GPUs provided. |
| `--pipeline-parallel-size` / `--pp-size` | `1` | Pipeline parallelism: split model layers across N stages on N GPUs. |
| `--attention-context-parallel-size` / `--attn-cp-size` | `1` | Context parallelism for attention (splits long sequences across ranks). |
| `--moe-data-parallel-size` / `--moe-dp-size` | `1` | Data parallelism for MoE layers independently of FFN TP. |
| `--stream-interval` | `1` | Decode steps between streaming token flushes to the client. `1` = minimum TTFT / inter-chunk delay. `4–8` = higher throughput for non-interactive batch use. |
| `--random-seed` | `None` | Fix the random seed for reproducible sampling output. |
| `--watchdog-timeout` | `300` | Seconds before the watchdog kills the server if a forward batch hangs. Prevents infinite hangs. |
| `--soft-watchdog-timeout` | `None` | Seconds before a soft timeout dumps debug info without crashing. |
| `--download-dir` | `None` | Override the HuggingFace model download directory. |
| `--base-gpu-id` | `0` | Start GPU allocation from this device index. Useful for running multiple server instances on the same machine. |
| `--gpu-id-step` | `1` | Step between consecutive GPU IDs. E.g. `2` uses GPUs 0, 2, 4, … |
| `--num-continuous-decode-steps` | `1` | Decode steps per scheduler tick before yielding to check for new requests. Values > 1 reduce per-step scheduling overhead and can raise throughput at the cost of increased TTFT variance. |
| `--sleep-on-idle` | `false` | Reduce CPU usage when the server is idle. |
| `--allow-auto-truncate` | `false` | Automatically truncate requests exceeding max input length instead of returning an error. |
| `--enable-deterministic-inference` | `false` | Enable batch-invariant operations for reproducible output across different batch sizes. |

---

### Logging and observability

| Argument | Default | Description |
|---|---|---|
| `--log-level` | `info` | Global logging level: `debug`, `info`, `warning`, `error`. |
| `--log-level-http` | same as `--log-level` | HTTP server log level, set independently from the rest. |
| `--log-requests` | `false` | Log metadata/inputs/outputs for every request. Verbosity controlled by `--log-requests-level`. |
| `--log-requests-level` | `2` | `0` = metadata only; `1` = + sampling params; `2` = + partial I/O; `3` = full I/O for every request. |
| `--log-requests-format` | `text` | `text` (human-readable) or `json` (structured, for log aggregation pipelines). |
| `--log-requests-target` | `None` | Output target(s) for request logs: `stdout`, a directory path, or both. |
| `--enable-metrics` | `false` | **Expose Prometheus `/metrics` endpoint.** Must be set to `true` for AIPerf to collect server-side metrics and for `sg_lang_bench_report.py` to produce server charts. |
| `--enable-metrics-for-all-schedulers` | `false` | Record per-TP-rank request metrics separately. Especially useful with `--enable-dp-attention` where all metrics otherwise appear from TP rank 0. |
| `--bucket-time-to-first-token` | defaults | Custom histogram bucket boundaries (list of floats) for the TTFT Prometheus metric. |
| `--bucket-inter-token-latency` | defaults | Custom histogram bucket boundaries for the ITL metric. |
| `--bucket-e2e-request-latency` | defaults | Custom histogram bucket boundaries for the E2E latency metric. |
| `--collect-tokens-histogram` | `false` | Collect prompt and generation token count histograms in Prometheus. |
| `--prompt-tokens-buckets` | defaults | Custom bucket rule for prompt token histogram: `default`, `tse <center> <step> <count>`, or `custom <v1> <v2> ...`. |
| `--generation-tokens-buckets` | defaults | Same as above for generation token histogram. |
| `--decode-log-interval` | `40` | Log decode batch stats every N steps. |
| `--show-time-cost` | `false` | Print timing annotations for custom instrumentation marks. |
| `--enable-trace` | `false` | Enable OpenTelemetry distributed tracing. |
| `--otlp-traces-endpoint` | `localhost:4317` | OpenTelemetry collector endpoint (host:port) when tracing is enabled. |
| `--crash-dump-folder` | `None` | Save recent requests to this folder before a crash. Helps reproduce issues that cause server crashes. |
| `--gc-warning-threshold-secs` | `0.0` | Log a warning if GC takes longer than this threshold. `0` = disabled. |
| `--export-metrics-to-file` | `false` | Write per-request performance metrics to local files for external ingestion. |
| `--export-metrics-to-file-dir` | `None` | Directory for per-request metric files (required when `--export-metrics-to-file` is enabled). |
| `--enable-request-time-stats-logging` | `false` | Enable detailed per-request timing stats in logs. |

---

### API and chat templates

| Argument | Default | Description |
|---|---|---|
| `--api-key` | `None` | API authentication key. Applied to all requests including OpenAI-compatible endpoints. |
| `--admin-api-key` | `None` | Separate key for admin/control endpoints (weight update, cache flush, `/get_server_info`). |
| `--served-model-name` | same as `--model-path` | Override the name returned by `/v1/models`. Set this to match your `aiperf --model` argument exactly. |
| `--chat-template` | `None` | Built-in template name or path to a custom chat template file. Only applies to OpenAI-compatible API server. |
| `--hf-chat-template-name` | first available | When the HF tokenizer has multiple named templates (e.g. `default`, `tool_use`, `rag`), select one by name. |
| `--completion-template` | `None` | Built-in or custom completion template path. Currently only for code completion use cases. |
| `--enable-cache-report` | `false` | Include cached token count in `usage.prompt_tokens_details` for each OpenAI-compatible request. |
| `--reasoning-parser` | `None` | Parser for reasoning model output (chain-of-thought extraction). Options: `deepseek-r1`, `deepseek-v3`, `gpt-oss`, `qwen3`, `qwen3-thinking`, `kimi`, `glm45`, `step3`. |
| `--tool-call-parser` | `None` | Parser for tool/function-call interactions. Options: `gpt-oss`, `llama3`, `mistral`, `qwen`, `qwen25`, `deepseekv3`, `deepseekv31`, `pythonic`, `glm47`, `kimi_k2`, `step3`, and others. |
| `--sampling-defaults` | `model` | Where to get default sampling parameters: `model` (uses `generation_config.json` from the model) or `openai` (temperature=1.0, top_p=1.0 etc.). |
| `--enable-custom-logit-processor` | `false` | Allow API clients to pass custom logit processors. **Disabled by default for security** — enable only in trusted environments. |
| `--weight-version` | `default` | Version identifier for model weights, returned in model info. |

---

### Data parallelism

| Argument | Default | Description |
|---|---|---|
| `--data-parallel-size` / `--dp-size` | `1` | Replicate the full model N times across N GPU groups. Better for throughput when sufficient memory is available. Use with SGLang Router (`sglang_router.launch_server`). |
| `--load-balance-method` | `auto` | DP load balancing strategy: `auto`, `round_robin`, `follow_bootstrap_room`, `total_requests`, `total_tokens`. `total_tokens` requires DP attention and balances based on real-time token load per worker. |

---

### Multi-node distributed serving

| Argument | Default | Description |
|---|---|---|
| `--dist-init-addr` / `--nccl-init-addr` | `None` | `host:port` for distributed backend initialisation, e.g. `192.168.0.2:25000`. Required for multi-node. |
| `--nnodes` | `1` | Total number of nodes in the job. |
| `--node-rank` | `0` | This node's rank (0-indexed). |

**Example — 2-node, 4-GPU total (TP=4):**

```bash
# Node 0
python -m sglang.launch_server \
  --model-path meta-llama/Meta-Llama-3-8B-Instruct \
  --tp 4 --dist-init-addr sgl-dev-0:50000 --nnodes 2 --node-rank 0

# Node 1
python -m sglang.launch_server \
  --model-path meta-llama/Meta-Llama-3-8B-Instruct \
  --tp 4 --dist-init-addr sgl-dev-0:50000 --nnodes 2 --node-rank 1
```

> If you hit a deadlock during multi-node startup, add `--disable-cuda-graph`.

---

### Kernel backends

These choose the low-level CUDA kernels used for the most compute-intensive operations.

#### Attention

| Argument | Default | Options | Notes |
|---|---|---|---|
| `--attention-backend` | auto | `triton`, `flashinfer`, `fa3`, `fa4`, `torch_native`, `flex_attention`, `cutlass_mla`, `flashmla`, `trtllm_mha`, `trtllm_mla`, `nsa`, `aiter`, `wave`, `ascend` | Global attention backend for all phases. |
| `--prefill-attention-backend` | same | same | Overrides `--attention-backend` for prefill phase only. |
| `--decode-attention-backend` | same | same | Overrides `--attention-backend` for decode phase only. |

**Choosing an attention backend:**

| Backend | Best for |
|---|---|
| `triton` | Safe default — works on all CUDA GPUs |
| `flashinfer` | Fastest on A100/H100 for most workloads |
| `fa3` / `fa4` | Long contexts on Hopper+ (Flash Attention 3/4) |
| `torch_native` | Debugging only — slow |
| `cutlass_mla` / `flashmla` | DeepSeek MLA (Multi-head Latent Attention) |
| `trtllm_mha` / `trtllm_mla` | Highest throughput on H100 with TRT-LLM installed |

#### Sampling and grammar

| Argument | Default | Options | Description |
|---|---|---|---|
| `--sampling-backend` | auto | `flashinfer`, `pytorch`, `ascend` | Token sampling kernel. `flashinfer` is faster for large vocabularies (100k+). |
| `--grammar-backend` | auto | `xgrammar`, `outlines`, `llguidance`, `none` | Backend for grammar-guided / JSON-constrained decoding. |

#### GEMM kernels

| Argument | Default | Options | Description |
|---|---|---|---|
| `--fp8-gemm-backend` | `auto` | `auto`, `deep_gemm`, `flashinfer_trtllm`, `flashinfer_cutlass`, `flashinfer_deepgemm`, `cutlass`, `triton`, `aiter` | FP8 blockwise GEMM kernel. `auto` selects DeepGEMM on Hopper/Blackwell when installed. `cutlass` is optimal for high-throughput H100. `triton` is the widest compatibility fallback. |
| `--fp4-gemm-backend` | `flashinfer_cutlass` | `auto`, `flashinfer_cudnn`, `flashinfer_cutlass`, `flashinfer_trtllm` | NVFP4 GEMM for FP4-weight models. `flashinfer_cudnn` is optimal on CUDA 13+ with cuDNN 9.15+. `flashinfer_trtllm` requires differently shuffled weights. |
| `--moe-runner-backend` | `auto` | `auto`, `deep_gemm`, `triton`, `triton_kernel`, `flashinfer_trtllm`, `flashinfer_cutlass`, `flashinfer_mxfp4`, `cutlass` | MoE expert routing + GEMM. `flashinfer_cutlass` is recommended for most MoE models (Mixtral, GPT-MoE variants). `deep_gemm` is best when DeepGEMM is installed on Hopper. |

---

### Speculative decoding

Uses a small draft model to propose tokens verified in parallel by the target model. Improves throughput by 1.5–3× at small batch sizes.

| Argument | Default | Description |
|---|---|---|
| `--speculative-algorithm` | `None` | `EAGLE`, `EAGLE3`, `NEXTN`, `STANDALONE`, `NGRAM`. |
| `--speculative-draft-model-path` | `None` | HF repo ID or local path for the draft model. |
| `--speculative-draft-model-revision` | default | Specific draft model version (branch/tag/commit). |
| `--speculative-num-steps` | auto | Draft steps per speculative decode cycle. |
| `--speculative-eagle-topk` | auto | Top-K candidates per draft step for EAGLE2. |
| `--speculative-num-draft-tokens` | auto | Total draft tokens proposed per cycle. |
| `--speculative-accept-threshold-single` | `1.0` | Accept a draft token if its probability in the target model exceeds this (0–1). Lower = more aggressive (more accepted, more risk of quality loss). |
| `--speculative-accept-threshold-acc` | `1.0` | Raises acceptance probability: `min(1, p / threshold)`. Adjusts global acceptance rate. |
| `--speculative-attention-mode` | `prefill` | Attention mode for speculative operations: `prefill` or `decode`. |
| `--speculative-draft-model-quantization` | `None` | Quantization for the draft model (same options as `--quantization`). |

**N-gram speculative decoding** — no draft model needed; uses the prompt as the draft source:

| Argument | Default | Description |
|---|---|---|
| `--speculative-ngram-min-match-window-size` | `1` | Min window for n-gram pattern matching. |
| `--speculative-ngram-max-match-window-size` | `12` | Max window. Larger = potentially better draft quality. |
| `--speculative-ngram-min-bfs-breadth` | `1` | Min BFS breadth for candidate exploration. |
| `--speculative-ngram-max-bfs-breadth` | `10` | Max BFS breadth. |
| `--speculative-ngram-match-type` | `BFS` | `BFS` (recency-based expansion) or `PROB` (frequency-based). |
| `--speculative-ngram-branch-length` | `18` | Draft branch length. |
| `--speculative-ngram-capacity` | `10000000` | N-gram cache capacity (entries). |

---

### Mixture-of-Experts (MoE)

| Argument | Default | Description |
|---|---|---|
| `--expert-parallel-size` / `--ep-size` | `1` | Distribute experts across N GPU groups. Commonly set equal to `--tp-size`. |
| `--moe-a2a-backend` | `none` | All-to-all comm backend for expert parallelism: `deepep`, `mooncake`, `mori`, `nixl`, `ascend_fuseep`. |
| `--moe-runner-backend` | `auto` | MoE GEMM backend — see [Kernel backends](#kernel-backends). |
| `--deepep-mode` | `auto` | `normal`, `low_latency`, or `auto` (low_latency for decode batches, normal for prefill). |
| `--ep-num-redundant-experts` | `0` | Allocate N extra redundant experts per rank to improve expert utilisation under skewed routing. |
| `--enable-eplb` | `false` | Enable Expert Parallel Load Balancing to dynamically rebalance hot experts. |
| `--eplb-rebalance-num-iterations` | `1000` | Trigger an EPLB rebalance every N forward passes. |
| `--eplb-min-rebalancing-utilization-threshold` | `1.0` | GPU utilization must be below this to trigger rebalancing (0.0–1.0). |
| `--enable-expert-distribution-metrics` | `false` | Log expert balancedness metrics to Prometheus. |
| `--enable-flashinfer-allreduce-fusion` | `false` | Fuse FlashInfer allreduce with Residual RMSNorm. Reduces communication overhead for MoE. |
| `--enable-aiter-allreduce-fusion` | `false` | ROCm equivalent of the above for AMD GPUs. |
| `--moe-dense-tp-size` | `None` | TP size for the dense MLP layers within a MoE model. Set when large TP causes dimension errors in GEMM (weight dimensions smaller than GEMM minimum). |
| `--elastic-ep-backend` | `none` | Collective backend for elastic EP: `mooncake`. |

---

### LoRA

| Argument | Default | Description |
|---|---|---|
| `--enable-lora` | `false` | Enable LoRA adapter support. Automatically `true` when `--lora-paths` is provided. |
| `--lora-paths` | `None` | Adapters to load at startup. Format: `<PATH>`, `<name>=<PATH>`, or JSON `{"lora_name": ..., "lora_path": ..., "pinned": bool}`. Multiple values allowed. |
| `--max-lora-rank` | auto from adapters | Maximum LoRA rank to support. Set explicitly to allow dynamically loading higher-rank adapters after startup. |
| `--lora-target-modules` | auto from adapters | Which modules to apply LoRA to: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, `qkv_proj`, `gate_up_proj`, `all`. |
| `--max-loras-per-batch` | `8` | Maximum concurrent adapters in a single batch, including base-only requests. |
| `--max-loaded-loras` | unlimited | Max adapters held in CPU memory at once. Must be ≥ `--max-loras-per-batch`. |
| `--lora-eviction-policy` | `lru` | Adapter eviction policy when GPU memory pool is full: `lru` or `fifo`. |
| `--lora-backend` | `csgmv` | Multi-LoRA GEMM kernel: `triton`, `csgmv` (ChunkedSGMV), `torch_native`, `ascend`. |
| `--max-lora-chunk-size` | `16` | Chunk size for ChunkedSGMV backend. Options: `16`, `32`, `64`, `128`. Larger = faster but more memory. |
| `--enable-lora-overlap-loading` | `false` | Async LoRA weight loading to overlap H2D transfers with GPU compute. Enable when adapters are large and swapped frequently. |

---

### Hierarchical KV cache

Spills overflow KV entries to CPU memory or external storage, enabling much higher effective context without OOM.

| Argument | Default | Description |
|---|---|---|
| `--enable-hierarchical-cache` | `false` | Enable CPU/storage-backed overflow KV cache. |
| `--hicache-ratio` | `2.0` | Host KV cache size as a multiple of the device pool size. |
| `--hicache-size` | `0` | Host KV pool size in GB. Overrides `--hicache-ratio` when set. |
| `--hicache-write-policy` | `write_through` | `write_through` (write to host immediately), `write_back` (lazy), `write_through_selective`. |
| `--hicache-io-backend` | `kernel` | CPU↔GPU transfer mechanism: `direct`, `kernel`, `kernel_ascend`. |
| `--hicache-mem-layout` | `layer_first` | Host memory layout: `layer_first`, `page_first`, `page_first_direct`, `page_first_kv_split`, `page_head`. |
| `--hicache-storage-backend` | `None` | External storage backend: `file`, `mooncake`, `hf3fs`, `nixl`, `aibrix`. |
| `--hicache-storage-prefetch-policy` | `best_effort` | When to stop prefetching from storage: `best_effort`, `wait_complete`, `timeout`. |
| `--hicache-storage-backend-extra-config` | `None` | JSON string or `@<filepath>` of extra config for the storage backend. |
| `--enable-lmcache` | `false` | Use LMCache as an alternative hierarchical KV cache solution. |

---

### Mamba cache

Only relevant for Mamba / hybrid SSM-Transformer models. No effect on pure Transformer architectures.

| Argument | Default | Description |
|---|---|---|
| `--max-mamba-cache-size` | auto | Maximum number of Mamba state cache slots. |
| `--mamba-ssm-dtype` | `float32` | SSM state precision: `float32`, `bfloat16`, `float16`. |
| `--mamba-full-memory-ratio` | `0.9` | Ratio of Mamba state memory budget to the full KV cache memory budget. |
| `--mamba-scheduler-strategy` | `auto` | `no_buffer` (default, no overlap scheduling) or `extra_buffer` (enables overlap at 2× Mamba memory cost; strictly better for non-KV-bound workloads). |
| `--mamba-track-interval` | `256` | Interval (tokens) at which to track Mamba state during decode when using `extra_buffer`. Must be divisible by `--page-size` and ≥ speculative draft tokens when using spec decoding. |

---

### Offloading

For models that don't fit entirely in GPU memory — spills weight layers to CPU RAM.

| Argument | Default | Description |
|---|---|---|
| `--cpu-offload-gb` | `0` | GB of CPU RAM to reserve for weight offloading. `0` = disabled. |
| `--offload-group-size` | `-1` | Number of model layers per offload group. `-1` = auto. |
| `--offload-num-in-group` | `1` | Layers to offload per group. |
| `--offload-prefetch-step` | `1` | Steps ahead to prefetch the next offload group. Higher = lower stall at the cost of more CPU memory pinned. |
| `--offload-mode` | `cpu` | Offloading target. |

---

### Multi-modal

For vision-language models (VLMs) processing images, video, or audio alongside text.

| Argument | Default | Description |
|---|---|---|
| `--enable-multimodal` | `false` | Enable multimodal inputs. No effect on text-only models. |
| `--mm-max-concurrent-calls` | `32` | Max concurrent async multimodal preprocessing calls. Increase for high image/video throughput. |
| `--mm-per-request-timeout` | `10.0` | Timeout in seconds for multimodal preprocessing per request. |
| `--mm-attention-backend` | auto | Attention backend for multimodal attention layers: `sdpa`, `fa3`, `fa4`, `triton_attn`, `ascend_attn`, `aiter_attn`. |
| `--limit-mm-data-per-request` | `None` | JSON dict capping multimodal inputs per request, e.g. `{"image": 1, "video": 2}`. |
| `--enable-mm-global-cache` | `false` | Mooncake-backed cache for ViT embeddings. Repeated images reuse cached embeddings instead of re-running the encoder. |
| `--mm-enable-dp-encoder` | `false` | Data-parallel multimodal encoder (dp size is set to tp size automatically). |
| `--keep-mm-feature-on-device` | `false` | Keep multimodal feature tensors on GPU after processing to avoid costly D2H copies. |
| `--enable-prefix-mm-cache` | `false` | Prefix cache for multimodal inputs (currently supports mm-only prompts). |
| `--mm-process-config` | `{}` | JSON config for multimodal preprocessing per modality: `{"image": {...}, "video": {...}, "audio": {...}}`. |

---

### PD disaggregation

Splits Prefill and Decode phases onto separate server instances for maximum throughput at production scale.

| Argument | Default | Description |
|---|---|---|
| `--disaggregation-mode` | `null` | `prefill` for a prefill-only server; `decode` for a decode-only server; `null` for combined operation. |
| `--disaggregation-transfer-backend` | `mooncake` | KV cache transfer protocol between P and D nodes: `mooncake`, `nixl`, `ascend`, `fake` (for testing). |
| `--disaggregation-bootstrap-port` | `8998` | Bootstrap server port on the prefill node. |
| `--disaggregation-ib-device` | auto | InfiniBand device(s) for transfer, e.g. `mlx5_0` or `mlx5_0,mlx5_1`. |
| `--num-reserved-decode-tokens` | `512` | Memory slots reserved for decode tokens when a new request joins the running batch on the decode server. |
| `--disaggregation-decode-enable-offload-kvcache` | `false` | Enable async KV cache offloading on the decode server. |
| `--disaggregation-decode-polling-interval` | `1` | Polling interval (steps) on the decode server. Increase to reduce overhead. |

---

### Optimization and debug

| Argument | Default | Description |
|---|---|---|
| `--disable-radix-cache` | `false` | Disable RadixAttention prefix caching. Normally leave enabled — only disable to isolate prefix caching effects. |
| `--disable-cuda-graph` | `false` | Disable CUDA graphs for decode. Required for some MoE models with dynamic expert shapes. Increases decode overhead. |
| `--cuda-graph-max-bs` | auto | Extend CUDA graph capture to cover this max batch size. |
| `--cuda-graph-bs` | auto | Explicit list of batch sizes to capture CUDA graphs for. |
| `--disable-cuda-graph-padding` | `false` | Disable CUDA graphs only when padding is needed; still uses them for unpadded batches. |
| `--enable-mixed-chunk` | `false` | Batch prefill and decode tokens together. Improves GPU utilisation at moderate concurrency. |
| `--enable-dp-attention` | `false` | Data-parallel attention + tensor-parallel FFN. Currently supports DeepSeek-V2/V3 and Qwen2/3 MoE. DP size must equal TP size. |
| `--enable-two-batch-overlap` | `false` | Overlap computation of two micro-batches for pipeline parallelism. |
| `--enable-single-batch-overlap` | `false` | Overlap computation and communication within a single micro-batch. |
| `--disable-overlap-schedule` | `false` | Disable the overlap scheduler that runs CPU scheduling concurrently with GPU execution. Use for debugging scheduling issues. |
| `--enable-torch-compile` | `false` | Accelerate with `torch.compile`. Experimental; best for small models at small batch sizes. |
| `--torch-compile-max-bs` | `32` | Max batch size when using `torch.compile`. |
| `--torchao-config` | `""` | TorchAO post-training optimisation: `int8dq`, `int8wo`, `int4wo-<group_size>`, `fp8wo`, `fp8dq-per_tensor`, `fp8dq-per_row`. |
| `--enable-nan-detection` | `false` | Log NaN logprobs for debugging numerical instability. |
| `--enable-p2p-check` | `false` | Verify GPU-to-GPU P2P access at startup. Recommended for all multi-GPU setups. |
| `--triton-attention-reduce-in-fp32` | `false` | Cast intermediate Triton attention accumulations to FP32. Prevents FP16 overflow crashes on certain hardware. |
| `--triton-attention-num-kv-splits` | `8` | Number of KV splits in the Triton flash-decode kernel. Larger values improve long-context decode throughput. |
| `--enable-nccl-nvls` | `false` | Enable NCCL NVLS (NVLink SHARP) for prefill-heavy requests when hardware supports it. |
| `--disable-custom-all-reduce` | `false` | Fall back to standard NCCL allreduce instead of the custom kernel. Use when debugging communication issues. |
| `--delete-ckpt-after-loading` | `false` | Delete checkpoint files from disk after weights are loaded. Frees disk space on shared storage. |
| `--enable-memory-saver` | `false` | Allow `release_memory_occupation` / `resume_memory_occupation` API calls for dynamic memory management. |
| `--json-model-override-args` | `{}` | JSON string to override model config fields, e.g. `{"num_hidden_layers": 12, "num_attention_heads": 16}`. |
| `--schedule-conservativeness` | `1.0` | Higher = more conservative scheduling decisions. Raise if you see frequent request retractions in logs. |
| `--disable-outlines-disk-cache` | `false` | Disable grammar backend disk cache. Use when the cache causes crashes under high concurrency or filesystem issues. |
| `--enable-tokenizer-batch-encode` | `false` | Batch-encode multiple requests simultaneously. Do not use with image inputs, pre-tokenized `input_ids`, or `input_embeds`. |

---

### Deprecated arguments

Do not use these in new configs — they have been replaced:

| Deprecated | Replacement |
|---|---|
| `--enable-ep-moe` | `--ep-size <N>` (set to same value as `--tp-size`) |
| `--enable-deepep-moe` | `--moe-a2a-backend deepep` |
| `--enable-flashinfer-cutlass-moe` | `--moe-runner-backend flashinfer_cutlass` |
| `--enable-flashinfer-cutedsl-moe` | `--moe-runner-backend flashinfer_cutedsl` |
| `--enable-flashinfer-trtllm-moe` | `--moe-runner-backend flashinfer_trtllm` |
| `--enable-triton-kernel-moe` | `--moe-runner-backend triton_kernel` |
| `--enable-flashinfer-mxfp4-moe` | `--moe-runner-backend flashinfer_mxfp4` |
| `--prefill-round-robin-balance` | (removed) |

---

### Flag quick-reference

Boolean flags that have no argument value — include to enable, omit to disable:

```
--skip-server-warmup               Skip startup warmup (dev only)
--trust-remote-code                Allow custom HF Hub model code
--is-embedding                     Embedding mode
--enable-multimodal                Image/video/audio inputs
--enable-metrics                   Prometheus /metrics  ← needed for bench report
--log-requests                     Log all request I/O
--enable-deterministic-inference   Reproducible batch-invariant output
--enable-p2p-check                 Verify NVLink P2P  ← recommended multi-GPU
--disable-cuda-graph               No CUDA graphs  ← needed for many MoE models
--disable-radix-cache              No prefix caching
--enable-mixed-chunk               Co-batch prefill + decode
--enable-dp-attention              DP attention + TP FFN (DeepSeek/Qwen MoE)
--enable-lora                      LoRA adapter support
--enable-hierarchical-cache        CPU/storage KV cache overflow
--enable-torch-compile             torch.compile (experimental)
--allow-auto-truncate              Silently truncate overlong requests
--enable-custom-logit-processor    Client-provided logit processors (security risk)
--delete-ckpt-after-loading        Free disk space after weight load
--disable-overlap-schedule         Debug: disable CPU/GPU scheduling overlap
--disable-outlines-disk-cache      Debug: no grammar cache
--collect-tokens-histogram         Token count histograms in Prometheus
--enable-nan-detection             Debug NaN logprobs
--enable-memory-saver              Enable memory release/resume API
```

---

### Launch examples

#### Minimal single-GPU

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 --port 30000 \
  --enable-metrics
```

#### Two-GPU tensor parallel with FP4 MoE model

```bash
python -m sglang.launch_server \
  --model-path openai/gpt-oss-20b \
  --served-model-name gpt-oss-20b \
  --tp 2 \
  --mem-fraction-static 0.92 \
  --max-running-requests 32 \
  --max-queued-requests 64 \
  --attention-backend triton \
  --sampling-backend pytorch \
  --moe-runner-backend flashinfer_cutlass \
  --fp4-gemm-backend flashinfer_cutlass \
  --enable-mixed-chunk \
  --enable-metrics \
  --enable-p2p-check \
  --disable-cuda-graph
```

#### High-throughput long-context batch serving

```bash
python -m sglang.launch_server \
  --model-path mistralai/Mistral-7B-Instruct-v0.3 \
  --tp 1 \
  --mem-fraction-static 0.90 \
  --chunked-prefill-size 4096 \
  --max-prefill-tokens 32768 \
  --stream-interval 4 \
  --max-running-requests 64 \
  --max-queued-requests 256 \
  --schedule-policy lpm \
  --num-continuous-decode-steps 4 \
  --attention-backend flashinfer \
  --sampling-backend flashinfer \
  --enable-metrics
```

#### Interactive low-latency serving

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.2-3B-Instruct \
  --tp 1 \
  --mem-fraction-static 0.82 \
  --chunked-prefill-size 512 \
  --max-prefill-tokens 4096 \
  --stream-interval 1 \
  --max-running-requests 8 \
  --schedule-policy fcfs \
  --num-continuous-decode-steps 1 \
  --enable-metrics
```

#### LoRA multi-adapter serving with prefix cache

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --tp 1 \
  --schedule-policy lpm \
  --enable-lora \
  --lora-paths customer-a=/adapters/a customer-b=/adapters/b \
  --max-loras-per-batch 4 \
  --lora-backend csgmv \
  --enable-metrics
```

#### FP8 quantization with FP8 KV cache

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-70B-Instruct \
  --tp 4 \
  --quantization fp8 \
  --kv-cache-dtype fp8_e4m3 \
  --quantization-param-path /path/to/kv_scaling_factors.json \
  --fp8-gemm-backend cutlass \
  --mem-fraction-static 0.88 \
  --enable-metrics
```

#### EAGLE speculative decoding

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --tp 1 \
  --speculative-algorithm EAGLE \
  --speculative-draft-model-path lmzheng/sglang-EAGLE-llama3.1-Instruct-8B \
  --speculative-num-draft-tokens 5 \
  --enable-metrics
```

#### YAML config file

```yaml
# server_config.yaml
model-path: openai/gpt-oss-20b
served-model-name: gpt-oss-20b
host: 0.0.0.0
port: 30000
tensor-parallel-size: 2
mem-fraction-static: 0.92
max-running-requests: 32
attention-backend: triton
sampling-backend: pytorch
moe-runner-backend: flashinfer_cutlass
enable-mixed-chunk: true
enable-metrics: true
enable-p2p-check: true
disable-cuda-graph: true
log-level: info
```

```bash
python -m sglang.launch_server --config server_config.yaml
```

---

## sg_lang_bench_report.py — Benchmark summariser

Reads AIPerf artifact directories, extracts metrics, and prints a summary table. Optionally generates matplotlib plots and a Markdown report. Supports both **SGLang** (`sglang:*`) and **vLLM** (`vllm:*`) — backend auto-detected from the server metrics CSV.

### Directory convention

```
benchmarks/
└── <MODEL>_<YYYYMMDD>_<HHMMSS>/       ← --model-ts filter value
    └── <suite>_c<C>_req<R>_in<I>_out<O>/
        └── aiperf_artifacts/
            ├── profile_export_aiperf.json   ← required
            ├── profile_export_aiperf.csv
            ├── profile_export.jsonl         ← used by --per-request
            └── server_metrics_export.csv    ← SGLang/vLLM Prometheus export
```

Run directories must match `<suite>_c<C>_req<R>_in<I>_out<O>` where suite is `baseline`, `concurrency`, `longctx`, or `stress`.

### CLI arguments

| Argument | Default | Description |
|---|---|---|
| `--benchmarks-dir DIR` | `benchmarks` | Root directory with `<MODEL>_<TIMESTAMP>/` subdirectories. |
| `--model-ts STRING` | *(all)* | Filter to one model+timestamp directory, e.g. `gpt-oss-20b_20260322_080523`. |
| `--suite SUITE` | *(all)* | Filter to one suite: `baseline`, `concurrency`, `longctx`, `stress`. |
| `--limit N` | `50` | Max rows to print in the summary table. |
| `--sort KEYS` | `suite,c` | Comma-separated sort keys: `suite`, `c`, `in`, `out`, `req`, `ttft_p95`, `out_tps`, `rps`. |
| `--per-request` | off | Parse `profile_export.jsonl` for per-request p99/max tails and OSL mismatch rate. Slower. |
| `--osl-mismatch-threshold-pct N` | `5.0` | Count a request as mismatched if output length differs from requested by more than this %. |
| `--plot` | off | Generate matplotlib scatter plots (`pip install matplotlib` required). |
| `--by-suite` | off | Also generate per-suite plots under `<plot-out>/<suite>/` in addition to `all/`. |
| `--plot-out DIR` | auto | Plot output directory. Defaults to `<benchmarks-dir>/<model-ts>/_plots`. |
| `--report` | off | Write a `report.md` Markdown summary into the plot directory. |
| `--report-name NAME` | `report.md` | Filename for the Markdown report. |

### Output columns explained

| Column | Source | Unit | Direction |
|---|---|---|---|
| `run` | directory path | — | — |
| `suite` | run name | — | — |
| `c` | run name | requests | — |
| `req` | run name | total requests | — |
| `in` / `out` | run name | tokens | — |
| `backend` | server_metrics CSV | — | — |
| `rps` | AIPerf profile JSON | req/s | ↑ higher |
| `out_tps` | AIPerf profile JSON | tok/s | ↑ higher |
| `ttft_avg_ms` | AIPerf JSON or SGLang histogram | ms | ↓ lower |
| `ttft_p95_ms` | AIPerf JSON or SGLang histogram | ms | ↓ lower |
| `itl_avg_ms` | AIPerf JSON or SGLang histogram p50 | ms | ↓ lower |
| `itl_p95_ms` | AIPerf JSON or SGLang histogram | ms | ↓ lower |
| `req_lat_avg_ms` | AIPerf profile JSON | ms | ↓ lower |
| `req_lat_p95_ms` | AIPerf profile JSON | ms | ↓ lower |
| `kv_avg%` | SGLang `token_usage`×100 | % | ↓ lower = more headroom |
| `kv_p95%` | SGLang `token_usage` p95×100 | % | ↓ lower = more headroom |
| `running_avg` | SGLang `num_running_reqs` | requests | — |
| `waiting_avg` | SGLang `num_queue_reqs` | requests | ↓ lower |
| `prefill_p95_ms` | SGLang `per_stage_req_latency_seconds` prefill_waiting | ms | ↓ lower |
| `decode_p95_ms` | SGLang: proxy via ITL p95 | ms | ↓ lower |
| `e2e_p95_ms` | SGLang `e2e_request_latency_seconds` | ms | ↓ lower |
| `queue_p95_ms` | SGLang `queue_time_seconds` | ms | ↓ lower |
| `prompt_tps` | SGLang `prompt_tokens` counter rate | tok/s | ↑ higher |
| `gen_tps` | SGLang `generation_tokens` counter rate | tok/s | ↑ higher |
| `pcache_hit` | SGLang `cache_hit_rate` gauge | ratio 0–1 | ↑ higher |
| `pr_ttft_p99` | per-request JSONL (`--per-request`) | ms | ↓ lower |
| `pr_ttft_max` | per-request JSONL | ms | ↓ lower |
| `pr_req_p99` | per-request JSONL | ms | ↓ lower |
| `pr_req_max` | per-request JSONL | ms | ↓ lower |
| `pr_wait_p99` | per-request JSONL | ms | ↓ lower |
| `pr_wait_max` | per-request JSONL | ms | ↓ lower |
| `pr_osl_mismatch` | per-request JSONL | fraction 0–1 | ↓ lower |
| `little_L` | derived: `rps × (req_lat_avg_sec)` | — | diagnostic |
| `q_pressure` | derived: `waiting_avg / concurrency` | — | ↓ lower |

> `decode_p95_ms` (SGLang): SGLang has no per-request decode histogram. This column shows ITL p95 as a proxy.

### SGLang metric mapping

| Logical field | SGLang metric | vLLM metric |
|---|---|---|
| KV cache usage | `sglang:token_usage` (×100 → %) | `vllm:kv_cache_usage_perc` |
| Running requests | `sglang:num_running_reqs` | `vllm:num_requests_running` |
| Waiting requests | `sglang:num_queue_reqs` | `vllm:num_requests_waiting` |
| Cache hit ratio | `sglang:cache_hit_rate` gauge | `vllm:prefix_cache_hits / queries` |
| Prompt token rate | `sglang:prompt_tokens` counter rate | `vllm:prompt_tokens` counter rate |
| Generation token rate | `sglang:generation_tokens` counter rate | `vllm:generation_tokens` counter rate |
| E2E latency | `sglang:e2e_request_latency_seconds` | `vllm:e2e_request_latency_seconds` |
| TTFT | `sglang:time_to_first_token_seconds` | (from AIPerf JSON) |
| ITL | `sglang:inter_token_latency_seconds` | (from AIPerf JSON) |
| Prefill latency | `sglang:per_stage_req_latency_seconds` stage=`prefill_waiting` tp_rank=0 | `vllm:request_prefill_time_seconds` |
| Decode latency | *(proxy via ITL p95)* | `vllm:request_decode_time_seconds` |
| Queue latency | `sglang:queue_time_seconds` tp_rank=0 | `vllm:request_queue_time_seconds` |

### Examples

```bash
# Print summary for all runs
python3 sg_lang_bench_report.py --benchmarks-dir benchmarks

# Full plots + report for one model timestamp, split by suite
python3 sg_lang_bench_report.py \
  --benchmarks-dir benchmarks \
  --model-ts gpt-oss-20b_20260322_080523 \
  --plot --by-suite --report

# Long-context suite only, sorted by TTFT p95
python3 sg_lang_bench_report.py \
  --benchmarks-dir benchmarks \
  --model-ts gpt-oss-20b_20260322_080523 \
  --suite longctx --sort ttft_p95

# Include per-request p99/max tails and tighter OSL mismatch threshold
python3 sg_lang_bench_report.py \
  --benchmarks-dir benchmarks \
  --model-ts gpt-oss-20b_20260322_080523 \
  --per-request --osl-mismatch-threshold-pct 3.0 --plot

# Custom output directory and report filename
python3 sg_lang_bench_report.py \
  --benchmarks-dir /data/benchmarks \
  --plot --plot-out /reports/gpt-oss-20b \
  --report --report-name summary.md

# Compare all model timestamps side by side (omit --model-ts)
python3 sg_lang_bench_report.py \
  --benchmarks-dir benchmarks \
  --suite concurrency --sort suite,c \
  --plot --plot-out benchmarks/_plots_compare
```

---

## Benchmark suites

| Suite | Concurrency levels | Input tokens | Output tokens | Purpose |
|---|---|---|---|---|
| `baseline` | `1 2 4 6 16 32` | 256 | 128 | Short-context latency floor; establishes minimum TTFT and ITL |
| `concurrency` | `1 2 4 6 16 32` | 512 | 160 | Mid-context throughput sweep; finds saturation knee |
| `longctx` | `64 128` | 4000 | 2000 | Long-context stress; tests KV cache pressure and eviction |
| `stress` | `1 2 4 8 16 32` | 1024 | 512 | Sustained mixed load; checks stability under extended operation |

---

## Tuning guide

### Finding the right `MAX_RUNNING_REQUESTS`

1. Start conservative (e.g. 16 for a 20B model on 2×H100s).
2. Run the `concurrency` suite.
3. Check `kv_p95%` — if it stays below ~85%, raise `MAX_RUNNING_REQUESTS`.
4. Check `waiting_avg` — when it grows linearly with concurrency while throughput plateaus, you've hit saturation. Back off one step.

### Choosing `SCHEDULE_POLICY`

- **`lpm`** — best when all requests share a common system prompt or prefix. `pcache_hit` will jump from ~0 to 0.5+ and TTFT drops significantly.
- **`fcfs`** — safe default for mixed/unpredictable prompt distributions.
- **`priority`** — use with `--enable-priority-scheduling` to guarantee latency SLOs for high-priority request classes.
- **`dfs-weight`** — better for long-context batch jobs where KV reuse depth matters more than arrival order.

### Mixed chunk (`ENABLE_MIXED_CHUNK`)

Enable when decode throughput drops noticeably at higher concurrency. Decode tokens piggyback on prefill batches, keeping the GPU busier. Slight increase in per-token prefill latency is the tradeoff.

### Attention backend selection

- Start with `triton` (compatible everywhere).
- Switch to `flashinfer` on A100/H100 — typically 10–25% faster for both prefill and decode.
- Use `fa3`/`fa4` for very long contexts (>32k tokens) on Hopper.
- For DeepSeek-V2/V3/R1 MLA models use `flashmla` or `cutlass_mla`.
- For maximum throughput on H100 with TRT-LLM installed, use `trtllm_mha`.

### Speculative decoding

Add `--speculative-algorithm EAGLE` with a draft model to improve throughput by 1.5–3× at small batch sizes. Most effective when output is much longer than input and batch size is small (< 8). For workloads without a suitable draft model, try `--speculative-algorithm NGRAM` — no extra model needed.

### When to disable CUDA graphs

Set `DISABLE_CUDA_GRAPH=1` when:
- Running an MoE model with dynamic expert routing shapes.
- Hitting unexplained CUDA errors at decode time.
- You need to reduce startup VRAM during development.
- Running on a multi-node job that deadlocks at startup.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ERROR: HF_TOKEN is not set` | Token not exported | `export HF_TOKEN=hf_xxxx` before running |
| Container starts but `/health` returns 503 | Model still loading | Watch `docker logs -f sglang-gptoss-server`; wait for "Server is ready" |
| `kv_p95%` shows `N/A` in report | `ENABLE_METRICS=0` | Set `ENABLE_METRICS=1` and restart server |
| All server metric columns are `N/A` | Metrics not scraped | Verify `/metrics` is reachable; check AIPerf scrape endpoint config |
| OOM in container logs | `MEM_FRACTION_STATIC` too high | Lower by 0.03–0.05 |
| Graphs empty / "No numeric data" | Only one concurrency level | Run ≥ 2 concurrency levels; plots need multiple data points |
| `decode_p95_vs_concurrency.png` flat | SGLang proxy limitation | Expected — shows ITL p95 as a proxy for decode; not a bug |
| High `waiting_avg` at all concurrencies | `MAX_RUNNING_REQUESTS` too low | Increase it; keep `kv_p95%` under 90% |
| `pr_ttft_p99` always `N/A` | `--per-request` not passed | Add `--per-request` to the report command |
| "peer access is not supported" error | P2P NVLink not verified | Add `ENABLE_P2P_CHECK=1` to env and restart |
| Deadlock on multi-node startup | CUDA graph / NCCL conflict | Add `--disable-cuda-graph` to both nodes |
| High TTFT spikes at low concurrency | Missing warmup | Remove `SKIP_SERVER_WARMUP=1`; restart with warmup enabled |
| Requests retracted frequently in logs | Over-aggressive scheduling | Increase `--schedule-conservativeness` to e.g. `1.5` |
| Grammar-constrained output crashes | Disk cache corruption | Add `--disable-outlines-disk-cache` |
| Slow first request on LoRA adapter | Adapter not pre-loaded | List adapters in `--lora-paths`; mark hot ones with `"pinned": true` |
| CUDA graph capture fails | MoE dynamic shapes | Set `DISABLE_CUDA_GRAPH=1`; MoE models often need this |
| Very high prefill latency | Chunked prefill too large | Reduce `CHUNKED_PREFILL_SIZE` to 512–1024 |
| Throughput doesn't improve past a concurrency level | Saturation reached | Lower `MAX_RUNNING_REQUESTS` by 20%; check `kv_p95%` for headroom |
