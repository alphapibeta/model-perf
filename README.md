## GPT-OSS vLLM Server & Benchmarks

This folder contains a small set of scripts and configs to:

- **Start a vLLM OpenAI-compatible server** for `openai/gpt-oss-20b` in Docker.
- **Configure the server** via `.env` files (GPU, context length, caching, etc.).
- **Run latency/throughput benchmarks** using `aiperf`.

Everything is driven by environment files and two Bash entrypoints: `run_gptoss.sh` and `benchmark.sh`.

---

### Files

- **`gptoss.env`**: Env file for running the `openai/gpt-oss-20b` model with a **long context configuration** (e.g. large `MAX_MODEL_LEN`). Good for experimentation and long-context scenarios.
- **`responsive-gptoss.env`**: Env file tuned for **more responsive benchmarking** (shorter `MAX_MODEL_LEN`, explicit `BENCHMARK_HOST=127.0.0.1`, and matching benchmark parameters).
- **`run_gptoss.sh`**: Starts the vLLM container using Docker, based on a given env file.
- **`benchmark.sh`**: Runs `aiperf` benchmark suites (baseline, concurrency, long context, stress) against a running server.
- **`benchmarks/`**: Auto-created directory where benchmark runs are stored (one subfolder per model/timestamp).

You can maintain multiple `.env` files (e.g. different GPUs, models, or context sizes) and point both scripts at the one you want to use.

---

### Prerequisites

- **Linux with NVIDIA GPUs** and drivers installed.
- **Docker** with the NVIDIA runtime available (so `--gpus` and `--runtime=nvidia` work).
- **Internet access** to pull the model and Docker image (by default `vllm/vllm-openai:latest`).
- **Hugging Face token** with access to `openai/gpt-oss-20b`.
- For benchmarking:
  - `aiperf` in your `PATH`.
  - `curl` in your `PATH`.

---

### 1. Configure your environment

Pick one of the provided env files as a starting point:

- **Long-context / general use**: `gptoss.env`
- **Benchmark-focused / more responsive**: `responsive-gptoss.env`

Recommended workflow:

```bash
cd /root/exp/nims-model-loader/model-builder

# Make a copy so you can safely edit
cp responsive-gptoss.env my-gptoss.env
```

Then open `my-gptoss.env` and update at least:

- **`CUDA_DEVICES`**: GPU indices, e.g. `"0,1"`.
- **`TENSOR_PARALLEL_SIZE`**: Should match the number of GPUs you actually use.
- **`HF_CACHE_HOST_PATH`**: Path on the host where you want HF cache persisted.
- **`PORT`**: Host port to expose the OpenAI-compatible endpoint on.
- **`HF_TOKEN`**:
  - Prefer **not** to hard-code secrets in the file.
  - Instead, export it in your shell:  
    `export HF_TOKEN=...`

You can also adjust:

- **`MAX_MODEL_LEN`**, `MAX_NUM_SEQS`, `MAX_NUM_BATCHED_TOKENS`, `GPU_MEMORY_UTILIZATION` for memory/throughput tradeoffs.
- **Benchmark parameters** (request counts, token sizes, concurrency levels) at the bottom of the env file.

---

### 2. Start the vLLM server

Use `run_gptoss.sh` to start the container:

```bash
cd /root/exp/nims-model-loader/model-builder

# Using your own env file
bash run_gptoss.sh my-gptoss.env

# Or use the default responsive env if you don't pass anything
bash run_gptoss.sh          # defaults to responsive-gptoss.env
```

What `run_gptoss.sh` does:

- Loads and exports variables from the env file.
- Verifies that `HF_TOKEN` is set.
- Removes any old container with the same `CONTAINER_NAME`.
- Starts `vllm/vllm-openai:latest` with:
  - GPU selection (`CUDA_DEVICES`).
  - Shared memory and NCCL debug envs.
  - HF cache mounted from `HF_CACHE_HOST_PATH` to `HF_HOME` inside the container.
  - vLLM arguments based on your env (`MODEL_NAME`, `TENSOR_PARALLEL_SIZE`, `MAX_MODEL_LEN`, etc.).
- Prints:
  - The final `docker run` command.
  - `docker ps` output for the container.
  - A reminder of the OpenAI endpoint, e.g. `http://localhost:${PORT}/v1`.

Once the container is running, you can:

- Tail logs: `docker logs -f gptoss-server`
- Hit the health/model list endpoint: `curl http://localhost:${PORT}/v1/models`

---

### 3. Run benchmarks with `aiperf`

After the server is up and serving your model, use `benchmark.sh` to run synthetic benchmarks.

Basic usage:

```bash
cd /root/exp/nims-model-loader/model-builder

# Syntax:
#   benchmark.sh <env_file> [baseline|concurrency|longctx|stress|all]

# Example: run all suites using the responsive env
bash benchmark.sh responsive-gptoss.env all

# Example: run only concurrency tests
bash benchmark.sh responsive-gptoss.env concurrency
```

What `benchmark.sh` does:

- Sources the env file and derives:
  - `MODEL` from `SERVED_MODEL_NAME`.
  - `HOST_FOR_URL` (defaults to `BENCHMARK_HOST` or `localhost`).
  - `URL` (defaults to `http://${HOST_FOR_URL}:${PORT}`).
  - `TOKENIZER`, `ENDPOINT_TYPE`, `STREAMING`.
  - Request counts, concurrency levels, and input/output token sizes.
- Verifies dependencies:
  - `aiperf` exists.
  - `curl` exists.
- Verifies the server:
  - Calls `${URL}/v1/models`.
  - Saves the response as `models.json` in the artifact directory.
  - Fails with a helpful message if the server is unreachable.
- Runs one or more suites (depending on the mode):
  - **`baseline`**: Single-user baseline.
  - **`concurrency`**: Multiple concurrency levels (`CONCURRENCY_LEVELS`).
  - **`longctx`**: Long-context runs at lower concurrency.
  - **`stress`**: High-load runs (`STRESS_CONCURRENCY_LEVELS`).
- For each run:
  - Calls `aiperf profile` with synthetic token settings.
  - Saves `command.txt` (the exact `aiperf` command).
  - Saves `run.log` with start/end timestamps and full output.

Artifacts are stored under:

```text
benchmarks/<MODEL>_<TIMESTAMP>/
  baseline_c1_req10_in256_out128/        # example; numbers depend on your env
  concurrency_c4_req20_in512_out160/
  longctx_c2_req8_in4000_out2000/
  stress_c6_req30_in1200_out180/
  ...
```

The directory names are built as:

- **`<suite>_c<concurrency>_req<request_count>_in<input_tokens>_out<output_tokens>`**
  - `suite`: one of `baseline`, `concurrency`, `longctx`, `stress`.
  - `concurrency`: number of in-flight requests used by `aiperf`.
  - `request_count`: total requests for that run.
  - `in` / `out`: mean synthetic input/output tokens per request.

---

### 3.1 Summarize and plot runs (Python helper)

This repo includes `bench_report.py` to quickly compare runs across `benchmarks/` and generate a few standard plots.

Examples:

```bash
cd /root/exp/nims-model-loader/model-builder

# Print a summary table of recent runs (default limit 50)
python3 bench_report.py

# Filter to a specific timestamp directory
python3 bench_report.py --model-ts gpt-oss-20b_20260318_054504

# Filter to one suite and sort by concurrency
python3 bench_report.py --suite concurrency --sort suite,c

# Generate plots (writes PNGs under benchmarks/_plots by default)
python3 bench_report.py --suite concurrency --plot

# Generate plots for a specific benchmark timestamp
# (writes under benchmarks/<model-ts>/_plots)
python3 bench_report.py --model-ts gpt-oss-20b_20260318_064711 --suite concurrency --plot
```

Plotting requires `matplotlib`:

```bash
python3 -m pip install matplotlib
```

What it shows (per run):

- Core AIPerf metrics from `profile_export_aiperf.json`:
  - TTFT (avg/p95), inter-token latency (avg/p95), request latency (avg/p95)
  - request throughput (rps), output token throughput (tokens/sec)
- Selected vLLM server metrics (best-effort) from `server_metrics_export.csv`:
  - **KV & queueing**
    - `vllm:kv_cache_usage_perc` (avg/p95)
    - `vllm:num_requests_running` / `vllm:num_requests_waiting` (avg)
  - **Prefix cache effectiveness**
    - `vllm:prefix_cache_hits` and `vllm:prefix_cache_queries` → reports **hit ratio** \(hits / queries\)
  - **Token rates**
    - `vllm:prompt_tokens` rate → prompt tokens/sec
    - `vllm:generation_tokens` rate → generation tokens/sec
  - **Phase timing (server-side, p95)**
    - `vllm:request_prefill_time_seconds` (p95)
    - `vllm:request_decode_time_seconds` (p95)
    - `vllm:e2e_request_latency_seconds` (p95)

At the end it prints a high-level summary and reminds you to focus on:

- Time to First Token
- Inter-Token Latency
- Request Latency
- Output Token Throughput Per User
- Request Throughput

---

### 4. Using the endpoint

The container exposes an **OpenAI-compatible API** on:

- **Base URL**: `http://localhost:${PORT}/v1`
- **Model ID**: `SERVED_MODEL_NAME` from your env (default `gpt-oss-20b`).

Example `curl` snippet (chat completion-style):

```bash
curl "http://localhost:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $HF_TOKEN" \
  -d '{
    "model": "gpt-oss-20b",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "stream": false
  }'
```

Adjust the `model` field to match your `SERVED_MODEL_NAME` if you change it.

---

### 5. Tuning guide: important knobs

This section explains the main knobs you care about and how they affect **memory**, **latency**, and **throughput**, especially around **KV cache** and **concurrency**.

---

#### Model / tokens / KV cache

- **`MAX_MODEL_LEN`**
  - **What it is**: Maximum total tokens per request (input + output) that vLLM will reserve KV cache for.
  - **Effect**:
    - Higher = can handle longer prompts or outputs.
    - But KV cache memory scales roughly with \( \text{num_layers} \times \text{hidden_dim} \times \text{MAX\_MODEL\_LEN} \times \text{batch\_size} \).
    - So **doubling `MAX_MODEL_LEN` roughly doubles KV cache usage** per sequence.
  - **Guidance**:
    - For **interactive / responsive** use, keep this as **low as your use case allows** (e.g. 16k–32k).
    - For **long-context experiments**, raise it (e.g. 64k–120k), but expect:
      - Fewer concurrent sequences.
      - More OOM risk if `MAX_NUM_SEQS` / `MAX_NUM_BATCHED_TOKENS` stay high.

- **`MAX_NUM_SEQS`**
  - **What it is**: Max number of sequences vLLM will decode in parallel.
  - **Effect**:
    - Higher = more concurrent requests sharing the GPU at the same time.
    - But each active sequence consumes KV cache; more sequences = more memory.
  - **Guidance**:
    - If you are **hitting OOMs**, try **reducing** `MAX_NUM_SEQS`.
    - If you want **higher throughput** and have headroom, you can **increase** it a bit and re-benchmark.

- **`MAX_NUM_BATCHED_TOKENS`**
  - **What it is**: Upper bound on total tokens (sum over sequences) per decode step.
  - **Effect**:
    - Higher = more work per GPU step, which can **increase throughput** (better GPU utilization).
    - But larger batches can increase **latency** for individual requests and memory usage.
  - **Guidance**:
    - For **low-latency / interactive** scenarios, keep this moderate.
    - For **throughput benchmarking**, increase it (within memory limits) to see if throughput improves.

- **`GPU_MEMORY_UTILIZATION`**
  - **What it is**: Fraction of visible GPU memory vLLM will try to use for KV cache and weights.
  - **Effect**:
    - Higher (e.g. `0.90`) = more memory available for KV cache → potentially higher `MAX_NUM_SEQS`/`MAX_NUM_BATCHED_TOKENS`.
    - But too high can make the system more fragile to other processes.
  - **Guidance**:
    - 0.85–0.90 is a common range when the GPU is mostly dedicated to vLLM.

---

#### Concurrency & GPU parallelism

- **`CUDA_DEVICES`**
  - **What it is**: Which GPU IDs to expose to the container, e.g. `"0,1"`.
  - **Effect**:
    - Controls how many GPUs vLLM can see and therefore how many GPUs can participate in tensor parallelism.

- **`TENSOR_PARALLEL_SIZE`**
  - **What it is**: Number of GPUs over which the model weights are sharded.
  - **Effect**:
    - Must not exceed the number of visible GPUs (length of `CUDA_DEVICES`).
    - Higher TP size:
      - Lets you fit larger models or longer contexts by spreading work/memory.
      - Adds some communication overhead (NCCL), especially at small batch sizes.
  - **Guidance**:
    - If you set `CUDA_DEVICES="0,1"`, then `TENSOR_PARALLEL_SIZE=2` is typical.
    - For single-GPU runs, set `CUDA_DEVICES="0"` and `TENSOR_PARALLEL_SIZE=1`.

---

#### Benchmark knobs (aiperf)

These live in the env files and are consumed by `benchmark.sh`.

- **Request counts**
  - `REQUEST_COUNT_BASELINE`, `REQUEST_COUNT_CONCURRENCY`, `REQUEST_COUNT_LONGCTX`, `REQUEST_COUNT_STRESS`
  - **What they control**: How many **total requests** each suite sends.
  - **Effect**:
    - Higher counts = more stable statistics but longer benchmark runs.
  - **Important constraint**:
    - `aiperf` requires **`request_count ≥ concurrency`**.
    - `benchmark.sh` will automatically **bump** the per-run request count if needed (and the run directory name will reflect the effective `req...`).

- **Concurrency levels**
  - `CONCURRENCY_LEVELS`, `STRESS_CONCURRENCY_LEVELS`
  - **What they control**: How many **in-flight requests** `aiperf` will maintain.
  - **Effect**:
    - Higher concurrency exposes **throughput limits** and queueing behavior.
    - Too high can make latency explode or cause timeouts if the server is saturated.
  - **Guidance**:
    - Use lower concurrency levels for baselines (`1–4`).
    - Use higher levels for stress to see how the system behaves near saturation.

- **Token sizes**
  - `INPUT_TOKENS_BASELINE`, `OUTPUT_TOKENS_BASELINE`
  - `INPUT_TOKENS_CONCURRENCY`, `OUTPUT_TOKENS_CONCURRENCY`
  - `INPUT_TOKENS_LONGCTX`, `OUTPUT_TOKENS_LONGCTX`
  - `INPUT_TOKENS_STRESS`, `OUTPUT_TOKENS_STRESS`
  - **What they control**: Mean input and output token counts per request for synthetic prompts.
  - **Effect**:
    - Larger input tokens stress **KV cache and prefill throughput**.
    - Larger output tokens stress **decode throughput** and TTFB/ITL.
  - **Guidance**:
    - Match these to your **realistic workloads** (short chat vs. long documents, etc.).
    - For long-context tests, ensure `INPUT_TOKENS_LONGCTX + OUTPUT_TOKENS_LONGCTX ≤ MAX_MODEL_LEN`.

---

#### How the LLM setup and KV cache work (intuitively)

- vLLM keeps a **KV cache** of attention keys/values on the GPU for each active sequence.
- When you:
  - Increase **`MAX_MODEL_LEN`**, each sequence has a potentially larger window, so its KV cache footprint grows.
  - Increase **`MAX_NUM_SEQS`** or effective concurrency, you maintain **more sequences simultaneously**, multiplying total KV cache usage.
  - Increase **token sizes** in benchmarks, you push more tokens into this cache and work the GPU harder per request.
- The goal of tuning is to balance:
  - **Latency** (TTFB, inter-token latency, request latency).
  - **Throughput** (tokens/sec, requests/sec).
  - **Memory** (no OOMs, minimal swapping/failures).

In practice:

- For **interactive / product-like** workloads:
  - Keep `MAX_MODEL_LEN` aligned to real needs (e.g. 16k–32k).
  - Moderate `MAX_NUM_SEQS` and `MAX_NUM_BATCHED_TOKENS`.
  - Benchmark with input/output tokens similar to production.
- For **max-throughput experiments**:
  - Increase `MAX_NUM_SEQS` and `MAX_NUM_BATCHED_TOKENS` until:
    - You hit OOM, or
    - Latency becomes unacceptably high.
  - Use higher concurrency levels in `benchmark.sh` and observe where throughput plateaus.

---

### 6. Tips & customization

- **Secrets**: Prefer exporting `HF_TOKEN` in your shell (or using a secret manager) instead of committing it to `.env` files.
- **Multiple configs**: Create separate env files per GPU layout or model variant and pass the one you want into `run_gptoss.sh` / `benchmark.sh`.
- **Benchmark host**: If you hit issues with `localhost` vs `0.0.0.0` in some environments, use the pattern from `responsive-gptoss.env` and set `BENCHMARK_HOST=127.0.0.1`.
- **Model changes**: To try another model supported by the vLLM image, update `MODEL_NAME`, `SERVED_MODEL_NAME`, and `TOKENIZER` consistently in your env file.

This README should give you a clear path from zero to:

1. Running a vLLM server for `openai/gpt-oss-20b`.
2. Hitting it via an OpenAI-compatible API.
3. Benchmarking it with `aiperf` and analyzing latency/throughput.

