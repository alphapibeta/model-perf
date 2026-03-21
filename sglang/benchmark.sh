#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <env_file> [baseline|concurrency|longctx|stress|all]" >&2
  exit 1
fi

ENV_FILE="$1"
MODE="${2:-all}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: env file not found: $ENV_FILE" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

MODEL="${SERVED_MODEL_NAME:-gpt-oss-20b}"
HOST_FOR_URL="${BENCHMARK_HOST:-localhost}"
PORT="${PORT:-9000}"
URL="${URL:-http://${HOST_FOR_URL}:${PORT}}"
TOKENIZER="${TOKENIZER:-openai/gpt-oss-20b}"
ENDPOINT_TYPE="${ENDPOINT_TYPE:-chat}"
STREAMING="${STREAMING:-true}"
STREAMING_BASELINE="${STREAMING_BASELINE:-$STREAMING}"
STREAMING_CONCURRENCY="${STREAMING_CONCURRENCY:-$STREAMING}"
STREAMING_LONGCTX="${STREAMING_LONGCTX:-$STREAMING}"
STREAMING_STRESS="${STREAMING_STRESS:-$STREAMING}"

REQUEST_COUNT_BASELINE="${REQUEST_COUNT_BASELINE:-10}"
REQUEST_COUNT_CONCURRENCY="${REQUEST_COUNT_CONCURRENCY:-20}"
REQUEST_COUNT_LONGCTX="${REQUEST_COUNT_LONGCTX:-8}"
REQUEST_COUNT_STRESS="${REQUEST_COUNT_STRESS:-30}"

CONCURRENCY_LEVELS=(${CONCURRENCY_LEVELS:-1 2 4})
STRESS_CONCURRENCY_LEVELS=(${STRESS_CONCURRENCY_LEVELS:-2 4 6})
LONGCTX_CONCURRENCY_LEVELS=(${LONGCTX_CONCURRENCY_LEVELS:-1 2})

INPUT_TOKENS_BASELINE="${INPUT_TOKENS_BASELINE:-256}"
OUTPUT_TOKENS_BASELINE="${OUTPUT_TOKENS_BASELINE:-128}"

INPUT_TOKENS_CONCURRENCY="${INPUT_TOKENS_CONCURRENCY:-512}"
OUTPUT_TOKENS_CONCURRENCY="${OUTPUT_TOKENS_CONCURRENCY:-160}"

INPUT_TOKENS_LONGCTX="${INPUT_TOKENS_LONGCTX:-4000}"
OUTPUT_TOKENS_LONGCTX="${OUTPUT_TOKENS_LONGCTX:-200}"

INPUT_TOKENS_STRESS="${INPUT_TOKENS_STRESS:-1200}"
OUTPUT_TOKENS_STRESS="${OUTPUT_TOKENS_STRESS:-180}"

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
ARTIFACT_DIR="${ARTIFACT_DIR:-$SCRIPT_DIR/benchmarks/${MODEL}_${TIMESTAMP}}"
mkdir -p "$ARTIFACT_DIR"

log() {
  echo
  echo "================================================================"
  echo "$1"
  echo "================================================================"
}

check_deps() {
  command -v aiperf >/dev/null 2>&1 || { echo "ERROR: aiperf not found in PATH"; exit 1; }
  command -v curl >/dev/null 2>&1 || { echo "ERROR: curl not found in PATH"; exit 1; }
}

check_server() {
  log "Checking already-running model server"

  local models_json
  if ! models_json="$(curl -fsS "${URL}/v1/models")"; then
    echo "ERROR: Could not reach ${URL}/v1/models" >&2
    echo "Start the server first with:" >&2
    echo "  bash run_gptoss.sh ${ENV_FILE}" >&2
    exit 1
  fi

  echo "$models_json" > "${ARTIFACT_DIR}/models.json"

  if echo "$models_json" | grep -q "\"id\":\"${MODEL}\""; then
    echo "Server reachable and model '${MODEL}' is visible at ${URL}"
  else
    echo "WARNING: Server is reachable at ${URL}, but model '${MODEL}' was not found."
    echo "Response saved to ${ARTIFACT_DIR}/models.json"
  fi
}

run_aiperf() {
  local suite_name="$1"
  local concurrency="$2"
  local request_count="$3"
  local input_tokens="$4"
  local output_tokens="$5"
  local streaming_override="${6:-}"

  # AIPerf requires: concurrency <= request_count.
  # If the env requests an invalid combo (e.g., c32 with req20), bump request_count
  # so a sweep doesn't fail mid-run. The run directory name reflects the effective request count.
  local effective_request_count="$request_count"
  if (( concurrency > request_count )); then
    echo "WARNING: concurrency (${concurrency}) > request_count (${request_count}); bumping request_count to ${concurrency} for this run."
    effective_request_count="$concurrency"
  fi

  # More descriptive run name so the directory encodes key knobs
  local name="${suite_name}_c${concurrency}_req${effective_request_count}_in${input_tokens}_out${output_tokens}"
  local run_dir="${ARTIFACT_DIR}/${name}"
  mkdir -p "$run_dir"

  log "Running benchmark: ${name}"
  echo "Env file : ${ENV_FILE}"
  echo "Model    : ${MODEL}"
  echo "URL      : ${URL}"
  echo "Run dir  : ${run_dir}"

  local cmd=(
    aiperf profile
    --model "${MODEL}"
    --endpoint-type "${ENDPOINT_TYPE}"
    --tokenizer "${TOKENIZER}"
    --url "${URL}"
    --concurrency "${concurrency}"
    --request-count "${effective_request_count}"
    --synthetic-input-tokens-mean "${input_tokens}"
    --output-tokens-mean "${output_tokens}"
    --extra-inputs "min_tokens:${output_tokens}"
    --extra-inputs "ignore_eos:true"
    --artifact-dir "${run_dir}/aiperf_artifacts"
  )

  local use_streaming="${STREAMING}"
  if [[ -n "${streaming_override}" ]]; then
    use_streaming="${streaming_override}"
  fi

  if [[ "${use_streaming}" == "true" ]]; then
    cmd+=(--streaming)
  fi

  printf '%q ' "${cmd[@]}" > "${run_dir}/command.txt"
  echo >> "${run_dir}/command.txt"

  {
    echo "Started at: $(date)"
    echo "Env file: ${ENV_FILE}"
    echo "Command:"
    printf '%q ' "${cmd[@]}"
    echo
    echo
    "${cmd[@]}"
    echo
    echo "Finished at: $(date)"
  } | tee "${run_dir}/run.log"
}

baseline_suite() {
  run_aiperf "baseline" "1" "${REQUEST_COUNT_BASELINE}" "${INPUT_TOKENS_BASELINE}" "${OUTPUT_TOKENS_BASELINE}" "${STREAMING_BASELINE}"
}

concurrency_suite() {
  for c in "${CONCURRENCY_LEVELS[@]}"; do
    run_aiperf "concurrency" "${c}" "${REQUEST_COUNT_CONCURRENCY}" "${INPUT_TOKENS_CONCURRENCY}" "${OUTPUT_TOKENS_CONCURRENCY}" "${STREAMING_CONCURRENCY}"
  done
}

long_context_suite() {
  for c in "${LONGCTX_CONCURRENCY_LEVELS[@]}"; do
    run_aiperf "longctx" "${c}" "${REQUEST_COUNT_LONGCTX}" "${INPUT_TOKENS_LONGCTX}" "${OUTPUT_TOKENS_LONGCTX}" "${STREAMING_LONGCTX}"
  done
}

stress_suite() {
  for c in "${STRESS_CONCURRENCY_LEVELS[@]}"; do
    run_aiperf "stress" "${c}" "${REQUEST_COUNT_STRESS}" "${INPUT_TOKENS_STRESS}" "${OUTPUT_TOKENS_STRESS}" "${STREAMING_STRESS}"
  done
}

print_summary() {
  log "Benchmark summary"
  cat <<EOF
Env file used:
  ${ENV_FILE}

Server URL:
  ${URL}

Artifacts saved under:
  ${ARTIFACT_DIR}

Focus on:
  - Time to First Token
  - Inter Token Latency
  - Request Latency
  - Output Token Throughput Per User
  - Request Throughput
EOF
}

main() {
  check_deps
  check_server

  case "${MODE}" in
    baseline) baseline_suite ;;
    concurrency) concurrency_suite ;;
    longctx) long_context_suite ;;
    stress) stress_suite ;;
    all)
      baseline_suite
      concurrency_suite
      long_context_suite
      stress_suite
      ;;
    *)
      echo "Usage: $0 <env_file> [baseline|concurrency|longctx|stress|all]" >&2
      exit 1
      ;;
  esac

  print_summary
}

main "$@"