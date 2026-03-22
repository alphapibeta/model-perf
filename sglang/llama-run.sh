#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${1:-$SCRIPT_DIR/sglang-gpt.env}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: env file not found: $ENV_FILE" >&2
  exit 1
fi

# Prefer token from shell over env file
HF_TOKEN_FROM_SHELL="${HF_TOKEN:-}"

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

HF_TOKEN_VALUE="${HF_TOKEN_FROM_SHELL:-${HF_TOKEN:-}}"
if [[ -z "$HF_TOKEN_VALUE" ]]; then
  echo "ERROR: HF_TOKEN is not set." >&2
  echo "Run: export HF_TOKEN=hf_xxxxx" >&2
  exit 1
fi

# Defaults
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
CONTAINER_NAME="${CONTAINER_NAME:-sglang-gptoss-server}"
IMAGE_NAME="${IMAGE_NAME:-lmsysorg/sglang:latest}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH must be set}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL_PATH}"

CUDA_DEVICES="${CUDA_DEVICES:-0}"
TP_SIZE="${TP_SIZE:-1}"

MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-2048}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-16384}"
STREAM_INTERVAL="${STREAM_INTERVAL:-2}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-16}"
MAX_QUEUED_REQUESTS="${MAX_QUEUED_REQUESTS:-128}"
SCHEDULE_POLICY="${SCHEDULE_POLICY:-fcfs}"
NUM_CONTINUOUS_DECODE_STEPS="${NUM_CONTINUOUS_DECODE_STEPS:-1}"

ATTENTION_BACKEND="${ATTENTION_BACKEND:-triton}"
SAMPLING_BACKEND="${SAMPLING_BACKEND:-pytorch}"
MOE_RUNNER_BACKEND="${MOE_RUNNER_BACKEND:-}"

HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
HF_CACHE_HOST_PATH="${HF_CACHE_HOST_PATH:-$HOME/.cache/huggingface}"

SHM_SIZE="${SHM_SIZE:-16g}"
NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}"
NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,ENV}"
LOG_LEVEL="${LOG_LEVEL:-info}"

ENABLE_MIXED_CHUNK="${ENABLE_MIXED_CHUNK:-0}"
ENABLE_METRICS="${ENABLE_METRICS:-0}"
ENABLE_P2P_CHECK="${ENABLE_P2P_CHECK:-0}"
DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-0}"
SKIP_SERVER_WARMUP="${SKIP_SERVER_WARMUP:-0}"

FP4_GEMM_BACKEND="${FP4_GEMM_BACKEND:-flashinfer_cutlass}"
mkdir -p "$HF_CACHE_HOST_PATH"

echo "[INFO] Removing old container..."
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true







CMD=(
  docker run -d
  --name "$CONTAINER_NAME"
  --gpus all
  --runtime=nvidia
  --network host
  --ipc=host
  --shm-size "$SHM_SIZE"
  -e "HF_TOKEN=${HF_TOKEN_VALUE}"
  -e "HF_HOME=${HF_HOME}"
  -e "NVIDIA_VISIBLE_DEVICES=${CUDA_DEVICES}"
  -e "NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES}"
  -e "NCCL_DEBUG=${NCCL_DEBUG}"
  -e "NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS}"
  -v "${HF_CACHE_HOST_PATH}:${HF_HOME}"
  "$IMAGE_NAME"
  python3 -m sglang.launch_server
  --model-path "$MODEL_PATH"
  --host "$HOST"
  --port "$PORT"
  --served-model-name "$SERVED_MODEL_NAME"
  --tensor-parallel-size "$TP_SIZE"
  --mem-fraction-static "$MEM_FRACTION_STATIC"
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
  --max-prefill-tokens "$MAX_PREFILL_TOKENS"
  --stream-interval "$STREAM_INTERVAL"
  --max-running-requests "$MAX_RUNNING_REQUESTS"
  --max-queued-requests "$MAX_QUEUED_REQUESTS"
  --schedule-policy "$SCHEDULE_POLICY"
  --attention-backend "$ATTENTION_BACKEND"
  --sampling-backend "$SAMPLING_BACKEND"
  --num-continuous-decode-steps "$NUM_CONTINUOUS_DECODE_STEPS"
)

if [[ -n "${MOE_RUNNER_BACKEND}" ]]; then
  CMD+=(--moe-runner-backend "$MOE_RUNNER_BACKEND")
fi

if [[ -n "${FP4_GEMM_BACKEND}" ]]; then
  CMD+=(--fp4-gemm-backend "$FP4_GEMM_BACKEND")
fi


if [[ "${ENABLE_MIXED_CHUNK}" == "1" ]]; then
  CMD+=(--enable-mixed-chunk)
fi

if [[ "${ENABLE_METRICS}" == "1" ]]; then
  CMD+=(--enable-metrics)
fi

if [[ "${ENABLE_P2P_CHECK}" == "1" ]]; then
  CMD+=(--enable-p2p-check)
fi

if [[ "${DISABLE_CUDA_GRAPH}" == "1" ]]; then
  CMD+=(--disable-cuda-graph)
fi

if [[ "${SKIP_SERVER_WARMUP}" == "1" ]]; then
  CMD+=(--skip-server-warmup)
fi

if [[ -n "${LOG_LEVEL}" ]]; then
  CMD+=(--log-level "$LOG_LEVEL")
fi

echo "[DEBUG] Final argv:"
for i in "${!CMD[@]}"; do
  printf 'CMD[%d]=<%s>\n' "$i" "${CMD[$i]}"
done

echo
echo "[INFO] Launching container..."
"${CMD[@]}"

echo
echo "[INFO] Running container:"
docker ps --filter "name=${CONTAINER_NAME}"

echo
echo "[INFO] Logs:"
echo "docker logs -f ${CONTAINER_NAME}"

echo
echo "[INFO] Health:"
echo "http://localhost:${PORT}/health"

echo
echo "[INFO] API:"
echo "http://localhost:${PORT}/v1"