#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${1:-$SCRIPT_DIR/sglang-llama.env}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: env file not found: $ENV_FILE" >&2
  exit 1
fi

set -a
source "$ENV_FILE"
set +a

HF_TOKEN_VALUE="${HF_TOKEN:-}"
if [[ -z "$HF_TOKEN_VALUE" ]]; then
  echo "ERROR: HF_TOKEN is not set." >&2
  echo "Run: export HF_TOKEN=hf_xxxxx" >&2
  exit 1
fi

echo "[INFO] Removing old container..."
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

GPU_ARG="device=${CUDA_DEVICES}"

CMD=(
  docker run -d
  --name "$CONTAINER_NAME"
  --gpus "$GPU_ARG"
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
  --tp-size "$TP_SIZE"
  --mem-fraction-static "$MEM_FRACTION_STATIC"
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
  --max-prefill-tokens "$MAX_PREFILL_TOKENS"
)

if [[ "${DISABLE_CUDA_GRAPH:-0}" == "1" ]]; then
  CMD+=(--disable-cuda-graph)
fi

if [[ -n "${LOG_LEVEL:-}" ]]; then
  CMD+=(--log-level "$LOG_LEVEL")
fi

echo "[INFO] Launching container..."
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"

echo
echo "[INFO] Running container:"
docker ps --filter "name=${CONTAINER_NAME}"

echo
echo "[INFO] Logs:"
echo "docker logs -f ${CONTAINER_NAME}"

echo
echo "[INFO] API:"
echo "http://localhost:${PORT}/v1"
