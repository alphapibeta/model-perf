#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${1:-$SCRIPT_DIR/responsive-gptoss.env}"

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
  echo "Set it in gptoss.env or export HF_TOKEN in your shell." >&2
  exit 1
fi

echo "[INFO] Removing old container if present..."
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

GPU_ARG="\"device=${CUDA_DEVICES}\""

CMD=(
  docker run -d
  --name "$CONTAINER_NAME"
  --gpus "$GPU_ARG"
  --runtime=nvidia
  --ipc=host
  --shm-size "$SHM_SIZE"
  -p "${PORT}:8000"
  -e "HF_TOKEN=${HF_TOKEN_VALUE}"
  -e "HF_HOME=${HF_HOME}"
  -e "TORCH_COMPILE_DISABLE=1"
  -e "VLLM_TORCH_COMPILE_LEVEL=0"
  -e "NVIDIA_DRIVER_CAPABILITIES=compute,utility,video"
  -e "NCCL_DEBUG=${NCCL_DEBUG}"
  -e "NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS}"
  -v "${HF_CACHE_HOST_PATH}:${HF_HOME}"
  "$IMAGE_NAME"
  --model "$MODEL_NAME"
  --host "$HOST"
  --port 8000
  --served-model-name "$SERVED_MODEL_NAME"
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
)

if [[ "${ENFORCE_EAGER:-0}" == "1" ]]; then
  CMD+=(--enforce-eager)
fi

if [[ "${ENABLE_AUTO_TOOL_CHOICE:-0}" == "1" ]]; then
  CMD+=(--tool-call-parser "$TOOL_CALL_PARSER" --enable-auto-tool-choice)
fi

echo "[INFO] Final command:"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"

echo
echo "[INFO] Container started:"
docker ps --filter "name=${CONTAINER_NAME}"

echo
echo "[INFO] Follow logs with:"
echo "docker logs -f ${CONTAINER_NAME}"

echo
echo "[INFO] Endpoint:"
echo "http://localhost:${PORT}/v1"