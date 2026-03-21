# 🚀 SGLang LLaMA 3.1 Deployment (Script-Based)

---

## 📦 Overview

This setup runs **Meta LLaMA 3.1 8B Instruct** using:

- Docker
- NVIDIA GPUs
- SGLang server
- Hugging Face model loading

Endpoint:
http://localhost:30000

---

## 📁 Files

- `sglang-llama.env` → configuration
- `llama-run.sh` → launcher script

---

## ⚡ Setup (Step-by-Step)

### 1. Install Requirements

Verify GPU:

```bash
nvidia-smi
```

Verify Docker GPU:

```bash
docker run --rm --runtime=nvidia --gpus all nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi
```

---

### 2. Hugging Face Setup

Install CLI:

```bash
pip install huggingface_hub
```

Login:

```bash
huggingface-cli login
```

Accept license:
https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct

---

### 3. Set Token (IMPORTANT)

DO NOT store token in file.

Run:

```bash
export HF_TOKEN=hf_xxxxx
```

---

### 4. Run Server

```bash
bash llama-run.sh
```

---

### 5. Check Logs

```bash
docker logs -f sglang-llama-server
```

---

## 🌐 API Usage

### Health

```bash
curl http://localhost:30000/health
```

### List Models

```bash
curl http://localhost:30000/v1/models
```

### Chat

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Explain AI simply"}]
  }'
```

---

## ⚙️ Parameters Explained

### MODEL

- `MODEL_PATH`
  → Hugging Face model repo

- `SERVED_MODEL_NAME`
  → API model name

---

### GPU

- `CUDA_DEVICES`
  → GPUs to use (0 / 1 / 0,1)

- `TP_SIZE`
  → Tensor parallel size

---

### MEMORY

- `MEM_FRACTION_STATIC`
  → GPU memory usage fraction

- `CHUNKED_PREFILL_SIZE`
  → Prompt chunking size

- `MAX_PREFILL_TOKENS`
  → Max prompt tokens

---

### CACHE

- `HF_HOME`
  → container cache path

- `HF_CACHE_HOST_PATH`
  → host cache mount

---

### SYSTEM

- `SHM_SIZE`
  → shared memory

- `NVIDIA_DRIVER_CAPABILITIES`
  → GPU capabilities

---

## 🔥 Presets

### Single GPU

```env
CUDA_DEVICES=0
TP_SIZE=1
```

### Multi GPU

```env
CUDA_DEVICES=0,1
TP_SIZE=2
```

---

## ❗ Common Issues

### HF_TOKEN not set

Fix:

```bash
export HF_TOKEN=hf_xxxxx
```

---

### No weights downloading

Fix:
- Accept Meta license
- Check token access

---

### GPU not detected

Fix:
- Ensure `--runtime=nvidia`
- Test CUDA container

---

## 🧠 Notes

- Do NOT use raw `.pth` files
- First run downloads ~15GB

---

## 🔐 Security

Rotate any exposed HF token:
https://huggingface.co/settings/tokens

---

## 🚀 Next Steps

- Multi-GPU scaling
- KV cache tuning
- LiteLLM integration

---

