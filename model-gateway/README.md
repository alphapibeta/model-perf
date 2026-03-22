# Model Gateway

A single OpenAI-compatible gateway in front of multiple model backends, served by **LiteLLM** and optionally exposed publicly via **ngrok**.

---

## What this does

One endpoint routes to multiple backends. Clients only need the gateway URL, the LiteLLM master key, and a public model name — they never interact with internal infrastructure directly.

| Public model name | Backend |
|---|---|
| `local-oss-20b` | Local SGLang server — `http://192.168.1.33:30000/v1` |
| `nvidia-llama3.1-70b` | NVIDIA NIM hosted API |
| `nvidia-llama3.1-405b` | NVIDIA NIM hosted API |

---

## Repository structure

```
model-gateway/
├── config.yaml     # LiteLLM routing config
├── .env            # Runtime secrets and environment variables
├── start.sh        # Launch script for running LiteLLM directly
└── README.md
```

---

## config.yaml — explained

```yaml
model_list:
  - model_name: local-oss-20b
    litellm_params:
      model: openai/gpt-oss-20b
      api_base: http://192.168.1.33:30000/v1
      api_key: "dummy"
    model_info:
      supports_function_calling: true
      supports_tool_choice: true

  - model_name: nvidia-llama3.1-70b
    litellm_params:
      model: nvidia_nim/meta/llama-3.1-70b-instruct
      api_key: os.environ/NVIDIA_API_KEY

  - model_name: nvidia-llama3.1-405b
    litellm_params:
      model: nvidia_nim/meta/llama-3.1-405b-instruct
      api_key: os.environ/NVIDIA_API_KEY

mcp_servers:
  kite:
    url: "http://localhost:8080/mcp"
    transport: "http"

litellm_settings:
  success_callback: ["langfuse_otel"]
  failure_callback: ["langfuse_otel"]
  num_retries: 2
  request_timeout: 300
```

### `model_list`

Each entry defines a public model name that the gateway exposes. The `model_name` is what clients pass as the `model` field in their requests. The `litellm_params` block defines how LiteLLM routes that request internally.

---

#### `local-oss-20b`

```yaml
- model_name: local-oss-20b
  litellm_params:
    model: openai/gpt-oss-20b
    api_base: http://192.168.1.33:30000/v1
    api_key: "dummy"
  model_info:
    supports_function_calling: true
    supports_tool_choice: true
```

**`model: openai/gpt-oss-20b`** — The `openai/` prefix tells LiteLLM to use its OpenAI-compatible HTTP handler for this backend. SGLang exposes a fully OpenAI-compatible API, so this is the correct prefix. The part after the slash (`gpt-oss-20b`) is the model ID passed to SGLang in the request body.

**`api_base`** — The internal SGLang server address. All `local-oss-20b` requests are routed here.

**`api_key: "dummy"`** — SGLang does not require authentication, but LiteLLM requires this field to be present. Any non-empty string is valid.

**`model_info.supports_function_calling: true`** and **`supports_tool_choice: true`** — These flags explicitly declare that this model supports tool calling. Without them, LiteLLM leaves these as `null` and routes tool call responses through a fallback parsing path that mishandles SGLang's output. SGLang's responses are correct and OpenAI-compatible — these flags ensure LiteLLM processes them through the right code path.

---

#### `nvidia-llama3.1-70b` and `nvidia-llama3.1-405b`

```yaml
- model_name: nvidia-llama3.1-70b
  litellm_params:
    model: nvidia_nim/meta/llama-3.1-70b-instruct
    api_key: os.environ/NVIDIA_API_KEY
```

**`model: nvidia_nim/...`** — The `nvidia_nim/` prefix routes requests through LiteLLM's NVIDIA NIM provider. The full path after the prefix is the canonical NVIDIA model ID.

**`api_key: os.environ/NVIDIA_API_KEY`** — LiteLLM reads the API key from the `NVIDIA_API_KEY` environment variable at startup, sourced from `.env`.

---

### `mcp_servers`

```yaml
mcp_servers:
  kite:
    url: "http://localhost:8080/mcp"
    transport: "http"
```

Registers the `kite` MCP (Model Context Protocol) server with LiteLLM. This makes the MCP server available to the gateway environment for tool routing and context injection. `transport: "http"` means LiteLLM communicates with it over HTTP rather than stdio.

---

### `litellm_settings`

```yaml
litellm_settings:
  success_callback: ["langfuse_otel"]
  failure_callback: ["langfuse_otel"]
  num_retries: 2
  request_timeout: 300
```

**`success_callback` / `failure_callback`** — LiteLLM emits telemetry to Langfuse via OpenTelemetry on both successful and failed requests. Langfuse connection details are configured in `.env`.

**`num_retries: 2`** — Failed requests are retried up to 2 times before an error is returned to the client.

**`request_timeout: 300`** — Backend requests time out after 300 seconds. This is set generously to accommodate the local 20B model under load.

---

## .env

```bash
# LiteLLM
LITELLM_MASTER_KEY=sk-your-litellm-master-key

# NVIDIA
NVIDIA_API_KEY=nvapi-your-nvidia-key

# Langfuse
LANGFUSE_SECRET_KEY=sk-lf-your-secret
LANGFUSE_PUBLIC_KEY=pk-lf-your-public
LANGFUSE_BASE_URL=http://localhost:3002
OTEL_EXPORTER_OTLP_ENDPOINT=https://cloud.langfuse.com/api/public/otel

# Misc
PYTHONUNBUFFERED=1
```

---

## start.sh

For running LiteLLM directly without Docker:

```bash
#!/usr/bin/env bash
set -euo pipefail

source /root/exp/kite-mcp-clinet/kite_mcp/bin/activate

set -a
source .env
set +a

echo "Starting LiteLLM proxy on port 4000..."
litellm --config config.yaml --port 4000
```

```bash
chmod +x start.sh
./start.sh
```

---

## Docker (preferred)

```bash
docker rm -f litellm-proxy 2>/dev/null || true

docker run -d --rm \
  --name litellm-proxy \
  --network host \
  --env-file /root/exp/model-gateway/.env \
  -v /root/exp/model-gateway/config.yaml:/app/config.yaml \
  ghcr.io/berriai/litellm:main-stable \
  --config /app/config.yaml \
  --port 4000
```

`--network host` is required so the container can reach the local SGLang server at `192.168.1.33:30000`.

```bash
# Follow logs
docker logs -f litellm-proxy

# Stop
docker stop litellm-proxy
```

---

## Startup order

1. **Start SGLang** and verify it is healthy:

```bash
curl -s http://192.168.1.33:30000/v1/models | jq
```

Expected:
```json
{
  "object": "list",
  "data": [{"id": "gpt-oss-20b", "object": "model", "owned_by": "sglang"}]
}
```

2. **Start optional services** — MCP server, Langfuse, Postgres.

3. **Start LiteLLM** via `./start.sh` or Docker.

4. **Verify all three models are registered:**

```bash
curl -s http://localhost:4000/models \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" | jq
```

5. **Start ngrok** if a public endpoint is needed.

---

## Testing

### List models

```bash
curl -s 'https://<ngrok-url>/models' \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" | jq
```

### Plain chat — local model

```bash
curl -X POST 'https://<ngrok-url>/chat/completions' \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -d '{
    "model": "local-oss-20b",
    "messages": [{"role": "user", "content": "Say hello and a short poem"}],
    "max_tokens": 120,
    "temperature": 0.7
  }'
```

### Tool call — local model

```bash
curl -X POST 'https://<ngrok-url>/chat/completions' \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -d '{
    "model": "local-oss-20b",
    "messages": [{"role": "user", "content": "login"}],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "kite_login",
          "description": "Login tool",
          "parameters": {"type": "object", "properties": {}, "required": []}
        }
      }
    ],
    "max_tokens": 100
  }'
```

Expected response:
```json
{
  "choices": [{
    "finish_reason": "tool_calls",
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "type": "function",
        "function": {"name": "kite_login", "arguments": "{}"}
      }]
    }
  }]
}
```

### NVIDIA models

```bash
# 70B
curl -X POST 'https://<ngrok-url>/chat/completions' \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -d '{"model": "nvidia-llama3.1-70b", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 50}'

# 405B
curl -X POST 'https://<ngrok-url>/chat/completions' \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -d '{"model": "nvidia-llama3.1-405b", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 50}'
```

---

## Optional: Postgres

```bash
docker network create litellm-net || true
docker rm -f my-postgres 2>/dev/null || true

docker run --name my-postgres \
  --network litellm-net \
  -e POSTGRES_PASSWORD=secret \
  -p 5433:5432 \
  -d postgres
```

---

## Sanity checklist

- [ ] SGLang responds on `192.168.1.33:30000`
- [ ] LiteLLM starts without `Error creating deployment` in logs
- [ ] `/models` returns all three model IDs
- [ ] `local-oss-20b` plain chat works
- [ ] `local-oss-20b` tool call returns `finish_reason: tool_calls`
- [ ] `nvidia-llama3.1-70b` responds correctly
- [ ] `nvidia-llama3.1-405b` responds correctly

---

## Troubleshooting

### `'dict object' has no attribute 'function'`

LiteLLM's tool call response parser falls back to an incompatible code path when it has no explicit declaration that a model supports function calling. SGLang's responses are valid and OpenAI-compatible — the issue is LiteLLM's internal routing logic.

Ensure `config.yaml` includes the following under `local-oss-20b`:

```yaml
model_info:
  supports_function_calling: true
  supports_tool_choice: true
```

---

### `Unsupported provider - sglang`

`custom_llm_provider: sglang` is not a recognised value in LiteLLM v1.82.3. When present, the deployment is silently dropped and the model does not appear in `/models`.

Use `model: openai/gpt-oss-20b` without a `custom_llm_provider` field. The `openai/` prefix selects the OpenAI-compatible HTTP handler, which is the correct choice for SGLang.

---

### `No healthy deployments` for local model

Either the deployment was dropped due to a config error, or SGLang is not reachable at `192.168.1.33:30000`. Check the SGLang server first, then inspect the LiteLLM startup logs:

```bash
docker logs litellm-proxy 2>&1 | grep -i error
```