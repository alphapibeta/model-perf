from openai import OpenAI

client = OpenAI(
    base_url="https://b8b6-98-5-241-170.ngrok-free.app/v1",
    api_key="sk-1234",
)

tests = [
    {
        "label": "LOCAL OSS 20B",
        "model": "vllm-llama-3.1-8b-instruct",
        "prompt": "Say exactly: This is running on my local GPU! Hello hello hello.",
        "temperature": 0.8,
        "max_tokens": 150,
    },
    {
        "label": "SGLANG 8B",
        "model": "sglang-llama-3.1-8b-instruct",
        "prompt": "Write a short poem about snow in Buffalo.",
        "temperature": 0.5,
        "max_tokens": 100,
    },
    {
        "label": "NVIDIA 70B",
        "model": "nvidia-llama3.1-70b",
        "prompt": "Write a short poem about snow in Buffalo.",
        "temperature": 0.4,
        "max_tokens": 100,
    },
    {
        "label": "NVIDIA 405B",
        "model": "nvidia-llama3.1-405b",
        "prompt": "Write a short poem about snow in Buffalo.",
        "temperature": 0.4,
        "max_tokens": 100,
    },
]

for test in tests:
    print(f"\n=== Testing {test['label']} ({test['model']}) ===")
    try:
        resp = client.chat.completions.create(
            model=test["model"],
            messages=[
                {"role": "user", "content": test["prompt"]}
            ],
            temperature=test["temperature"],
            max_tokens=test["max_tokens"],
        )

        if resp.choices and resp.choices[0].message:
            print(resp.choices[0].message.content)
        else:
            print("No choices returned.")
            print(resp)

    except Exception as e:
        print(f"ERROR for {test['model']}: {e}")