# RKLLM-Server Demo

## Before Run
Before running the demo, you need to prepare the following files:
- The transformed RKLLM model file on the board.
- Check the IP address of the board with the `ifconfig` command.

---

## 1. Build & Deploy

### Flask Server (OpenAI-compatible API)

```bash
# Usage: ./build_rkllm_server_flask.sh --workshop [Working Path] --model_path [Absolute Path of RKLLM Model on Board] --platform [Target Platform: rk3588/rk3576] [--lora_model_path [Lora Model Path]] [--prompt_cache_path [Prompt Cache File Path]] [--adb_device [ADB Device Serial]]
./build_rkllm_server_flask.sh --workshop /userdata --model_path /userdata/model.rkllm --platform rk3588 --adb_device 1234567890abcdef
```

The Flask server provides an **OpenAI-compatible API** with the following endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/models` | GET | List available models |
| `/v1/chat/completions` | POST | Chat completions (streaming & non-streaming) |

### Gradio Server (Web UI)

```bash
# Usage: ./build_rkllm_server_gradio.sh --workshop [Working Path] --model_path [Absolute Path of RKLLM Model on Board] --platform [Target Platform: rk3588/rk3576] [--lora_model_path [Lora Model Path]] [--prompt_cache_path [Prompt Cache File Path]] [--adb_device [ADB Device Serial]]
./build_rkllm_server_gradio.sh --workshop /userdata --model_path /userdata/model.rkllm --platform rk3588 --adb_device 1234567890abcdef
```

After deployment, open `http://<board-ip>:8080` in a browser to access the Gradio chat UI.

---

## 2. API Usage Guide

The Flask server exposes an OpenAI-compatible HTTP API supporting both regular chat and Function Calling (tool use).

### 2.1 Regular Chat

Regular chat is the standard Q&A interaction: the client sends messages, the server returns model responses.

#### Quick Start (Recommended)

Use the `RKLLMClient` wrapper class for minimal code:

```python
from chat_api_flask import RKLLMClient

client = RKLLMClient(base_url="http://x.x.x.x:8080")

# Non-streaming
reply = client.chat_simple("Hello, introduce yourself please.")
print(reply)

# Streaming
for chunk in client.chat(messages=[{"role": "user", "content": "Hello"}], stream=True):
    print(chunk["content"], end="", flush=True)
```

Run the built-in interactive demo:

```bash
# Interactive chat (Demo 1)
python chat_api_flask.py --server http://<board-ip>:8080

# Disable streaming
python chat_api_flask.py --server http://<board-ip>:8080 --no-stream
```

> **Note:** Replace `<board-ip>` with the actual IP address of the board (check with `ifconfig`).

#### Manual HTTP Calls

If you need a custom implementation, follow these steps:

**1) Set the server URL**

```python
server_url = "http://x.x.x.x:8080/v1/chat/completions"
```

**2) Create a session**

```python
import requests

session = requests.Session()
session.keep_alive = False
adapter = requests.adapters.HTTPAdapter(max_retries=3)
session.mount("https://", adapter)
session.mount("http://", adapter)
```

**3) Build the request body**

```python
data = {
    "model": "rkllm",                    # Model identifier (customizable)
    "messages": [
        {"role": "user", "content": "Hello"}
    ],
    "stream": False,                     # True = streaming, False = non-streaming
    "temperature": 0.8,                  # Sampling temperature (default: 0.8)
    "top_p": 0.9,                        # Nucleus sampling threshold (default: 0.9)
    "top_k": 1,                          # Top-k sampling (default: 1)
    "max_tokens": 4096,                  # Max tokens to generate (default: 4096)
    "repeat_penalty": 1.1,               # Repeat penalty (default: 1.1)
    "frequency_penalty": 0.0,            # Frequency penalty (default: 0.0)
    "presence_penalty": 0.0,             # Presence penalty (default: 0.0)
    "enable_thinking": False,            # Enable deep thinking mode
}
```

**4) Send the request**

```python
headers = {"Content-Type": "application/json", "Authorization": "not_required"}
resp = session.post(server_url, json=data, headers=headers, stream=data["stream"])
```

**5) Parse the response**

```python
import json

# Non-streaming
if resp.status_code == 200:
    result = resp.json()
    print("A:", result["choices"][0]["message"]["content"])
else:
    print("Error:", resp.text)

# Streaming (SSE)
if resp.status_code == 200:
    for line in resp.iter_lines(decode_unicode=True):
        if line.startswith("data: ") and line[6:].strip() != "[DONE]":
            chunk = json.loads(line[6:])
            delta = chunk["choices"][0].get("delta", {})
            if delta.get("content"):
                print(delta["content"], end="", flush=True)
```

---

### 2.2 Function Calling (Tool Use)

> **Note:** This section uses **Qwen3** series models as an example. Other models (e.g., DeepSeek, LLaMA) may output tool calls in different formats — adjust `parse_tool_calls()` accordingly.

Function Calling allows the model to automatically invoke predefined functions based on user requests, enabling access to real-time or precise information (e.g., weather, calculations).

#### Scenario

Take weather query as an example: a user asks "What's the temperature in San Francisco today and tomorrow?" The model cannot answer real-time questions on its own, but can call `get_current_temperature` and `get_temperature_date` to retrieve accurate data.

#### Quick Start (Recommended)

```python
from chat_api_flask import RKLLMClient, TOOLS, parse_tool_calls, execute_tool_calls

client = RKLLMClient(base_url="http://x.x.x.x:8080")

messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\nCurrent Date: 2024-09-30"},
    {"role": "user", "content": "What's the temperature in San Francisco now? How about tomorrow?"},
]

# Step 1: Model returns which tools to call and with what arguments
resp = client.chat(messages=messages, tools=TOOLS, stream=False)
tool_calls = parse_tool_calls(resp["choices"][0]["message"]["content"])

# Step 2: Execute tool calls and append results to messages
assistant_msg, tool_msgs = execute_tool_calls(tool_calls)
messages.append(assistant_msg)
messages.extend(tool_msgs)

# Step 3: Model synthesizes the final answer from tool results
resp = client.chat(messages=messages, tools=None, stream=False)
print("A:", resp["choices"][0]["message"]["content"])
```

Run the built-in function calling demo:

```bash
python chat_api_flask.py --server http://<board-ip>:8080 --demo 2
```

#### Manual HTTP Calls

**1) Define tool functions**

```python
def get_current_temperature(location: str, unit: str = "celsius"):
    return {"temperature": 26.1, "location": location, "unit": unit}

def get_temperature_date(location: str, date: str, unit: str = "celsius"):
    return {"temperature": 25.9, "location": location, "date": date, "unit": unit}

FUNCTION_MAP = {
    "get_current_temperature": get_current_temperature,
    "get_temperature_date": get_temperature_date,
}
```

**2) Define tool descriptions (OpenAI Function Calling format)**

```python
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_temperature",
            "description": "Get current temperature at a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City, State, Country"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_temperature_date",
            "description": "Get temperature at a location and date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City, State, Country"},
                    "date": {"type": "string", "description": "YYYY-MM-DD"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location", "date"],
            },
        },
    },
]
```

**3) First call: model returns tool invocation instructions**

```python
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\nCurrent Date: 2024-09-30"},
    {"role": "user", "content": "What's the temperature in San Francisco now? How about tomorrow?"},
]

data = {
    "model": "rkllm",
    "messages": messages,
    "stream": False,
    "tools": TOOLS,
}

resp = session.post(server_url, json=data, headers=headers)
server_answer = resp.json()["choices"][0]["message"]["content"]

# Example model output:
# <tool_call>
# {"name": "get_current_temperature", "arguments": {"location": "San Francisco"}}
# </tool_call>
# <tool_call>
# {"name": "get_temperature_date", "arguments": {"location": "San Francisco", "date": "2024-10-01"}}
# </tool_call>
```

**4) Parse tool calls and execute**

```python
import re, json

# Supports both JSON and XML-like tool_call formats
matches = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", server_answer, re.DOTALL)
tool_calls = [json.loads(m) for m in matches]

# Execute functions and build messages
for tc in tool_calls:
    name, args = tc["name"], tc["arguments"]
    result = FUNCTION_MAP[name](**args)
    messages.append({"role": "tool", "name": name, "content": json.dumps(result)})
```

**5) Second call: get the final answer**

```python
data["messages"] = messages
resp = session.post(server_url, json=data, headers=headers)
print("A:", resp.json()["choices"][0]["message"]["content"])

# Example output:
# A: The current temperature in San Francisco is 26.1°C.
#     Tomorrow, the temperature is expected to be 25.9°C.
```

#### Important Notes

| Item | Description |
|------|-------------|
| **Model Compatibility** | This demo is based on Qwen3 series. For other models, adjust `parse_tool_calls()` regex and the `tool_response_str` parameter |
| **Tool Call Format** | Qwen3 outputs `<tool_call>...</tool_call>` wrapped JSON. XML-like format (`<function=xxx><parameter=xxx>...</parameter></function>`) is also supported |
| **Multi-Tool Merging** | Multiple tool results are merged into a JSON array `[{...}, {...}]` before being sent to the model |
| **Model Capability** | Weaker models (e.g., Qwen3-0.6B) may fail to return accurate tool call parameters |
| **System Prompt** | The system prompt template is Qwen-specific; replace it with the appropriate template for your model |

---

## 3. Client Scripts

| Script | Description |
|--------|-------------|
| `chat_api_flask.py` | OpenAI-compatible client for the Flask server (supports chat + function calling) |
| `chat_api_gradio.py` | Client for the Gradio server (chat UI via Gradio API) |
