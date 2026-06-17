"""
RKLLM OpenAI-compatible Chat API Client

Usage:
    python chat_api_flask.py [--server http://x.x.x.x:8080] [--no-stream]

Demos:
    Demo1: Interactive multi-turn chat conversation
    Demo2: Function calling (tool use) demonstration
"""

import sys
import requests
import json
import re
import argparse
import textwrap


class RKLLMClient:
    """OpenAI-compatible client for RKLLM Server."""

    def __init__(self, base_url="http://x.x.x.x:8080", timeout=120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.keep_alive = False
        adapter = requests.adapters.HTTPAdapter(max_retries=3)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": "not_required",
        }

    def list_models(self):
        """GET /v1/models — list available models."""
        r = self.session.get(f"{self.base_url}/v1/models", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def chat(
        self,
        messages,
        model="rkllm",
        stream=False,
        temperature=0.8,
        top_p=0.9,
        top_k=1,
        max_tokens=4096,
        repeat_penalty=1.1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        enable_thinking=False,
        tools=None,
    ):
        """POST /v1/chat/completions — send a chat completion request.

        Returns:
            Non-streaming: dict (OpenAI-style response)
            Streaming:     generator yielding dict chunks
        """
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_tokens": max_tokens,
            "repeat_penalty": repeat_penalty,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "enable_thinking": enable_thinking,
        }
        if tools:
            payload["tools"] = tools

        r = self.session.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            headers=self._headers,
            stream=stream,
            timeout=self.timeout,
            verify=False,
        )
        r.raise_for_status()

        if not stream:
            return r.json()
        else:
            return self._parse_stream(r)

    def chat_simple(self, content, role="user", **kwargs):
        """Convenience: send a single message and return the assistant's reply text."""
        response = self.chat(messages=[{"role": role, "content": content}], **kwargs)
        return response["choices"][0]["message"]["content"]

    def _parse_stream(self, response):
        """Generator that yields delta text chunks from an SSE stream."""
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    finish = chunk["choices"][0].get("finish_reason")
                    yield {"content": content, "finish_reason": finish}
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue


# ---------------------------------------------------------------------------
# Demo 1: Interactive multi-turn chat
# ---------------------------------------------------------------------------

def demo_chat(client, stream=True):
    """Interactive chat loop.  Maintains full message history for multi-turn."""
    print("=" * 60)
    print("RKLLM Interactive Chat (OpenAI-compatible)")
    print(f"Server: {client.base_url}")
    print("Type 'exit' to quit, 'clear' to reset history.")
    print("=" * 60)

    # Check server connectivity
    try:
        models = client.list_models()
        model_name = models["data"][0]["id"] if models.get("data") else "unknown"
        print(f"Connected.  Model: {model_name}\n")
    except Exception as e:
        print(f"Warning: Could not reach server — {e}\n")

    messages = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() == "exit":
                print("Goodbye!")
                break
            if user_input.lower() == "clear":
                messages.clear()
                print("[History cleared]")
                continue

            messages.append({"role": "user", "content": user_input})

            if stream:
                print("Assistant: ", end="", flush=True)
                full_reply = ""
                for chunk in client.chat(messages=messages, stream=True):
                    content = chunk["content"]
                    if content:
                        print(content, end="", flush=True)
                        full_reply += content
                print()
                messages.append({"role": "assistant", "content": full_reply})
            else:
                resp = client.chat(messages=messages, stream=False)
                reply = resp["choices"][0]["message"]["content"]
                print(f"Assistant: {reply}")
                messages.append({"role": "assistant", "content": reply})

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except requests.exceptions.RequestException as e:
            print(f"\n[Error] {e}")


# ---------------------------------------------------------------------------
# Demo 2: Function calling (tool use)
# ---------------------------------------------------------------------------

# --- Tool definitions ---

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_temperature",
            "description": "Get current temperature at a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": 'The location, e.g. "San Francisco, CA, USA".',
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit (default: celsius).",
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_temperature_date",
            "description": "Get temperature at a location on a specific date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": 'The location, e.g. "San Francisco, CA, USA".',
                    },
                    "date": {
                        "type": "string",
                        "description": 'Date in "YYYY-MM-DD" format.',
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit (default: celsius).",
                    },
                },
                "required": ["location", "date"],
            },
        },
    },
]


# --- Tool implementations ---

def get_current_temperature(location: str, unit: str = "celsius"):
    return {"temperature": 26.1, "location": location, "unit": unit}


def get_temperature_date(location: str, date: str, unit: str = "celsius"):
    return {"temperature": 25.9, "location": location, "date": date, "unit": unit}


FUNCTION_MAP = {
    "get_current_temperature": get_current_temperature,
    "get_temperature_date": get_temperature_date,
}


def parse_tool_calls(text: str):
    """Extract tool calls from model output.

    Supports two formats:
    1. JSON:   <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    2. XML-like:
       <tool_call>
         <function=get_temperature_date>
         <parameter=location>San Francisco, CA, USA</parameter>
         <parameter=date>2024-09-30</parameter>
         <parameter=unit>celsius</parameter>
         </function>
       </tool_call>
    """
    # --- Try JSON format first ---
    json_pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
    json_matches = re.findall(json_pattern, text, re.DOTALL)
    if json_matches:
        return [json.loads(m) for m in json_matches]

    # --- Fall back to XML-like format ---
    results = []
    # Match each <tool_call>...</tool_call> block
    blocks = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
    for block in blocks:
        # Match each <function=NAME>...</function> block
        func_blocks = re.findall(r"<function=(\S+)>\s*(.*?)\s*</function>", block, re.DOTALL)
        for func_name, params_block in func_blocks:
            arguments = {}
            # Match each <parameter=NAME>VALUE</parameter>
            for param_name, param_value in re.findall(
                r"<parameter=(\S+)>\s*(.*?)\s*</parameter>", params_block, re.DOTALL
            ):
                arguments[param_name] = param_value.strip()
            results.append({"name": func_name, "arguments": arguments})

    return results


def execute_tool_calls(tool_calls):
    """Execute parsed tool calls and return (assistant_msg, tool_msgs).

    Supports two input formats:
    1. {"name": "...", "arguments": {...}}          (from parse_tool_calls)
    2. {"function": {"name": "...", "arguments": {...}}}  (OpenAI-style)
    """
    normalized = []
    tool_msgs = []

    for tc in tool_calls:
        # Normalize: extract name & arguments regardless of input format
        fn = tc.get("function", tc)
        name = fn.get("name", "")
        args = fn.get("arguments", {})

        # Execute the function
        if name in FUNCTION_MAP:
            result = FUNCTION_MAP[name](**args)
        else:
            result = {"error": f"Unknown function: {name}"}

        # Build normalized tool_call entry
        normalized.append({
            "id": f"call_{name}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(args, ensure_ascii=False),
            },
        })
        tool_msgs.append({"role": "tool", "name": name, "content": json.dumps(result, ensure_ascii=False)})

    assistant_msg = {"role": "assistant", "content": None, "tool_calls": normalized}
    return assistant_msg, tool_msgs


def demo_function_call(client, stream=True):
    """Demonstrate function calling (tool use) with the RKLLM server."""
    print("=" * 60)
    print("RKLLM Function Calling Demo")
    print("=" * 60)

    messages = [
        {
            "role": "system",
            "content": (
                "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n"
                "Current Date: 2024-09-30"
            ),
        },
        {
            "role": "user",
            "content": "What's the temperature in San Francisco now? How about tomorrow?",
        },
    ]

    # --- Step 1: First call — model decides which tools to invoke ---
    print(f"\nQ: {messages[-1]['content']}\n")

    resp = client.chat(messages=messages, tools=TOOLS, stream=False)
    server_answer = resp["choices"][0]["message"]["content"]
    print(f"[Model tool-call response]:\n{textwrap.indent(server_answer, '  ')}\n")

    tool_calls = parse_tool_calls(server_answer)
    if not tool_calls:
        print("No tool calls detected — model answered directly.")
        print(f"A: {server_answer}")
        return
    
    # --- Step 2: Execute tool calls ---
    assistant_msg, tool_msgs = execute_tool_calls(tool_calls)
    messages.append(assistant_msg)
    messages.extend(tool_msgs)

    print(f"[Executed {len(tool_calls)} tool call(s)]:")
    for t in tool_calls:
        print(f"  - {t.get('function', t).get('name', '?')}")
    print()
   
    # --- Step 3: Second call — model synthesizes final answer ---
    if stream:
        print("A: ", end="", flush=True)
        for chunk in client.chat(messages=messages, tools=None, stream=True):
            content = chunk["content"]
            if content:
                print(content, end="", flush=True)
        print()
    else:
        resp = client.chat(messages=messages, tools=None, stream=False)
        print(f"A: {resp['choices'][0]['message']['content']}")

    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RKLLM OpenAI-compatible Chat Client")
    parser.add_argument(
        "--server",
        default="http://x.x.x.x:8080",
        help="RKLLM server base URL (default: http://x.x.x.x:8080)",
    )
    parser.add_argument(
        "--no-stream", action="store_true", help="Disable streaming mode"
    )
    parser.add_argument(
        "--demo", type=int, choices=[1, 2], default=1, help="Demo to run (1=chat, 2=function-call)"
    )
    args = parser.parse_args()

    client = RKLLMClient(base_url=args.server)
    stream = not args.no_stream

    if args.demo == 1:
        demo_chat(client, stream=stream)
    else:
        demo_function_call(client, stream=stream)
    

