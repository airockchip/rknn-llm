import ctypes
import sys
import os
import subprocess
import resource
import threading
import time
import argparse
import json
import uuid
import signal
from flask import Flask, request, jsonify, Response, stream_with_context
import re

app = Flask(__name__)

# Set the dynamic library path
rkllm_lib = ctypes.CDLL('lib/librkllmrt.so')
# Define the structures from the library
RKLLM_Handle_t = ctypes.c_void_p
userdata = ctypes.c_void_p(None)

LLMCallState = ctypes.c_int
LLMCallState.RKLLM_RUN_NORMAL  = 0
LLMCallState.RKLLM_RUN_WAITING  = 1
LLMCallState.RKLLM_RUN_FINISH  = 2
LLMCallState.RKLLM_RUN_ERROR   = 3

RKLLMInputType = ctypes.c_int
RKLLMInputType.RKLLM_INPUT_PROMPT      = 0
RKLLMInputType.RKLLM_INPUT_TOKEN       = 1
RKLLMInputType.RKLLM_INPUT_EMBED       = 2
RKLLMInputType.RKLLM_INPUT_MULTIMODAL  = 3

RKLLMInferMode = ctypes.c_int
RKLLMInferMode.RKLLM_INFER_GENERATE = 0
RKLLMInferMode.RKLLM_INFER_GET_LAST_HIDDEN_LAYER = 1
RKLLMInferMode.RKLLM_INFER_GET_LOGITS = 2
class RKLLMExtendParam(ctypes.Structure):
    _fields_ = [
        ("base_domain_id", ctypes.c_int32),
        ("embed_flash", ctypes.c_int8),
        ("enabled_cpus_num", ctypes.c_int8),
        ("enabled_cpus_mask", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint8),
        ("use_cross_attn", ctypes.c_int8),
        ("reserved", ctypes.c_uint8 * 104)
    ]

class RKLLMParam(ctypes.Structure):
    _fields_ = [
        ("model_path", ctypes.c_char_p),
        ("max_context_len", ctypes.c_int32),
        ("max_new_tokens", ctypes.c_int32),
        ("top_k", ctypes.c_int32),
        ("n_keep", ctypes.c_int32),
        ("top_p", ctypes.c_float),
        ("temperature", ctypes.c_float),
        ("repeat_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("mirostat", ctypes.c_int32),
        ("mirostat_tau", ctypes.c_float),
        ("mirostat_eta", ctypes.c_float),
        ("skip_special_token", ctypes.c_bool),
        ("ignore_eos_token", ctypes.c_bool),
        ("is_async", ctypes.c_bool),
        ("extend_param", RKLLMExtendParam),
    ]

class RKLLMLoraAdapter(ctypes.Structure):
    _fields_ = [
        ("lora_adapter_path", ctypes.c_char_p),
        ("lora_adapter_name", ctypes.c_char_p),
        ("scale", ctypes.c_float)
    ]

class RKLLMEmbedInput(ctypes.Structure):
    _fields_ = [
        ("embed", ctypes.POINTER(ctypes.c_float)),
        ("n_tokens", ctypes.c_size_t)
    ]

class RKLLMTokenInput(ctypes.Structure):
    _fields_ = [
        ("input_ids", ctypes.POINTER(ctypes.c_int32)),
        ("n_tokens", ctypes.c_size_t)
    ]

class RKLLMImageInput(ctypes.Structure):
    _fields_ = [
        ("image_embed", ctypes.POINTER(ctypes.c_float)),
        ("n_image_tokens", ctypes.c_size_t),
        ("n_image", ctypes.c_size_t),
        ("image_start", ctypes.c_char_p),
        ("image_end", ctypes.c_char_p),
        ("image_content", ctypes.c_char_p),
        ("image_width", ctypes.c_size_t),
        ("image_height", ctypes.c_size_t),
    ]

class RKLLMVideoInput(ctypes.Structure):
    _fields_ = [
        ("video_embed", ctypes.POINTER(ctypes.c_float)),
        ("n_frame_tokens", ctypes.c_size_t),
        ("n_frame_per_video", ctypes.c_size_t),
        ("n_video", ctypes.c_size_t),
        ("video_start", ctypes.c_char_p),
        ("video_end", ctypes.c_char_p),
        ("video_content", ctypes.c_char_p),
        ("frame_width", ctypes.c_size_t),
        ("frame_height", ctypes.c_size_t),
    ]

class RKLLMMultiModalInput(ctypes.Structure):
    _fields_ = [
        ("prompt", ctypes.c_char_p),
        ("image", RKLLMImageInput),
        ("video", RKLLMVideoInput),
    ]

class RKLLMInputUnion(ctypes.Union):
    _fields_ = [
        ("prompt_input", ctypes.c_char_p),
        ("embed_input", RKLLMEmbedInput),
        ("token_input", RKLLMTokenInput),
        ("multimodal_input", RKLLMMultiModalInput)
    ]

class RKLLMInput(ctypes.Structure):
    _anonymous_ = ("input_data",)
    _fields_ = [
        ("role", ctypes.c_char_p),
        ("enable_thinking", ctypes.c_bool),
        ("input_type", RKLLMInputType),
        ("input_data", RKLLMInputUnion)
    ]

class RKLLMLoraParam(ctypes.Structure):
    _fields_ = [
        ("lora_adapter_name", ctypes.c_char_p)
    ]

class RKLLMPromptCacheParam(ctypes.Structure):
    _fields_ = [
        ("save_prompt_cache", ctypes.c_int),
        ("prompt_cache_path", ctypes.c_char_p)
    ]

class RKLLMSamplingParam(ctypes.Structure):
    _fields_ = [
        ("top_k", ctypes.c_int32),
        ("top_p", ctypes.c_float),
        ("temperature", ctypes.c_float),
        ("repeat_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("mirostat", ctypes.c_int32),
        ("mirostat_tau", ctypes.c_float),
        ("mirostat_eta", ctypes.c_float),
    ]

class RKLLMInferParam(ctypes.Structure):
    _fields_ = [
        ("mode", RKLLMInferMode),
        ("lora_params", ctypes.POINTER(RKLLMLoraParam)),
        ("prompt_cache_params", ctypes.POINTER(RKLLMPromptCacheParam)),
        ("sampling_params", ctypes.POINTER(RKLLMSamplingParam)),
        ("keep_history", ctypes.c_int),
        ("max_new_tokens", ctypes.c_int32),
    ]

class RKLLMResultLastHiddenLayer(ctypes.Structure):
    _fields_ = [
        ("hidden_states", ctypes.POINTER(ctypes.c_float)),
        ("embd_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int)
    ]

class RKLLMResultLogits(ctypes.Structure):
    _fields_ = [
        ("logits", ctypes.POINTER(ctypes.c_float)),
        ("vocab_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int)
    ]

class RKLLMPerfStat(ctypes.Structure):
    _fields_ = [
        ("prefill_time_ms", ctypes.c_float),
        ("prefill_tokens", ctypes.c_int),
        ("generate_time_ms", ctypes.c_float),
        ("generate_tokens", ctypes.c_int),
        ("memory_usage_mb", ctypes.c_float)
    ]

class RKLLMResult(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("token_id", ctypes.c_int),
        ("last_hidden_layer", RKLLMResultLastHiddenLayer),
        ("logits", RKLLMResultLogits),
        ("perf", RKLLMPerfStat)
    ]

# Create a lock to control multi-user access to the server.
lock = threading.Lock()

# Create a global variable to indicate whether the server is currently in a blocked state.
is_blocking = False

# Define global variables to store the callback function output
system_prompt = ''
global_text = []
global_state = -1
split_byte_data = bytes(b"") # Used to store the segmented byte data

# Per-request context for function calling
tool_call_context = {"pending_tools": None, "tool_response_str": "tool_response"}

# Define the callback function
def callback_impl(result, userdata, state):
    global global_text, global_state, split_byte_data
    if state == LLMCallState.RKLLM_RUN_FINISH:
        global_state = state
        sys.stdout.flush()
    elif state == LLMCallState.RKLLM_RUN_ERROR:
        global_state = state
        print("run error")
        sys.stdout.flush()
    elif state == LLMCallState.RKLLM_RUN_NORMAL:
        global_state = state
        global_text += result.contents.text.decode('utf-8')
    return 0
    

# Connect the callback function between the Python side and the C++ side
LLMResultCallback_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int)
LLMTokenizerCallback_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32)
LLMGetEmbedCallback_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64)

class RKLLMCallback(ctypes.Structure):
    _fields_ = [
        ("result_callback", LLMResultCallback_type),
        ("result_userdata", ctypes.c_void_p),
        ("tokenizer_callback", LLMTokenizerCallback_type),
        ("tokenizer_userdata", ctypes.c_void_p),
        ("embed_callback", LLMGetEmbedCallback_type),
        ("embed_userdata", ctypes.c_void_p),
    ]

callback = LLMResultCallback_type(callback_impl)

# Define the RKLLM class, which includes initialization, inference, and release operations for the RKLLM model in the dynamic library
class RKLLM(object):
    def __init__(self, model_path, lora_model_path = None, prompt_cache_path = None, platform = "rk3588"):
        rkllm_param = RKLLMParam()
        rkllm_param.model_path = bytes(model_path, 'utf-8')

        rkllm_param.max_context_len = 4096
        rkllm_param.max_new_tokens = 4096
        rkllm_param.skip_special_token = True
        rkllm_param.n_keep = -1
        rkllm_param.top_k = 1
        rkllm_param.top_p = 0.9
        rkllm_param.temperature = 0.8
        rkllm_param.repeat_penalty = 1.1
        rkllm_param.frequency_penalty = 0.0
        rkllm_param.presence_penalty = 0.0

        rkllm_param.mirostat = 0
        rkllm_param.mirostat_tau = 5.0
        rkllm_param.mirostat_eta = 0.1

        rkllm_param.is_async = False

        rkllm_param.ignore_eos_token = False

        rkllm_param.extend_param.base_domain_id = 0
        rkllm_param.extend_param.embed_flash = 1
        rkllm_param.extend_param.n_batch = 1
        rkllm_param.extend_param.use_cross_attn = 0
        rkllm_param.extend_param.enabled_cpus_num = 4
        if platform.lower() in ["rk3576", "rk3588"]:
            rkllm_param.extend_param.enabled_cpus_mask = (1 << 4)|(1 << 5)|(1 << 6)|(1 << 7)
        else:
            rkllm_param.extend_param.enabled_cpus_mask = (1 << 0)|(1 << 1)|(1 << 2)|(1 << 3)

        self.handle = RKLLM_Handle_t()

        self.rkllm_init = rkllm_lib.rkllm_init
        self.rkllm_init.argtypes = [ctypes.POINTER(RKLLM_Handle_t), ctypes.POINTER(RKLLMParam), ctypes.POINTER(RKLLMCallback)]
        self.rkllm_init.restype = ctypes.c_int
        self.callback = RKLLMCallback()
        self.callback.result_callback = callback
        self.callback.result_userdata = None
        self.callback.tokenizer_callback = LLMTokenizerCallback_type()
        self.callback.tokenizer_userdata = None
        self.callback.embed_callback = LLMGetEmbedCallback_type()
        self.callback.embed_userdata = None
        ret = self.rkllm_init(ctypes.byref(self.handle), ctypes.byref(rkllm_param), ctypes.byref(self.callback))
        if (ret != 0):
            print("\nrkllm init failed\n")
            exit(0)
        else:
            print("\nrkllm init success!\n")

        self.rkllm_run = rkllm_lib.rkllm_run
        self.rkllm_run.argtypes = [RKLLM_Handle_t, ctypes.POINTER(RKLLMInput), ctypes.POINTER(RKLLMInferParam), ctypes.c_void_p]
        self.rkllm_run.restype = ctypes.c_int
        
        self.set_chat_template = rkllm_lib.rkllm_set_chat_template
        self.set_chat_template.argtypes = [RKLLM_Handle_t, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
        self.set_chat_template.restype = ctypes.c_int
        
        self.set_function_tools_ = rkllm_lib.rkllm_set_function_tools
        self.set_function_tools_.argtypes = [RKLLM_Handle_t, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
        self.set_function_tools_.restype = ctypes.c_int

        self.rkllm_destroy = rkllm_lib.rkllm_destroy
        self.rkllm_destroy.argtypes = [RKLLM_Handle_t]
        self.rkllm_destroy.restype = ctypes.c_int
        
        self.rkllm_abort = rkllm_lib.rkllm_abort

        rkllm_lora_params = None
        if lora_model_path:
            lora_adapter_name = "test"
            lora_adapter = RKLLMLoraAdapter()
            ctypes.memset(ctypes.byref(lora_adapter), 0, ctypes.sizeof(RKLLMLoraAdapter))
            lora_adapter.lora_adapter_path = ctypes.c_char_p((lora_model_path).encode('utf-8'))
            lora_adapter.lora_adapter_name = ctypes.c_char_p((lora_adapter_name).encode('utf-8'))
            lora_adapter.scale = 1.0

            rkllm_load_lora = rkllm_lib.rkllm_load_lora
            rkllm_load_lora.argtypes = [RKLLM_Handle_t, ctypes.POINTER(RKLLMLoraAdapter)]
            rkllm_load_lora.restype = ctypes.c_int
            rkllm_load_lora(self.handle, ctypes.byref(lora_adapter))
            rkllm_lora_params = RKLLMLoraParam()
            rkllm_lora_params.lora_adapter_name = ctypes.c_char_p((lora_adapter_name).encode('utf-8'))
        
        self.rkllm_infer_params = RKLLMInferParam()
        ctypes.memset(ctypes.byref(self.rkllm_infer_params), 0, ctypes.sizeof(RKLLMInferParam))
        self.rkllm_infer_params.mode = RKLLMInferMode.RKLLM_INFER_GENERATE
        self.rkllm_infer_params.lora_params = ctypes.pointer(rkllm_lora_params) if rkllm_lora_params else None
        self.rkllm_infer_params.keep_history = 0

        self.prompt_cache_path = None
        if prompt_cache_path:
            self.prompt_cache_path = prompt_cache_path

            rkllm_load_prompt_cache = rkllm_lib.rkllm_load_prompt_cache
            rkllm_load_prompt_cache.argtypes = [RKLLM_Handle_t, ctypes.c_char_p]
            rkllm_load_prompt_cache.restype = ctypes.c_int
            rkllm_load_prompt_cache(self.handle, ctypes.c_char_p((prompt_cache_path).encode('utf-8')))
        
        self.tools = None
        self.model_name = os.path.basename(model_path).rsplit('.', 1)[0]

    def set_chat_template(self, system_prompt, prompt_prefix, prompt_postfix):
        self.set_chat_template_(self.handle, ctypes.c_char_p(system_prompt.encode('utf-8')), ctypes.c_char_p(prompt_prefix.encode('utf-8')), ctypes.c_char_p(prompt_postfix.encode('utf-8')))

    def set_function_tools(self, system_prompt, tools, tool_response_str):
        if self.tools is None or not self.tools == tools:
            self.tools = tools
            self.set_function_tools_(self.handle, ctypes.c_char_p(system_prompt.encode('utf-8')), ctypes.c_char_p(tools.encode('utf-8')),  ctypes.c_char_p(tool_response_str.encode('utf-8')))

    def run(self, role, enable_thinking, prompt, sampling_params=None, max_new_tokens=None):
        rkllm_input = RKLLMInput()
        rkllm_input.role = role.encode('utf-8') if role is not None else "user".encode('utf-8')
        rkllm_input.enable_thinking = ctypes.c_bool(enable_thinking if enable_thinking is not None else False)
        rkllm_input.input_type = RKLLMInputType.RKLLM_INPUT_PROMPT
        rkllm_input.prompt_input = ctypes.c_char_p(prompt.encode('utf-8'))

        # Apply per-request sampling params and max_new_tokens
        if sampling_params is not None:
            self.rkllm_infer_params.sampling_params = ctypes.pointer(sampling_params)
        if max_new_tokens is not None:
            self.rkllm_infer_params.max_new_tokens = max_new_tokens

        self.rkllm_run(self.handle, ctypes.byref(rkllm_input), ctypes.byref(self.rkllm_infer_params), None)

        # Reset sampling_params to NULL after run (avoid dangling pointer)
        if sampling_params is not None:
            self.rkllm_infer_params.sampling_params = None
        # Reset max_new_tokens to 0 after run (<=0 means use init value)
        if max_new_tokens is not None:
            self.rkllm_infer_params.max_new_tokens = 0
        return
    
    def abort(self):
        return self.rkllm_abort(self.handle)
    
    def release(self):
        self.rkllm_destroy(self.handle)


# ---------------------------------------------------------------------------
# OpenAI-compatible helpers
# ---------------------------------------------------------------------------

def build_openai_response(model_name, content, finish_reason="stop", prompt_tokens=0, completion_tokens=0):
    """Build a standard OpenAI chat completion response."""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content
            },
            "logprobs": None,
            "finish_reason": finish_reason
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    }


def build_openai_stream_chunk(model_name, delta_content, finish_reason=None, index=0):
    """Build a standard OpenAI streaming chat completion chunk."""
    chunk = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": index,
            "delta": {"role": "assistant"} if delta_content is None else {"content": delta_content},
            "logprobs": None,
            "finish_reason": finish_reason
        }]
    }
    return f"data: {json.dumps(chunk)}\n\n"


def build_openai_error(message, error_type="invalid_request_error", status_code=400):
    """Build a standard OpenAI error response."""
    return jsonify({
        "error": {
            "message": message,
            "type": error_type,
            "param": None,
            "code": None
        }
    }), status_code


def extract_system_prompt_and_tools(messages):
    """Extract system prompt and tools from messages."""
    system_prompt = ""
    tools = None
    for msg in messages:
        if msg.get("role") == "system":
            system_prompt = msg.get("content", "")
    return system_prompt, tools


# Track previous messages to detect newly added ones across requests
_last_messages = []


def get_last_input(messages):
    """Extract the latest new input for inference.

    Tracks message history to detect newly added messages since last call.
    When multiple consecutive 'tool' messages are detected in the new batch,
    merges their content into a JSON array, e.g.:
        [{"temperature": 26.1, ...}, {"temperature": 25.9, ...}]

    Otherwise returns the last user/tool message's role and content directly.
    """
    global _last_messages

    role = "user"
    content = ""

    # Detect newly added messages by comparing with tracked history
    prev_len = len(_last_messages)
    new_messages = messages[prev_len:] if prev_len < len(messages) else []

    # Update tracking for next call
    _last_messages = list(messages)

    if not new_messages:
        # No new messages — fall back to last user/tool in full history
        for msg in reversed(messages):
            r = msg.get("role", "")
            if r in ("user", "tool"):
                role = r
                content = _extract_text_content(msg)
                break
        return role, content

    # Filter new messages to only user/tool roles
    new_inputs = [m for m in new_messages if m.get("role", "") in ("user", "tool")]

    if not new_inputs:
        # New messages exist but none are user/tool (e.g. only assistant)
        for msg in reversed(messages):
            r = msg.get("role", "")
            if r in ("user", "tool"):
                role = r
                content = _extract_text_content(msg)
                break
        return role, content

    # Check if all new inputs are 'tool' role → merge into JSON array
    if all(m.get("role") == "tool" for m in new_inputs):
        tool_contents = []
        for m in new_inputs:
            c = m.get("content", "")
            try:
                parsed = json.loads(c)
                tool_contents.append(parsed)
            except (json.JSONDecodeError, TypeError):
                tool_contents.append(c)
        role = "tool"
        content = json.dumps(tool_contents, ensure_ascii=False)
    else:
        # Use the last new user/tool message
        last = new_inputs[-1]
        role = last.get("role", "user")
        content = _extract_text_content(last)

    return role, content


def _extract_text_content(msg):
    """Extract plain text content from a message dict."""
    c = msg.get("content", "")
    if isinstance(c, list):
        text_parts = [p.get("text", "") for p in c if p.get("type") == "text"]
        return " ".join(text_parts)
    return c or ""


def run_inference(rkllm_model, prompt, role="user", enable_thinking=False, sampling_params=None, max_new_tokens=None):
    """Run inference in a thread and wait for completion. Returns output text."""
    global global_text, global_state
    global_text = []
    global_state = -1

    model_thread = threading.Thread(target=rkllm_model.run, args=(role, enable_thinking, prompt, sampling_params, max_new_tokens))
    model_thread.start()

    output = ""
    model_thread_finished = False
    while not model_thread_finished:
        while len(global_text) > 0:
            output += global_text.pop(0)
            time.sleep(0.005)
        model_thread.join(timeout=0.005)
        model_thread_finished = not model_thread.is_alive()
    return output


def generate_stream(rkllm_model, prompt, role="user", enable_thinking=False, sampling_params=None, max_new_tokens=None):
    """Generator that yields output tokens as they are produced."""
    global global_text, global_state
    global_text = []
    global_state = -1

    model_thread = threading.Thread(target=rkllm_model.run, args=(role, enable_thinking, prompt, sampling_params, max_new_tokens))
    model_thread.start()

    model_thread_finished = False
    while not model_thread_finished:
        while len(global_text) > 0:
            yield global_text.pop(0)
        model_thread.join(timeout=0.005)
        model_thread_finished = not model_thread.is_alive()


# ---------------------------------------------------------------------------
# OpenAI-compatible routes
# ---------------------------------------------------------------------------

@app.route('/v1/models', methods=['GET'])
def list_models():
    """OpenAI-compatible model listing endpoint."""
    return jsonify({
        "object": "list",
        "data": [{
            "id": rkllm_model.model_name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "rkllm"
        }]
    })


@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """OpenAI-compatible chat completions endpoint."""
    global global_text, global_state, system_prompt, is_blocking

    # If the server is in a blocking state, return a specific response.
    if is_blocking or global_state == 0:
        return build_openai_error("RKLLM_Server is busy! Maybe you can try again later.", "server_error", 503)

    with lock:
        try:
            is_blocking = True
            data = request.json
            if not data or "messages" not in data:
                return build_openai_error("Missing 'messages' in request body.")

            messages = data["messages"]
            model_name = data.get("model", rkllm_model.model_name)
            stream = data.get("stream", False)
            temperature = data.get("temperature", 0.8)
            top_p = data.get("top_p", 0.9)
            top_k = data.get("top_k", 1)
            max_tokens = data.get("max_tokens", 4096)
            repeat_penalty = data.get("repeat_penalty", 1.1)
            frequency_penalty = data.get("frequency_penalty", 0.0)
            presence_penalty = data.get("presence_penalty", 0.0)
            enable_thinking = data.get("enable_thinking", False)
            tools = data.get("tools", None)

            # Build per-request sampling params
            sampling_params = RKLLMSamplingParam()
            sampling_params.top_k = int(top_k)
            sampling_params.top_p = float(top_p)
            sampling_params.temperature = float(temperature)
            sampling_params.repeat_penalty = float(repeat_penalty)
            sampling_params.frequency_penalty = float(frequency_penalty)
            sampling_params.presence_penalty = float(presence_penalty)
            sampling_params.mirostat = 0
            sampling_params.mirostat_tau = 5.0
            sampling_params.mirostat_eta = 0.1

            # Extract system prompt
            sys_prompt = ""
            for msg in messages:
                if msg.get("role") == "system":
                    sys_prompt = msg.get("content", "")

            # Configure function calling if tools are provided
            if tools:
                rkllm_model.set_function_tools(
                    system_prompt=sys_prompt,
                    tools=json.dumps(tools),
                    tool_response_str="tool_response"
                )

            # Build prompt from messages
            role, prompt = get_last_input(messages)
            if not stream:
                # Non-streaming mode
                output = run_inference(rkllm_model, prompt, role, enable_thinking, sampling_params, max_tokens)
                response = build_openai_response(
                    model_name=model_name,
                    content=output,
                    finish_reason="stop",
                    prompt_tokens=len(prompt.split()),
                    completion_tokens=len(output.split())
                )
                return jsonify(response)
            else:
                # Streaming mode (SSE)
                def sse_generate():
                    # Send initial chunk with role
                    yield build_openai_stream_chunk(model_name, None, None)

                    for token in generate_stream(rkllm_model, prompt, role, enable_thinking, sampling_params, max_tokens):
                        yield build_openai_stream_chunk(model_name, token, None)

                    # Send final chunk with finish_reason
                    yield build_openai_stream_chunk(model_name, "", "stop")
                    yield "data: [DONE]\n\n"

                return Response(
                    stream_with_context(sse_generate()),
                    content_type='text/event-stream',
                    headers={
                        'Cache-Control': 'no-cache',
                        'Connection': 'keep-alive',
                        'X-Accel-Buffering': 'no'
                    }
                )

        finally:
            is_blocking = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rkllm_model_path', type=str, required=True, help='Absolute path of the converted RKLLM model on the Linux board;')
    parser.add_argument('--target_platform', type=str, required=True, help='Target platform: e.g., rk3588/rk3576;')
    parser.add_argument('--lora_model_path', type=str, help='Absolute path of the lora_model on the Linux board;')
    parser.add_argument('--prompt_cache_path', type=str, help='Absolute path of the prompt_cache file on the Linux board;')
    args = parser.parse_args()

    if not os.path.exists(args.rkllm_model_path):
        print("Error: Please provide the correct rkllm model path, and ensure it is the absolute path on the board.")
        sys.stdout.flush()
        exit()

    if not (args.target_platform in ["rk3588", "rk3576", "rv1126b", "rk3562"]):
        print("Error: Please specify the correct target platform: rk3588/rk3576/rv1126b/rk3562.")
        sys.stdout.flush()
        exit()

    if args.lora_model_path:
        if not os.path.exists(args.lora_model_path):
            print("Error: Please provide the correct lora_model path, and advise it is the absolute path on the board.")
            sys.stdout.flush()
            exit()

    if args.prompt_cache_path:
        if not os.path.exists(args.prompt_cache_path):
            print("Error: Please provide the correct prompt_cache_file path, and advise it is the absolute path on the board.")
            sys.stdout.flush()
            exit()

    # Fix frequency
    command = "sudo bash fix_freq_{}.sh".format(args.target_platform)
    subprocess.run(command, shell=True)

    # Set resource limit
    resource.setrlimit(resource.RLIMIT_NOFILE, (102400, 102400))

    # Initialize RKLLM model
    print("=========init....===========")
    sys.stdout.flush()
    model_path = args.rkllm_model_path
    rkllm_model = RKLLM(model_path, args.lora_model_path, args.prompt_cache_path, args.target_platform)
    print(f"Model name: {rkllm_model.model_name}")
    print("==============================")
    sys.stdout.flush()

    # Graceful shutdown on Ctrl+C
    def shutdown_handler(signum, frame):
        print("\n====================")
        print("Received interrupt signal, releasing RKLLM model resources...")
        rkllm_model.release()
        print("====================")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    try:
        # Start the Flask application.
        app.run(host='0.0.0.0', port=8080, threaded=False, debug=False)
    finally:
        print("====================")
        print("RKLLM model inference completed, releasing RKLLM model resources...")
        rkllm_model.release()
        print("====================")
