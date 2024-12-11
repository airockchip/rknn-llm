import ctypes
import sys
import os
import subprocess
import resource
import threading
import time
import argparse
import json
from flask import Flask, request, jsonify, Response

app = Flask(__name__)

PROMPT_TEXT_PREFIX = "<|im_start|>system You are a helpful assistant. <|im_end|> <|im_start|>user"
PROMPT_TEXT_POSTFIX = "<|im_end|><|im_start|>assistant"

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
LLMCallState.RKLLM_RUN_GET_LAST_HIDDEN_LAYER = 4

RKLLMInputMode = ctypes.c_int
RKLLMInputMode.RKLLM_INPUT_PROMPT      = 0
RKLLMInputMode.RKLLM_INPUT_TOKEN       = 1
RKLLMInputMode.RKLLM_INPUT_EMBED       = 2
RKLLMInputMode.RKLLM_INPUT_MULTIMODAL  = 3

RKLLMInferMode = ctypes.c_int
RKLLMInferMode.RKLLM_INFER_GENERATE = 0
RKLLMInferMode.RKLLM_INFER_GET_LAST_HIDDEN_LAYER = 1

class RKLLMExtendParam(ctypes.Structure):
    _fields_ = [
        ("base_domain_id", ctypes.c_int32),
        ("reserved", ctypes.c_uint8 * 112)
    ]

class RKLLMParam(ctypes.Structure):
    _fields_ = [
        ("model_path", ctypes.c_char_p),
        ("max_context_len", ctypes.c_int32),
        ("max_new_tokens", ctypes.c_int32),
        ("top_k", ctypes.c_int32),
        ("top_p", ctypes.c_float),
        ("temperature", ctypes.c_float),
        ("repeat_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("mirostat", ctypes.c_int32),
        ("mirostat_tau", ctypes.c_float),
        ("mirostat_eta", ctypes.c_float),
        ("skip_special_token", ctypes.c_bool),
        ("is_async", ctypes.c_bool),
        ("img_start", ctypes.c_char_p),
        ("img_end", ctypes.c_char_p),
        ("img_content", ctypes.c_char_p),
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

class RKLLMMultiModelInput(ctypes.Structure):
    _fields_ = [
        ("prompt", ctypes.c_char_p),
        ("image_embed", ctypes.POINTER(ctypes.c_float)),
        ("n_image_tokens", ctypes.c_size_t)
    ]

class RKLLMInputUnion(ctypes.Union):
    _fields_ = [
        ("prompt_input", ctypes.c_char_p),
        ("embed_input", RKLLMEmbedInput),
        ("token_input", RKLLMTokenInput),
        ("multimodal_input", RKLLMMultiModelInput)
    ]

class RKLLMInput(ctypes.Structure):
    _fields_ = [
        ("input_mode", ctypes.c_int),
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

class RKLLMInferParam(ctypes.Structure):
    _fields_ = [
        ("mode", RKLLMInferMode),
        ("lora_params", ctypes.POINTER(RKLLMLoraParam)),
        ("prompt_cache_params", ctypes.POINTER(RKLLMPromptCacheParam))
    ]

class RKLLMResultLastHiddenLayer(ctypes.Structure):
    _fields_ = [
        ("hidden_states", ctypes.POINTER(ctypes.c_float)),
        ("embd_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int)
    ]

class RKLLMResult(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("size", ctypes.c_int),
        ("last_hidden_layer", RKLLMResultLastHiddenLayer)
    ]


# Create a lock to control multi-user access to the server.
lock = threading.Lock()

# Create a global variable to indicate whether the server is currently in a blocked state.
is_blocking = False

# Define global variables to store the callback function output for displaying in the Gradio interface
global_text = []
global_state = -1
split_byte_data = bytes(b"") # Used to store the segmented byte data

# Define the callback function
def callback_impl(result, userdata, state):
    global global_text, global_state, split_byte_data
    if state == LLMCallState.RKLLM_RUN_FINISH:
        global_state = state
        print("\n")
        sys.stdout.flush()
    elif state == LLMCallState.RKLLM_RUN_ERROR:
        global_state = state
        print("run error")
        sys.stdout.flush()
    elif state == LLMCallState.RKLLM_RUN_GET_LAST_HIDDEN_LAYER:
        '''
        If using the GET_LAST_HIDDEN_LAYER function, the callback interface will return the memory pointer: last_hidden_layer, the number of tokens: num_tokens, and the size of the hidden layer: embd_size.
        With these three parameters, you can retrieve the data from last_hidden_layer.
        Note: The data needs to be retrieved during the current callback; if not obtained in time, the pointer will be released by the next callback.
        '''
        if result.last_hidden_layer.embd_size != 0 and result.last_hidden_layer.num_tokens != 0:
            data_size = result.last_hidden_layer.embd_size * result.last_hidden_layer.num_tokens * ctypes.sizeof(ctypes.c_float)
            print(f"data_size: {data_size}")
            global_text.append(f"data_size: {data_size}\n")
            output_path = os.getcwd() + "/last_hidden_layer.bin"
            with open(output_path, "wb") as outFile:
                data = ctypes.cast(result.last_hidden_layer.hidden_states, ctypes.POINTER(ctypes.c_float))
                float_array_type = ctypes.c_float * (data_size // ctypes.sizeof(ctypes.c_float))
                float_array = float_array_type.from_address(ctypes.addressof(data.contents))
                outFile.write(bytearray(float_array))
                print(f"Data saved to {output_path} successfully!")
                global_text.append(f"Data saved to {output_path} successfully!")
        else:
            print("Invalid hidden layer data.")
            global_text.append("Invalid hidden layer data.")
        global_state = state
        time.sleep(0.05) # Delay for 0.05 seconds to wait for the output result
        sys.stdout.flush()
    else:
        # Save the output token text and the RKLLM running state
        global_state = state
        # Monitor if the current byte data is complete; if incomplete, record it for later parsing
        try:
            global_text.append((split_byte_data + result.contents.text).decode('utf-8'))
            print((split_byte_data + result.contents.text).decode('utf-8'), end='')
            split_byte_data = bytes(b"")
        except:
            split_byte_data += result.contents.text
        sys.stdout.flush()

# Connect the callback function between the Python side and the C++ side
callback_type = ctypes.CFUNCTYPE(None, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int)
callback = callback_type(callback_impl)

# Define the RKLLM class, which includes initialization, inference, and release operations for the RKLLM model in the dynamic library
class RKLLM(object):
    def __init__(self, model_path, lora_model_path = None, prompt_cache_path = None):
        rkllm_param = RKLLMParam()
        rkllm_param.model_path = bytes(model_path, 'utf-8')

        rkllm_param.max_context_len = 512
        rkllm_param.max_new_tokens = -1
        rkllm_param.skip_special_token = True

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

        rkllm_param.img_start = "".encode('utf-8')
        rkllm_param.img_end = "".encode('utf-8')
        rkllm_param.img_content = "".encode('utf-8')

        rkllm_param.extend_param.base_domain_id = 0
        
        self.handle = RKLLM_Handle_t()

        self.rkllm_init = rkllm_lib.rkllm_init
        self.rkllm_init.argtypes = [ctypes.POINTER(RKLLM_Handle_t), ctypes.POINTER(RKLLMParam), callback_type]
        self.rkllm_init.restype = ctypes.c_int
        self.rkllm_init(ctypes.byref(self.handle), ctypes.byref(rkllm_param), callback)

        self.rkllm_run = rkllm_lib.rkllm_run
        self.rkllm_run.argtypes = [RKLLM_Handle_t, ctypes.POINTER(RKLLMInput), ctypes.POINTER(RKLLMInferParam), ctypes.c_void_p]
        self.rkllm_run.restype = ctypes.c_int

        self.rkllm_destroy = rkllm_lib.rkllm_destroy
        self.rkllm_destroy.argtypes = [RKLLM_Handle_t]
        self.rkllm_destroy.restype = ctypes.c_int

        self.lora_adapter_path = None
        self.lora_model_name = None
        if lora_model_path:
            self.lora_adapter_path = lora_model_path
            self.lora_adapter_name = "test"

            lora_adapter = RKLLMLoraAdapter()
            ctypes.memset(ctypes.byref(lora_adapter), 0, ctypes.sizeof(RKLLMLoraAdapter))
            lora_adapter.lora_adapter_path = ctypes.c_char_p((self.lora_adapter_path).encode('utf-8'))
            lora_adapter.lora_adapter_name = ctypes.c_char_p((self.lora_adapter_name).encode('utf-8'))
            lora_adapter.scale = 1.0

            rkllm_load_lora = rkllm_lib.rkllm_load_lora
            rkllm_load_lora.argtypes = [RKLLM_Handle_t, ctypes.POINTER(RKLLMLoraAdapter)]
            rkllm_load_lora.restype = ctypes.c_int
            rkllm_load_lora(self.handle, ctypes.byref(lora_adapter))
        
        self.prompt_cache_path = None
        if prompt_cache_path:
            self.prompt_cache_path = prompt_cache_path

            rkllm_load_prompt_cache = rkllm_lib.rkllm_load_prompt_cache
            rkllm_load_prompt_cache.argtypes = [RKLLM_Handle_t, ctypes.c_char_p]
            rkllm_load_prompt_cache.restype = ctypes.c_int
            rkllm_load_prompt_cache(self.handle, ctypes.c_char_p((prompt_cache_path).encode('utf-8')))

    def run(self, prompt):
        rkllm_lora_params = None
        if self.lora_model_name:
            rkllm_lora_params = RKLLMLoraParam()
            rkllm_lora_params.lora_adapter_name = ctypes.c_char_p((self.lora_model_name).encode('utf-8'))
        
        rkllm_infer_params = RKLLMInferParam()
        ctypes.memset(ctypes.byref(rkllm_infer_params), 0, ctypes.sizeof(RKLLMInferParam))
        rkllm_infer_params.mode = RKLLMInferMode.RKLLM_INFER_GENERATE
        rkllm_infer_params.lora_params = ctypes.byref(rkllm_lora_params) if rkllm_lora_params else None

        rkllm_input = RKLLMInput()
        rkllm_input.input_mode = RKLLMInputMode.RKLLM_INPUT_PROMPT
        rkllm_input.input_data.prompt_input = ctypes.c_char_p((PROMPT_TEXT_PREFIX + prompt + PROMPT_TEXT_POSTFIX).encode('utf-8'))
        self.rkllm_run(self.handle, ctypes.byref(rkllm_input), ctypes.byref(rkllm_infer_params), None)
        return

    def release(self):
        self.rkllm_destroy(self.handle)

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

    if not (args.target_platform in ["rk3588", "rk3576"]):
        print("Error: Please specify the correct target platform: rk3588/rk3576.")
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
    rkllm_model = RKLLM(model_path, args.lora_model_path, args.prompt_cache_path)
    print("RKLLM Model has been initialized successfully！")
    print("==============================")
    sys.stdout.flush()

    # Create a function to receive data sent by the user using a request
    @app.route('/rkllm_chat', methods=['POST'])
    def receive_message():
        # Link global variables to retrieve the output information from the callback function
        global global_text, global_state
        global is_blocking

        # If the server is in a blocking state, return a specific response.
        if is_blocking or global_state==0:
            return jsonify({'status': 'error', 'message': 'RKLLM_Server is busy! Maybe you can try again later.'}), 503
        
        lock.acquire()
        try:
            # Set the server to a blocking state.
            is_blocking = True

            # Get JSON data from the POST request.
            data = request.json
            if data and 'messages' in data:
                # Reset global variables.
                global_text = []
                global_state = -1

                # Define the structure for the returned response.
                rkllm_responses = {
                    "id": "rkllm_chat",
                    "object": "rkllm_chat",
                    "created": None,
                    "choices": [],
                    "usage": {
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None
                    }
                }

                if not "stream" in data.keys() or data["stream"] == False:
                    # Process the received data here.
                    messages = data['messages']
                    print("Received messages:", messages)
                    for index, message in enumerate(messages):
                        input_prompt = message['content']
                        rkllm_output = ""
                        
                        # Create a thread for model inference.
                        model_thread = threading.Thread(target=rkllm_model.run, args=(input_prompt,))
                        model_thread.start()

                        # Wait for the model to finish running and periodically check the inference thread of the model.
                        model_thread_finished = False
                        while not model_thread_finished:
                            while len(global_text) > 0:
                                rkllm_output += global_text.pop(0)
                                time.sleep(0.005)

                            model_thread.join(timeout=0.005)
                            model_thread_finished = not model_thread.is_alive()
                        
                        rkllm_responses["choices"].append(
                            {"index": index,
                            "message": {
                                "role": "assistant",
                                "content": rkllm_output,
                            },
                            "logprobs": None,
                            "finish_reason": "stop"
                            }
                        )
                    return jsonify(rkllm_responses), 200
                else:
                    messages = data['messages']
                    print("Received messages:", messages)
                    for index, message in enumerate(messages):
                        input_prompt = message['content']
                        rkllm_output = ""
                        
                        def generate():
                            model_thread = threading.Thread(target=rkllm_model.run, args=(input_prompt,))
                            model_thread.start()

                            model_thread_finished = False
                            while not model_thread_finished:
                                while len(global_text) > 0:
                                    rkllm_output = global_text.pop(0)

                                    rkllm_responses["choices"].append(
                                        {"index": index,
                                        "delta": {
                                            "role": "assistant",
                                            "content": rkllm_output,
                                        },
                                        "logprobs": None,
                                        "finish_reason": "stop" if global_state == 1 else None,
                                        }
                                    )
                                    yield f"{json.dumps(rkllm_responses)}\n\n"

                                model_thread.join(timeout=0.005)
                                model_thread_finished = not model_thread.is_alive()

                    return Response(generate(), content_type='text/plain')
            else:
                return jsonify({'status': 'error', 'message': 'Invalid JSON data!'}), 400
        finally:
            lock.release()
            is_blocking = False
        
    # Start the Flask application.
    # app.run(host='0.0.0.0', port=8080)
    app.run(host='0.0.0.0', port=8080, threaded=True, debug=False)

    print("====================")
    print("RKLLM model inference completed, releasing RKLLM model resources...")
    rkllm_model.release()
    print("====================")
