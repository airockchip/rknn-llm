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

# 创建一个锁，用于控制多人访问Server
lock = threading.Lock()

# 创建一个全局变量，用于标识服务器当前是否处于阻塞状态
is_blocking = False

# 设置动态库路径
rkllm_lib = ctypes.CDLL('lib/librkllmrt.so')

# 定义全局变量，用于保存回调函数的输出，便于在gradio界面中输出
global_text = []
global_state = -1
split_byte_data = bytes(b"") # 用于保存分割的字节数据

# 定义动态库中的结构体
class Token(ctypes.Structure):
    _fields_ = [
        ("logprob", ctypes.c_float),
        ("id", ctypes.c_int32)
    ]

class RKLLMResult(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("tokens", ctypes.POINTER(Token)),
        ("num", ctypes.c_int32)
    ]


# 定义回调函数
def callback(result, userdata, state):
    global global_text, global_state, split_byte_data
    if state == 0:
        # 保存输出的token文本及RKLLM运行状态
        global_state = state
        # 需要监控当前的字节数据是否完整，不完整则进行记录，后续进行解析
        try:
            global_text.append((split_byte_data + result.contents.text).decode('utf-8'))
            print((split_byte_data + result.contents.text).decode('utf-8'), end='')
            split_byte_data = bytes(b"")
        except:
            split_byte_data += result.contents.text
        sys.stdout.flush()
    elif state == 1:
        # 保存RKLLM运行状态
        global_state = state
        print("\n")
        sys.stdout.flush()
    else:
        print("run error")

# Python端与C++端的回调函数连接
callback_type = ctypes.CFUNCTYPE(None, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int)
c_callback = callback_type(callback)

# 定义动态库中的结构体
class RKNNllmParam(ctypes.Structure):
    _fields_ = [
        ("model_path", ctypes.c_char_p),
        ("num_npu_core", ctypes.c_int32),
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
        ("logprobs", ctypes.c_bool),
        ("top_logprobs", ctypes.c_int32),
        ("use_gpu", ctypes.c_bool)
    ]

# 定义RKLLM_Handle_t和userdata
RKLLM_Handle_t = ctypes.c_void_p
userdata = ctypes.c_void_p(None)

# 设置提示文本
PROMPT_TEXT_PREFIX = "<|im_start|>system You are a helpful assistant. <|im_end|> <|im_start|>user"
PROMPT_TEXT_POSTFIX = "<|im_end|><|im_start|>assistant"

# 定义Python端的RKLLM类，其中包括了对动态库中RKLLM模型的初始化、推理及释放操作
class RKLLM(object):
    def __init__(self, model_path, target_platform):
        rknnllm_param = RKNNllmParam()
        rknnllm_param.model_path = bytes(model_path, 'utf-8')
        if target_platform == "rk3588":
            rknnllm_param.num_npu_core = 3
        elif target_platform == "rk3576":
            rknnllm_param.num_npu_core = 1
        rknnllm_param.max_context_len = 320
        rknnllm_param.max_new_tokens = 512
        rknnllm_param.top_k = 1
        rknnllm_param.top_p = 0.9
        rknnllm_param.temperature = 0.8
        rknnllm_param.repeat_penalty = 1.1
        rknnllm_param.frequency_penalty = 0.0
        rknnllm_param.presence_penalty = 0.0
        rknnllm_param.mirostat = 0
        rknnllm_param.mirostat_tau = 5.0
        rknnllm_param.mirostat_eta = 0.1
        rknnllm_param.logprobs = False
        rknnllm_param.top_logprobs = 5
        rknnllm_param.use_gpu = True
        self.handle = RKLLM_Handle_t()

        self.rkllm_init = rkllm_lib.rkllm_init
        self.rkllm_init.argtypes = [ctypes.POINTER(RKLLM_Handle_t), ctypes.POINTER(RKNNllmParam), callback_type]
        self.rkllm_init.restype = ctypes.c_int
        self.rkllm_init(ctypes.byref(self.handle), rknnllm_param, c_callback)

        self.rkllm_run = rkllm_lib.rkllm_run
        self.rkllm_run.argtypes = [RKLLM_Handle_t, ctypes.POINTER(ctypes.c_char), ctypes.c_void_p]
        self.rkllm_run.restype = ctypes.c_int

        self.rkllm_destroy = rkllm_lib.rkllm_destroy
        self.rkllm_destroy.argtypes = [RKLLM_Handle_t]
        self.rkllm_destroy.restype = ctypes.c_int

    def run(self, prompt):
        prompt = bytes(PROMPT_TEXT_PREFIX + prompt + PROMPT_TEXT_POSTFIX, 'utf-8')
        self.rkllm_run(self.handle, prompt, ctypes.byref(userdata))
        return

    def release(self):
        self.rkllm_destroy(self.handle)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_platform', help='目标平台: 如rk3588/rk3576;')
    parser.add_argument('--rkllm_model_path', help='Linux板端上已转换好的rkllm模型的绝对路径')
    args = parser.parse_args()

    if not (args.target_platform in ["rk3588", "rk3576"]):
        print("====== Error: 请指定正确的目标平台: rk3588/rk3576 ======")
        sys.stdout.flush()
        exit()

    if not os.path.exists(args.rkllm_model_path):
        print("====== Error: 请给出准确的rkllm模型路径，需注意是板端的绝对路径 ======")
        sys.stdout.flush()
        exit()

    # 定频设置
    command = "sudo bash fix_freq_{}.sh".format(args.target_platform)
    subprocess.run(command, shell=True)

    # 设置文件描述符限制
    resource.setrlimit(resource.RLIMIT_NOFILE, (102400, 102400))

    # 初始化RKLLM模型
    print("=========init....===========")
    sys.stdout.flush()
    target_platform = args.target_platform
    model_path = args.rkllm_model_path
    rkllm_model = RKLLM(model_path, target_platform)
    print("RKLLM初始化成功！")
    print("==============================")
    sys.stdout.flush()

    # 创建一个函数用于接受用户使用 request 发送的数据
    @app.route('/rkllm_chat', methods=['POST'])
    def receive_message():
        # 链接全局变量，获取回调函数的输出信息
        global global_text, global_state
        global is_blocking

        # 如果服务器正在阻塞状态，则返回特定响应
        if is_blocking or global_state==0:
            return jsonify({'status': 'error', 'message': 'RKLLM_Server is busy! Maybe you can try again later.'}), 503
        
        # 加锁
        lock.acquire()
        try:
            # 设置服务器为阻塞状态
            is_blocking = True

            # 获取 POST 请求中的 JSON 数据
            data = request.json
            if data and 'messages' in data:
                # 重置全局变量
                global_text = []
                global_state = -1

                # 定义返回的结构体
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
                    # 在这里处理收到的数据
                    messages = data['messages']
                    print("Received messages:", messages)
                    for index, message in enumerate(messages):
                        input_prompt = message['content']
                        rkllm_output = ""
                        
                        # 创建模型推理的线程
                        model_thread = threading.Thread(target=rkllm_model.run, args=(input_prompt,))
                        model_thread.start()

                        # 等待模型运行完成，定时检查模型的推理线程
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
                    # 在这里处理收到的数据
                    messages = data['messages']
                    print("Received messages:", messages)
                    for index, message in enumerate(messages):
                        input_prompt = message['content']
                        rkllm_output = ""
                        
                        def generate():
                            # 创建模型推理的线程
                            model_thread = threading.Thread(target=rkllm_model.run, args=(input_prompt,))
                            model_thread.start()

                            # 等待模型运行完成，定时检查模型的推理线程
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
            # 释放锁
            lock.release()
            # 将服务器状态设置为非阻塞
            is_blocking = False
        
    # 启动 Flask 应用程序
    # app.run(host='0.0.0.0', port=8080)
    app.run(host='0.0.0.0', port=8080, threaded=True, debug=False)

    print("====================")
    print("RKLLM模型推理结束, 释放RKLLM模型资源...")
    rkllm_model.release()
    print("====================")
