import ctypes
import sys
import os
import subprocess
import resource
import threading
import time
import gradio as gr
import argparse

# 设定环境变量
os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"
os.environ["GRADIO_SERVER_PORT"] = "8080"

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
            rknnllm_param.num_npu_core = 2
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

    # 记录用户输入的prompt         
    def get_user_input(user_message, history):
        history = history + [[user_message, None]]
        return "", history

    # 获取RKLLM模型的输出并进行流式打印
    def get_RKLLM_output(history):
        # 链接全局变量，获取回调函数的输出信息
        global global_text, global_state
        global_text = []
        global_state = -1

        # 创建模型推理的线程
        model_thread = threading.Thread(target=rkllm_model.run, args=(history[-1][0],))
        model_thread.start()

        # history[-1][1]表示当前的输出对话
        history[-1][1] = ""
        
        # 等待模型运行完成，定时检查模型的推理线程
        model_thread_finished = False
        while not model_thread_finished:
            while len(global_text) > 0:
                history[-1][1] += global_text.pop(0)
                time.sleep(0.005)
                # gradio在调用then方法式自动将yield返回的结果推进行输出
                yield history

            model_thread.join(timeout=0.005)
            model_thread_finished = not model_thread.is_alive()

    # 创建gradio界面
    with gr.Blocks(title="Chat with RKLLM") as chatRKLLM:
        gr.Markdown("<div align='center'><font size='70'> Chat with RKLLM </font></div>")
        gr.Markdown("### 在 inputTextBox 输入您的问题，按下 Enter 键，即可与 RKLLM 模型进行对话。")
        # 创建一个Chatbot组件，用于显示对话历史
        rkllmServer = gr.Chatbot(height=600)
        # 创建一个Textbox组件，让用户输入消息
        msg = gr.Textbox(placeholder="Please input your question here...", label="inputTextBox")
        # 创建一个Button组件，用于清除聊天历史
        clear = gr.Button("清除")

        # 将用户输入的消息提交给get_user_input函数，并且立即更新聊天历史
        # 然后调用get_RKLLM_output函数，进一步更新聊天历史
        # queue=False参数确保这些更新不会被放入队列，而是立即执行
        msg.submit(get_user_input, [msg, rkllmServer], [msg, rkllmServer], queue=False).then(get_RKLLM_output, rkllmServer, rkllmServer)
        # 当点击清除按钮时，执行一个空操作（lambda: None），并且立即清除聊天历史
        clear.click(lambda: None, None, rkllmServer, queue=False)

    # 启用事件队列系统
    chatRKLLM.queue()
    # 启动Gradio应用程序
    chatRKLLM.launch()

    print("====================")
    print("RKLLM模型推理结束, 释放RKLLM模型资源...")
    rkllm_model.release()
    print("====================")