import sys
import requests
import json
  
# 设置 Server 服务器的地址
server_url = 'http://172.16.10.102:8080/rkllm_chat'
# 设置是否开启流式对话
is_streaming = True

# 创建一个会话对象
session = requests.Session()
session.keep_alive = False  # 关闭连接池，保持长连接
adapter = requests.adapters.HTTPAdapter(max_retries=5)
session.mount('https://', adapter)
session.mount('http://', adapter)

if __name__ == '__main__':
    print("============================")
    print("在终端中输入您的问题，即可与 RKLLM 模型进行对话....")
    print("============================")
    # 进入循环，持续获取用户输入，并与RKLLM模型进行对话
    while True:
        try:
            user_message = input("请输入您的问题：")
            if user_message == "exit":
                print("============================")
                print("程序正在退出......")
                print("============================")
                break
            else:
                # 设置请求头，此处的请求头实际并无作用，仅为模拟OpenAI接口设计
                headers = {
                    'Content-Type': 'application/json',
                    'Authorization': 'not_required'
                }

                # 准备要发送的数据
                # model: 为用户在设置RKLLM-Server时定义的模型，此处并无作用
                # messages: 用户输入的问题，RKLLM-Server将会把它作为输入，并返回模型的回复；支持在 messags 加入多个问题
                # stream: 是否开启流式对话，与OpenAI接口相同
                data = {
                    "model": 'your_model_deploy_with_RKLLM_Server',
                    "messages": [{"role": "user", "content": user_message}],
                    "stream": is_streaming
                }

                # 发送 POST 请求
                responses = session.post(server_url, json=data, headers=headers, stream=is_streaming, verify=False)

                if not is_streaming:
                    # 解析响应
                    if responses.status_code == 200:
                        print("Q:", data["messages"][-1]["content"])
                        print("A:", json.loads(responses.text)["choices"][-1]["message"]["content"])
                    else:
                        print("Error:", responses.text)
                else:
                    if responses.status_code == 200:
                        print("Q:", data["messages"][-1]["content"])
                        print("A:", end="")
                        for line in responses.iter_lines():
                            if line:
                                line = json.loads(line.decode('utf-8'))
                                if line["choices"][-1]["finish_reason"] != "stop":
                                    print(line["choices"][-1]["delta"]["content"], end="")
                                    sys.stdout.flush()
                    else:
                        print('Error:', responses.text)


                
                
        except KeyboardInterrupt:
            # 捕获 Ctrl-C 信号，关闭会话
            session.close()

            print("\n")
            print("============================")
            print("程序正在退出......")
            print("============================")
            break
