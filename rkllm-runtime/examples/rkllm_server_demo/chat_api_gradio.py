from gradio_client import Client

# 该函数通过调用Gradio Client API与RKLLM模型进行交互
def chat_with_rkllm(user_message, history=[]):
    # 实例化Gradio Client，用户需要根据自己部署的具体网址进行修改
    client = Client("http://172.16.10.102:8080/")

    # 调用Gradio Client API进行交互，内部的API主要包括：
    # /get_user_input：模型获取用户输入，并将输入添加至历史记录history
    # /get_RKLLM_output：RKLLM利用已包含输入的历史记录history生成回复
    _, history = client.predict(user_message=user_message, history=history, api_name="/get_user_input")
    result_history = client.predict(history=history, api_name="/get_RKLLM_output")
    return result_history

if __name__ == '__main__':
    #初始化聊天记录
    result_history = []

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
                # 调用chat_with_rkllm函数，获取模型的回复
                result_history = chat_with_rkllm(user_message, result_history)

                # 打印模型输出
                print("Q:", result_history[-1][0])
                print("A:", result_history[-1][1])
        except KeyboardInterrupt:
            print("\n")
            print("============================")
            print("程序正在退出......")
            print("============================")
            break
            