#!/bin/bash

#*****************************************************************************************#
# 该脚本为 RKLLM-Server-Gradio 服务的一键设置脚本
# 用户可以运行该脚本实现Linux板端的 RKLLM-Server-Gradio 服务的自动化部署。
# 使用说明: ./build_rkllm_server_gradio.sh [目标平台:rk3588/rk3576] [RKLLM-Server工作路径] [已转换的rkllm模型在板端的绝对路径]
# example: ./build_rkllm_server_gradio.sh rk3588 /user/data/rkllm_server /user/data/rkllm_server/model.rkllm
#*****************************************************************************************#

#################### 检查板端是否已经安装了 pip/gradio 库 ####################
# 1.准备板端的gradio环境
adb shell << EOF

# 检查是否安装了 pip3
if ! command -v pip3 &> /dev/null; then
    echo "-------- pip3 未安装，将进行安装... --------"
    # 安装 pip3
    sudo apt update
    sudo apt install python3-pip -y
else
    echo "-------- pip3 已经安装 --------"
fi

# 检查是否安装了 gradio
if ! python3 -c "import gradio" &> /dev/null; then
    echo "-------- Gradio 未安装，将进行安装... --------"
    # 安装 Gradio
    pip3 install gradio>=4.24.0 -i https://pypi.tuna.tsinghua.edu.cn/simple/
else
    echo "-------- Gradio 已经安装 --------"
fi

exit

EOF

#################### 推送 server 运行的相关文件进入板端 ####################
# 2.检查需要推送进板端的路径是否存在
adb shell ls $2 > /dev/null 2>&1
if [ $? -ne 0 ]; then
    # 如果路径不存在，则创建路径
    adb shell mkdir -p $2
    echo "-------- rkllm_server 工作目录不存在，已创建目录 --------"
else
    echo "-------- rkllm_server 工作目录已存在 --------"
fi

# 3.更新 ./rkllm_server/lib 中的 librkllmrt.so 文件
cp ../../runtime/Linux/librkllm_api/aarch64/librkllmrt.so  ./rkllm_server/lib/

# 4.推送文件到 Linux 板端
adb push ./rkllm_server $2

#################### 进入板端并启动 server 服务 ####################
# 5.进入板端启动 server 服务
adb shell << EOF

cd $2/rkllm_server/
python3 gradio_server.py --target_platform $1 --rkllm_model_path $3

EOF
