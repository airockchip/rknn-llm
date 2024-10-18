#!/bin/bash

#*****************************************************************************************#
# This script is an automated setup script for the RKLLM-Server-Flask service.
# Users can run this script to automate the deployment of the RKLLM-Server-Flask service on a Linux board.
# Usage: ./build_rkllm_server_flask.sh --workshop [RKLLM-Server Working Path] --model_path [Absolute Path of Converted RKLLM Model on Board] --platform [Target Platform: rk3588/rk3576] [--lora_model_path [Lora Model Path]] [--prompt_cache_path [Prompt Cache File Path]]
# example: ./build_rkllm_server_flask.sh --workshop /user/data --model_path /user/data/model.rkllm --platform rk3588
#*****************************************************************************************#

LORA_PATH=""
PROMPT_FILE_PATH=""

# Function to display help
function show_help {
    echo "Usage: ./build_rkllm_server_flask.sh --workshop [RKLLM-Server Working Path] --model_path [Absolute Path of Converted RKLLM Model on Board] --platform [Target Platform: rk3588/rk3576] [--lora_path [Lora Model Path]] [--prompt_cache_path [Prompt Cache File Path]]"
}

# Parse command-line options
while [[ $# -gt 0 ]]; do
    case "$1" in
        --workshop)
            WORKING_PATH="$2"
            shift 2
            ;;
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --platform)
            TARGET_PLATFORM="$2"
            shift 2
            ;;
        --lora_model_path)
            LORA_PATH="$2"
            shift 2
            ;;
        --prompt_cache_path)
            PROMPT_FILE_PATH="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *) 
            echo "无效的选项: $1" 1>&2
            show_help
            exit 1
            ;;
    esac
done

#################### Check if pip and the Flask library are already installed on the board. ####################
adb shell << EOF

if ! command -v pip3 &> /dev/null; then
    echo "-------- pip3 is not installed. Installing it now... --------"
    sudo apt update
    sudo apt install python3-pip -y
else
    echo "-------- pip3 is already installed. --------"
fi

if ! python3 -c "import flask" &> /dev/null; then
    echo "-------- flask is not installed. Installing it now... --------"
    # 安装 flask
    pip install flask==2.2.2 Werkzeug==2.2.2 -i https://pypi.tuna.tsinghua.edu.cn/simple --break-system-packages
else
    echo "-------- flask is already installed. --------"
fi

exit

EOF

#################### Push the relevant files for the server to the board. ####################
adb shell ls $WORKING_PATH > /dev/null 2>&1

if [ $? -ne 0 ]; then
    adb shell mkdir -p $WORKING_PATH
    echo "-------- The rkllm_server working directory does not exist, so it has been created. --------"
else
    echo "-------- The rkllm_server working directory already exists. --------"
fi

# Update the `librkllmrt.so` file in the `./rkllm_server/lib` directory.
cp ../../runtime/Linux/librkllm_api/aarch64/librkllmrt.so  ./rkllm_server/lib/

adb push ./rkllm_server $WORKING_PATH

#################### Enter the board terminal and start the server service. ####################
CMD="python3 flask_server.py --rkllm_model_path $MODEL_PATH --target_platform $TARGET_PLATFORM"
if [[ -n "$LORA_PATH" ]]; then
    CMD="$CMD --lora_model_path $LORA_PATH"
fi

if [[ -n "$PROMPT_FILE_PATH" ]]; then
    CMD="$CMD --prompt_cache_path $PROMPT_FILE_PATH"
fi

adb shell << EOF

cd $WORKING_PATH/rkllm_server/
$CMD

EOF
