# RKLLM-Server Demo
## Before Run
Before running the demo, you need to prepare the following files:
- The transformed RKLLM model file in board.
- check the IP address of the board with 'ifconfig' command.
  
## RKLLM-Server-Flask Demo
### Build
You can run the demo with the only command:
```bash
# Usage: ./build_rkllm_server_flask.sh --workshop [RKLLM-Server Working Path] --model_path [Absolute Path of Converted RKLLM Model on Board] --platform [Target Platform: rk3588/rk3576] [--lora_model_path [Lora Model Path]] [--prompt_cache_path [Prompt Cache File Path]]
./build_rkllm_server_flask.sh --workshop /user/data --model_path /user/data/model.rkllm --platform rk3588
```
### Access with API 
After building the RKLLM-Server-Flask, You can use ‘chat_api_flask.py’ to access the RKLLM-Server-Flask and get the answser of RKLLM models.

Attention: you should check the IP address of the board with 'ifconfig' command and replace the IP address in the ‘chat_api_flask.py’.

## RKLLM-Server-Gradio Demo
### Build
You can run the demo with the only command:
```bash
# Usage: ./build_rkllm_server_gradio.sh --workshop [RKLLM-Server Working Path] --model_path [Absolute Path of Converted RKLLM Model on Board] --platform [Target Platform: rk3588/rk3576] [--lora_model_path [Lora Model Path]] [--prompt_cache_path [Prompt Cache File Path]]
./build_rkllm_server_gradio.sh --workshop /user/data --model_path /user/data/model.rkllm --platform rk3588
```
### Access the Server
After running the demo, You can access the RKLLM-Server-Gradio with two ways:
1. Just Start your browser and access the URL: ‘http://[board_ip]:8080/’. You can chat with the RKLLM models in visual interface.
2. Use the 'chat_api_gradio.py'(you need fix the IP address in the code previously) and get the answser of RKLLM models.
   