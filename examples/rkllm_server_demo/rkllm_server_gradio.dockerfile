FROM python:3.12-slim

WORKDIR /app

RUN pip install gradio>=4.24.0 -i https://pypi.tuna.tsinghua.edu.cn/simple/

COPY examples/rkllm_server_demo/rkllm_server /app
COPY runtime/Linux/librkllm_api/aarch64/librkllmrt.so /app/lib

EXPOSE 8080

ENTRYPOINT [ "python3", "-m", "gradio_server" ]
