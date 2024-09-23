FROM python:3.12-slim AS base

WORKDIR /app

COPY examples/rkllm_server_demo/rkllm_server /app
COPY runtime/Linux/librkllm_api/aarch64/librkllmrt.so /app/lib

ENV PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
EXPOSE 8080


FROM base AS flask-server

RUN pip install "flask==2.2.2" "Werkzeug==2.2.2"

ENTRYPOINT [ "python3", "-m", "flask_server" ]


FROM base AS gradio-server

RUN pip install "gradio>=4.24.0"

ENTRYPOINT [ "python3", "-m", "gradio_server" ]
