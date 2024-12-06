# Description

  RKLLM software stack can help users to quickly deploy AI models to Rockchip chips. The overall framework is as follows:
    <center class="half">
        <div style="background-color:#ffffff;">
        <img src="res/framework.jpg" title="RKLLM"/>
    </center>

  In order to use RKNPU, users need to first run the RKLLM-Toolkit tool on the computer, convert the trained model into an RKLLM format model, and then inference on the development board using the RKLLM C API.

- RKLLM-Toolkit is a software development kit for users to perform model conversionand quantization on PC.

- RKLLM Runtime provides C/C++ programming interfaces for Rockchip NPU platform to help users deploy RKLLM models and accelerate the implementation of LLM applications.

- RKNPU kernel driver is responsible for interacting with NPU hardware. It has been open source and can be found in the Rockchip kernel code.

# Support Platform

- RK3588 Series
- RK3576 Series

# Support Models

- [x] [LLAMA models](https://huggingface.co/meta-llama) 
- [x] [TinyLLAMA models](https://huggingface.co/TinyLlama) 
- [x] [Qwen models](https://huggingface.co/models?search=Qwen/Qwen)
- [x] [Phi models](https://huggingface.co/models?search=microsoft/phi)
- [x] [ChatGLM3-6B](https://huggingface.co/THUDM/chatglm3-6b/tree/103caa40027ebfd8450289ca2f278eac4ff26405)
- [x] [Gemma models](https://huggingface.co/collections/google/gemma-2-release-667d6600fd5220e7b967f315)
- [x] [InternLM2 models](https://huggingface.co/collections/internlm/internlm2-65b0ce04970888799707893c)
- [x] [MiniCPM models](https://huggingface.co/collections/openbmb/minicpm-65d48bf958302b9fd25b698f)
- [x] [TeleChat models](https://huggingface.co/Tele-AI)

# Model Performance Benchmark

| model          | dtype      | seqlen | max_context | new_tokens | TTFT(ms) | Tokens/s | memory(G) | platform |
|:-------------- |:---------- |:------:|:-----------:|:----------:|:--------:|:--------:|:---------:|:--------:|
| TinyLLAMA-1.1B | w4a16      | 64     | 320         | 256        | 345.00   | 21.10    | 0.77      | RK3576   |
|                | w4a16_g128 | 64     | 320         | 256        | 410.00   | 18.50    | 0.8       | RK3576   |
|                | w8a8       | 64     | 320         | 256        | 140.46   | 24.21    | 1.25      | RK3588   |
|                | w8a8_g512  | 64     | 320         | 256        | 195.00   | 20.08    | 1.29      | RK3588   |
| Qwen2-1.5B     | w4a16      | 64     | 320         | 256        | 512.00   | 14.40    | 1.75      | RK3576   |
|                | w4a16_g128 | 64     | 320         | 256        | 550.00   | 12.75    | 1.76      | RK3576   |
|                | w8a8       | 64     | 320         | 256        | 206.00   | 16.46    | 2.47      | RK3588   |
|                | w8a8_g128  | 64     | 320         | 256        | 725.00   | 7.00     | 2.65      | RK3588   |
| Phi-3-3.8B     | w4a16      | 64     | 320         | 256        | 975.00   | 6.60     | 2.16      | RK3576   |
|                | w4a16_g128 | 64     | 320         | 256        | 1180.00  | 5.85     | 2.23      | RK3576   |
|                | w8a8       | 64     | 320         | 256        | 516.00   | 7.44     | 3.88      | RK3588   |
|                | w8a8_g512  | 64     | 320         | 256        | 610.00   | 6.13     | 3.95      | RK3588   |
| ChatGLM3-6B    | w4a16      | 64     | 320         | 256        | 1168.00  | 4.62     | 3.86      | RK3576   |
|                | w4a16_g128 | 64     | 320         | 256        | 1582.56  | 3.82     | 3.96      | RK3576   |
|                | w8a8       | 64     | 320         | 256        | 800.00   | 4.95     | 6.69      | RK3588   |
|                | w8a8_g128  | 64     | 320         | 256        | 2190.00  | 2.70     | 7.18      | RK3588   |
| Gemma2-2B      | w4a16      | 64     | 320         | 256        | 628.00   | 8.00     | 3.63      | RK3576   |
|                | w4a16_g128 | 64     | 320         | 256        | 776.20   | 7.40     | 3.63      | RK3576   |
|                | w8a8       | 64     | 320         | 256        | 342.29   | 9.67     | 4.84      | RK3588   |
|                | w8a8_g128  | 64     | 320         | 256        | 1055.00  | 5.49     | 5.14      | RK3588   |
| InternLM2-1.8B | w4a16      | 64     | 320         | 256        | 475.00   | 13.30    | 1.59      | RK3576   |
|                | w4a16_g128 | 64     | 320         | 256        | 572.00   | 11.95    | 1.62      | RK3576   |
|                | w8a8       | 64     | 320         | 256        | 205.97   | 15.66    | 2.38      | RK3588   |
|                | w8a8_g512  | 64     | 320         | 256        | 298.00   | 12.66    | 2.45      | RK3588   |
| MiniCPM3-4B    | w4a16      | 64     | 320         | 256        | 1397.00  | 4.80     | 2.7       | RK3576   |
|                | w4a16_g128 | 64     | 320         | 256        | 1645.00  | 4.39     | 2.8       | RK3576   |
|                | w8a8       | 64     | 320         | 256        | 702.18   | 6.15     | 4.65      | RK3588   |
|                | w8a8_g128  | 64     | 320         | 256        | 1691.00  | 3.42     | 5.06      | RK3588   |
| llama3-8B      | w4a16      | 64     | 320         | 256        | 1607.98  | 3.60     | 5.63      | RK3576   |
|                | w4a16_g128 | 64     | 320         | 256        | 2010.00  | 3.00     | 5.76      | RK3576   |
|                | w8a8       | 64     | 320         | 256        | 1128.00  | 3.79     | 9.21      | RK3588   |
|                | w8a8_g512  | 64     | 320         | 256        | 1281.35  | 3.05     | 9.45      | RK3588   |

- This performance data were collected based on the maximum CPU and NPU frequencies of each platform with version 1.1.0. 
- The script for setting the frequencies is located in the scripts directory.

# Download

You can download the latest package, docker image, example, documentation, and platform-tool from [RKLLM_SDK](https://console.zbox.filez.com/l/RJJDmB), fetch code: rkllm

# Note

- The modifications in version 1.1 are significant, making it incompatible with older version models. Please use the latest toolchain for model conversion and inference.

- The supported Python versions are:
  
  - Python 3.8
  
  - Python 3.10

- Latest version: [ <u>v1.1.3](https://github.com/airockchip/rknn-llm/releases/tag/release-v1.1.3)</u>

# RKNN Toolkit2

If you want to deploy additional AI model, we have introduced a SDK called RKNN-Toolkit2. For details, please refer to:

https://github.com/airockchip/rknn-toolkit2

# CHANGELOG

## v1.1.0

- Support group-wise quantization (w4a16 group sizes of 32/64/128, w8a8 group sizes of 128/256/512).
- Support joint inference with LoRA model loading
- Support storage and preloading of prompt cache.
- Support gguf model conversion (currently only support q4_0 and fp16).
- Optimize initialization, prefill, and decode time.
- Support four input types: prompt, embedding, token, and multimodal.
- Add PC-based simulation accuracy testing and inference interface support for rkllm-toolkit.
- Add gdq algorithm to improve 4-bit quantization accuracy.
- Add mixed quantization algorithm, supporting a combination of grouped and non-grouped quantization based on specified ratios.
- Add support for models such as Llama3, Gemma2, and MiniCPM3.
- Resolve catastrophic forgetting issue when the number of tokens exceeds max_context.

for older version, please refer [CHANGELOG](CHANGELOG.md)