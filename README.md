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
  - [X] [LLAMA models](https://huggingface.co/meta-llama) 
  - [X] [TinyLLAMA models](https://huggingface.co/TinyLlama) 
  - [X] [Qwen models](https://huggingface.co/models?search=Qwen/Qwen)
  - [X] [Phi models](https://huggingface.co/models?search=microsoft/phi)
  - [X] [ChatGLM3-6B](https://huggingface.co/THUDM/chatglm3-6b/tree/103caa40027ebfd8450289ca2f278eac4ff26405)
  - [X] [Gemma models](https://huggingface.co/collections/google/gemma-2-release-667d6600fd5220e7b967f315)
  - [X] [InternLM2 models](https://huggingface.co/collections/internlm/internlm2-65b0ce04970888799707893c)
  - [X] [MiniCPM models](https://huggingface.co/collections/openbmb/minicpm-65d48bf958302b9fd25b698f)

# Download
You can download the latest package, docker image, example, documentation, and platform-tool from [RKLLM_SDK](https://console.zbox.filez.com/l/RJJDmB), fetch code: rkllm

# Note

The modifications in version 1.1.0 are significant, making it incompatible with older version models. Please use the latest toolchain for model conversion and inference.

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