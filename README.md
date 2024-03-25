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

# Download
- You can also download all packages, docker image, examples, docs and platform-tools from [RKLLM_SDK](https://console.zbox.filez.com/l/RJJDmB), fetch code: rkllm

# RKNN Toolkit2
If you want to deploy additional AI model, we have introduced a new SDK called RKNN-Toolkit2. For details, please refer to:

https://github.com/airockchip/rknn-toolkit2

# Notes

Due to recent updates to the Phi2 model, the current version of the RKLLM SDK does not yet support these changes. 
Please ensure to download a version of the [Phi2](https://hf-mirror.com/microsoft/phi-2/tree/834565c23f9b28b96ccbeabe614dd906b6db551a) model that is supported. 

# CHANGELOG

## v1.0.0-beta
 - Supports the conversion and deployment of LLM models on RK3588/RK3576 platforms
 - Compatible with Hugging Face model architectures
 - Currently supports the models LLaMA, Qwen, Qwen2, and Phi-2
 - Supports quantization with w8a8 and w4a16 precision