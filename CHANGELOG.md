# CHANGELOG
## v1.0.1
 - Optimize model conversion memory occupation
 - Optimize inference memory occupation
 - Increase prefill speed
 - Reduce initialization time
 - Improve quantization accuracy
 - Add support for Gemma, ChatGLM3, MiniCPM, InternLM2, and Phi-3
 - Add Server invocation
 - Add inference interruption interface
 - Add logprob and token_id to the return value

## v1.0.0
 - Supports the conversion and deployment of LLM models on RK3588/RK3576 platforms
 - Compatible with Hugging Face model architectures
 - Currently supports the models Llama, Qwen, Qwen2, and Phi-2
 - Supports quantization with w8a8 and w4a16 precision