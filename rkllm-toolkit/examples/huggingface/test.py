from rkllm.api import RKLLM

'''
https://huggingface.co/Qwen/Qwen-1_8B-Chat
Download the Qwen model from the above website.
'''

modelpath = '/path/to/your/model'
llm = RKLLM()

# Load model
ret = llm.load_huggingface(model = modelpath)
if ret != 0:
    print('Load model failed!')
    exit(ret)

# Build model
ret = llm.build(do_quantization=True, optimization_level=1, quantized_dtype='w8a8', target_platform='rk3588')
if ret != 0:
    print('Build model failed!')
    exit(ret)

# Export rknn model
ret = llm.export_rkllm("./qwen.rkllm")
if ret != 0:
    print('Export model failed!')
    exit(ret)


