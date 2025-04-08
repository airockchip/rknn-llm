from rkllm.api import RKLLM
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

'''
https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

Download the DeepSeek R1 model from the above url.
'''

modelpath = '/path/to/DeepSeek-R1-Distill-Qwen-1.5B'
llm = RKLLM()

# Load model
# Use 'export CUDA_VISIBLE_DEVICES=0' to specify GPU device
# device options ['cpu', 'cuda']
# dtype  options ['float32', 'float16', 'bfloat16']
# Using 'bfloat16' or 'float16' can significantly reduce memory consumption but at the cost of lower precision  
# compared to 'float32'. Choose the appropriate dtype based on your hardware and model requirements. 
ret = llm.load_huggingface(model=modelpath, model_lora = None, device='cuda', dtype="float32", custom_config=None, load_weight=True)
# ret = llm.load_gguf(model = modelpath)
if ret != 0:
    print('Load model failed!')
    exit(ret)

# Build model
dataset = "./data_quant.json"
# Json file format, please note to add prompt in the input，like this:
# [{"input":"Human: 你好！\nAssistant: ", "target": "你好！我是人工智能助手KK！"},...]
# Different quantization methods are optimized for different algorithms:  
# w8a8/w8a8_gx   is recommended to use the normal algorithm.  
# w4a16/w4a16_gx is recommended to use the grq algorithm.
qparams = None # Use extra_qparams
target_platform = "RK3588"
optimization_level = 1
quantized_dtype = "W8A8"
quantized_algorithm = "normal"
num_npu_core = 3

ret = llm.build(do_quantization=True, optimization_level=optimization_level, quantized_dtype=quantized_dtype,
                quantized_algorithm=quantized_algorithm, target_platform=target_platform, num_npu_core=num_npu_core, extra_qparams=qparams, dataset=dataset, hybrid_rate=0, max_context=4096)
if ret != 0:
    print('Build model failed!')
    exit(ret)

# Export rkllm model
ret = llm.export_rkllm(f"./{os.path.basename(modelpath)}_{quantized_dtype}_{target_platform}.rkllm")
if ret != 0:
    print('Export model failed!')
    exit(ret)
