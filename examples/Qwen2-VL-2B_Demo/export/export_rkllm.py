import os
from rkllm.api import RKLLM
from datasets import load_dataset
from transformers import  AutoTokenizer
from tqdm import tqdm
import torch
from torch import nn

modelpath = "/path/to/Qwen2-VL-2B-Instruct" 
savepath = './Qwen2-VL-2B-Instruct.rkllm'
llm = RKLLM()

# Load model
# Use 'export CUDA_VISIBLE_DEVICES=2' to specify GPU device
ret = llm.load_huggingface(model=modelpath, device='cpu')
if ret != 0:
    print('Load model failed!')
    exit(ret)

# Build model
dataset = 'data/inputs.json'

qparams = None
ret = llm.build(do_quantization=True, optimization_level=1, quantized_dtype='w8a8',
                quantized_algorithm='normal', target_platform='rk3588', num_npu_core=3, extra_qparams=qparams, dataset=dataset)

if ret != 0:
    print('Build model failed!')
    exit(ret)

# # Export rkllm model
ret = llm.export_rkllm(savepath)
if ret != 0:
    print('Export model failed!')
    exit(ret)


