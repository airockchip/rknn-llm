import os
from rkllm.api import RKLLM
from datasets import load_dataset
from transformers import  AutoTokenizer
from tqdm import tqdm
import torch
from torch import nn
import argparse

argparse = argparse.ArgumentParser()
argparse.add_argument('--path', type=str, default='Qwen/Qwen2-VL-2B-Instruct', help='model path', required=False)
argparse.add_argument('--target-platform', type=str, default='rk3588', help='target platform', required=False)
argparse.add_argument('--num_npu_core', type=int, default=3, help='npu core num', required=False)
argparse.add_argument('--quantized_dtype', type=str, default='w8a8', help='quantized dtype', required=False)
argparse.add_argument('--device', type=str, default='cpu', help='device', required=False)
argparse.add_argument('--savepath', type=str, default='qwen2_vl_2b_instruct.rkllm', help='save path', required=False)
args = argparse.parse_args()

modelpath = args.path
target_platform = args.target_platform
num_npu_core = args.num_npu_core
quantized_dtype = args.quantized_dtype
savepath = args.savepath
llm = RKLLM()

# Load model
# Use 'export CUDA_VISIBLE_DEVICES=2' to specify GPU device
ret = llm.load_huggingface(model=modelpath, device=args.device)
if ret != 0:
    print('Load model failed!')
    exit(ret)

# Build model
dataset = 'data/inputs.json'

qparams = None
ret = llm.build(do_quantization=True, optimization_level=1, quantized_dtype=quantized_dtype,
                quantized_algorithm='normal', target_platform=target_platform, num_npu_core=num_npu_core, extra_qparams=qparams, dataset=dataset)

if ret != 0:
    print('Build model failed!')
    exit(ret)

# # Export rkllm model
ret = llm.export_rkllm(savepath)
if ret != 0:
    print('Export model failed!')
    exit(ret)


