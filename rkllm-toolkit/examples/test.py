from rkllm.api import RKLLM
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import torch
from torch import nn
import os
# os.environ['CUDA_VISIBLE_DEVICES']='1'

'''
https://huggingface.co/Qwen/Qwen-1_8B-Chat
从上面网址中下载Qwen模型
'''

modelpath = './path/to/model'
# modelpath = "./path/to/Qwen-1.8B-F16.gguf"
llm = RKLLM()

# Load model
# Use 'export CUDA_VISIBLE_DEVICES=2' to specify GPU device
# options ['cpu', 'cuda']
ret = llm.load_huggingface(model=modelpath, model_lora = None, device='cpu')
# ret = llm.load_gguf(model = modelpath)
if ret != 0:
    print('Load model failed!')
    exit(ret)

# Build model
dataset = "./data_quant.json"
# Json file format, please note to add prompt in the input，like this:
# [{"input":"Human: 你好！\nAssistant: ", "target": "你好！我是人工智能助手KK！"},...]

qparams = None
# qparams = 'gdq.qparams' # Use extra_qparams
ret = llm.build(do_quantization=True, optimization_level=1, quantized_dtype='w8a8',
                quantized_algorithm='normal', target_platform='rk3588', num_npu_core=3, extra_qparams=qparams, dataset=dataset)

if ret != 0:
    print('Build model failed!')
    exit(ret)

# Evaluate Accuracy
def eval_wikitext(llm):
    seqlen = 512
    tokenizer = AutoTokenizer.from_pretrained(
        modelpath, trust_remote_code=True)
    # Dataset download link:
    # https://huggingface.co/datasets/Salesforce/wikitext/tree/main/wikitext-2-raw-v1
    testenc = load_dataset(
        "parquet", data_files='./wikitext/wikitext-2-raw-1/test-00000-of-00001.parquet', split='train')
    testenc = tokenizer("\n\n".join(
        testenc['text']), return_tensors="pt").input_ids
    nsamples = testenc.numel() // seqlen
    nlls = []
    for i in tqdm(range(nsamples), desc="eval_wikitext: "):
        batch = testenc[:, (i * seqlen): ((i + 1) * seqlen)]
        inputs = {"input_ids": batch}
        lm_logits = llm.get_logits(inputs)
        if lm_logits is None:
            print("get logits failed!")
            return
        shift_logits = lm_logits[:, :-1, :]
        shift_labels = batch[:, 1:].to(lm_logits.device)
        loss_fct = nn.CrossEntropyLoss().to(lm_logits.device)
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    print(f'wikitext-2-raw-1-test ppl: {round(ppl.item(), 2)}')

# eval_wikitext(llm)


# Chat with model
messages = "<|im_start|>system You are a helpful assistant.<|im_end|><|im_start|>user你好！\n<|im_end|><|im_start|>assistant"
kwargs = {"max_length": 128, "top_k": 1, "top_p": 0.8,
          "temperature": 0.8, "do_sample": True, "repetition_penalty": 1.1}
# print(llm.chat_model(messages, kwargs))


# Export rkllm model
ret = llm.export_rkllm("./qwen.rkllm")
if ret != 0:
    print('Export model failed!')
    exit(ret)
