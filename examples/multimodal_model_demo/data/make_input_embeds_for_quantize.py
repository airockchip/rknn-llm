import torch
import os
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
import json
import pickle
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoProcessor, Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration, Qwen3_5ForConditionalGeneration
import argparse

class StopForward(Exception):
    """Used to stop forward pass intentionally."""
    pass

argparse = argparse.ArgumentParser()
argparse.add_argument('--path', type=str, help='model path', required=True)
argparse.add_argument('--model_type', type=str, choices=['qwen2vl', 'qwen2.5vl', 'qwen3vl', 'qwen3.5'],
                      help='Model type to use', required=True)
args = argparse.parse_args()

## 模型类型映射
MODEL_CLASSES = {
    'qwen2vl': Qwen2VLForConditionalGeneration,
    'qwen2.5vl': Qwen2_5_VLForConditionalGeneration,
    'qwen3vl': Qwen3VLForConditionalGeneration,
    'qwen3.5': Qwen3_5ForConditionalGeneration,
}

## 加载模型
model_class = MODEL_CLASSES[args.model_type]
print(f"Loading model: {args.model_type} from {args.path}")
model = model_class.from_pretrained(
    args.path, torch_dtype="auto", device_map="cpu",
    trust_remote_code=True).eval()


processor = AutoProcessor.from_pretrained(args.path)

## 定义hook
captured = []
def model_pre_hook(module, args, kwargs):
    captured.append(kwargs)
    raise StopForward

handle = model.model.language_model.register_forward_pre_hook(
    model_pre_hook,
    with_kwargs=True
)


## 生成量化校准数据
info = []
info_path = "data/llm_inputs.json"
datasets = json.load(open("data/datasets.json", 'r'))
for idx, data in enumerate(tqdm(datasets)):
    image_name = data["image"].split(".")[0]
    imgp = os.path.join(data["image_path"], data["image"])
    image = Image.open(imgp)

    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {"type": "text", "text": data["input"]},
            ],
        }
    ]
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(
        text=[text_prompt], images=[image], padding=True, return_tensors="pt"
    )
    inputs = inputs.to(model.device)
    try:
        output = model(**inputs)
    except StopForward:
        pass
    temp = captured[-1]
    sample_name = "sample_{}".format(idx)
    os.makedirs("data/llm_inputs/", exist_ok=True)
    pickle_path = "data/llm_inputs/{}".format(sample_name)
    info.append({"sample":"llm_inputs/{}".format(sample_name), "token_nums":temp["inputs_embeds"].shape[1]})
    with open(pickle_path, 'wb') as f:
        pickle.dump(temp, f)
    captured.pop()
    
handle.remove()
with open(info_path, 'w', encoding='utf-8') as f:
    json.dump(info, f, indent=2, ensure_ascii=False)
