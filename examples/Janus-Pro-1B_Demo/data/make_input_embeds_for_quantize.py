from PIL import Image
import torch
import sys
import os
import json
import numpy as np
from tqdm import tqdm
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images

# 设置模型路径
model_path = "/path/to/Janus-Pro-1B/"

# 加载处理器和分词器
vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

# 加载模型并确保所有参数为float16
vl_gpt = MultiModalityCausalLM.from_pretrained(
    model_path, 
    trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

# 加载数据集
datasets = json.load(open("data/datasets.json", 'r'))

# 创建保存嵌入的目录
os.makedirs("data/inputs_embeds/", exist_ok=True)

# 遍历数据集进行推理
for data in tqdm(datasets):
    image_name = data["image"].split(".")[0]
    imgp = os.path.join(data["image_path"], data["image"])
    image = Image.open(imgp)

    # 构建对话格式
    question = data["input"]
    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image>\n{question}",
            "images": [imgp],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    # 处理输入
    pil_images = load_pil_images(conversation)
    prepare_inputs = vl_chat_processor(
        conversations=conversation, 
        images=pil_images, 
        force_batchify=True,
    ).to(vl_gpt.device)

    # 准备输入嵌入
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    # 保存嵌入数据
    np.save("data/inputs_embeds/{}".format(image_name), inputs_embeds.to(dtype=torch.float16).cpu().detach().numpy())

# 创建并写入JSON文件
with open('data/inputs.json', 'w') as json_file:
    json_file.write('[\n')
    first = True
    for data in tqdm(datasets):
        # 读取保存的嵌入数据
        input_embed = np.load(os.path.join("data/inputs_embeds", data["image"].split(".")[0] + '.npy'))
        target = data["target"]
        
        # 构建保存的数据字典
        input_dict = {
            "input_embed": input_embed.tolist(),
            "target": target
        }
        
        # 写入到JSON文件中
        if not first:
            json_file.write(',\n')
        else:
            first = False
        json.dump(input_dict, json_file)
    
    json_file.write('\n]')
    
print("Done")
