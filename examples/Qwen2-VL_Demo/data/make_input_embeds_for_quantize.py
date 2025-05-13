import torch
import os
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
import json
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoProcessor, Qwen2VLForConditionalGeneration
import argparse

argparse = argparse.ArgumentParser()
argparse.add_argument('--path', type=str, default='Qwen/Qwen2-VL-2B-Instruct', help='model path', required=False)
args = argparse.parse_args()

path = args.path
model = Qwen2VLForConditionalGeneration.from_pretrained(
    path, torch_dtype="auto", device_map="cpu",
    low_cpu_mem_usage=True,
    trust_remote_code=True).eval()

processor = AutoProcessor.from_pretrained(path)

datasets = json.load(open("data/datasets.json", 'r'))
for data in datasets:
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
    inputs_embeds = model.model.embed_tokens(inputs["input_ids"])
    pixel_values = inputs["pixel_values"].type(model.visual.get_dtype())
    image_mask = inputs["input_ids"] == model.config.image_token_id
    image_embeds = model.visual(pixel_values, grid_thw=inputs["image_grid_thw"]).to(inputs_embeds.device)
    inputs_embeds[image_mask] = image_embeds
    print("inputs_embeds", inputs_embeds.shape)
    os.makedirs("data/inputs_embeds/", exist_ok=True)
    np.save("data/inputs_embeds/{}".format(image_name), inputs_embeds.to(dtype=torch.float16).cpu().detach().numpy())
    
with open('data/inputs.json', 'w') as json_file:
    json_file.write('[\n')
    first = True
    for data in tqdm(datasets):
        input_embed = np.load(os.path.join("data/inputs_embeds", data["image"].split(".")[0]+'.npy'))
        target = data["target"]
        input_dict = {
            "input_embed": input_embed.tolist(),
            "target": target
        }
        if not first:
            json_file.write(',\n')
        else:
            first = False
        json.dump(input_dict, json_file)
    json_file.write('\n]')

print("Done")
