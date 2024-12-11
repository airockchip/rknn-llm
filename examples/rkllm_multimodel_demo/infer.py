from PIL import Image
import requests
import torch
from torchvision import io
from typing import Dict
from transformers import AutoModel, AutoProcessor, Qwen2VLForConditionalGeneration

# Load the model in half-precision on the available device(s)
path = "/home/mnt/bd_mount/models/Qwen2-VL-2B-Instruct"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    path, torch_dtype="auto", device_map="auto",
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(path)

image = Image.open('./data/demo.jpg')
image = image.resize((392, 392), 3)
conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Preprocess the inputs
text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
# Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n'

inputs = processor(
    text=[text_prompt], images=[image], padding=True, return_tensors="pt"
)
inputs = inputs.to("cuda")
print("inputs.pixel_values: ", inputs['pixel_values'].shape)

# Inference: Generation of the output
output_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids = [
    output_ids[len(input_ids) :]
    for input_ids, output_ids in zip(inputs.input_ids, output_ids)
]
output_text = processor.batch_decode(
    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
)
print(output_text)
