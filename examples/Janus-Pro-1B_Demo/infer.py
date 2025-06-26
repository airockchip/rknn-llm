from PIL import Image
import torch
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images

# 添加模型路径
model_path = "/path/to/Janus-Pro-1B/"

# 加载处理器和分词器
vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

# 加载模型
vl_gpt = MultiModalityCausalLM.from_pretrained(
    model_path, 
    trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

# 加载图像
images_path="data/demo.jpg"
# 构建对话格式
question = "请详细描述这张图片。"
conversation = [
    {
        "role": "<|User|>",
        "content": f"<image_placeholder>\n{question}",
        "images": [images_path],
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

# 生成回答
outputs = vl_gpt.language_model.generate(
    inputs_embeds=inputs_embeds,
    attention_mask=prepare_inputs.attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=256,
    do_sample=False,
    use_cache=True,
)

# 解码输出
answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
print(f"{prepare_inputs['sft_format'][0]}", answer)