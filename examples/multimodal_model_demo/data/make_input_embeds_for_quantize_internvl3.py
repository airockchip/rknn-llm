import argparse
import copy
import json
import os

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_START_TOKEN = "<image>"
IMG_END_TOKEN = "</image>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate InternVL3-1B calibration/validation dataset."
    )
    parser.add_argument(
        "--path",
        type=str,
        default="OpenGVLab/InternVL3-1B",
        help="InternVL3-1B model path.",
    )
    return parser.parse_args()


def resolve_torch_dtype(dtype_name):
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_name]


def build_transform(input_size):
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    grid_w = target_width // image_size
    for i in range(blocks):
        box = (
            (i % grid_w) * image_size,
            (i // grid_w) * image_size,
            ((i % grid_w) + 1) * image_size,
            ((i // grid_w) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))

    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(tile) for tile in images]
    return torch.stack(pixel_values)


def build_query(model, tokenizer, question, num_patches):
    if "<image>" not in question:
        question = "<image>\n" + question

    template = copy.deepcopy(model.conv_template)
    template.system_message = model.system_message
    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()

    image_tokens = (
        IMG_START_TOKEN + IMG_CONTEXT_TOKEN * model.num_image_token * num_patches + IMG_END_TOKEN
    )
    query = query.replace("<image>", image_tokens, 1)
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    if img_context_token_id is None or img_context_token_id < 0:
        raise ValueError(f"Failed to resolve token id for {IMG_CONTEXT_TOKEN}.")
    model.img_context_token_id = img_context_token_id
    return query


@torch.inference_mode()
def make_input_embeds(model, tokenizer, pixel_values, question):
    num_patches = pixel_values.shape[0]
    query = build_query(model, tokenizer, question, num_patches)
    model_inputs = tokenizer(query, return_tensors="pt")
    input_ids = model_inputs["input_ids"].to(model.device)

    input_embeds = model.language_model.get_input_embeddings()(input_ids).clone()
    visual_embeds = model.extract_feature(pixel_values).to(input_embeds.device)

    selected = input_ids.reshape(-1) == model.img_context_token_id
    flat_input_embeds = input_embeds.reshape(-1, input_embeds.shape[-1])
    flat_visual_embeds = visual_embeds.reshape(-1, visual_embeds.shape[-1])

    if selected.sum().item() != flat_visual_embeds.shape[0]:
        raise ValueError(
            f"Image token count mismatch: selected={selected.sum().item()}, "
            f"visual_embeds={flat_visual_embeds.shape[0]}"
        )

    flat_input_embeds[selected] = flat_visual_embeds.to(flat_input_embeds.dtype)
    input_embeds = flat_input_embeds.reshape_as(input_embeds)
    return input_embeds, query, num_patches


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    demo_dir = os.path.dirname(script_dir)
    dataset_path = os.path.join(script_dir, "datasets.json")
    output_json = os.path.join(script_dir, "inputs_internvl3.json")
    embed_dir = os.path.join(script_dir, "inputs_embeds_internvl3")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dtype = resolve_torch_dtype("bfloat16" if torch.cuda.is_available() else "float32")
    max_num = 12
    save_dtype = np.float16

    with open(dataset_path, "r", encoding="utf-8") as f:
        datasets = json.load(f)

    model = AutoModel.from_pretrained(
        args.path,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.path, trust_remote_code=True, use_fast=False
    )

    input_size = getattr(model.config, "force_image_size", None) or model.config.vision_config.image_size

    os.makedirs(embed_dir, exist_ok=True)
    output_records = []

    for item in tqdm(datasets, desc="Generating InternVL3 calibration data"):
        image_name = os.path.splitext(item["image"])[0]
        image_path = os.path.join(demo_dir, item["image_path"], item["image"])
        pixel_values = load_image(image_path, input_size=input_size, max_num=max_num)
        pixel_values = pixel_values.to(device=device, dtype=model_dtype)

        input_embeds, query, num_patches = make_input_embeds(
            model, tokenizer, pixel_values, item["input"]
        )

        embed_path = os.path.join(embed_dir, f"{image_name}.npy")
        embed_np = input_embeds.detach().cpu().to(torch.float32).numpy().astype(save_dtype)
        np.save(embed_path, embed_np)

        output_records.append(
            {
                "input_embed": embed_np.tolist(),
                "target": item["target"],
                "image": item["image"],
                "num_patches": num_patches,
                "prompt": query.replace(IMG_CONTEXT_TOKEN, ""),
            }
        )

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output_records, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(output_records)} samples to {output_json}")
    print(f"Per-sample embeddings saved in {embed_dir}")


if __name__ == "__main__":
    main()
