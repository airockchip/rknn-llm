import os
import warnings
import argparse
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class InternVL3VisionEncoder(nn.Module):
    """
    Vision encoder module separated from InternVL3, containing:
    - vision_model (InternViT): visual Transformer encoder
    - pixel_shuffle: spatial downsampling
    - mlp1: projection layer from visual features to LLM embedding space
    """

    def __init__(self, model):
        super().__init__()
        self.vision_model = model.vision_model
        self.mlp1 = model.mlp1
        self.select_layer = model.select_layer
        self.downsample_ratio = model.downsample_ratio
        self.ps_version = model.ps_version

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(
            n,
            int(h * scale_factor),
            int(w * scale_factor),
            int(c / (scale_factor * scale_factor)),
        )
        if self.ps_version == "v1":
            warnings.warn(
                "In ps_version 'v1', the height and width have not been swapped back, "
                "which results in a transposed image."
            )
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def forward(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values, output_hidden_states=False, return_dict=True
            ).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values, output_hidden_states=True, return_dict=True
            ).hidden_states[self.select_layer]

        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds


def main():
    parser = argparse.ArgumentParser(
        description="export InternVL3 vision encoder to ONNX format"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="OpenGVLab/InternVL3-1B",
        help="model path of InternVL3 (HuggingFace or local path)",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="onnx/internvl3_vision.onnx",
        help="save path of exported ONNX file",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="batch size (corresponding to the number of image tiles)",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=448,
        help="input image size (InternVL3 default 448)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="inference device, cpu or cuda"
    )
    parser.add_argument(
        "--opset_version", type=int, default=15, help="ONNX opset version"
    )
    parser.add_argument(
        "--dynamic_batch",
        action="store_true",
        help="whether to use dynamic batch dimension",
    )
    args = parser.parse_args()

    print(f"loading model: {args.model_path}")
    print(
        f"note: using float32 precision, because the edge-end RKNN only supports float32"
    )

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    config.vision_config.use_flash_attn = False

    model = (
        AutoModel.from_pretrained(
            args.model_path,
            config=config,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        .eval()
        .to(args.device)
    )

    vision_encoder = InternVL3VisionEncoder(model).to(torch.float32).eval()

    print(f"\n===== model structure =====")
    print(f"select_layer: {vision_encoder.select_layer}")
    print(f"downsample_ratio: {vision_encoder.downsample_ratio}")
    print(f"ps_version: {vision_encoder.ps_version}")

    image_size = args.image_size
    patch_size = config.vision_config.patch_size
    num_patches = (image_size // patch_size) ** 2
    num_output_tokens = int(num_patches * (vision_encoder.downsample_ratio**2))
    vit_hidden_size = config.vision_config.hidden_size
    llm_hidden_size = config.llm_config.hidden_size

    print(f"image_size: {image_size}")
    print(f"patch_size: {patch_size}")
    print(f"num_patches (ViT): {num_patches}")
    print(f"num_output_tokens (after pixel_shuffle): {num_output_tokens}")
    print(f"vit_hidden_size: {vit_hidden_size}")
    print(f"llm_hidden_size (output dimension): {llm_hidden_size}")

    pixel_values = torch.randn(
        args.batch_size,
        3,
        image_size,
        image_size,
        device=args.device,
        dtype=torch.float32,
    )

    print(f"\n===== test forward inference =====")
    with torch.no_grad():
        output = vision_encoder(pixel_values)
    print(f"input pixel_values shape: {pixel_values.shape}")
    print(f"output vit_embeds shape:   {output.shape}")
    print(
        f"  -> [batch_size={output.shape[0]}, num_tokens={output.shape[1]}, hidden_dim={output.shape[2]}]"
    )

    os.makedirs(
        os.path.dirname(args.save_path) if os.path.dirname(args.save_path) else ".",
        exist_ok=True,
    )

    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {
            "pixel_values": {0: "batch_size"},
            "visual_features": {0: "batch_size"},
        }

    print(f"\n===== export ONNX =====")
    print(f"save path: {args.save_path}")
    print(f"opset_version: {args.opset_version}")
    print(f"dynamic_batch: {args.dynamic_batch}")

    torch.onnx.export(
        vision_encoder,
        pixel_values,
        args.save_path,
        input_names=["pixel_values"],
        output_names=["visual_features"],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset_version,
        do_constant_folding=True,
    )


if __name__ == "__main__":
    main()
