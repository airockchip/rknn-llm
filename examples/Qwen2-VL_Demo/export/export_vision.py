import numpy as np
import os
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer
import torch.nn.functional as F
import argparse

argparse = argparse.ArgumentParser()
argparse.add_argument('--step', type=int, help='export step', required=True)
argparse.add_argument('--path', type=str, default='Qwen/Qwen2-VL-2B-Instruct', help='model path', required=False)
argparse.add_argument('--batch', type=int, default=1, help='batch size', required=False)
argparse.add_argument('--height', type=int, default=392, help='image height', required=False)
argparse.add_argument('--width', type=int, default=392, help='image width', required=False)
argparse.add_argument('--savepath', type=str, default='qwen2-vl-2b/qwen2_vl_2b_vision.onnx', help='save path', required=False)
args = argparse.parse_args()

step = args.step
# 加载本地模型
path = args.path
model = Qwen2VLForConditionalGeneration.from_pretrained(
    path,
    torch_dtype=torch.float32, # 注意此处的数据类型，由于 rknn 目前仅支持 float32 ，因此需要指定；若是在加载权重时限制了数据类型，需要自行修改config.json中的 "use_flash_attn" 参数为 false
    low_cpu_mem_usage=True,
    trust_remote_code=True).eval()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

N = args.batch                           # batch size
channel = 3                                 # 3 for RGB
H = args.height                         # image height, must be divisible by (merge_size * patch_size)
W = args.width                          # image width, must be divisible by (merge_size * patch_size)
merge_size = 2
temporal_patch_size = 2
patch_size = 14
grid_t = N // temporal_patch_size if N%temporal_patch_size == 0 else N // temporal_patch_size + 1
grid_h = H // patch_size
grid_w = W // patch_size

def export_onnx(image):
    if N == 1:
        images = image.repeat(temporal_patch_size, 1, 1, 1)
    elif N % temporal_patch_size != 0:
        repeat_time = temporal_patch_size - N % temporal_patch_size
        repeat_image = image[-1:, ...].repeat(repeat_time, 1, 1, 1)
        images = torch.cat((image, repeat_image), dim=0)
    patches = images.reshape(grid_t, temporal_patch_size, channel, grid_h//merge_size, merge_size, patch_size, grid_w//merge_size, merge_size, patch_size)
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flatten_patches = patches.reshape(grid_t * grid_h * grid_w, channel * temporal_patch_size * patch_size * patch_size)
    model.visual.forward = forward_new(model.visual)
    if step == 1:
        feature = model.visual(flatten_patches, torch.tensor([grid_t, grid_h, grid_w]).unsqueeze(0))
    else:
        feature = model.visual(flatten_patches)
    return feature

def forward_new(self):
    def tmp (hidden_states, grid_thw=None):
        hidden_states = self.patch_embed(hidden_states)
        if grid_thw is not None:
            rotary_pos_emb = self.rot_pos_emb(grid_thw)
            cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
                dim=0, dtype=torch.int32
            )
            cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
            np.save("./rotary_pos_emb.npy", rotary_pos_emb.cpu().detach().numpy())
            np.save("./cu_seqlens.npy", cu_seqlens.cpu().detach().numpy())
        else:
            rotary_pos_emb = torch.from_numpy(np.load("./rotary_pos_emb.npy")).to(dtype=hidden_states.dtype, device=hidden_states.device)
            cu_seqlens = torch.from_numpy(np.load("./cu_seqlens.npy")).to(dtype=torch.int32, device=hidden_states.device)
        
        for blk in self.blocks:
            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)

        return self.merger(hidden_states)
    return tmp

# 导出 Vison 部分所对应的 onnx 模型，假设输入是2x3x392x392->(28x28)x(3x2x14x14)
# pixel_values = torch.randn(784, 1176, device="cuda", dtype=torch.float32)
pixel_values = torch.randn(N, channel, H, W, device="cpu", dtype=torch.float32)
model.forward = export_onnx
model = model.to(torch.float32).eval()
if step == 1:
    print("==========================================================")
    print("Generating the rotary_pos_emb and cu_seqlens done.")
    feature = model(pixel_values)
else:
    print("==========================================================")
    print(f"Exporting the vision part of {path} to onnx format.")
    os.makedirs(os.path.dirname(args.savepath), exist_ok=True)
    torch.onnx.export(model, pixel_values, args.savepath, opset_version=18)
