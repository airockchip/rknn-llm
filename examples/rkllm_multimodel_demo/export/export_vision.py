import numpy as np
import os
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer
import torch.nn.functional as F

# 加载本地模型
path = '/path/to/Qwen2-VL-2B-Instruct'
model = Qwen2VLForConditionalGeneration.from_pretrained(
    path,
    torch_dtype=torch.float32, # 注意此处的数据类型，由于 rknn 目前仅支持 float32 ，因此需要指定；若是在加载权重时限制了数据类型，需要自行修改config.json中的 "use_flash_attn" 参数为 false
    low_cpu_mem_usage=True,
    trust_remote_code=True).eval()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

grid_t = 1
grid_h = 28
grid_w = 28
merge_size = 2
channel = 3
temporal_patch_size = 2
patch_size = 14

def export_onnx(image):
    patches = image.repeat(temporal_patch_size, 1, 1, 1)
    patches = patches.reshape(grid_t, temporal_patch_size, channel, grid_h//merge_size, merge_size, patch_size, grid_w//merge_size, merge_size, patch_size)
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flatten_patches = patches.reshape(grid_t * grid_h * grid_w, channel * temporal_patch_size * patch_size * patch_size)
    model.visual.forward = forward_new(model.visual)
    # feature = model.visual(flatten_patches, torch.tensor([grid_t, grid_h, grid_w]).unsqueeze(0))
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
pixel_values = torch.randn(1, 3, 392, 392, device="cpu", dtype=torch.float32)
model.forward = export_onnx
model = model.to(torch.float32).eval()
os.makedirs("onnx", exist_ok=True)
torch.onnx.export(model, pixel_values, "./onnx/qwen2_vl_2b_vision.onnx", opset_version=18)
