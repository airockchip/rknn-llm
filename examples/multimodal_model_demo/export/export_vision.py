import torch
import numpy as np
import os
import math
import argparse
import torch.nn.functional as F
from transformers import AutoModel

class minicpm_v_2_6_vision(torch.nn.Module):
    def __init__(self, vlm, batch_size, in_h, in_w):
        super(minicpm_v_2_6_vision, self).__init__()
        self.vpm = vlm.vpm
        self.resampler = vlm.resampler
        patch_size = vlm.config.patch_size
        num_patches_per_side = vlm.vpm.embeddings.num_patches_per_side
        tgt_sizes = torch.Tensor([[(in_h // patch_size), math.ceil(in_w / patch_size)]]).type(torch.int32)
        patch_attention_mask = torch.ones(
            size=(batch_size, in_h // patch_size, in_w // patch_size),
            dtype=torch.bool, device=vlm.device,
        )
        max_im_h, max_im_w = in_h, in_w
        max_nb_patches_h, max_nb_patches_w = max_im_h // patch_size, max_im_w // patch_size
        boundaries = torch.arange(1 / num_patches_per_side, 1.0, 1 / num_patches_per_side)
        position_ids = torch.full(
            size=(batch_size, max_nb_patches_h * max_nb_patches_w),
            fill_value=0,
        )
        for batch_idx, p_attn_mask in enumerate(patch_attention_mask):
            if tgt_sizes is not None:
                nb_patches_h = tgt_sizes[batch_idx][0]
                nb_patches_w = tgt_sizes[batch_idx][1]
            else:
                nb_patches_h = p_attn_mask[:, 0].sum()
                nb_patches_w = p_attn_mask[0].sum()

            fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / nb_patches_h)
            fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / nb_patches_w)

            bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
            bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)

            pos_ids = (bucket_coords_h[:, None] * num_patches_per_side + bucket_coords_w).flatten()
            position_ids[batch_idx][p_attn_mask.view(-1).cpu()] = pos_ids

            position_ids = position_ids.to(vlm.device)
        self.position_ids = position_ids
        
        patch_len = tgt_sizes[:, 0] * tgt_sizes[:, 1]
        max_patch_len = torch.max(patch_len)
        key_padding_mask = torch.zeros((batch_size, max_patch_len), dtype=torch.bool, device=vlm.device)
        pos_embed = []
        for i in range(batch_size):
            tgt_h, tgt_w = tgt_sizes[i]
            pos_embed.append(self.resampler.pos_embed[:tgt_h, :tgt_w, :].reshape((tgt_h * tgt_w, -1)).to(torch.float32))  # patches * D
            key_padding_mask[i, patch_len[i]:] = True

        self.pos_embed = torch.nn.utils.rnn.pad_sequence(
            pos_embed, batch_first=True, padding_value=0.0).permute(1, 0, 2)  # BLD => L * B * D
    
    def forward(self, pixel_values):
        batch_size = pixel_values.size(0)
        # patch embedding
        patch_embeds = self.vpm.embeddings.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        hidden_states = embeddings + self.vpm.embeddings.position_embedding(self.position_ids)
        # encoder
        encoder_outputs = self.vpm.encoder(inputs_embeds=hidden_states)
        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.vpm.post_layernorm(last_hidden_state)
        # resampler
        x = self.resampler.kv_proj(last_hidden_state)  # B * L * D
        x = self.resampler.ln_kv(x).permute(1, 0, 2)  # L * B * D

        q = self.resampler.ln_q(self.resampler.query)  # Q * D

        out = self.resampler.attn(
            self.resampler._repeat(q, batch_size),  # Q * B * D
            x + self.pos_embed,  # L * B * D +  L * B * D
            x)[0]
        #  out: Q * B * D
        x = out.permute(1, 0, 2)  # B * Q * D

        x = self.resampler.ln_post(x)
        x = x @ self.resampler.proj
        return x

class qwen2_5_vl_3b_vision(torch.nn.Module):
    def __init__(self, vlm, batch_size):
        super(qwen2_5_vl_3b_vision, self).__init__()
        self.merge_size = 2
        self.temporal_patch_size = 2
        self.patch_size = 14
        self.channel = 3
        self.vpm = vlm.visual
        self.batch_size = batch_size

    def forward(self, pixel_value, grid_thw):
        if self.batch_size == 1:
            patches = pixel_value.repeat(self.temporal_patch_size, 1, 1, 1)
        elif self.batch_size % self.temporal_patch_size == 1:
            repeat_image = pixel_value[-1:, ...].repeat(2, 1, 1, 1)
            patches = torch.cat((pixel_value, repeat_image), dim=0)
        else:
            patches = pixel_value
        grid_t, grid_h, grid_w = grid_thw[0][0], grid_thw[0][1], grid_thw[0][2]
        patches = patches.reshape(grid_t, self.temporal_patch_size, self.channel, 
                                  grid_h//self.merge_size, self.merge_size, self.patch_size, grid_w//self.merge_size, self.merge_size, self.patch_size)
        patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = patches.reshape(grid_t * grid_h * grid_w, self.channel * self.temporal_patch_size * self.patch_size * self.patch_size)
        
        return self.vpm(flatten_patches, grid_thw)

class qwen3_vl_vision(torch.nn.Module):
    def __init__(self, vlm, batch_size):
        super(qwen3_vl_vision, self).__init__()
        self.merge_size = 2
        self.temporal_patch_size = 2
        self.patch_size = 16
        self.channel = 3
        self.vpm = vlm.visual
        self.batch_size = batch_size

    def forward(self, pixel_value, grid_thw):
        if self.batch_size == 1:
            patches = pixel_value.repeat(self.temporal_patch_size, 1, 1, 1)
        elif self.batch_size % self.temporal_patch_size == 1:
            repeat_image = pixel_value[-1:, ...].repeat(2, 1, 1, 1)
            patches = torch.cat((pixel_value, repeat_image), dim=0)
        else:
            patches = pixel_value
        grid_t, grid_h, grid_w = grid_thw[0][0], grid_thw[0][1], grid_thw[0][2]
        patches = patches.reshape(grid_t, self.temporal_patch_size, self.channel, 
                                  grid_h//self.merge_size, self.merge_size, self.patch_size, grid_w//self.merge_size, self.merge_size, self.patch_size)
        patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = patches.reshape(grid_t * grid_h * grid_w, self.channel * self.temporal_patch_size * self.patch_size * self.patch_size)
        
        return self.vpm(flatten_patches, grid_thw)

class qwen3_5_vl_vision(torch.nn.Module):
    def __init__(self, vlm, batch_size):
        super(qwen3_5_vl_vision, self).__init__()
        self.merge_size = 2
        self.temporal_patch_size = 2
        self.patch_size = 16
        self.channel = 3
        self.vpm = vlm.visual
        self.batch_size = batch_size

    def forward(self, pixel_value, grid_thw):
        if self.batch_size == 1:
            patches = pixel_value.repeat(self.temporal_patch_size, 1, 1, 1)
        elif self.batch_size % self.temporal_patch_size == 1:
            repeat_image = pixel_value[-1:, ...].repeat(2, 1, 1, 1)
            patches = torch.cat((pixel_value, repeat_image), dim=0)
        else:
            patches = pixel_value
        grid_t, grid_h, grid_w = grid_thw[0][0], grid_thw[0][1], grid_thw[0][2]
        patches = patches.reshape(grid_t, self.temporal_patch_size, self.channel, 
                                  grid_h//self.merge_size, self.merge_size, self.patch_size, grid_w//self.merge_size, self.merge_size, self.patch_size)
        patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = patches.reshape(grid_t * grid_h * grid_w, self.channel * self.temporal_patch_size * self.patch_size * self.patch_size)
        vision_output = self.vpm(flatten_patches, grid_thw)
        image_embeds = vision_output.pooler_output
        split_sizes = (grid_thw.prod(-1) // self.merge_size**2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        return image_embeds

class smolvlm_vision(torch.nn.Module):
    def __init__(self, vlm):
        super(smolvlm_vision, self).__init__()
        self.vpm = vlm.model.vision_model
        self.connector = vlm.model.connector
        
    def forward(self, pixel_values):
        # Get sequence from the vision encoder
        image_hidden_states = self.vpm(pixel_values).last_hidden_state
        # Modality projection & resampling
        image_hidden_states = self.connector(image_hidden_states)
        print("image_features:", image_hidden_states.shape)
        return image_hidden_states

class vila1_5_3b_vision(torch.nn.Module):
    def __init__(self, vlm):
        super(vila1_5_3b_vision, self).__init__()
        self.vlm = vlm

    def forward(self, pixel_values):
        # Get sequence from the vision encoder
        out = self.vlm.encode_images(pixel_values)
        return out


class deepseekocr_vision(torch.nn.Module):
    def __init__(self, model):
        super(deepseekocr_vision, self).__init__()
        self.sam_model = model.sam_model
        self.vision_model = model.vision_model
        self.view_seperator = model.view_seperator
        self.image_newline = model.image_newline
        self.projector = model.projector

    def forward(self, pixel_value):
        global_features_1 = self.sam_model(pixel_value)
        global_features_2 = self.vision_model(pixel_value, global_features_1) 
        global_features = torch.cat((global_features_2[:, 1:], global_features_1.flatten(2).permute(0, 2, 1)), dim=-1) 
        global_features = self.projector(global_features)
        print('=====================')
        print('BASE: ', global_features.shape)
        print('NO PATCHES')
        print('=====================')
        _, hw, n_dim = global_features.shape
        h = w = int(hw ** 0.5)
        global_features = global_features.view(h, w, n_dim)
        global_features = torch.cat(
            [global_features, self.image_newline[None, None, :].expand(h, 1, n_dim)], dim=1
        )
        global_features = global_features.view(-1, n_dim)
        global_local_features = torch.cat([global_features, self.view_seperator[None, :]], dim=0)
        return global_local_features

if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--path', type=str, default='CKPT/MiniCPM-V-2_6', help='model path', required=False)
    argparse.add_argument('--model_name', type=str, default='minicpm-v-2_6',
                        choices=['minicpm-v-2_6', 'qwen2_5-vl-3b', 'qwen3-vl', 'qwen3.5', 'smolvlm', 'internvl3-1b', 'deepseekocr'],
                        help='model name', required=True)
    argparse.add_argument('--batch_size', type=int, default=1, help='batch size', required=False)
    argparse.add_argument('--height', type=int, default=448, help='image height', required=False)
    argparse.add_argument('--width', type=int, default=448, help='image width', required=False)
    argparse.add_argument('--device', type=str, default="cpu", help='cpu or cuda', required=False)
    args = argparse.parse_args()

    path = args.path
    model_name = args.model_name
    savepath = os.path.join("./onnx", model_name + "_vision.onnx")
    device_type = args.device
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    if model_name == 'minicpm-v-2_6':
        model = AutoModel.from_pretrained(
            path, trust_remote_code=True, dtype=torch.float32,
        )
        model = model.to(device=device_type, dtype=torch.float32)
        model.eval()
        model = minicpm_v_2_6_vision(model, args.batch_size, args.height, args.width)
        pixel_values = torch.randn(args.batch_size, 3, args.height, args.width, device=model.device, dtype=torch.float32)
        out = model(pixel_values)
        print("Output shape:", out.shape)
        torch.onnx.export(model, 
                    pixel_values, 
                    savepath,
                    input_names=['pixel'],
                    opset_version=15)
    elif model_name == 'qwen2_5-vl-3b':
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            path,
            dtype=torch.float32, # 注意此处的数据类型，由于 rknn 目前仅支持 float32 ，因此需要指定；若是在加载权重时限制了数据类型，需要自行修改config.json中的 "use_flash_attn" 参数为 false
            low_cpu_mem_usage=True, _attn_implementation="eager",
            trust_remote_code=True).eval().to(device_type)
        pixel_values = torch.randn(args.batch_size, 3, args.height, args.width, device=model.device, dtype=torch.float32)
        grid_thw = torch.tensor([[args.batch_size // 2 if args.batch_size% 2 == 0 else args.batch_size // 2 + 1, args.height//14, args.width//14]], dtype=torch.int64)
        model.eval()
        model = qwen2_5_vl_3b_vision(model, args.batch_size)
        out = model(pixel_values, grid_thw)
        print("Output shape:", out.shape)
        torch.onnx.export(model, 
                    (pixel_values, grid_thw), 
                    savepath,
                    input_names=['pixel', 'grid_thw'],
                    dynamic_axes={'pixel': {2: 'height', 3: 'width'}},
                    opset_version=15)
    elif model_name == 'qwen3-vl':
        from transformers import Qwen3VLForConditionalGeneration
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            path,
            dtype=torch.float32, # 注意此处的数据类型，由于 rknn 目前仅支持 float32 ，因此需要指定；若是在加载权重时限制了数据类型，需要自行修改config.json中的 "use_flash_attn" 参数为 false
            low_cpu_mem_usage=True, _attn_implementation="eager",
            trust_remote_code=True).eval().to(device_type)
        pixel_values = torch.randn(args.batch_size, 3, args.height, args.width, device=model.device, dtype=torch.float32)
        grid_thw = torch.tensor([[args.batch_size // 2 if args.batch_size% 2 == 0 else args.batch_size // 2 + 1, args.height//16, args.width//16]], dtype=torch.int64)
        model.eval()
        model = qwen3_vl_vision(model, args.batch_size)
        out = model(pixel_values, grid_thw)
        print("Output shape:", out[0].shape)
        torch.onnx.export(model, 
                    (pixel_values, grid_thw), 
                    savepath,
                    input_names=['pixel', 'grid_thw'],
                    dynamic_axes={'pixel': {2: 'height', 3: 'width'}},
                    opset_version=15)
    elif model_name == 'smolvlm':
        from transformers import SmolVLMForConditionalGeneration
        model = SmolVLMForConditionalGeneration.from_pretrained(
            path,
            dtype=torch.float32,
            _attn_implementation="eager",
        ).to(device_type)
        pixel_values = torch.randn(args.batch_size, 3, args.height, args.width, device=model.device, dtype=torch.float32)
        print("pixel_values:", pixel_values.shape)
        model = smolvlm_vision(model)
        model = model.to(torch.float32).eval()
        out = model(pixel_values)
        torch.onnx.export(model, 
                    pixel_values, 
                    savepath,
                    input_names=['pixel'],
                    dynamic_axes={'pixel': {2: 'height', 3: 'width'}},
                    opset_version=15)
    elif model_name == 'internvl3-1b':
        model = AutoModel.from_pretrained(
        path,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True).eval().to(device_type)
        pixel_values = torch.randn(args.batch_size, 3, args.height, args.width, device=model.device, dtype=torch.float32)
        model.forward = model.extract_feature
        model = model.to(torch.float32).eval()
        torch.onnx.export(model, pixel_values, savepath, input_names=['pixel'])
    elif model_name == 'deepseekocr':
        model = AutoModel.from_pretrained(
        path,
        _attn_implementation='eager',
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True).eval().to(device_type)
        pixel_values = torch.randn(args.batch_size, 3, args.height, args.width, device=model.device, dtype=torch.float32)
        model = deepseekocr_vision(model.model)
        model = model.to(torch.float32).eval()
        torch.onnx.export(model, pixel_values, savepath, input_names=['pixel'], opset_version=18)
    elif model_name == 'qwen3.5':
        from transformers import AutoModel
        model = AutoModel.from_pretrained(
            path,
            dtype=torch.float32, # 注意此处的数据类型，由于 rknn 目前仅支持 float32 ，因此需要指定；若是在加载权重时限制了数据类型，需要自行修改config.json中的 "use_flash_attn" 参数为 false
            low_cpu_mem_usage=True, _attn_implementation="eager",
            trust_remote_code=True).eval().to(device_type)
        pixel_values = torch.randn(args.batch_size, 3, args.height, args.width, device=model.device, dtype=torch.float32)
        grid_thw = torch.tensor([[args.batch_size // 2 if args.batch_size% 2 == 0 else args.batch_size // 2 + 1, args.height//16, args.width//16]], dtype=torch.int64)
        model.eval()
        model = qwen3_5_vl_vision(model, args.batch_size)
        out = model(pixel_values, grid_thw)
        print("Output shape:", out[0].shape)
        torch.onnx.export(model, 
                    (pixel_values, grid_thw), 
                    savepath,
                    input_names=['pixel', 'grid_thw'],
                    dynamic_axes={'pixel': {2: 'height', 3: 'width'}},
                    opset_version=15)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
        exit(1)
        
    print(f"Exported to {savepath}")
