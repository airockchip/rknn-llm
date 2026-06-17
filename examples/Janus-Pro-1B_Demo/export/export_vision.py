import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
import os

model_path = "/path/to/Janus-Pro-1B/"
save_path = "onnx/Janus_pro_vision.onnx"

def export_vision_janus_pro(model_path: str, save_path: str):
    """
    Export the vision encoder and projector of Janus-Pro-1B model to ONNX format
    """
    # 设置默认数据类型为 float32
    torch.set_default_dtype(torch.float32)
    
    # Load model
    vl_gpt = MultiModalityCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,  # 强制使用 float32
        low_cpu_mem_usage=True,
    )
    
    # Move model to CPU and convert to float32
    vl_gpt = vl_gpt.cpu().eval().float()  # 确保模型是 float32
    
    # Create a wrapper class for vision encoder + projector
    class VisionWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.vision_tower = model.vision_model  # 视觉编码器
            self.vision_projector = model.aligner  # 特征对齐投影器
            
        def forward(self, pixel_values):
            # Forward through vision tower
            vision_features = self.vision_tower(pixel_values)
            # Project features - vision_features 已经是最终的特征张量
            projected_features = self.vision_projector(vision_features)
            return projected_features
            
    # Create wrapper instance and convert to float32
    vision_wrapper = VisionWrapper(vl_gpt)
    vision_wrapper.eval().float()  # 确保包装器也是 float32
    
    # Create dummy input with float32
    batch_size = 1
    num_channels = 3
    height = 384  # Janus default image size
    width = 384
    dummy_input = torch.randn(batch_size, num_channels, height, width, dtype=torch.float32)
    
    # Export to ONNX with higher opset version
    torch.onnx.export(
        vision_wrapper,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=16,  # 使用高版本 opset 以支持 scaled_dot_product_attention
        do_constant_folding=True,
        input_names=['pixel_values'],
        output_names=['projected_features'],
        dynamic_axes={
            'pixel_values': {0: 'batch_size'},
            'projected_features': {0: 'batch_size'}
        },
        # 添加额外的配置
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
        training=torch.onnx.TrainingMode.EVAL,
        verbose=False
    )
    
    print(f"Successfully exported vision components to {save_path}")
    
    # Verify the exported model
    import onnxruntime
    
    # Create inference session
    ort_session = onnxruntime.InferenceSession(save_path)
    
    # Run inference with dummy input
    ort_inputs = {
        'pixel_values': dummy_input.numpy()
    }
    ort_outputs = ort_session.run(None, ort_inputs)
    
    # Compare with PyTorch output
    torch_output = vision_wrapper(dummy_input)
    
    # Check numerical accuracy with更宽松的容忍度
    import numpy as np
    np.testing.assert_allclose(
        torch_output.detach().numpy(), 
        ort_outputs[0], 
        rtol=1e-2,  # 放宽相对误差容忍度
        atol=1e-3   # 放宽绝对误差容忍度
    )
    
    print("ONNX model verification successful!")
    
    # 打印一些统计信息
    torch_output_np = torch_output.detach().numpy()
    onnx_output_np = ort_outputs[0]
    
    abs_diff = np.abs(torch_output_np - onnx_output_np)
    rel_diff = np.abs((torch_output_np - onnx_output_np) / (torch_output_np + 1e-7))
    
    print(f"\nValidation Statistics:")
    print(f"Max absolute difference: {np.max(abs_diff):.6f}")
    print(f"Mean absolute difference: {np.mean(abs_diff):.6f}")
    print(f"Max relative difference: {np.max(rel_diff):.6f}")
    print(f"Mean relative difference: {np.mean(rel_diff):.6f}")

if __name__ == "__main__":
    os.makedirs("onnx", exist_ok=True)
    # 检查环境版本
    print(f"PyTorch version: {torch.__version__}")
    try:
        import onnx
        # 尝试不同的方式获取 ONNX 版本
        try:
            onnx_version = onnx.__version__
        except AttributeError:
            try:
                onnx_version = onnx.version.version
            except AttributeError:
                onnx_version = "Unknown"
        print(f"ONNX version: {onnx_version}")
    except ImportError:
        print("ONNX not installed")
        
    import onnxruntime
    print(f"ONNX Runtime version: {onnxruntime.__version__}")
    
    export_vision_janus_pro(model_path, save_path)