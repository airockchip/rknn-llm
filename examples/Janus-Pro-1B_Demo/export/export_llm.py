import torch
from transformers import AutoConfig, AutoModelForCausalLM, LlamaTokenizer
from janus.models import MultiModalityCausalLM, VLChatProcessor
import os

model_path = "/path/to/Janus-Pro-1B/"  # 输入原始模型路径
save_path = "hf_model/Janus_pro_llm"  # 输出路径（Hugging Face 格式）

def export_llm_janus_pro(model_path: str, save_path: str):
    """
    Export the LLM part of the Janus-Pro-1B model to Hugging Face format
    """
    # 设置默认数据类型为 float32
    torch.set_default_dtype(torch.float32)
    
    # Load model
    llm_model = MultiModalityCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,  # 强制使用 float32
        low_cpu_mem_usage=True,
    )
    
    # Move model to CPU and convert to float32
    llm_model = llm_model.cpu().eval().float()  # 确保模型是 float32

    # 提取语言模型部分
    language_model = llm_model.language_model  # 获取语言模型部分
    
    # 保存语言模型为 Hugging Face 格式
    language_model.save_pretrained(save_path)
    print(f"Successfully exported LLM components to {save_path}")

    # 保存 Tokenizer 文件
    vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer
    tokenizer.save_pretrained(save_path)  # 保存 tokenizer 到同一目录
    print(f"Tokenizer saved to {save_path}")

    # 验证模型和 tokenizer 是否成功加载
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # 加载保存的模型和 tokenizer
    loaded_model = AutoModelForCausalLM.from_pretrained(save_path)
    loaded_tokenizer = AutoTokenizer.from_pretrained(save_path)
    
    # 创建一个 dummy 输入进行推理验证
    batch_size = 1
    sequence_length = 128  # 根据需要的最大长度调整
    dummy_input_ids = torch.randint(0, 1000, (batch_size, sequence_length), dtype=torch.long)
    dummy_attention_mask = torch.ones(batch_size, sequence_length, dtype=torch.long)  # Attention mask 全部为 1
    
    # 使用加载的模型进行推理
    outputs = loaded_model(input_ids=dummy_input_ids, attention_mask=dummy_attention_mask)
    
    # 打印输出结果
    print(f"Logits shape: {outputs.logits.shape}")
    print("Model export and verification successful!")

if __name__ == "__main__":
    # 检查环境版本
    print(f"PyTorch version: {torch.__version__}")
    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")
    except ImportError:
        print("Transformers not installed")
    
    # 调用导出函数
    export_llm_janus_pro(model_path, save_path)
