from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoConfig, AutoProcessor
import torch
import os

# 配置路径
pretrain_model_path = "Qwen/Qwen2.5-VL-3B-Instruct" 
output_model_path = "./checkpoints/Qwen2.5-VL-3B-Instruct-resize"

print(f"正在加载模型和processor: {pretrain_model_path}")
config = AutoConfig.from_pretrained(pretrain_model_path)
tokenizer = AutoTokenizer.from_pretrained(pretrain_model_path)
processor = AutoProcessor.from_pretrained(pretrain_model_path)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(pretrain_model_path)

print(f"Qwen2.5-VL原始字典大小为： {len(tokenizer)}")
# ==========================================
# 1. 添加特殊 Token (关键修改：special_tokens=True)
# ==========================================
print("正在添加201个特殊token...")
new_tokens = [f'<extra_id_{i}>' for i in range(201)]
# 关键：设置为 special_tokens=True，防止被切分
num_added_toks = tokenizer.add_tokens(new_tokens, special_tokens=True)

# ==========================================
# 2. 调整模型 Embedding 大小
# ==========================================
model.resize_token_embeddings(len(tokenizer))

# ==========================================
# 3. 初始化新 Token 的 Embeddings
# ==========================================
print("正在改进value token embeddings的初始化...")
input_embeddings = model.get_input_embeddings()
output_embeddings = model.get_output_embeddings() # 获取 lm_head

dtype = input_embeddings.weight.dtype
device = input_embeddings.weight.device

# 计算现有 embedding 的均值和方差
# 注意：只统计原始词表部分，避免受 padding 0 的影响
original_vocab_limit = len(tokenizer) - num_added_toks
with torch.no_grad():
    existing_weights = input_embeddings.weight[:original_vocab_limit]
    emb_mean = existing_weights.mean(dim=0)
    emb_std = existing_weights.std(dim=0)

    # 获取新添加 token 的 ID 列表
    # convert_tokens_to_ids 返回的是 list
    new_token_ids = tokenizer.convert_tokens_to_ids(new_tokens)

    for i, token_id in enumerate(new_token_ids):
        # 生成随机初始化向量
        torch.manual_seed(42 + i)
        # 必须转为由于模型一致的 dtype 和 device
        new_emb = (torch.randn_like(emb_mean) * emb_std * 0.1 + emb_mean).to(device=device, dtype=dtype)
        
        # 赋值给 input embeddings
        input_embeddings.weight[token_id] = new_emb
        
        # 如果 output embeddings (lm_head) 是独立的（未绑定权重），也需要初始化
        # Qwen 模型通常 input/output 可能是绑定的，也可能不是，显式赋值最安全
        if output_embeddings is not None and not torch.equal(input_embeddings.weight, output_embeddings.weight):
             # 只有当它们不共享内存时才赋值，避免重复操作（虽然重复赋值也没错）
             if token_id < output_embeddings.weight.shape[0]:
                output_embeddings.weight[token_id] = new_emb

print("新 token 初始化完成。")

# ==========================================
# 4. 同步 Processor 并保存 (关键修改)
# ==========================================
processor.tokenizer = tokenizer

print(f"正在保存模型到: {output_model_path}")
os.makedirs(output_model_path, exist_ok=True)

# 保存所有组件
config.save_pretrained(output_model_path)
tokenizer.save_pretrained(output_model_path)
processor.save_pretrained(output_model_path)
model.save_pretrained(output_model_path)