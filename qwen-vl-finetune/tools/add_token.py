from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import torch
import os

pretrain_model_path = "/mnt/lijunhao/output/value_function/checkpoints/Qwen2.5-VL-3B-Instruct" 
output_model_path = "/mnt/lijunhao/output/value_function/checkpoints/Qwen2.5-VL-3B-Instruct-resize"

print(f"Loading model and processor: {pretrain_model_path}")

tokenizer = AutoTokenizer.from_pretrained(pretrain_model_path)
processor = AutoProcessor.from_pretrained(pretrain_model_path)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(pretrain_model_path)

print(f"{pretrain_model_path} 原始字典大小为： {len(tokenizer)}")

# 添加特殊字符
new_tokens = [f'<extra_id_{i}>' for i in range(201)]
num_added_toks = tokenizer.add_tokens(new_tokens, special_tokens=True)
if num_added_toks != len(new_tokens):
    raise ValueError(
        f"Expected to add {len(new_tokens)} tokens, but tokenizer.add_tokens returned {num_added_toks}. "
        "Make sure you are starting from the original checkpoint instead of an already-resized tokenizer."
    )
# 修改模型中embedding和lm_head这两层的维度
model.resize_token_embeddings(len(tokenizer))

# 初始化新token的embeddings
input_embeddings = model.get_input_embeddings()
output_embeddings = model.get_output_embeddings()
embeddings_tied = output_embeddings.weight.data_ptr() == input_embeddings.weight.data_ptr()

dtype = input_embeddings.weight.dtype
device = input_embeddings.weight.device

original_vocab_limit = len(tokenizer) - num_added_toks
with torch.no_grad():
    existing_weights = input_embeddings.weight[:original_vocab_limit]
    emb_mean = existing_weights.mean(dim=0)
    emb_std = existing_weights.std(dim=0)

    new_token_ids = tokenizer.convert_tokens_to_ids(new_tokens)

    for i, token_id in enumerate(new_token_ids):
        torch.manual_seed(42 + i)
        init_weight = (torch.randn_like(emb_mean) * emb_std * 0.1 + emb_mean).to(device=device, dtype=dtype)
        input_embeddings.weight[token_id] = init_weight
        if not embeddings_tied:
            output_embeddings.weight[token_id] = init_weight

print(f"Saving model: {output_model_path}")

os.makedirs(output_model_path, exist_ok=True)

processor.tokenizer = tokenizer
tokenizer.save_pretrained(output_model_path)
processor.save_pretrained(output_model_path)
model.save_pretrained(output_model_path)
