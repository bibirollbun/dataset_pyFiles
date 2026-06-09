# from transformers import AutoModelForCausalLM, AutoTokenizer
# from huggingface_hub import login




# login("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")  # ⚡改成你想要的路径
# model_name = "Qwen/Qwen3-8B"

# # 下载并缓存到自定义目录
# tokenizer = AutoTokenizer.from_pretrained(
#     model_name,
#     use_auth_token=True
# )

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     use_auth_token=True
# )

# print(f"✅ 模型已缓存到: {cache_dir}")

# save_dir = "Qwen3-8B"

# tokenizer.save_pretrained(save_dir)
# model.save_pretrained(save_dir)

# print(f"✅ 模型已保存到: {save_dir}")


!pip download transformers==4.56.2

