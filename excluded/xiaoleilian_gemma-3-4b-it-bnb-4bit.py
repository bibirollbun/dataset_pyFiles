!pip install --no-index --find-links=/kaggle/input/unsloth-pip/unsloth-wheels unsloth


# from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported

# os.environ['HF_XET_CHUNK_CACHE_SIZE_BYTES'] = '0'

# max_length = 2048
# lora_rank = 8
# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name = "google/gemma-3-4b-it",
#     # model_name = "Qwen/Qwen2.5-3B-Instruct",
#     cache_dir="/kaggle/working/gemma-3-4b-it-bnb-4bit",
#     max_seq_length = 2048,
#     load_in_4bit = True, # False for LoRA 16bit
#     # fast_inference = True, # Enable vLLM fast inference
#     # use_fast=True,
#     max_lora_rank = lora_rank,
#     gpu_memory_utilization = 0.7, 
# )



from huggingface_hub import snapshot_download
from unsloth import FastLanguageModel

local_dir = "/kaggle/working/gemma-3-4b-it-bnb-4bit"
snapshot_download(
    repo_id="unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
    local_dir=local_dir,
    local_dir_use_symlinks=False,     # 关键：生成真实文件而不是指向 blobs 的链接
)

max_length = 2048
lora_rank = 8
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = local_dir,
    max_seq_length = 2048,
    load_in_4bit = True,
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.7,
)

