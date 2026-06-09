!pip -q install "transformers>=4.39" trl peft bitsandbytes accelerate datasets einops
!pip -q install flash-attn --no-build-isolation --no-cache-dir
# Optional: Unsloth for extra speed / lower VRAM
!pip -q install unsloth


import subprocess, os, json, torch, psutil, platform, typing, sys, time, pathlib, math
!nvidia-smi
print(f"Torch versionÂ : {torch.__version__}\nGPU countÂ : {torch.cuda.device_count()}\nSystem RAMÂ : {psutil.virtual_memory().total/1e9:.1f}Â GB")


from datasets import load_dataset
RAW_PATH = "/kaggle/input/visual-scene-instructions-for-generative-llms/combined_train.json"
raw_ds = load_dataset("json", data_files=RAW_PATH, split="train")
print(raw_ds[0])
print(f"Total examplesÂ : {len(raw_ds):,}")


SYSTEM_PROMPT = "You are Qwen, a helpful assistant that writes succinct, standardsâ€‘compliant SVG code given a scene description."

def to_chatml(example):
    user_msg = f"Generate an SVG for the following scene description:\n{example['description']}"
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": example["svg"]},
        ]
    }

chatml_ds = raw_ds.map(to_chatml, remove_columns=raw_ds.column_names)
chatml_ds = chatml_ds.train_test_split(test_size=0.05, seed=42)
print(chatml_ds)


# ===== Qwenâ€‘14BÂ QLoRA fineâ€‘tune â€“ P100â€‘friendly =====
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model
import torch, gc

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

# ---------- quantâ€‘loader (4â€‘bit NF4, fp16 matmuls) ----------
bnb_conf = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,   # saves ~1Â GB vs fp32
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

base = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_conf,
    device_map="auto",
    trust_remote_code=True,
    max_memory={0: "15GiB", "cpu": "48GiB"},   # hardâ€‘cap GPU 0
)

tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tok.pad_token = tok.eos_token

# ---------- LoRA ----------
lora_cfg = LoraConfig(
    r=64, lora_alpha=128,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(base, lora_cfg)

# ğŸ”‘Â make inputs require grad so checkpointing keeps graph
model.enable_input_require_grads()
model.gradient_checkpointing_enable()

# ---------- SFT config ----------
sft_cfg = SFTConfig(
    output_dir="qwen25_vsp_lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,   # eff. batchÂ =Â 16
    learning_rate=2e-4,
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    num_train_epochs=3,
    bf16=True,
    logging_steps=25,
    save_strategy="steps",
    save_steps=500,
    max_seq_length=1024,   # fits 16Â GB w/ checkpointing
    packing=True,
    max_grad_norm=0.3,
    report_to=[],          # no wandb popâ€‘ups
    gradient_checkpointing=True,  # matches model.enable...
)

# ---------- Trainer ----------
trainer = SFTTrainer(
    model=model,
    processing_class=tok,
    train_dataset=chatml_ds["train"],
    eval_dataset=chatml_ds["test"],
    args=sft_cfg,
)

# free any cached VRAM before training
gc.collect(); torch.cuda.empty_cache()

# ---------- Start training ----------
trainer.train()


# ===== Merge LoRA into full model & quick demo =====
from pathlib import Path
from peft import AutoPeftModelForCausalLM
from transformers import pipeline

ckpt_root = Path("qwen25_vsp_lora")
ckpts = list(ckpt_root.glob("checkpoint-*"))

# pick latest checkpoint if they exist, else root dir
if ckpts:
    ckpt_dir = max(ckpts, key=lambda p: int(p.name.split("-")[-1]))
    print("Merging from", ckpt_dir)
    peft_path = ckpt_dir
else:
    print("No checkpoint-* folders found; merging from output_dir root")
    peft_path = ckpt_root

merged = AutoPeftModelForCausalLM.from_pretrained(
    peft_path,
    low_cpu_mem_usage=True,
)
merged = merged.merge_and_unload()
merged.save_pretrained("qwen25_vsp_full", safe_serialization=True)
tok.save_pretrained("qwen25_vsp_full")

# ---- inference demo ----
pipe = pipeline(
    "text-generation",
    model="qwen25_vsp_full",
    tokenizer=tok,
    device_map="auto",
    max_new_tokens=1024,
)
TEST_PROMPT = "Generate an SVG of a playful kitten chasing a bright red ball."
print(pipe(TEST_PROMPT)[0]["generated_text"])


# ğŸ”¦ optional cleanup â€” delete the LoRA-only folder to save space
import shutil, pathlib, os, gc, torch

lora_dir = pathlib.Path("qwen25_vsp_lora")          # name on disk
merged_dir = pathlib.Path("qwen25_vsp_full")        # keep this!

if lora_dir.exists():
    print(f"Removing {lora_dir} â€¦")
    shutil.rmtree(lora_dir)
else:
    print("LoRA folder already gone.")

# free any file handles / CUDA memory
gc.collect(); torch.cuda.empty_cache()

