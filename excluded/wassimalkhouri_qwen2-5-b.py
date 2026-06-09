# 1 ▸ Library installs   (internet must be ON *only for this cell*)
!pip -q install "transformers>=4.39" trl peft bitsandbytes accelerate datasets einops
!pip -q install flash-attn --no-build-isolation --no-cache-dir   # optional
!pip -q install unsloth                                          # optional


import torch, psutil, subprocess
!nvidia-smi
print(f"Torch {torch.__version__} | GPUs: {torch.cuda.device_count()} | "
      f"RAM: {psutil.virtual_memory().total/1e9:.1f} GB")


from datasets import load_dataset
CSV_PATH = "/kaggle/input/scored-svg-11k/svg_dataset_scored_11k.csv"

raw_ds = load_dataset("csv", data_files=CSV_PATH, split="train")
SYSTEM_PROMPT = (
    "You are Qwen, a helpful assistant that writes succinct, "
    "standards-compliant SVG code given a scene description."
)

def to_chatml(ex):
    svg = ex["svg_code"]
    if not svg or svg == "nan":
        return None                       # skip rows without SVG
    user = f"Generate an SVG:\n{ex['sentence']}"
    return {"messages":[
        {"role":"system",    "content":SYSTEM_PROMPT},
        {"role":"user",      "content":user},
        {"role":"assistant", "content":svg},
    ]}

chatml_ds = (raw_ds
             .map(to_chatml, remove_columns=raw_ds.column_names)
             .filter(lambda r: r is not None))

print("Rows after cleaning:", len(chatml_ds))
chatml_ds = chatml_ds.train_test_split(test_size=0.05, seed=42)


from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import torch, gc

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

bnb_conf = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

base = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_conf,
    device_map="auto",
    trust_remote_code=True,
    max_memory={0: "15GiB", "cpu": "48GiB"},
)

tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tok.pad_token = tok.eos_token

lora_cfg = LoraConfig(
    r=64, lora_alpha=128, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
    bias="none", task_type="CAUSAL_LM",
)

model = get_peft_model(base, lora_cfg)
model.enable_input_require_grads()
model.gradient_checkpointing_enable()


from trl import SFTTrainer, SFTConfig

def chatml_to_str(example):
    return tok.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False
    )

sft_cfg = SFTConfig(
    output_dir="qwen25_vsp_lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    num_train_epochs=3,
    bf16=True,
    logging_steps=25,
    save_strategy="steps",
    save_steps=500,
    max_seq_length=1024,
    packing=True,
    max_grad_norm=0.3,
    report_to=[],
    gradient_checkpointing=True,
)

trainer = SFTTrainer(
    model           = model,
    args            = sft_cfg,
    train_dataset   = chatml_ds["train"],
    eval_dataset    = chatml_ds["test"],
    tokenizer       = tok,
    formatting_func = chatml_to_str,
)

gc.collect(); torch.cuda.empty_cache()
trainer.train()


from pathlib import Path
from peft import AutoPeftModelForCausalLM
from transformers import pipeline

peft_dir = Path("qwen25_vsp_lora")
ckpts    = list(peft_dir.glob("checkpoint-*"))
src      = max(ckpts, key=lambda p:int(p.name.split("-")[-1])) if ckpts else peft_dir
print("Merging from", src)

merged = AutoPeftModelForCausalLM.from_pretrained(src, low_cpu_mem_usage=True)
merged = merged.merge_and_unload()
merged.save_pretrained("qwen25_vsp_full", safe_serialization=True)
tok.save_pretrained("qwen25_vsp_full")

pipe = pipeline("text-generation",
                model="qwen25_vsp_full",
                tokenizer=tok,
                device_map="auto",
                max_new_tokens=1024)

print(pipe("Generate an SVG of a playful kitten chasing a bright red ball.")[0]["generated_text"])


import shutil, gc, torch
lora_folder = Path("qwen25_vsp_lora")
if lora_folder.exists():
    print("Removing", lora_folder)
    shutil.rmtree(lora_folder)

gc.collect(); torch.cuda.empty_cache()

