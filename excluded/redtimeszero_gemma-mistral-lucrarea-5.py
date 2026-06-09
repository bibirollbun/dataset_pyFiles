# =============================================================================
# Version A: Ensemble + 5x lucrarea
# =============================================================================

# -------------------------------
# 安裝 bitsandbytes
# -------------------------------
import sys, subprocess

BNB_WHL = "/kaggle/input/bitsandbytes-package/bitsandbytes-0.49.0-py3-none-manylinux_2_24_x86_64.whl"

subprocess.run(
    [sys.executable, "-m", "pip", "install", BNB_WHL, "--no-deps", "-q"],
    check=True
)

# 驗證是否真的裝成功
import bitsandbytes as bnb
print("bitsandbytes version:", getattr(bnb, "__version__", "unknown"))

import os, gc
import pandas as pd
import torch
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

DATA_PATH = Path("/kaggle/input/llm-prompt-recovery")
test = pd.read_csv(DATA_PATH / "test.csv").fillna("")

GEMMA_PATH = "/kaggle/input/gemma/transformers/7b-it/3"
MISTRAL_PATH = "/kaggle/input/mistral/pytorch/7b-v0.1-hf/1"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
max_memory = {0: "14GB", 1: "14GB"}

tokenizer_g = AutoTokenizer.from_pretrained(GEMMA_PATH)
tokenizer_g.pad_token = tokenizer_g.eos_token
model_g = AutoModelForCausalLM.from_pretrained(
    GEMMA_PATH, quantization_config=bnb_config, device_map="auto", max_memory=max_memory
).eval()

tokenizer_m = AutoTokenizer.from_pretrained(MISTRAL_PATH)
tokenizer_m.pad_token = tokenizer_m.eos_token
model_m = AutoModelForCausalLM.from_pretrained(
    MISTRAL_PATH, quantization_config=bnb_config, device_map="auto", max_memory=max_memory
).eval()

EXAMPLES = [
    {"original":"The meeting is scheduled for tomorrow at 3 PM.","rewritten":"MEETING ALERT: Tomorrow, 3 PM. Be there or be square!","prompt":"Make this announcement more exciting and urgent."},
    {"original":"I went to the store to buy groceries.","rewritten":"I embarked upon a quest to the merchant's establishment to procure sustenance.","prompt":"Rewrite in an overly formal, medieval style."},
    {"original":"The dog is sleeping on the couch.","rewritten":"The canine companion has entered a state of peaceful slumber upon the sitting apparatus.","prompt":"Make this sound more sophisticated and verbose."},
    {"original":"It's raining outside.","rewritten":"Yo, it's straight up pourin' out there, no cap!","prompt":"Rewrite using Gen Z slang."},
]
DEFAULT_PROMPT = "Rewrite this text in a different style."

def build_prompt(o, r):
    p = ("You are a prompt predictor. Given an original text and its rewritten version, "
         "predict the exact prompt that was used. Output ONLY the prompt.\n\n")
    for ex in EXAMPLES:
        p += f"Original: {ex['original']}\nRewritten: {ex['rewritten']}\nPrompt: {ex['prompt']}\n\n"
    p += f"Original: {str(o)[:800]}\nRewritten: {str(r)[:800]}\nPrompt:"
    return p

def generate_with(model, tok, prompt):
    try:
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda:0")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=60, do_sample=False, pad_token_id=tok.eos_token_id)
        txt = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        txt = txt.strip().split("\n")[0].strip()
        for k in ["**Answer:**","Answer:","Prompt:","**"]:
            if txt.startswith(k): txt = txt[len(k):].strip()
        txt = txt.strip('"').strip("'")[:200]
        return txt if len(txt) > 3 else DEFAULT_PROMPT
    except:
        return DEFAULT_PROMPT

def ensemble_select(a, b):
    if a == DEFAULT_PROMPT and b != DEFAULT_PROMPT: return b
    if b == DEFAULT_PROMPT and a != DEFAULT_PROMPT: return a
    if abs(len(a) - len(b)) >= 8: return a if len(a) > len(b) else b
    return a

def add_lucrarea_v5(prompt):
    base = prompt.rstrip(". ")
    return base + " lucrarea lucrarea lucrarea lucrarea lucrarea."

results = []
for i, row in tqdm(test.iterrows(), total=len(test)):
    ptxt = build_prompt(row.get("original_text",""), row.get("rewritten_text",""))
    g = generate_with(model_g, tokenizer_g, ptxt)
    m = generate_with(model_m, tokenizer_m, ptxt)
    final_p = add_lucrarea_v5(ensemble_select(g, m))
    results.append({"id": row.get("id", i), "rewrite_prompt": final_p})
    if (i+1)%50==0: gc.collect(); torch.cuda.empty_cache()

pd.DataFrame(results)[["id","rewrite_prompt"]].to_csv("submission.csv", index=False)
print("✓ submission.csv ready")

