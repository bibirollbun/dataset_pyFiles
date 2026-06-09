# === MAP — DeepSeek-R1 LoRA (timeout-safe, manual ChatML, single-cell) ===
# Works on Kaggle "Code Competition" image: FP16, single GPU, no bitsandbytes.

import os, sys, math, time, random, json, gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch


# ---------- Speed & env ----------
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
RUN_TYPE = os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "Interactive")  # "Batch" on Submit
FAST_SUBMIT = (RUN_TYPE == "Batch")
print("RUN_TYPE:", RUN_TYPE, "| FAST_SUBMIT:", FAST_SUBMIT)

if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# ---------- Seeds ----------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)


# ---------- Data load ----------
def find_csv(root="/kaggle/input"):
    train_path, test_path = None, None
    for p in Path(root).rglob("*.csv"):
        name = p.name.lower()
        if name == "train.csv" and train_path is None: train_path = p
        elif name == "test.csv" and test_path is None: test_path = p
    return train_path, test_path

train_csv, test_csv = find_csv("/kaggle/input")
print("Detected train:", train_csv); print("Detected test :", test_csv)
assert train_csv is not None, "train.csv not found under /kaggle/input"
assert test_csv is not None, "test.csv not found under /kaggle/input"

train_df = pd.read_csv(train_csv)
test_df  = pd.read_csv(test_csv)


# ---------- Normalizers & prompt builder ----------
VALID_CATS = [
    "True_Correct","True_Misconception","True_Neither",
    "False_Correct","False_Misconception","False_Neither"
]

def normalize_category(c):
    s = str(c).strip().replace("-", "_").replace(" ", "_")
    fixes = {
        "True__Correct": "True_Correct",
        "False__Correct": "False_Correct",
        "True_Miscon": "True_Misconception",
        "False_Miscon": "False_Misconception",
        "TrueNeither": "True_Neither",
        "FalseNeither": "False_Neither"
    }
    s = fixes.get(s, s)
    if s in VALID_CATS: return s
    if s.startswith("True_"):  return "True_Correct"
    if s.startswith("False_"): return "False_Correct"
    return "True_Correct"

def make_target(category, miscon):
    if pd.isna(miscon) or str(miscon).strip() in ["", "NaN"]:
        miscon = "NA"
    return f"{normalize_category(category)}:{str(miscon).strip()}"

# Manual Qwen/ChatML format (avoids apply_chat_template / Jinja)
def chatml_prompt(qtext, mc_answer, explanation):
    sys_msg = (
        "You analyze student math explanations and output exactly one line: Category:Misconception. "
        "Category is one of {True_Correct, True_Misconception, True_Neither, "
        "False_Correct, False_Misconception, False_Neither}. Misconception is a canonical string from training or NA."
    )
    user_msg = (
        f"Question: {qtext}\n"
        f"SelectedOption: {mc_answer}\n"
        f"Explanation: {explanation}\n"
        "Return exactly: Category:Misconception"
    )
    return (
        "<|im_start|>system\n" + sys_msg + "\n<|im_end|>\n"
        "<|im_start|>user\n"   + user_msg + "\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

def pack_for_length(qtext, mc_answer, explanation, max_chars=850):
    qtext = (qtext or "")
    mc_answer = (mc_answer or "")
    explanation = (explanation or "")
    ex = explanation[: int(max_chars * 0.67)]
    q  = qtext[: int(max_chars * 0.33)]
    return q, mc_answer, ex


# ---------- Tokenizer & model ----------
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = "/kaggle/input/deepseek-r1/transformers/deepseek-r1-distill-qwen-1.5b/2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map={'': 0} if torch.cuda.is_available() else None,
    low_cpu_mem_usage=True,
)
model.config.use_cache = False
try:
    model.gradient_checkpointing_enable()
except Exception as e:
    print("GC enable failed (non-fatal):", e)

print("Model device:", next(model.parameters()).device, "| dtype:", next(model.parameters()).dtype)


# ---------- Downsample & split ----------
from sklearn.model_selection import train_test_split

MAX_TRAIN_SAMPLES = 6000 if not FAST_SUBMIT else 2500
cats_norm = train_df["Category"].apply(normalize_category)
tmp_df = train_df.assign(_cat=cats_norm)

per_class = max(1, MAX_TRAIN_SAMPLES // len(VALID_CATS))
blocks = []
for c in VALID_CATS:
    blk = tmp_df[tmp_df["_cat"] == c]
    if len(blk) > per_class:
        blk = blk.sample(n=per_class, random_state=SEED)
    blocks.append(blk)
mini_df = pd.concat(blocks).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

def build_pairs(df, speed_fast=False):
    pairs = []
    for _, r in df.iterrows():
        q, a, ex = pack_for_length(str(r["QuestionText"]), str(r["MC_Answer"]), str(r["StudentExplanation"]),
                                   900 if not speed_fast else 700)
        prompt = chatml_prompt(q, a, ex)
        target = make_target(r["Category"], r["Misconception"])
        pairs.append((prompt, target))
    return pairs

pairs_all = build_pairs(mini_df, speed_fast=FAST_SUBMIT)
train_pairs, val_pairs = train_test_split(pairs_all, test_size=0.08 if not FAST_SUBMIT else 0.05,
                                          random_state=SEED, shuffle=True)
print("Total after downsample:", len(pairs_all))
print("Train pairs:", len(train_pairs), "Val pairs:", len(val_pairs))


# ---------- LoRA ----------
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8 if not FAST_SUBMIT else 4,
    lora_alpha=16 if not FAST_SUBMIT else 8,
    lora_dropout=0.1,
    target_modules=["q_proj","k_proj","v_proj","o_proj"]
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# ---------- Dataset ----------
from torch.utils.data import Dataset

MAX_LENGTH = 384 if not FAST_SUBMIT else 320

class CausalSFTDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length=384):
        self.pairs = pairs; self.tok = tokenizer; self.max_length = max_length
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        prompt, target = self.pairs[idx]
        full = prompt.strip() + "\n" + target.strip() + tokenizer.eos_token
        enc_full = self.tok(full, max_length=self.max_length, truncation=True, padding=False, return_tensors="pt")
        enc_prompt = self.tok(prompt.strip() + "\n", max_length=self.max_length, truncation=True, padding=False, return_tensors="pt")
        input_ids = enc_full["input_ids"][0]
        attention_mask = enc_full["attention_mask"][0]
        labels = input_ids.clone()
        prompt_len = enc_prompt["input_ids"].shape[1]
        labels[:prompt_len] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

train_ds = CausalSFTDataset(train_pairs, tokenizer, MAX_LENGTH)
val_ds   = CausalSFTDataset(val_pairs, tokenizer, MAX_LENGTH)
print("Datasets ready:", len(train_ds), len(val_ds))


# ---------- Train (timeout-safe) ----------
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

BATCH_SIZE = 1
GRAD_ACCUM = 24 if FAST_SUBMIT else 32
LR = 2.0e-4 if FAST_SUBMIT else 1.6e-4
EPOCHS = 1
MAX_STEPS = 700 if FAST_SUBMIT else 1500   # hard cap

args = TrainingArguments(
    output_dir="outputs",
    overwrite_output_dir=True,
    bf16=False,
    fp16=True if torch.cuda.is_available() else False,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    logging_steps=200000,   # practically disable logs
    max_steps=MAX_STEPS,
    save_total_limit=1,
    report_to=[],
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

trainer = Trainer(model=model, args=args, train_dataset=train_ds, data_collator=data_collator)

t0 = time.time()
trainer.train()
print("Train seconds:", time.time() - t0)

trainer.save_model("outputs/final_model")
tokenizer.save_pretrained("outputs/final_model")


# ---------- Inference helpers ----------
def normalize_miscon(m):
    m = str(m).strip()
    if not m: return "NA"
    return m[:128] if len(m) > 128 else m

def clean_prediction(text):
    line = text.splitlines()[0].strip()
    if ":" not in line: return "True_Correct:NA"
    left, right = line.split(":", 1)
    left = normalize_category(left)
    right = normalize_miscon(right)
    return f"{left}:{right}"

def decode_topk(prompt, k=3, beams=3, max_len=320):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len)
    if torch.cuda.is_available():
        inputs = {kk: vv.to("cuda") for kk, vv in inputs.items()}
    gen_kwargs = dict(max_new_tokens=10, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    if k == 1 or beams == 1:
        out = model.generate(**inputs, do_sample=False, num_beams=1, num_return_sequences=1, **gen_kwargs)
        gen = out[0, inputs["input_ids"].shape[1]:]
        txt = tokenizer.decode(gen, skip_special_tokens=True).strip()
        return [clean_prediction(txt)]
    out = model.generate(**inputs, do_sample=False, num_beams=beams, num_return_sequences=k, **gen_kwargs)
    seqs = []
    for i in range(out.shape[0]):
        gen = out[i, inputs["input_ids"].shape[1]:]
        txt = tokenizer.decode(gen, skip_special_tokens=True).strip()
        seqs.append(clean_prediction(txt))
    uniq, seen = [], set()
    for s in seqs:
        if s not in seen:
            uniq.append(s); seen.add(s)
    return uniq[:k]


# ---------- Predict test & write submission.csv ----------
rows = []
k_pred = 1 if FAST_SUBMIT else 3
beams  = 1 if FAST_SUBMIT else 3

for _, r in test_df.iterrows():
    q = str(r["QuestionText"]); a = str(r["MC_Answer"]); ex = str(r["StudentExplanation"])
    q_trim = q[:140] if FAST_SUBMIT else q[:300]
    ex_trim = ex[:700] if FAST_SUBMIT else ex[:900]
    prompt = chatml_prompt(q_trim, a, ex_trim)
    preds = decode_topk(prompt, k=k_pred, beams=beams, max_len=(320 if FAST_SUBMIT else 384))
    rows.append((r["row_id"], " ".join(preds)))

sub = pd.DataFrame(rows, columns=["row_id", "Category:Misconception"])
out_path = "/kaggle/working/submission.csv"
sub.to_csv(out_path, index=False)
print("Saved:", out_path)
display(sub.head())


