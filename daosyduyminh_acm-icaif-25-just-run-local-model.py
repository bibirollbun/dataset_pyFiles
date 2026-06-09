# -*- coding: utf-8 -*-
"""
Agentic Retrieval Grand Challenge â€” Single-GPU, OOM-safe â€¢ Train 2 LoRA (doc vs chunk) + Inference (Kaggle)

- Náº¿u CÃ“ bitsandbytes + triton.ops: Æ°u tiÃªn QLoRA 4-bit/8-bit vá»›i 7B.
- Náº¿u KHÃ”NG: Bá»� QUA 7B fp16 (dá»… OOM), tá»± Ä‘á»™ng chá»�n 3B -> 1.5B, vÃ  Ã©p PEFT KHÃ”NG dÃ¹ng bnb.
- Ghim 1 GPU (cuda:0) Ä‘á»ƒ trÃ¡nh lá»—i "cuda:0 vs cuda:1".
- Train 2 adapter riÃªng: DOC (lora_doc) & CHUNK (lora_chunk).
- Inference OOM-safe: truncate + two-stage cho chunk prompts.
- DÃ™NG TOÃ€N Bá»˜ TRAINING DATASET (khÃ´ng cáº¯t máº«u).
- Giá»›i háº¡n thá»�i gian train theo wall-clock (máº·c Ä‘á»‹nh â‰¤ 7 giá»�), sau Ä‘Ã³ tá»± inference Ä‘á»ƒ ká»‹p submit.
"""

import os, time, re, json, csv, gc, warnings
from typing import List, Dict, Optional, Tuple

# --- Pin 1 GPU + giáº£m á»“n TF/XLA
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments, TrainerCallback, set_seed,
    logging as hf_logging
)
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training

# ==== Quiet & CUDA mem tweaks
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()
torch.backends.cuda.matmul.allow_tf32 = True
try: torch.set_float32_matmul_precision("high")
except Exception: pass
if torch.cuda.is_available():
    try: torch.cuda.set_device(0)
    except Exception: pass

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEV_STR = "cuda:0" if DEVICE == "cuda" else "cpu"

BASE_DIR    = "/kaggle/input/acm-icaif-25-ai-agentic-retrieval-grand-challenge"
DOC_DEV     = f"{BASE_DIR}/document_ranking_kaggle_dev.jsonl"
CHUNK_DEV   = f"{BASE_DIR}/chunk_ranking_kaggle_dev.jsonl"
DOC_EVAL    = f"{BASE_DIR}/document_ranking_kaggle_eval.jsonl"
CHUNK_EVAL  = f"{BASE_DIR}/chunk_ranking_kaggle_eval.jsonl"

# Model env
ENV_MODEL  = os.environ.get("HF_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
LOCAL_ONLY = os.environ.get("LOCAL_ONLY", "0") == "1"

# ---------- Time budget ----------
MAX_TRAIN_HOURS   = float(os.environ.get("MAX_TRAIN_HOURS", "7"))
INFER_RESERVE_MIN = float(os.environ.get("INFER_RESERVE_MIN", "45"))
TIME_SPLIT        = os.environ.get("TIME_SPLIT", "auto")  # "auto" hoáº·c "a:b"

# Train config
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "2048"))
BATCH_SIZE  = int(os.environ.get("BATCH_SIZE", "1"))
GRAD_ACCUM  = int(os.environ.get("GRAD_ACCUM", "8"))
EPOCHS      = int(os.environ.get("EPOCHS", "1"))
LR          = float(os.environ.get("LR", "1e-4"))
WEIGHT_DECAY= float(os.environ.get("WEIGHT_DECAY", "0.01"))
BETA1       = float(os.environ.get("BETA1", "0.9"))
BETA2       = float(os.environ.get("BETA2", "0.999"))
EPS         = float(os.environ.get("EPS", "1e-8"))
SAVE_STEPS  = int(os.environ.get("SAVE_STEPS", "0"))  # 0 = chá»‰ save theo epoch
SAVE_TOTAL_LIMIT = int(os.environ.get("SAVE_TOTAL_LIMIT", "2"))
RESUME      = os.environ.get("RESUME_FROM", "")

# Inference
INFER_MAX_INPUT_TOKENS = int(os.environ.get("INFER_MAX_INPUT_TOKENS", "4096"))
MAX_NEW_TOKENS_INFER   = int(os.environ.get("MAX_NEW_TOKENS_INFER", "128"))

# Save
OUTPUT_DIR      = "./outputs_agentic_rag"
LORA_DIR_DOC    = os.path.join(OUTPUT_DIR, "lora_doc")
LORA_DIR_CHUNK  = os.path.join(OUTPUT_DIR, "lora_chunk")
MERGED_DIR_DOC  = os.path.join(OUTPUT_DIR, "merged_doc")
MERGED_DIR_CHUNK= os.path.join(OUTPUT_DIR, "merged_chunk")
SUBMISSION_CSV  = "./kaggle_submission.csv"
SAVE_MERGED_MODEL = os.environ.get("SAVE_MERGED_MODEL", "0") == "1"

SEED = int(os.environ.get("SEED", "42"))
set_seed(SEED)

print(f"ğŸ–¥ï¸�  Device: {DEVICE} ({DEV_STR})")
print(f"â�±ï¸�  Time budget: train â‰¤ {MAX_TRAIN_HOURS}h | reserve for infer â‰ˆ {INFER_RESERVE_MIN}m | split={TIME_SPLIT}")
print(f"ğŸ”§ Train cfg: epochs={EPOCHS}, batch={BATCH_SIZE}, grad_accum={GRAD_ACCUM}, lr={LR}, max_len={MAX_SEQ_LEN}")
print(f"ğŸ’¾ Output: DOC_LORA={LORA_DIR_DOC} | CHUNK_LORA={LORA_DIR_CHUNK} | save_merged={SAVE_MERGED_MODEL}")

# =========================
# Tokenizer
# =========================
tk_kwargs = {"trust_remote_code": True}
if LOCAL_ONLY: tk_kwargs["local_files_only"] = True
tokenizer = AutoTokenizer.from_pretrained(ENV_MODEL, **tk_kwargs)
tokenizer.padding_side = "right"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# =========================
# bitsandbytes detection (robust)  >>> FIX BNB/TRITON
# =========================
def _force_disable_bnb_in_peft():
    # Ã‰p PEFT coi nhÆ° khÃ´ng cÃ³ bnb, trÃ¡nh import bnb -> triton.ops
    try:
        from peft.utils import import_utils as _iu
        _iu.is_bnb_available = lambda: False
    except Exception:
        pass

def detect_bnb():
    """
    Tráº£ vá»� dict: {'available': bool, 'mode': '4bit'|'8bit'|None, 'config': cfg|None}
    Chá»‰ 'available' náº¿u cáº£ bitsandbytes vÃ  triton.ops import OK.
    """
    info = {'available': False, 'mode': None, 'config': None}
    try:
        import importlib
        # Pháº£i import Ä‘Æ°á»£c cáº£ triton vÃ  triton.ops (káº»o PEFT sáº½ lá»—i)
        spec_triton = importlib.util.find_spec("triton")
        spec_tops   = importlib.util.find_spec("triton.ops")
        if spec_triton is None or spec_tops is None:
            print("â„¹ï¸� bnb disabled: Triton or triton.ops not present.")
            _force_disable_bnb_in_peft()
            return info

        import bitsandbytes as _  # noqa
        from transformers import BitsAndBytesConfig
    except Exception as e:
        print(f"â„¹ï¸� bitsandbytes not available -> will train without k-bit. ({e})")
        _force_disable_bnb_in_peft()
        return info

    # Thá»­ 4-bit
    try:
        cfg4 = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        info.update({'available': True, 'mode': '4bit', 'config': cfg4})
        return info
    except Exception as e:
        print(f"â„¹ï¸� cannot make 4-bit config: {e}")
        try:
            cfg8 = BitsAndBytesConfig(load_in_8bit=True)
            info.update({'available': True, 'mode': '8bit', 'config': cfg8})
        except Exception as e2:
            print(f"â„¹ï¸� cannot make 8-bit config: {e2}")
            _force_disable_bnb_in_peft()
        return info

BNB = detect_bnb()

# Danh sÃ¡ch model theo tÃ¬nh tráº¡ng BNB
if BNB['available']:
    PREF_MODELS = [ENV_MODEL, "Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"]
else:
    PREF_MODELS = ["Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"]

def common_load_kwargs(quantize: bool):
    kw = {"trust_remote_code": True}
    if LOCAL_ONLY: kw["local_files_only"] = True
    kw["device_map"] = {"": 0} if DEVICE == "cuda" else "cpu"
    if quantize and BNB["available"] and BNB["config"] is not None and DEVICE == "cuda":
        kw["quantization_config"] = BNB["config"]
    else:
        kw["torch_dtype"] = torch.float16 if DEVICE == "cuda" else torch.float32
    return kw

# =========================
# Data utils
# =========================
def load_jsonl_all(path: str):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            data.append(json.loads(line))
    return data

def sort_indices_by_qrel(qrel: Dict[str,int]) -> List[int]:
    pairs = [(int(k), int(v)) for k,v in qrel.items()]
    pairs.sort(key=lambda x: (-x[1], x[0]))
    return [k for k,_ in pairs]

def build_chat_prompt(system_prompt: str, user_prompt: str) -> str:
    try:
        return tokenizer.apply_chat_template(
            [{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}],
            tokenize=False, add_generation_prompt=True
        )
    except Exception:
        sp = f"[SYSTEM]\n{system_prompt}\n\n" if system_prompt else ""
        return sp + f"[USER]\n{user_prompt}\n\n[ASSISTANT]\n"

def make_sft_sample(messages: List[Dict], target_list: List[int]) -> Dict[str,str]:
    user = "".join([(m.get("content") or "") for m in messages if m.get("role","user")=="user"])
    answer = "[" + ", ".join(str(x) for x in target_list) + "]"
    return {"user": user, "answer": answer}

def tok_chat(sample: Dict[str,str], max_len: int):
    sys_prompt = "You are a helpful financial analyst. Return only a Python-style list of integers in brackets."
    prompt = build_chat_prompt(sys_prompt, sample["user"])
    full = prompt + sample["answer"]
    enc = tokenizer(full, truncation=True, max_length=max_len, padding=False, add_special_tokens=False, return_tensors="pt")
    with tokenizer.as_target_tokenizer():
        ans_ids = tokenizer(sample["answer"], add_special_tokens=False).input_ids
    input_ids = enc.input_ids[0]
    labels = input_ids.clone()
    ans_len = len(ans_ids)
    labels[:-ans_len] = -100
    return {"input_ids": input_ids, "labels": labels, "attention_mask": enc.attention_mask[0]}

class RagRankDataset(Dataset):
    def __init__(self, samples, max_len=2048):
        self.samples = samples; self.max_len = max_len
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return tok_chat(self.samples[idx], self.max_len)

def data_collator(features):
    batch = {}
    keys = features[0].keys()
    max_len = max(f["input_ids"].shape[0] for f in features)
    for k in keys:
        pad_id = -100 if k=="labels" else (tokenizer.pad_token_id if k=="input_ids" else 0)
        batch[k] = torch.stack([
            torch.nn.functional.pad(f[k], (0, max_len - f[k].shape[0]), value=pad_id)
            for f in features
        ])
    return batch

def truncate_text_tokens(text: str, max_tokens: int) -> str:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens: return text
    ids = ids[-max_tokens:]
    return tokenizer.decode(ids, skip_special_tokens=True)

# =========================
# Build two train sets (USE ALL SAMPLES)
# =========================
print("ğŸ“¥ Loading full dev (train) sets...")
doc_train_samples, chunk_train_samples = [], []
for ex in load_jsonl_all(DOC_DEV):
    if "qrel" in ex:
        doc_train_samples.append(make_sft_sample(ex["messages"], sort_indices_by_qrel(ex["qrel"])))
for ex in load_jsonl_all(CHUNK_DEV):
    if "qrel" in ex:
        chunk_train_samples.append(make_sft_sample(ex["messages"], sort_indices_by_qrel(ex["qrel"])))
print(f"âœ… Train sets ready (FULL): doc={len(doc_train_samples)} | chunk={len(chunk_train_samples)}")

# =========================
# Model loaders (single-GPU) + fallback
# =========================
def try_load_model(name: str, quantize: bool, train_mode: bool):
    kw = common_load_kwargs(quantize=quantize)
    print(f"â�³ Loading base: {name} | quantize={quantize and BNB['mode']} | device_map={kw.get('device_map')} ...")
    model = AutoModelForCausalLM.from_pretrained(name, **kw)
    try: model.config.attn_implementation = "sdpa"
    except Exception: pass
    if train_mode and hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
    model.config.use_cache = False if train_mode else (True if DEVICE=="cuda" else False)
    return model

def pick_working_model(train_mode: bool):
    for mname in PREF_MODELS:
        # a) k-bit náº¿u cÃ³
        if DEVICE == "cuda" and BNB["available"] and BNB["config"] is not None:
            try:
                model = try_load_model(mname, quantize=True, train_mode=train_mode)
                print(f"âœ… Loaded (k-bit, single-GPU): {mname} ({BNB['mode']})")
                return mname, model, True
            except Exception as e:
                print(f"âš ï¸� Fail k-bit {mname}: {e}")
                if torch.cuda.is_available(): torch.cuda.empty_cache()
        # b) fp16 fallback
        try:
            model = try_load_model(mname, quantize=False, train_mode=train_mode)
            print(f"âœ… Loaded (fp16, single-GPU): {mname}")
            return mname, model, False
        except Exception as e:
            print(f"âš ï¸� Fail fp16 {mname}: {e}")
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    raise RuntimeError("â�Œ Could not load any model candidate on single GPU.")

# =========================
# LoRA builder
# =========================
def build_lora(model, is_kbit: bool):
    # Náº¿u lÃ  k-bit, chuáº©n bá»‹ theo k-bit; náº¿u khÃ´ng, train FP16/FP32 bÃ¬nh thÆ°á»�ng.
    if is_kbit:
        model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_cfg)

# =========================
# Time limiter callback
# =========================
class TimeLimiterCallback(TrainerCallback):
    def __init__(self, stop_time_epoch_seconds: float, label: str):
        self.stop_ts = stop_time_epoch_seconds
        self.label = label
    def on_step_end(self, args, state, control, **kwargs):
        if time.time() >= self.stop_ts:
            print(f"\nâ�³ Time budget reached for {self.label}. Stopping at global_step={state.global_step}.")
            control.should_training_stop = True
            control.should_epoch_stop = True
        return control

# =========================
# Trainer (Ä‘Æ°a tensors vá»� Ä‘Ãºng device)
# =========================
class SimpleCLMTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        dev = next(model.parameters()).device
        for k, v in list(inputs.items()):
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(dev)
        out = model(**inputs)
        loss = out.loss
        return (loss, out) if return_outputs else loss

def train_one(samples: List[Dict[str,str]], out_dir: str, stop_ts: float, phase_label: str):
    ds = RagRankDataset(samples, max_len=MAX_SEQ_LEN)
    steps_per_epoch = max(1, len(ds) // max(1, (BATCH_SIZE * GRAD_ACCUM)))
    warmup_steps = max(10, steps_per_epoch)

    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        logging_steps=10,
        save_strategy="epoch" if SAVE_STEPS == 0 else "steps",
        save_steps=None if SAVE_STEPS == 0 else SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        bf16=False,
        fp16=(DEVICE=="cuda"),
        optim="adamw_torch",
        adam_beta1=BETA1, adam_beta2=BETA2, adam_epsilon=EPS,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=0.3,
        gradient_checkpointing=True,
        report_to=[],
        dataloader_pin_memory=False,
        ddp_find_unused_parameters=False,
    )

    base_name, base, is_kbit = pick_working_model(train_mode=True)

    # Rebind tokenizer náº¿u base Ä‘á»•i
    global tokenizer
    if base_name != ENV_MODEL:
        tk_kwargs = {"trust_remote_code": True}
        if LOCAL_ONLY: tk_kwargs["local_files_only"] = True
        tokenizer = AutoTokenizer.from_pretrained(base_name, **tk_kwargs)
        tokenizer.padding_side = "right"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    model = build_lora(base, is_kbit=is_kbit)
    eff_bs = BATCH_SIZE * GRAD_ACCUM
    print(f"ğŸ§© LoRA attached [{phase_label}] | eff_batch={eff_bs} | k-bit={is_kbit}")

    trainer = SimpleCLMTrainer(model=model, args=args, train_dataset=ds, data_collator=data_collator)
    trainer.add_callback(TimeLimiterCallback(stop_ts, label=phase_label))

    resume = RESUME if (RESUME and os.path.isdir(RESUME)) else None
    trainer.train(resume_from_checkpoint=resume)

    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    if SAVE_MERGED_MODEL:
        print(f"ğŸ§ª Merging LoRA -> {out_dir} ...")
        merged = model.merge_and_unload()
        merged_dir = MERGED_DIR_DOC if out_dir == LORA_DIR_DOC else MERGED_DIR_CHUNK
        os.makedirs(merged_dir, exist_ok=True)
        merged.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        print(f"ğŸ’¾ Merged model saved: {merged_dir}")

    del trainer, model, base
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# =========================
# Inference helpers (OOM-safe)
# =========================
def build_chat_for_infer(user_prompt: str) -> str:
    sys_prompt = "You are a helpful financial analyst. Return only a Python-style list of integers in brackets."
    return build_chat_prompt(sys_prompt, user_prompt)

def parse_index_list(text: str) -> List[int]:
    if not text: return []
    m = re.search(r'\[([^\]]+)\]', text)
    if m:
        nums = re.findall(r'-?\d+', m.group(1))
        if nums: return [int(x) for x in nums]
    nums = re.findall(r'-?\d+', text)
    return [int(x) for x in nums] if nums else []

def msgs_to_user_text(messages: List[Dict]) -> str:
    return "".join([(m.get("content") or "") for m in messages if m.get("role","user")=="user"])

@torch.inference_mode()
def generate_indices_from_text(model, user_text: str, max_new_tokens: int = MAX_NEW_TOKENS_INFER) -> List[int]:
    prompt = build_chat_for_infer(user_text)
    prompt = truncate_text_tokens(prompt, INFER_MAX_INPUT_TOKENS)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=INFER_MAX_INPUT_TOKENS).to(next(model.parameters()).device)
    gen_kwargs = dict(
        max_new_tokens=min(max_new_tokens, 128),
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    try:
        if next(model.parameters()).is_cuda:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                out = model.generate(**inputs, **gen_kwargs)
        else:
            out = model.generate(**inputs, **gen_kwargs)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            gen_kwargs["max_new_tokens"] = 32
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                               max_length=INFER_MAX_INPUT_TOKENS//2).to(next(model.parameters()).device)
            out = model.generate(**inputs, **gen_kwargs)
        else:
            raise
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return parse_index_list(text)

def extract_top_k(arr: List[int], k: int, fallback_n: int) -> List[int]:
    seen, out = set(), []
    for x in arr:
        if x not in seen:
            out.append(x); seen.add(x)
        if len(out) >= k: break
    p = 0
    while len(out) < k and p < fallback_n:
        if p not in seen: out.append(p); seen.add(p)
        p += 1
    return out

# ---------- Chunk two-stage ----------
_CHUNK_PAT = re.compile(r'\[Chunk Index (\d+)\]\s*([\s\S]*?)(?=(?:\[Chunk Index |\nTask:|$))')

def _extract_question_and_chunks(full: str) -> Tuple[Optional[str], List[int], List[str]]:
    q = None
    m = re.search(r'Question:\s*(.*)', full)
    if m: q = m.group(1).strip()
    idxs, texts = [], []
    for idx_str, txt in _CHUNK_PAT.findall(full):
        idxs.append(int(idx_str)); texts.append(txt.strip())
    return q, idxs, texts

def _mk_chunk_prompt(question: str, chunk_texts: List[str], chunk_idxs: List[int], k: int) -> str:
    k = max(1, min(k, len(chunk_idxs)))
    s = [f"Identify the {k} most relevant text chunks for answering this question, then rank them best-first.",
         f"Question: {question}", "Text chunks:"]
    for ci, ct in zip(chunk_idxs, chunk_texts):
        s.append(f"[Chunk Index {ci}] {ct}")
    s.append(f"Task: Return indices only in the format: [{', '.join(['idx']*k)}]")
    return "\n".join(s)

def chunk_two_stage_infer(model, full_user_text: str, batch_size: int = 300, k_each: int = 10, final_k: int = 20) -> List[int]:
    q, idxs, texts = _extract_question_and_chunks(full_user_text)
    if not q or not idxs:
        return generate_indices_from_text(model, full_user_text, MAX_NEW_TOKENS_INFER)

    cand_idxs: List[int] = []
    N = len(idxs)
    for start in range(0, N, batch_size):
        sub_i = idxs[start:start+batch_size]
        sub_t = texts[start:start+batch_size]
        sub_prompt = _mk_chunk_prompt(q, sub_t, sub_i, k=min(k_each, len(sub_i)))
        sub_prompt = truncate_text_tokens(sub_prompt, INFER_MAX_INPUT_TOKENS)
        sub_pred = generate_indices_from_text(model, sub_prompt, max_new_tokens=64)
        okset = set(sub_i)
        sub_pred = [x for x in sub_pred if x in okset][:k_each]
        cand_idxs.extend(sub_pred)

    # dedup order
    seen, cand_unique = set(), []
    for x in cand_idxs:
        if x not in seen: seen.add(x); cand_unique.append(x)

    # re-rank candidates
    ci2txt = {i:t for i,t in zip(idxs,texts)}
    final_idxs = cand_unique[:final_k*5] if len(cand_unique) > final_k*5 else cand_unique
    final_texts = [ci2txt[i] for i in final_idxs]
    final_prompt = _mk_chunk_prompt(q, final_texts, final_idxs, k=min(final_k, len(final_idxs)))
    final_prompt = truncate_text_tokens(final_prompt, INFER_MAX_INPUT_TOKENS)
    final_pred = generate_indices_from_text(model, final_prompt, max_new_tokens=64)
    okset2 = set(final_idxs)
    final_pred = [x for x in final_pred if x in okset2]
    return final_pred

# =========================
# Inference pipeline
# =========================
def load_base_for_infer():
    name, base, is_kbit = pick_working_model(train_mode=False)
    print(f"ğŸ”§ Inference base: {name} | k-bit={is_kbit} | device={next(base.parameters()).device}")
    base.eval()
    return base

def attach_lora(base, lora_dir):
    model = PeftModel.from_pretrained(base, lora_dir)
    model.eval()
    return model

def predict_and_build_submission():
    rows = []
    print("\nâ�³ Loading base model for inference...")
    base_infer = load_base_for_infer()

    # ---------- Document ----------
    print(f"\nğŸ“„ Attach DOC LoRA -> {LORA_DIR_DOC}")
    model_doc = attach_lora(base_infer, LORA_DIR_DOC)
    print("ğŸ§ª Inference: document_ranking_kaggle_eval.jsonl")
    for ex in load_jsonl_all(DOC_EVAL):
        qid = ex.get("_id") or ex.get("id") or ex.get("sample_id")
        user_text = msgs_to_user_text(ex["messages"])
        user_text = truncate_text_tokens(user_text, INFER_MAX_INPUT_TOKENS)
        pred = generate_indices_from_text(model_doc, user_text, max_new_tokens=MAX_NEW_TOKENS_INFER)
        top5 = extract_top_k(pred, 5, fallback_n=10)
        for idx in top5:
            rows.append({"sample_id": qid, "target_index": idx})

    del model_doc
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ---------- Chunk ----------
    print(f"\nğŸ”— Attach CHUNK LoRA -> {LORA_DIR_CHUNK}")
    model_chunk = attach_lora(base_infer, LORA_DIR_CHUNK)
    print("ğŸ”� Inference: chunk_ranking_kaggle_eval.jsonl")
    for ex in load_jsonl_all(CHUNK_EVAL):
        qid = ex.get("_id") or ex.get("id") or ex.get("sample_id")
        full_user = msgs_to_user_text(ex["messages"])
        n_chunks = len(_CHUNK_PAT.findall(full_user))
        if n_chunks >= 200:
            pred = chunk_two_stage_infer(model_chunk, full_user, batch_size=300, k_each=10, final_k=20)
        else:
            full_user = truncate_text_tokens(full_user, INFER_MAX_INPUT_TOKENS)
            pred = generate_indices_from_text(model_chunk, full_user, max_new_tokens=MAX_NEW_TOKENS_INFER)
        top5 = extract_top_k(pred, 5, fallback_n=max(200, n_chunks if n_chunks>0 else 200))
        for idx in top5:
            rows.append({"sample_id": qid, "target_index": idx})

    del model_chunk, base_infer
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    with open(SUBMISSION_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["sample_id","target_index"])
        for r in rows: w.writerow([r["sample_id"], r["target_index"]])
    print(f"ğŸ’¾ Saved submission -> {SUBMISSION_CSV} | rows={len(rows)}")

# =========================
# Run (with time budgeting)
# =========================
if __name__ == "__main__":
    job_t0 = time.time()

    total_train_budget_sec = int(MAX_TRAIN_HOURS * 3600)
    infer_reserve_sec = int(INFER_RESERVE_MIN * 60)
    print(f"â�±ï¸�  Total train budget: {total_train_budget_sec//3600}h{(total_train_budget_sec%3600)//60}m | reserve: {INFER_RESERVE_MIN}m")

    # Sáºµn dataset (Ä‘Ã£ load á»Ÿ trÃªn)
    n_doc, n_chunk = len(doc_train_samples), len(chunk_train_samples)
    n_total = max(1, n_doc + n_chunk)

    # Chia thá»�i gian DOC vs CHUNK
    if TIME_SPLIT.lower() == "auto":
        doc_share = n_doc / n_total
        chunk_share = n_chunk / n_total
    else:
        try:
            a, b = TIME_SPLIT.split(":")
            a, b = float(a), float(b)
            s = max(1e-9, a + b)
            doc_share, chunk_share = a/s, b/s
        except Exception:
            doc_share = n_doc / n_total
            chunk_share = n_chunk / n_total

    doc_budget   = int(total_train_budget_sec * doc_share)
    chunk_budget = int(total_train_budget_sec * chunk_share)
    now = time.time()
    doc_deadline = now + doc_budget
    chunk_deadline = doc_deadline + chunk_budget

    print(f"ğŸ•’ Budget split â†’ DOC ~ {doc_budget//3600}h{(doc_budget%3600)//60}m | CHUNK ~ {chunk_budget//3600}h{(chunk_budget%3600)//60}m")

    # Train DOC
    print("ğŸš€ Train LoRA for Document Ranking (FULL DATA, time-limited) ...")
    train_one(doc_train_samples, LORA_DIR_DOC, stop_ts=doc_deadline, phase_label="DOC")

    # Train CHUNK (báº£o toÃ n quá»¹ thá»�i gian tá»•ng)
    now = time.time()
    chunk_stop_ts = min(job_t0 + total_train_budget_sec, now + max(0, chunk_deadline - now))
    print("ğŸš€ Train LoRA for Chunk Ranking (FULL DATA, time-limited) ...")
    train_one(chunk_train_samples, LORA_DIR_CHUNK, stop_ts=chunk_stop_ts, phase_label="CHUNK")

    # Inference
    print("â–¶ï¸�  Start inference & submission build ...")
    predict_and_build_submission()
    print("\nğŸ�‰ Done. Submit 'kaggle_submission.csv'.")
    print(f"ğŸ“¦ LoRA (doc)   @ {LORA_DIR_DOC}")
    print(f"ğŸ“¦ LoRA (chunk) @ {LORA_DIR_CHUNK}")
    if SAVE_MERGED_MODEL:
        print(f"ğŸ“¦ Merged (doc)   @ {MERGED_DIR_DOC}")
        print(f"ğŸ“¦ Merged (chunk) @ {MERGED_DIR_CHUNK}")


