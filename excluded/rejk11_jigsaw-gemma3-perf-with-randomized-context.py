%%capture
!pip install --upgrade -qqq uv
try: import numpy; get_numpy = f"numpy=={numpy.__version__}"
except: get_numpy = "numpy"
try: import subprocess; is_t4 = "Tesla T4" in str(subprocess.check_output(["nvidia-smi"]))
except: is_t4 = False
get_vllm, get_triton = ("vllm==0.10.1", "triton==3.2.0") if is_t4 else ("vllm", "triton")
!uv pip install -qqq --upgrade     unsloth {get_vllm} {get_numpy} torchvision bitsandbytes xformers 
!uv pip install -qqq {get_triton}
!uv pip install "huggingface_hub>=0.34.0" "datasets>=3.4.1,<4.0.
!uv pip install transformers==4.55.4
!pip install trl==0.19.1 -U



from unsloth import FastModel
import torch
model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-3-4b-it",
    max_seq_length = 1024, # Choose any for long context!
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    # token = "hf_...", # use one if using gated models
)


model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # Turn off for just text!
    finetune_language_layers   = True,  # Should leave on!
    finetune_attention_modules = True,  # Attention good for GRPO
    finetune_mlp_modules       = True,  # SHould leave on always!

    r = 8,           # Larger = higher accuracy, but might overfit
    lora_alpha = 8,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
)


from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3",
)


import datasets as ds 
dataset=ds.Dataset.from_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
SYS_PROMPT = """
You are given a comment on reddit. Your task is to classify if it violates the given rule. Only respond Yes/No.
"""
def remap(row):
    text = f"""
    r/{row['subreddit']}
    Rule: {row['rule']}
    
    1) {row['positive_example_1']}
    Violation: Yes
    
    2) {row['positive_example_2']}
    Violation: Yes
    
    3) {row['negative_example_1']}
    Violation: No
    
    4) {row['negative_example_2']}
    Violation: No
    
    5) {row['body']}
    """
    messages = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": text},
        {"role": "assistant", "content": "Yes" if int(row["rule_violation"]) == 1 else "No"},
    ]
    return {
        'conversations':messages
    }
import os 
dataset=dataset.map(remap,num_proc=os.cpu_count())


import os
import random
import zlib
import datasets as ds

SYS_PROMPT = """
You are given a comment on reddit. Your task is to classify if it violates the given rule. Only respond Yes/No.
"""

# -----------------------------
# RNG helpers (deterministic)
# -----------------------------

def _rng_from_row(row):
    """Deterministic RNG per row so multiprocessing doesn't scramble randomness."""
    key = f"{row.get('subreddit','')}\n{row.get('rule','')}\n{row.get('body','')}"
    seed = zlib.adler32(key.encode("utf-8"))
    return random.Random(seed)

def _rng_from_text(*parts):
    key = "\n".join([p for p in parts if isinstance(p, str)])
    return random.Random(zlib.adler32(key.encode("utf-8")))

# -----------------------------
# Prompt builder
# -----------------------------

def _make_prompt(subreddit, rule, pos_samples, neg_samples, body):
    """Build the in-context prompt with however many exemplars we keep.
    Enumerate 1..N: positives, negatives, then the query body last.
    """
    lines = [f"r/{subreddit}", f"Rule: {rule}", ""]
    idx = 1
    for s in pos_samples:
        lines += [f"{idx}) {s}", "Violation: Yes", ""]
        idx += 1
    for s in neg_samples:
        lines += [f"{idx}) {s}", "Violation: No", ""]
        idx += 1
    lines += [f"{idx}) {body}"]  # the actual query is always last; no label line for it
    return "\n".join(lines)

# -----------------------------
# Main mapping: randomly drop exemplars in the base sample
# -----------------------------

def remap_with_random_drop(row):
    """Create the main training example but randomly drop some pos/neg exemplars.
    Returns a single conversations entry per input row.
    """
    rng = _rng_from_row(row)

    pos_all = [row.get("positive_example_1", ""), row.get("positive_example_2", "")]
    neg_all = [row.get("negative_example_1", ""), row.get("negative_example_2", "")]
    pos_all = [s for s in pos_all if isinstance(s, str) and s.strip()]
    neg_all = [s for s in neg_all if isinstance(s, str) and s.strip()]

    # Randomly decide how many to keep from each side (0..len)
    k_pos = rng.randrange(0, len(pos_all) + 1) if pos_all else 0
    k_neg = rng.randrange(0, len(neg_all) + 1) if neg_all else 0

    # Sample without replacement, deterministically via rng above
    pos_keep = rng.sample(pos_all, k_pos) if k_pos else []
    neg_keep = rng.sample(neg_all, k_neg) if k_neg else []

    text = _make_prompt(
        subreddit=row.get("subreddit", ""),
        rule=row.get("rule", ""),
        pos_samples=pos_keep,
        neg_samples=neg_keep,
        body=row.get("body", ""),
    )

    # Original label for the given body
    label = "Yes" if int(row.get("rule_violation", 0)) == 1 else "No"

    messages = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": text},
        {"role": "assistant", "content": label},
    ]
    return {"conversations": messages}

# Apply exemplar-dropping map (one row -> one row)
dataset_base = dataset.map(remap_with_random_drop, num_proc=os.cpu_count())

# -----------------------------
# Augmentation: turn pos/neg examples into new samples, sometimes with few-shot exemplars
# -----------------------------

def _build_augmented_conversation(subreddit, rule, body_text, gold_label,
                                  pos_pool, neg_pool):
    """Construct one augmented sample. Sometimes include a few-shot mix of exemplars.
    Excludes the chosen body from exemplar pools to avoid echoing.
    """
    rng = _rng_from_text(subreddit, rule, body_text)

    include_fewshot = rng.random() < 0.6  # ~60% chance to include exemplars
    pos_keep, neg_keep = [], []

    if include_fewshot:
        pos_cand = [s for s in pos_pool if s != body_text]
        neg_cand = [s for s in neg_pool if s != body_text]

        # Bias toward compact prompts (0-1 examples per side most of the time)
        k_pos = 0 if not pos_cand else rng.choices([0, 1, 2], [0.5, 0.35, 0.15])[0]
        k_neg = 0 if not neg_cand else rng.choices([0, 1, 2], [0.5, 0.35, 0.15])[0]

        if k_pos:
            k_pos = min(k_pos, len(pos_cand))
            pos_keep = rng.sample(pos_cand, k_pos)
        if k_neg:
            k_neg = min(k_neg, len(neg_cand))
            neg_keep = rng.sample(neg_cand, k_neg)

        # Small chance to swap ordering to reduce position bias
        if rng.random() < 0.2:
            pos_keep, neg_keep = neg_keep, pos_keep

    user_text = _make_prompt(
        subreddit=subreddit or "",
        rule=rule or "",
        pos_samples=pos_keep,
        neg_samples=neg_keep,
        body=(body_text or "").strip(),
    )

    return {
        "conversations": [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": gold_label},
        ]
    }


def generate_posneg_augmentations(src_dataset):
    """Iterate rows, producing extra samples where a pos/neg example becomes the body.
    Returns a plain Python list of dicts suitable for Dataset.from_list.
    """
    augmented_rows = []

    # Use streaming iterator to control memory on large datasets
    for row in src_dataset.to_iterable_dataset():
        subreddit = row.get("subreddit", "")
        rule = row.get("rule", "")

        # Pools
        pos_pool = [row.get("positive_example_1", ""), row.get("positive_example_2", "")]
        neg_pool = [row.get("negative_example_1", ""), row.get("negative_example_2", "")]
        pos_pool = [s for s in pos_pool if isinstance(s, str) and s.strip()]
        neg_pool = [s for s in neg_pool if isinstance(s, str) and s.strip()]

        # Add up to four augmented samples (p1, p2 -> Yes; n1, n2 -> No)
        for ex in pos_pool:
            augmented_rows.append(
                _build_augmented_conversation(
                    subreddit, rule, ex, "Yes", pos_pool, neg_pool
                )
            )
        for ex in neg_pool:
            augmented_rows.append(
                _build_augmented_conversation(
                    subreddit, rule, ex, "No", pos_pool, neg_pool
                )
            )

    return augmented_rows

# Build augmented dataset from list
_aug_rows = generate_posneg_augmentations(dataset)
print(len(_aug_rows))

if _aug_rows:
    dataset_aug = ds.Dataset.from_list(_aug_rows)
    # Concatenate original (with random drop) + augmented rows
    dataset = ds.concatenate_datasets([dataset_base, dataset_aug])
else:
    dataset = dataset_base

# Optional: shuffle for training
# dataset = dataset.shuffle(seed=42)

# If you want to inspect a few rows:
# print(dataset[0]["conversations"])



from unsloth.chat_templates import standardize_data_formats
dataset = standardize_data_formats(dataset)


def formatting_prompts_func(examples):
   convos = examples["conversations"]
   texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False).removeprefix('<bos>') for convo in convos]
   return { "text" : texts }

dataset = dataset.map(formatting_prompts_func, batched = True)


import numpy as np

# Compute length in tokens for each example
lengths = [len(tokenizer(x["text"]).input_ids[0]) for x in dataset]
print("Max token length:", max(lengths))
print("Mean token length:", np.mean(lengths))
print("95th percentile:", np.percentile(lengths, 95))


tokenizer.apply_chat_template(dataset[0]["conversations"],tokenize = False, add_generation_prompt = False).removeprefix('<bos>')


from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = None, # Can set up evaluation!
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4, # Use GA to mimic batch size!
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 30,
        learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none", # Use this for WandB etc
    ),
)


from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<start_of_turn>user\n",
    response_part = "<start_of_turn>model\n",
)


trainer_stats = trainer.train()


model.save_pretrained("/kaggle/working/gemma-3")
tokenizer.save_pretrained("/kaggle/working/gemma-3")

