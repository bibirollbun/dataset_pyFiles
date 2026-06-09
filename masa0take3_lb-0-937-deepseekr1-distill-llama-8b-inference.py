import os
from tqdm import tqdm
import pandas as pd, numpy as np
from sklearn.preprocessing import LabelEncoder

import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import softmax

# ------------------
# Paths & constants
# ------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

TRAIN_PATH = "/kaggle/input/map-charting-student-math-misunderstandings/train.csv"
TEST_PATH  = "/kaggle/input/map-charting-student-math-misunderstandings/test.csv"

# Base model used during training (deepseek-r1-distill-llama)
BASE_MODEL = "/kaggle/input/deepseek-r1/transformers/deepseek-r1-distill-llama-8b/2"

# State dict (.pth) saved in the training notebook (only requires_grad parameters)
LORA_PTH   = "/kaggle/input/deepseekr1-distill-llama-8b-tpu-training/deepseekrllama_v1_epoch_2.pth"

# NEW: classes saved during training (np.save(..., le.classes_))
CLASSES_NPY = "/kaggle/input/deepseekr1-distill-llama-8b-tpu-training/label_classes.npy"  # <-- update if different

MAX_LEN    = 256
BATCH_SIZE = 16
LORA_RANK  = 8
LORA_ALPHA = 16
DROPOUT    = 0.05

# ------------------
# Data preparation (same preprocessing and label order as training)
# ------------------
def format_input(row):
    x = "This is Correct answer." if row['is_correct']==1 else "This is Incorrect answer."
    return (
        f"• Question: {row['QuestionText']}\n"
        f"• Answer: {row['MC_Answer']}\n"
        f"• Correctness: {x}\n"
        f"• Student Explanation: {row['StudentExplanation']}"
    )

def load_labelencoder_or_fit(train_df: pd.DataFrame, classes_npy_path: str) -> tuple[LabelEncoder, int]:
    """
    LabelEncoder fix A:
      - Prefer loading classes saved at training time (classes_npy_path)
      - Fallback to fitting on train if the file is missing
    """
    le = LabelEncoder()
    if os.path.exists(classes_npy_path):
        classes = np.load(classes_npy_path, allow_pickle=True)
        le.classes_ = classes
        n_classes = len(classes)
        print(f"[LE] Loaded {n_classes} classes from {classes_npy_path}")
    else:
        print(f"[LE] classes_npy not found: {classes_npy_path} -> fitting on train as fallback")
        td = train_df.copy()
        td["Misconception"] = td["Misconception"].fillna("NA")
        td["target"] = td["Category"] + ":" + td["Misconception"]
        le.fit(td["target"].values)
        n_classes = len(le.classes_)
        print(f"[LE] Fitted on train with {n_classes} classes")
    return le, n_classes

train = pd.read_csv(TRAIN_PATH)
train.Misconception = train.Misconception.fillna("NA")
train["target"] = train["Category"] + ":" + train["Misconception"]

# LabelEncoder fix A (load classes from training, else fit)
le, n_classes = load_labelencoder_or_fit(train, CLASSES_NPY)

# Attach is_correct to test (same logic as training)
idx_true = train["Category"].str.split("_").str[0].eq("True")
correct = train.loc[idx_true, ["QuestionId","MC_Answer"]].copy()
correct["c"] = correct.groupby(["QuestionId","MC_Answer"])["MC_Answer"].transform("count")
correct = correct.sort_values(["QuestionId","c"], ascending=[True, False]).drop_duplicates(["QuestionId"])
correct = correct[["QuestionId","MC_Answer"]].copy()
correct["is_correct"] = 1

test = pd.read_csv(TEST_PATH)
test = test.merge(correct, on=["QuestionId","MC_Answer"], how="left")
test["is_correct"] = test["is_correct"].fillna(0).astype(int)
test["text"] = test.apply(format_input, axis=1)

# ------------------
# Tokenizer (same settings as training)
# ------------------
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None and tokenizer.eos_token is not None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors=None
    )

ds_test = Dataset.from_pandas(test[["text"]])
ds_test = ds_test.map(tokenize, batched=True, remove_columns=["text"])

collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding="max_length",
    max_length=MAX_LEN,
    return_tensors="pt"
)

dataloader = DataLoader(
    ds_test,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collator,
    pin_memory=True,
    num_workers=2
)

# ------------------
# Wrapper: CausalLM + custom classification head (same as training)
# ------------------
class CausalLMSequenceClassifier(nn.Module):
    def __init__(self, base_lm, num_labels: int):
        super().__init__()
        self.base = base_lm
        self.config = base_lm.config
        self.num_labels = num_labels
        hidden_size = getattr(self.base.config, "hidden_size", None) or getattr(self.base.config, "hidden_dim", None)
        if hidden_size is None:
            raise ValueError("hidden_size not found in config.")
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        return_dict = return_dict if return_dict is not None else getattr(self.config, "use_return_dict", True)
        outputs = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
            **kwargs
        )
        last_hidden = outputs.hidden_states[-1]  # [B, T, H]
        if attention_mask is None:
            pooled = last_hidden.mean(dim=1)
        else:
            mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
            pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        logits = self.classifier(pooled)
        if not return_dict:
            return (logits,)
        return SequenceClassifierOutput(logits=logits)

# 1) Load base LM (CausalLM)
base_lm = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    torch_dtype=torch.float16,   # or bfloat16; on GPU fp16 is typically safer
    device_map="auto"
)

# 2) Wrap to classifier with num_labels matching the encoder classes
wrapped = CausalLMSequenceClassifier(base_lm, num_labels=n_classes)
wrapped.base.config.pad_token_id = tokenizer.pad_token_id

# 3) LoRA config identical to training (modules_to_save=['classifier'])
lora_cfg = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=DROPOUT,
    bias="none",
    inference_mode=False,               # keep False so modules_to_save is effective
    task_type=TaskType.SEQ_CLS,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    modules_to_save=["classifier"]
)

model = get_peft_model(wrapped, lora_cfg)

# 4) Load training state_dict (.pth) as-is
sd = torch.load(LORA_PTH, map_location="cpu")
missing, unexpected = model.load_state_dict(sd, strict=False)
print("missing keys (first 10):", missing[:10], " ...", len(missing))
print("unexpected keys (first 10):", unexpected[:10], " ...", len(unexpected))

# Make classifier dtype/device consistent with the base
base_dtype = next(model.parameters()).dtype
dev = next(model.parameters()).device
model.base_model.model.classifier.to(device=dev, dtype=base_dtype)

# Ensure inference mode
model.eval()
device = next(model.parameters()).device

# ------------------
# Inference
# ------------------
all_logits = []
with torch.no_grad():
    for batch in tqdm(dataloader, desc="Inference"):
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        logits = out.logits.float().cpu().numpy()
        all_logits.append(logits)

pred = np.concatenate(all_logits, axis=0)
probs = softmax(pred, axis=1)

# Decode top-3 using the exact same label space as training
top_idx = np.argsort(-probs, axis=1)[:, :3]
decoded = le.inverse_transform(top_idx.flatten()).reshape(top_idx.shape)
joined = [" ".join(row) for row in decoded]

sub = pd.DataFrame({
    "row_id": test["row_id"].values,
    "Category:Misconception": joined
})
sub.to_csv("submission.csv", index=False)
print(sub.head())


