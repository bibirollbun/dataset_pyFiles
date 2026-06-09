!pip install --upgrade --no-index --find-links=/kaggle/input/transformers-4-56-1-and-deps transformers -qq


import os
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from peft import PeftModel
from datasets import Dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# Configuration
config = {
    "train_path": "/kaggle/input/map-charting-student-math-misunderstandings/train.csv",
    "test_path": "/kaggle/input/map-charting-student/test.csv",
    "base_model_path": "/kaggle/input/hunyuan-7b-instruct-bf16",
    "inference_model_dir": "/kaggle/input/hunyuan-7b-instruct-map",
    "max_length": 256,
}


# Data Processing
class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.le = None
        self.correct_lookup = None
        self.train_df = None
        self.test_df = None
        self.is_preprocessed = False

    def load_data(self):
        self.train_df = pd.read_csv(self.config['train_path'])
        self.test_df = pd.read_csv(self.config['test_path'])

    def get_num_classes(self):
        if not self.is_preprocessed:
            raise RuntimeError("Data must be preprocessed first.")
        return self.train_df['label'].nunique()

    def get_label_encoder(self):
        if self.le is None:
            raise RuntimeError("LabelEncoder not initialized.")
        return self.le

    @staticmethod
    def format_input(row):
        correct_text = "Yes" if row['IsCorrect'] else "No"
        return (
            f"Question: {row['QuestionText']}\n"
            f"Answer: {row['MC_Answer']}\n"
            f"Correct? {correct_text}\n"
            f"Student Explanation: {row['StudentExplanation']}\n"
        )

    def preprocess(self):
        self.load_data()

        self.train_df['Misconception'] = self.train_df['Misconception'].fillna('NA')
        self.train_df['target'] = self.train_df['Category'] + ':' + self.train_df['Misconception']
        correct_samples = self.train_df[self.train_df['Category'].str.startswith('True', na=False)].copy()
        correct_samples['count'] = correct_samples.groupby(['QuestionId', 'MC_Answer'])['MC_Answer'].transform('count')
        most_popular_correct = correct_samples.sort_values('count', ascending=False).drop_duplicates(['QuestionId'])

        self.correct_lookup = most_popular_correct[['QuestionId', 'MC_Answer']].copy()
        self.correct_lookup['IsCorrect_flag'] = True

        self.train_df = self.train_df.merge(self.correct_lookup, on=['QuestionId', 'MC_Answer'], how='left')
        self.train_df['IsCorrect'] = self.train_df['IsCorrect_flag'].notna()
        self.train_df = self.train_df.drop(columns=['IsCorrect_flag'])

        self.le = LabelEncoder()
        self.train_df['label'] = self.le.fit_transform(self.train_df['target'])
        self.train_df['text'] = self.train_df.apply(self.format_input, axis=1)

        self.is_preprocessed = True
        print(f"Train shape: {self.train_df.shape} with {self.get_num_classes()} target classes")
        return self.train_df

    def process_for_inference(self):
        if not self.is_preprocessed:
            raise RuntimeError("Training data must be preprocessed first.")

        self.test_df = self.test_df.merge(self.correct_lookup, on=['QuestionId', 'MC_Answer'], how='left')
        self.test_df['IsCorrect'] = self.test_df['IsCorrect_flag'].notna()
        self.test_df = self.test_df.drop(columns=['IsCorrect_flag'])
        self.test_df['text'] = self.test_df.apply(self.format_input, axis=1)
        return self.test_df


print("="*60)
print("STEP 1: HUNYUAN-7B INFERENCE")
print("="*60)

data_processor = DataProcessor(config)
train_df = data_processor.preprocess()
tokenizer = AutoTokenizer.from_pretrained(config['inference_model_dir'], trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForSequenceClassification.from_pretrained(
    config['base_model_path'],
    num_labels=data_processor.get_num_classes(),
    device_map="auto",
    torch_dtype=torch.float16
)

model = PeftModel.from_pretrained(base_model, config['inference_model_dir'])
model.config.pad_token_id = tokenizer.pad_token_id

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=config['max_length']
    )

test_df = data_processor.process_for_inference()
ds_test = Dataset.from_pandas(test_df[['text']])
ds_test = ds_test.map(tokenize_function, batched=True)

inference_args = TrainingArguments(
    output_dir="./results_infer",
    do_train=False,
    do_eval=False,
    per_device_eval_batch_size=16,
    fp16=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=inference_args,
    tokenizer=tokenizer
)

predictions = trainer.predict(ds_test)
hunyuan_probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1).numpy()
print(f"Hunyuan complete: {hunyuan_probs.shape}")

del model, base_model, trainer, tokenizer
import gc
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()


print("\n" + "="*60)
print("STEP 2: DEEPSEEK-MATH-7B INFERENCE")
print("="*60)

deepseek_model_path = "/kaggle/input/deekseepmath-7b-map-competition/MAP_EXP_09_FULL"

def format_deepseek_input(row):
    x = "This answer is correct."
    if not row['is_correct']:
        x = "This is answer is incorrect."
    return (
        f"Question: {row['QuestionText']}\n"
        f"Answer: {row['MC_Answer']}\n"
        f"{x}\n"
        f"Student Explanation: {row['StudentExplanation']}"
    )

le_shared = data_processor.get_label_encoder()
n_classes = data_processor.get_num_classes()
correct_lookup_shared = data_processor.correct_lookup
test_ds = test_df.copy()
test_ds = test_ds.merge(correct_lookup_shared, on=['QuestionId', 'MC_Answer'], how='left')
test_ds['is_correct'] = test_ds['IsCorrect_flag'].notna()
test_ds = test_ds.drop(columns=['IsCorrect_flag'])
test_ds['text'] = test_ds.apply(format_deepseek_input, axis=1)

from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from scipy.special import softmax
from tqdm import tqdm

tokenizer_ds = AutoTokenizer.from_pretrained(deepseek_model_path)
model_ds = AutoModelForSequenceClassification.from_pretrained(
    deepseek_model_path,
    device_map="auto",
    torch_dtype=torch.float16
)
model_ds.config.pad_token_id = tokenizer_ds.pad_token_id
model_ds.eval()

def tokenize_ds(batch):
    return tokenizer_ds(batch["text"], truncation=True, max_length=256)

ds_test_ds = Dataset.from_pandas(test_ds[['text']])
ds_test_ds = ds_test_ds.map(tokenize_ds, batched=True, remove_columns=['text'])

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer_ds,
    padding=True,
    return_tensors="pt"
)

dataloader = DataLoader(
    ds_test_ds,
    batch_size=4,
    shuffle=False,
    collate_fn=data_collator,
    pin_memory=True,
    num_workers=0
)

all_logits = []
device = next(model_ds.parameters()).device

with torch.no_grad():
    for batch in tqdm(dataloader, desc="DeepSeek"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model_ds(**batch)
        logits = outputs.logits
        all_logits.append(logits.float().cpu().numpy())

predictions_ds = np.concatenate(all_logits, axis=0)
deepseek_probs = softmax(predictions_ds, axis=1)
print(f"DeepSeek complete: {deepseek_probs.shape}")

del model_ds, tokenizer_ds
import gc
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()



print("\n" + "="*60)
print("STEP 3: QWEN3-8B INFERENCE")
print("="*60)

qwen3_model_path = "/kaggle/input/qwen3-8b-map-competition/MAP_EXP_16_FULL"

test_qw = test_df.copy()
test_qw = test_qw.merge(correct_lookup_shared, on=['QuestionId', 'MC_Answer'], how='left')
test_qw['is_correct'] = test_qw['IsCorrect_flag'].notna()
test_qw = test_qw.drop(columns=['IsCorrect_flag'])
test_qw['text'] = test_qw.apply(format_deepseek_input, axis=1)

tokenizer_qw = AutoTokenizer.from_pretrained(qwen3_model_path)
model_qw = AutoModelForSequenceClassification.from_pretrained(
    qwen3_model_path,
    device_map="auto",
    torch_dtype=torch.float16
)
model_qw.config.pad_token_id = tokenizer_qw.pad_token_id
model_qw.eval()

def tokenize_qw(batch):
    return tokenizer_qw(batch["text"], truncation=True, max_length=256)

ds_test_qw = Dataset.from_pandas(test_qw[['text']])
ds_test_qw = ds_test_qw.map(tokenize_qw, batched=True, remove_columns=['text'])

data_collator_qw = DataCollatorWithPadding(
    tokenizer=tokenizer_qw,
    padding=True,
    return_tensors="pt"
)

dataloader_qw = DataLoader(
    ds_test_qw,
    batch_size=8,
    shuffle=False,
    collate_fn=data_collator_qw,
    pin_memory=True,
    num_workers=0
)

all_logits_qw = []
device_qw = next(model_qw.parameters()).device

with torch.no_grad():
    for batch in tqdm(dataloader_qw, desc="Qwen3"):
        batch = {k: v.to(device_qw) for k, v in batch.items()}
        outputs = model_qw(**batch)
        logits = outputs.logits
        all_logits_qw.append(logits.float().cpu().numpy())

predictions_qw = np.concatenate(all_logits_qw, axis=0)
qwen3_probs = softmax(predictions_qw, axis=1)
print(f"Qwen3 complete: {qwen3_probs.shape}")

del model_qw, tokenizer_qw
import gc
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()


print("\n" + "="*60)
print("STEP 4: GEMMA2-9B INFERENCE")
print("="*60)

gemma_lora_path = "/kaggle/input/gemma2-9b-it-cv945"

def format_gemma_input(row):
    x = "Yes"
    if not row['is_correct']:
        x = "No"
    return (
        f"Question: {row['QuestionText']}\n"
        f"Answer: {row['MC_Answer']}\n"
        f"Correct? {x}\n"
        f"Student Explanation: {row['StudentExplanation']}"
    )

test_gm = test_df.copy()
test_gm = test_gm.merge(correct_lookup_shared, on=['QuestionId', 'MC_Answer'], how='left')
test_gm['is_correct'] = test_gm['IsCorrect_flag'].notna()
test_gm = test_gm.drop(columns=['IsCorrect_flag'])
test_gm['text'] = test_gm.apply(format_gemma_input, axis=1)

tokenizer_gm = AutoTokenizer.from_pretrained(gemma_lora_path)
base_model_gm = AutoModelForSequenceClassification.from_pretrained(
    "/kaggle/input/gemma2-9b-it-bf16",
    num_labels=n_classes,
    torch_dtype=torch.float16,
    device_map="auto"
)
model_gm = PeftModel.from_pretrained(base_model_gm, gemma_lora_path)
model_gm.eval()

def tokenize_gm(batch):
    return tokenizer_gm(batch["text"], truncation=True, max_length=256)

ds_test_gm = Dataset.from_pandas(test_gm[['text']])
ds_test_gm = ds_test_gm.map(tokenize_gm, batched=True, remove_columns=['text'])

data_collator_gm = DataCollatorWithPadding(
    tokenizer=tokenizer_gm,
    max_length=256,
    return_tensors="pt"
)

dataloader_gm = DataLoader(
    ds_test_gm,
    batch_size=8,
    shuffle=False,
    collate_fn=data_collator_gm,
    pin_memory=True,
    num_workers=2
)

all_logits_gm = []
device_gm = next(model_gm.parameters()).device

with torch.no_grad():
    for batch in tqdm(dataloader_gm, desc="Gemma2"):
        batch = {k: v.to(device_gm) for k, v in batch.items()}
        outputs = model_gm(**batch)
        logits = outputs.logits
        all_logits_gm.append(logits.float().cpu().numpy())

predictions_gm = np.concatenate(all_logits_gm, axis=0)
gemma2_probs = softmax(predictions_gm, axis=1)
print(f"Gemma2 complete: {gemma2_probs.shape}")

del model_gm, base_model_gm, tokenizer_gm
torch.cuda.empty_cache()


print("\n" + "="*60)
print("STEP 5: 4-MODEL ENSEMBLE")
print("="*60)

from collections import defaultdict

row_ids = test_df.row_id.values
unique_labels = sorted(data_processor.get_label_encoder().classes_)

hunyuan_weight = 1.2
deepseek_weight = 1.0
qwen3_weight = 1.0
gemma2_weight = 0.9

final_predictions = []

for i in range(len(row_ids)):
    hunyuan_top = np.argsort(-hunyuan_probs[i])[:25]
    deepseek_top = np.argsort(-deepseek_probs[i])[:25]
    qwen3_top = np.argsort(-qwen3_probs[i])[:25]
    gemma2_top = np.argsort(-gemma2_probs[i])[:25]
    
    all_classes = set()
    class_votes = defaultdict(int)
    class_total_prob = defaultdict(float)
    class_max_prob = defaultdict(float)
    
    for idx in hunyuan_top:
        label = unique_labels[idx]
        all_classes.add(label)
        class_votes[label] += 1
        class_total_prob[label] += hunyuan_probs[i, idx] * hunyuan_weight
        class_max_prob[label] = max(class_max_prob[label], hunyuan_probs[i, idx] * hunyuan_weight)
    
    for idx in deepseek_top:
        label = unique_labels[idx]
        all_classes.add(label)
        class_votes[label] += 1
        class_total_prob[label] += deepseek_probs[i, idx] * deepseek_weight
        class_max_prob[label] = max(class_max_prob[label], deepseek_probs[i, idx] * deepseek_weight)
    
    for idx in qwen3_top:
        label = unique_labels[idx]
        all_classes.add(label)
        class_votes[label] += 1
        class_total_prob[label] += qwen3_probs[i, idx] * qwen3_weight
        class_max_prob[label] = max(class_max_prob[label], qwen3_probs[i, idx] * qwen3_weight)
    
    for idx in gemma2_top:
        label = unique_labels[idx]
        all_classes.add(label)
        class_votes[label] += 1
        class_total_prob[label] += gemma2_probs[i, idx] * gemma2_weight
        class_max_prob[label] = max(class_max_prob[label], gemma2_probs[i, idx] * gemma2_weight)
    
    final_scores = {}
    for label in all_classes:
        base_score = class_total_prob[label]
        agreement_bonus = class_votes[label] / 4  
        confidence_bonus = class_max_prob[label]
        
        final_scores[label] = (
            base_score * 0.6 +
            agreement_bonus * 0.3 +
            confidence_bonus * 0.1
        )
    
    sorted_classes = sorted(final_scores.items(), key=lambda x: -x[1])
    top3 = [label for label, _ in sorted_classes[:3]]
    final_predictions.append(" ".join(top3))

submission = pd.DataFrame({
    "row_id": row_ids,
    "Category:Misconception": final_predictions
})

submission.to_csv("submission.csv", index=False)

print("\n" + "="*60)
print("ENSEMBLE COMPLETE")
print("="*60)
print("Submission saved: submission.csv")
print(submission.head())

