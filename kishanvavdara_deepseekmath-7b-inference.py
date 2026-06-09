import os
from tqdm import tqdm
import pandas as pd, numpy as np
from IPython.display import display, Math, Latex
from sklearn.preprocessing import LabelEncoder

import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from scipy.special import softmax



os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

TRAIN_PATH = "/kaggle/input/map-charting-student-math-misunderstandings/train.csv"
TEST_PATH = "/kaggle/input/map-charting-student-math-misunderstandings/test.csv"
model_path = "/kaggle/input/deepseek-math-7b-instruct/transformers/main/1"
lora_path = "/kaggle/input/map-deepseekmath-7b-it-tpu-train-bf16/deepseekmath7b_v5_epoch_2.pth" 
MAX_LEN = 256
BATCH_SIZE = 16


# helpers
def format_input(row):
    x = "This is Correct answer."
    if not row['is_correct']:
        x = "This is Incorrect answer."
    return (
        f"Question: {row['QuestionText']}\n"
        f"Answer: {row['MC_Answer']}\n"
        f"Correctness: {x}\n"
        f"Student Explanation: {row['StudentExplanation']}"
    )


# Tokenization function
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=MAX_LEN)


le = LabelEncoder()

train = pd.read_csv(TRAIN_PATH)

train.Misconception = train.Misconception.fillna('NA')
train['target'] = train.Category+":"+train.Misconception
train['label'] = le.fit_transform(train['target'])
target_classes = le.classes_
n_classes = len(target_classes)
print(f"Train shape: {train.shape} with {n_classes} target classes")
idx = train.apply(lambda row: row.Category.split('_')[0],axis=1)=='True'
correct = train.loc[idx].copy()
correct['c'] = correct.groupby(['QuestionId','MC_Answer']).MC_Answer.transform('count')
correct = correct.sort_values('c',ascending=False)
correct = correct.drop_duplicates(['QuestionId'])
correct = correct[['QuestionId','MC_Answer']]
correct['is_correct'] = 1

# Prepare test data
test = pd.read_csv(TEST_PATH)
test = test.merge(correct, on=['QuestionId','MC_Answer'], how='left')
test.is_correct = test.is_correct.fillna(0)
test['text'] = test.apply(format_input, axis=1)

# sanity
print("Inference input:\n")
print(test['text'][0])


tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

# Tokenize dataset
ds_test = Dataset.from_pandas(test[['text']])
ds_test = ds_test.map(tokenize, batched=True, remove_columns=['text'])

# Create data collator for efficient batching with padding
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    max_length=MAX_LEN,
    padding='max_length',
    return_tensors="pt")

dataloader = DataLoader(
    ds_test,
    batch_size=BATCH_SIZE,  
    shuffle=False,
    collate_fn=data_collator,
    pin_memory=True,  
    num_workers=2)


model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    num_labels=n_classes,
    torch_dtype=torch.float16,
    device_map="auto")
model.config.pad_token_id = tokenizer.pad_token_id

# LoRa configuration
# better way would be to save adapter.json in training, maybe in next version
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias='none',
    inference_mode=True,
    task_type=TaskType.SEQ_CLS,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])


# Get peft
model = get_peft_model(model, lora_config)
# Load weights
model.load_state_dict(torch.load(lora_path), strict=False)
model.eval()


# Fast inference loop
all_logits = []
device = next(model.parameters()).device
with torch.no_grad():
    for batch in tqdm(dataloader, desc="Inference"):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        outputs = model(**batch)
        logits = outputs.logits
        
        all_logits.append(logits.float().cpu().numpy())

# Concatenate all logits
predictions = np.concatenate(all_logits, axis=0)

# Convert to probs
probs = softmax(predictions, axis=1)

# Get top predictions (all 65 classes ranked)
top_indices = np.argsort(-probs, axis=1)

# Decode to class names
decoded_labels = le.inverse_transform(top_indices.flatten())
top_labels = decoded_labels.reshape(top_indices.shape)

# Create submission (top 3)
joined_preds = [" ".join(row[:3]) for row in top_labels]
sub = pd.DataFrame({
    "row_id": test.row_id.values,
    "Category:Misconception": joined_preds})

sub.to_csv("submission.csv", index=False)
sub




