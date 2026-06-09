# ==================== CONFIGURATIONS ====================
import torch
class CFG:
    
    DATA_PATH             = "/kaggle/input/map-charting-student-math-misunderstandings/"
    BASE_DIRECTORY        = "models_for_competitions"
    CHALLENGE_NAME        = "MAP-Charting-Student-Math-Misunderstandings"
    MODEL_NAME            = "deberta-v3-large"
    CHECKPOINT            = "/kaggle/input/huggingfacedebertav3variants/deberta-v3-large"
    PRETRAINED_CHECKPOINT = "/kaggle/input/models-for-competitions/.models_for_competitions/MAP-Charting-Student-Math-Misunderstandings/deberta-v3-large_stage2/checkpoint-2295"
    
    TRAINING       = True # True: train models
    PUSH_TO_HUB    = False # True: push models to hub
    PRE_TRAINED    = False  # True: loead and use pre-trained model
    SEED           = 42    # Reproducibility
    VALID_SIZE     = 0.2
    EPOCHS         = 50
    PATIENCE       = 5
    DEVICE         = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    COLOR_1        = 'cornflowerblue'
    COLOR_2        = 'salmon'
    PALETTE        = 'viridis'


# ==================== IMPORT BASE LIBRARIES AND SET RANDOM SEED ====================
import os
import torch
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set Seed for Reproducibility
random.seed(CFG.SEED)
np.random.seed(CFG.SEED)
torch.manual_seed(CFG.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CFG.SEED)
# torch.use_deterministic_algorithms(False)

# Set seaborn style
sns.set_style('darkgrid')


# ==================== HELPER FUNCTIONS ====================

def print_with_sep(text,sep="=",n=30):
  print("\n")
  print(sep*n)
  print('\t',text)
  print(sep*n)

# ============================================================
def print_dataset_overview(datasets):

    # Check shapes
    print_with_sep("Shapes")
    for name, df in datasets.items():
      print(f"{name} shape: {df.shape}")
    
    # Check duplicates
    print_with_sep("Duplicates")
    for name, df in datasets.items():
      print(f"{name} duplicates: {df.duplicated().sum()}")
    
    # Check nans
    print_with_sep("NaNs")
    for name, df in datasets.items():
      print(f"{name} NaNs: {df.isnull().sum().sum()}")
    
    # Check col difference
    print_with_sep("Columns not in test")
    print(set(TRAIN_DF.columns).difference(set(TEST_DF.columns)))

    # Check descriptive stats
    print_with_sep("Descriptive Statistics")
    for name, df in datasets.items():
      print(f"{name} Description:")
      percentage_missing = df.isnull().sum()/df.shape[0]; percentage_missing.name = '% Missing'
      data_types = df.dtypes; data_types.name = 'd_type'
    
      display(
          pd.concat([
              df.describe(include='all').T,
              percentage_missing,
              data_types],
                    axis=1).replace(np.nan,'-').drop(['top','freq'], axis=1).style.background_gradient(cmap='Blues'))
      print("\n")

# ============================================================
from datetime import datetime 
# Unique directory builder
def make_model_dir(base_dir, challenge_name, model_name, stage_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    return os.path.join(base_dir, challenge_name, f"{model_name}_{stage_name}_{timestamp}")


# Load data
# TRAIN_DF = pd.read_csv(CFG.DATA_PATH+'train.csv')
# TEST_DF = pd.read_csv(CFG.DATA_PATH+'test.csv')
TRAIN_DF = pd.read_csv('/kaggle/input/mat-splited-dataset/train.csv')
TEST_DF = pd.read_csv('/kaggle/input/mat-splited-dataset/test.csv')

# Inspect first rows
display(TRAIN_DF.head())
display(TEST_DF.head())


# Dataset overview
datasets = {
    "TRAIN_DATA":TRAIN_DF, 
    "TEST_DATA": TEST_DF
}
print_dataset_overview(datasets)


import plotly.express as px
fig = px.sunburst(TRAIN_DF.fillna("None"), path=['Category', 'Misconception'], color='Category', title="Misconceptions Nested within Categories")
fig.update_traces(textinfo='label+percent parent')
fig.show(renderer='iframe')

print('==========MISCONCEPTIONS==========')
print(TRAIN_DF['Misconception'].unique())

print('\n==========CATEGORIES==========')
print(TRAIN_DF['Category'].unique())


# Check categories and misconception
TRAIN_DF.query("`Category` == 'True_Misconception'").head()


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
TRAIN_DF['target'] = TRAIN_DF.Category+":"+TRAIN_DF.Misconception.fillna('NA')
TRAIN_DF['label'] = le.fit_transform(TRAIN_DF['target'])
n_classes = len(le.classes_)

print(f"Train shape: {TRAIN_DF.shape} with {n_classes} target classes")
TRAIN_DF.head()


# id2label: maps integer IDs to class names
id2label = {i: c for i, c in enumerate(le.classes_)}

# label2id: maps class names to integer IDs
label2id = {c: i for i, c in enumerate(le.classes_)}


print(TRAIN_DF.iloc[10781]['StudentExplanation'])


idx = TRAIN_DF.apply(lambda row: row.Category.split('_')[0],axis=1)=='True'
correct = TRAIN_DF.loc[idx].drop_duplicates(['QuestionId'])
correct = correct[['QuestionId','MC_Answer']]
correct['is_correct'] = 1

# Apply to train df
TRAIN_DF = TRAIN_DF.merge(correct, on=['QuestionId','MC_Answer'], how='left')
TRAIN_DF.is_correct = TRAIN_DF.is_correct.fillna(0)
display(TRAIN_DF.head())

# Apply to train df
TEST_DF = TEST_DF.merge(correct, on=['QuestionId','MC_Answer'], how='left')
TEST_DF.is_correct = TEST_DF.is_correct.fillna(0)


def format_input(row):
    x = "This answer is correct."
    if not row['is_correct']:
        x = "This is answer is incorrect."
    return (
        f"Question: {row['QuestionText']}\n"
        f"Answer: {row['MC_Answer']}\n"
        f"{x}\n"
        f"Student Explanation: {row['StudentExplanation']}"
    )

TRAIN_DF['text'] = TRAIN_DF.apply(format_input,axis=1)
TEST_DF['text'] = TEST_DF.apply(format_input,axis=1)

print("Example prompt for our LLM:\n")
print(TRAIN_DF.text.values[0] )


from IPython.display import display, Math, Latex

# GET ANSWER CHOICES
tmp = TRAIN_DF.groupby(['QuestionId','MC_Answer']).size().reset_index(name='count')
tmp['rank'] = tmp.groupby('QuestionId')['count'].rank(method='dense', ascending=False).astype(int) - 1
tmp = tmp.drop('count',axis=1)
tmp = tmp.sort_values(['QuestionId','rank'])

# DISPLAY QUESTION AND ANSWER CHOICES
Q = tmp.QuestionId.unique()
for q in Q:
    question = TRAIN_DF.loc[TRAIN_DF.QuestionId==q].iloc[0].QuestionText
    choices = tmp.loc[tmp.QuestionId==q].MC_Answer.values
    labels="ABCD"
    choice_str = " ".join([f"({labels[i]}) {choice}" for i, choice in enumerate(choices)])
    
    print()
    display(Latex(f"QuestionId {q}: {question}") )
    display(Latex(f"MC Answers: {choice_str}"))


# ============================================================
# Progressive Unfreezing Helper Functions for DeBERTa (large)
# ============================================================

import torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch
import numpy as np

def unfreeze_last_n_layers(model, n_unfrozen=24):
    """Unfreeze the last n encoder layers."""
    # freeze everything
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze last n encoder layers
    for layer in model.deberta.encoder.layer[-n_unfrozen:]:
        for param in layer.parameters():
            param.requires_grad = True
    # always keep classifier trainable
    for param in model.classifier.parameters():
        param.requires_grad = True
        
    print(f"Unfroze last {n_unfrozen} layers + classifier head.")

def unfreeze_last_n_layers_JUST_A_MEMO(model, n_unfrozen=24):
    """
    Unfreeze the last n encoder layers.
    
    Same as above but with list comprehension: still works, but for-loop is clearer (and i guess more standard practice)
    """
    # freeze everything
    [setattr(param, "requires_grad", False) for param in model.parameters()]
    # Unfreeze last n encoder layers
    [setattr(param, "requires_grad", True) for layer in model.deberta.encoder.layer[-n_unfrozen:] for param in layer.parameters()]
    # always keep classifier trainable
    [setattr(param, "requires_grad", True) for param in model.classifier.parameters()]
    
    print(f"Unfroze last {n_unfrozen} layers + classifier head.")
    
# Optimizer setup
def get_optimizer(model, lr_head=5e-5, lr_base=1e-5, n_unfrozen=24, custom=True, llrd=False):
    """
    custom=True  -> sá»­ dá»¥ng setup 2 LR (head/base)
    llrd=True    -> sá»­ dá»¥ng Layer-wise LR Decay (Æ°u tiÃªn hÆ¡n)
    """
    if not custom:
        return None  # Trainer sáº½ tá»± táº¡o optimizer máº·c Ä‘á»‹nh

    if llrd:
        # ============ LLRD Optimizer ============
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = []

        layers = [model.deberta.embeddings] + list(model.deberta.encoder.layer)
        layers.reverse()
        lr = lr_base

        for layer in layers:
            optimizer_grouped_parameters.append({
                "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
                "lr": lr,
            })
            optimizer_grouped_parameters.append({
                "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": lr,
            })
            lr *= 0.95  # decay per layer

        optimizer_grouped_parameters.append({
            "params": model.classifier.parameters(),
            "lr": lr_head * 2,
        })

        print("âœ… Using Layer-wise Learning Rate Decay (LLRD)")
        return AdamW(optimizer_grouped_parameters, eps=1e-6)
    
    else:
        # ============ Basic Two-LR Optimizer ============
        optimizer_grouped_parameters = [
            {
                "params": model.classifier.parameters(),
                "lr": lr_head,
            },
            {
                "params": [p for layer in model.deberta.encoder.layer[-n_unfrozen:] for p in layer.parameters()],
                "lr": lr_base,
            },   
        ]
        print("âœ… Using 2-LR Optimizer (head/base)")
        return AdamW(optimizer_grouped_parameters, weight_decay=0.01)     

# Define function to get model info 
def count_trainable_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100 * trainable / total
    print(f"Total params: {total:,} | Trainable: {trainable:,} ({pct:.2f}%)")
    return total, trainable, pct

# MAP@3 METRIC
def compute_metrics(eval_pred):
    """
    TÃ­nh toÃ n bá»™ cÃ¡c chá»‰ sá»‘ cho bÃ i toÃ¡n MAP:
      - MAP@3
      - Accuracy
      - Macro Precision / Recall / F1
      - Cross Entropy Loss (CE)
    """
    logits, labels = eval_pred
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()

    # ---- MAP@3 ----
    top3 = np.argsort(-probs, axis=1)[:, :3]  # Top 3 predictions
    match = (top3 == labels[:, None])

    # Compute MAP@3 manually
    map3 = 0
    for i in range(len(labels)):
        if match[i, 0]:
            map3 += 1.0
        elif match[i, 1]:
            map3 += 1.0 / 2
        elif match[i, 2]:
            map3 += 1.0 / 3

    # ---- Cross Entropy Loss ----
    ce_loss = -np.mean(np.log(probs[np.arange(len(labels)), labels] + 1e-12))

    # ---- Top-1 metrics ----
    preds = np.argmax(probs, axis=1)
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='macro', zero_division=0)
    recall = recall_score(labels, preds, average='macro', zero_division=0)
    f1 = f1_score(labels, preds, average='macro', zero_division=0)

    return {
        "map@3": map3 / len(labels),
        "eval_accuracy": acc,
        "eval_precision": precision,
        "eval_recall": recall,
        "eval_f1": f1,
        "eval_cross_entropy": ce_loss
    }


# =======================================
# ğŸ”� Cosine Annealing with Hard Restarts
# =======================================
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup

def get_cosine_scheduler(optimizer, num_warmup_steps, num_training_steps, num_cycles=3):
    """
    Scheduler giÃºp model thoÃ¡t local minima báº±ng cÃ¡ch "restart" learning rate.
    """
    print(f"âœ… Using CosineAnnealingWarmRestarts ({num_cycles} cycles)")
    return get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles
    )


from transformers import (AutoTokenizer, AutoModelForSequenceClassification, 
                          AutoConfig, DataCollatorWithPadding, EarlyStoppingCallback, TrainerCallback)
import csv, os

# load the checkpoint's tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    CFG.CHECKPOINT)

# Initialize DataCollator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Set config
config = AutoConfig.from_pretrained(
    CFG.CHECKPOINT, 
    label2id=label2id, id2label=id2label, 
    num_labels=len(le.classes_))

# Initialize model
model = AutoModelForSequenceClassification.from_pretrained(
    CFG.CHECKPOINT,
    config=config
).to(CFG.DEVICE)

# Define early stopping
early_stopping = EarlyStoppingCallback(
    early_stopping_patience=CFG.PATIENCE, 
    early_stopping_threshold=0.0
)

class CSVLoggerCallback(TrainerCallback):
    def __init__(self, log_file="training_log.csv"):
        self.log_file = log_file
        # náº¿u chÆ°a cÃ³ file thÃ¬ táº¡o header
        if not os.path.exists(log_file):
            with open(log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "epoch", "loss", "eval_map@3", "eval_f1", "eval_accuracy", "time"])
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            with open(self.log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    state.global_step,
                    state.epoch,
                    logs.get("loss"),
                    logs.get("eval_map@3"),
                    logs.get("eval_f1"),
                    logs.get("eval_accuracy"),
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                ])


from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict

# Find classes with only 1 sample
class_counts = TRAIN_DF['label'].value_counts()
rare_classes = class_counts[class_counts == 1].index.to_list()

print('========== RARE TARGETS ==========')
for x in [id2label[c] for c in rare_classes]:
    print(x)

# Split rare and non-rare
rare_df = TRAIN_DF.query(f"`label` in {rare_classes}")
rest_df = TRAIN_DF.query(f"`label` not in {rare_classes}")

# Stratified split only on the rest
train_rest, val_rest = train_test_split(
    rest_df,
    test_size    = CFG.VALID_SIZE, 
    random_state = CFG.SEED,
    stratify     = rest_df['label']
)

# Add rare classes back into train
train_df = pd.concat([train_rest, rare_df]).reset_index(drop=True)
val_df = val_rest.reset_index(drop=True)

# Convert to Hugging Face Dataset
COLS = ['text','label']
train_ds = Dataset.from_pandas(train_df[COLS])
val_ds   = Dataset.from_pandas(val_df[COLS])
test_ds  = Dataset.from_pandas(TEST_DF[['text']])

# Create a DatasetDict
hf_dataset = DatasetDict({
    'train': train_ds, 
    'valid': val_ds,
    'test' : test_ds,
})


# Tokenization function
MAX_LEN = 256
def tokenize(batch):
    return tokenizer(batch["text"], padding=False, truncation=True, max_length=MAX_LEN)
    # NOTE: padding=False because we'll use DataCollatorWithPadding

# Tokenize whole dataset
dataset_encoded = hf_dataset.map(tokenize, batched=True)  

print(f"========== Dataset before tokenization: ==========\n {hf_dataset}\n")
print(f"========== Dataset after tokenization: ==========\n {dataset_encoded}\n")

# Check inputs' length (before truncation)
lengths = [len(tokenizer.encode(t, truncation=False)) for t in hf_dataset['train']["text"]]
L = (np.array(lengths)>MAX_LEN).sum()

plt.figure(figsize=(12,6))
sns.histplot(lengths, bins=50, color=CFG.COLOR_1, kde=True)
plt.title("Token Length Distribution")
plt.xlabel("Number of tokens")
plt.ylabel("Frequency")
plt.show()

print(f"There are {L} train sample(s) with more than {MAX_LEN} tokens: {np.sort( lengths )}")


from torch import nn
from transformers import Trainer
from sklearn.utils.class_weight import compute_class_weight

# Calculate class weights
labels = hf_dataset['train']['label']
class_weights = compute_class_weight(
    class_weight='balanced', 
    classes=np.unique(labels), 
    y=labels
)

# Convert to tensor and normalize to mean = 1
class_weights = torch.tensor(class_weights, dtype=torch.float)
normalized_weights = class_weights / class_weights.mean()

# Smooth: log1p compresses extremes, +1.0 keeps min > 1 to avoid too-small weights
smoothed_weights = torch.log1p(normalized_weights) + 1.0
smoothed_weights = smoothed_weights.to(CFG.DEVICE)

# Subclass Trainer with weighted loss
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Apply class-weighted CrossEntropy
        loss_fct = nn.CrossEntropyLoss(weight=smoothed_weights)
        loss = loss_fct(logits, labels)
        
        return (loss, outputs) if return_outputs else loss


# Weights check (before vs after smoothing)
print(f"Unsmoothed class weights range: {class_weights.min().item():.2f} â†’ {class_weights.max().item():.2f}")
print(f"Smoothed class weights range: {smoothed_weights.min().item():.2f} â†’ {smoothed_weights.max().item():.2f}")

plt.figure(figsize=(12,6))
plt.plot(sorted(class_weights.cpu()), label='Original')
plt.plot(sorted(smoothed_weights.cpu()), label='Smoothed')
plt.legend(); plt.title("Class Weight Smoothing Effect"); plt.show()


import os
from transformers import TrainingArguments

# Directory full path
DIR = os.path.join(CFG.BASE_DIRECTORY, CFG.CHALLENGE_NAME, CFG.MODEL_NAME)

# DIR = make_model_dir(
#     base_dir=       CFG.BASE_DIRECTORY,
#     challenge_name= CFG.CHALLENGE_NAME,
#     model_name=     CFG.MODEL_NAME,
#     stage_name=""
# )

# Stages configs
stages = {
    "stage_1": {
        "freeze_strategy"  : lambda model: unfreeze_last_n_layers(model, n_unfrozen=4),
        "optimizer"        : lambda model, llrd=False: get_optimizer(
                                model, lr_head=5e-5, lr_base=1e-5, n_unfrozen=4, custom=True, llrd=llrd
                             ),
        "train_args"       : TrainingArguments(
            output_dir=f'.{DIR}_stage1',
            learning_rate=2e-5,                       # overridden by custom optimizer
            num_train_epochs=CFG.EPOCHS,
            warmup_ratio=0.1, 
            per_device_train_batch_size=16*2,
            per_device_eval_batch_size=31*2,
            # gradient_accumulation_steps=4,
            weight_decay=0.01,
            max_grad_norm=1.0,
            seed=CFG.SEED,
            metric_for_best_model="map@3",
            load_best_model_at_end=False,
            greater_is_better=True,
            eval_strategy="epoch",
            save_strategy="no",                      # skip periodic checkpoint saving
            # save_total_limit=3,
            logging_steps=10,
            report_to="none",
        )
    },
    
    "stage_2": {
        "freeze_strategy"  : lambda model: unfreeze_last_n_layers(model, n_unfrozen=8),
        "optimizer"        : lambda model, llrd=False: get_optimizer(
                                model, lr_head=5e-5, lr_base=1e-5, n_unfrozen=4, custom=True, llrd=llrd
                             ),
        "train_args"       : TrainingArguments(
            output_dir=f'.{DIR}_stage2',
            learning_rate=2e-5,                     # should be overridden by custom optimizer
            num_train_epochs=CFG.EPOCHS,
            warmup_ratio=0.1,    
            per_device_train_batch_size=8,          # instead of 32
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=4,          # keeps effective batch 32
            fp16=True,                              # mixed precision
            gradient_checkpointing=False,           # True: recompute activations to save memory
            weight_decay=0.01,
            max_grad_norm=1.0,
            seed=CFG.SEED,
            metric_for_best_model="map@3",
            load_best_model_at_end=True,
            greater_is_better=True,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=3,
            logging_steps=10,
            report_to="none",
        )
    },
}


import time
from torch.optim.swa_utils import AveragedModel, update_bn
import numpy as np
import pandas as pd

all_logs = []
swa_model = None

if CFG.TRAINING:
    for name, cfg in stages.items():
        print(f"\n===== Running {name.upper()} =====\n")

        # Apply freezing strategy
        cfg["freeze_strategy"](model)
        count_trainable_params(model)

        # ---- Create optimizer via the lambda (pass llrd=True to enable LLRD) ----
        optimizer = cfg["optimizer"](model, llrd=True)

        # ---- Compute num_training_steps robustly from cfg["train_args"] and dataset length ----
        train_args = cfg["train_args"]
        train_dataset = dataset_encoded["train"]

        per_device_batch = getattr(train_args, "per_device_train_batch_size", None)
        if per_device_batch is None:
            per_device_batch = getattr(train_args, "train_batch_size", 8)

        grad_accum = getattr(train_args, "gradient_accumulation_steps", 1)
        num_epochs = int(getattr(train_args, "num_train_epochs", 1))

        # steps per epoch (accounting for grad accumulation)
        steps_per_epoch = int(np.ceil(len(train_dataset) / per_device_batch / grad_accum))
        num_training_steps = steps_per_epoch * num_epochs
        num_warmup_steps = int(getattr(train_args, "warmup_ratio", 0.1) * num_training_steps)

        print(f"num_training_steps={num_training_steps}, steps_per_epoch={steps_per_epoch}, num_epochs={num_epochs}, grad_accum={grad_accum}")

        # ---- Scheduler (cosine with hard restarts) ----
        scheduler = get_cosine_scheduler(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

        # ---- Initialize trainer (use processing_class if you want to silence deprecation) ----
        trainer = WeightedTrainer(
            model=model,
            tokenizer=tokenizer,   # or processing_class=tokenizer to avoid deprecation warning
            data_collator=data_collator,
            args=cfg["train_args"],
            train_dataset=train_dataset,
            eval_dataset=dataset_encoded["valid"],
            compute_metrics=compute_metrics,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=CFG.PATIENCE),
                CSVLoggerCallback("training_log.csv")
            ],
            optimizers=(optimizer, scheduler) if optimizer is not None else (None, None)
        )

        # Ensure best model will be loaded if requested in train_args
        try:
            trainer.args.load_best_model_at_end = getattr(train_args, "load_best_model_at_end", False)
            trainer.args.metric_for_best_model = getattr(train_args, "metric_for_best_model", "map@3")
            trainer.args.greater_is_better = getattr(train_args, "greater_is_better", True)
        except Exception:
            pass

        # ---- Train ----
        start = time.time()
        trainer.train()
        end = time.time()
        print(f"Training time {name}: {(end - start)/60:.2f} min")

        # ---- Save logs from trainer.state.log_history (if any) ----
        if hasattr(trainer.state, "log_history") and len(trainer.state.log_history) > 0:
            df_log = pd.DataFrame(trainer.state.log_history)
            df_log["stage"] = name
            df_log["training_time_min"] = round((end - start) / 60, 2)
            all_logs.append(df_log)

        # ---- Apply SWA: update or create averaged model, then update BN on swa model ----
        try:
            print("Applying SWA...")
            if swa_model is None:
                swa_model = AveragedModel(trainer.model)
            else:
                swa_model.update_parameters(trainer.model)

            # update BN using training dataloader (recompute batchnorm stats)
            update_bn(trainer.get_train_dataloader(), swa_model)
            print("SWA applied and BN updated.")
        except Exception as e:
            print("SWA skipped (error):", e)

        # ---- Save checkpoint for this stage (unchanged behavior) ----
        try:
            trainer.save_model(f"{DIR}_{name}")
        except Exception as e:
            print("Warning: failed to save model:", e)

    # After all stages, optionally replace model with swa_model
    if swa_model is not None:
        model = swa_model
        print("Final model replaced with SWA-averaged weights.")

# Save combined logs
if len(all_logs) > 0:
    train_logs = pd.concat(all_logs, ignore_index=True)
    train_logs.to_csv("training_log.csv", index=False)
    print("Saved combined training_log.csv")
else:
    print("No logs were collected.")


if CFG.TRAINING:
    # ğŸ§  Náº¿u báº¡n Ä‘Ã£ Ã¡p dá»¥ng SWA
    save_model = model
    if isinstance(model, torch.optim.swa_utils.AveragedModel):
        print("âš ï¸� Detected SWA model, extracting base model for saving...")
        save_model = model.module if hasattr(model, "module") else model.module if hasattr(model, "base_model") else model
        # Má»™t sá»‘ version dÃ¹ng model.base_model
    
    save_model.save_pretrained(f'.{DIR}')
    tokenizer.save_pretrained(f'.{DIR}')
    print(f"âœ… Model & tokenizer saved to .{DIR}")


from transformers import AutoTokenizer, AutoModelForSequenceClassification
if CFG.PRE_TRAINED:
    model = AutoModelForSequenceClassification.from_pretrained(CFG.PRETRAINED_CHECKPOINT,
                                                              config=config)
    tokenizer = AutoTokenizer.from_pretrained(CFG.PRETRAINED_CHECKPOINT)
    training_args = TrainingArguments(report_to="none")


from transformers import pipeline

# Náº¿u model lÃ  AveragedModel, unwrap nÃ³
if isinstance(model, torch.optim.swa_utils.AveragedModel):
    print("âš ï¸� Detected SWA model â†’ extracting base model for inference...")
    model = getattr(model, "module", None) or getattr(model, "base_model", None)
    if model is None:
        raise ValueError("Cannot extract base model from SWA wrapper.")

# Táº¡o pipeline
pipe = pipeline(
    task="text-classification",
    model=model,
    tokenizer=tokenizer,
    device=CFG.DEVICE
)

# Dá»± Ä‘oÃ¡n thá»­
print(pipe("This is a test sentence."))


ds_test = Dataset.from_pandas(TEST_DF[['text']])
ds_test = ds_test.map(tokenize, batched=True)

predictions = trainer.predict(ds_test)
probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1).numpy()


# Get top 3 predicted class indices
top3 = np.argsort(-probs, axis=1)[:, :3]   # shape: [num_samples, 3]

# Decode numeric class indices to original string labels 
flat_top3 = top3.flatten()
decoded_labels = le.inverse_transform(flat_top3)
top3_labels = decoded_labels.reshape(top3.shape)

# Join 3 labels per row with space
joined_preds = [" ".join(row) for row in top3_labels]

# Save submission
sub = pd.DataFrame({
    "row_id": TEST_DF.row_id.values,
    "Category:Misconception": joined_preds
})
sub.to_csv("submission.csv", index=False)
sub.head()


# ============================================
# ğŸ“Š VISUALIZE TRAINING RESULTS
# ============================================
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# Láº¥y log tá»« trainer
log_history = pd.DataFrame(trainer.state.log_history)

# Lá»�c cÃ¡c log epoch
log_history = log_history.dropna(subset=['epoch'])

# Biá»ƒu Ä‘á»“ Loss vÃ  MAP@3
plt.figure(figsize=(16,5))

# --- Loss ---
plt.subplot(1,2,1)
sns.lineplot(x='epoch', y='loss', data=log_history, label='Train Loss', marker='o')
sns.lineplot(x='epoch', y='eval_loss', data=log_history, label='Val Loss', marker='o')
plt.title('Training & Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# --- MAP@3 ---
plt.subplot(1,2,2)
sns.lineplot(x='epoch', y='eval_map@3', data=log_history, color='green', label='Val MAP@3', marker='o')
plt.title('Validation MAP@3')
plt.xlabel('Epoch')
plt.ylabel('MAP@3')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ============================================
# ğŸ“‹ Summary table by epoch
# ============================================
summary_df = log_history.groupby('epoch').agg({
    'loss': 'last',
    'eval_loss': 'last',
    'eval_map@3': 'last'
}).rename(columns={'loss': 'Train Loss', 'eval_loss': 'Val Loss', 'eval_map@3': 'Val MAP@3'})

display(summary_df.style.background_gradient(cmap='Blues').format("{:.4f}"))

# ============================================
# ğŸ§® Classification report & Confusion matrix
# ============================================
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

preds_output = trainer.predict(dataset_encoded["valid"])
y_true = preds_output.label_ids
y_pred = preds_output.predictions.argmax(axis=1)

# BÃ¡o cÃ¡o chi tiáº¿t
report_df = pd.DataFrame(classification_report(y_true, y_pred, output_dict=True)).transpose()
display(report_df.style.background_gradient(cmap='Purples').format("{:.3f}"))

# Ma tráº­n nháº§m láº«n
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# ============================================
# ğŸ�† Best Epoch
# ============================================
best_epoch = summary_df['Val MAP@3'].idxmax()
print(f"ğŸ�† Best Epoch: {best_epoch}")
print(summary_df.loc[best_epoch])

