# ==================== CONFIGURATIONS ====================
import torch
class CFG:
    
    DATA_PATH             = "/kaggle/input/map-charting-student-math-misunderstandings/"
    BASE_DIRECTORY        = "models_for_competitions"
    CHALLENGE_NAME        = "MAP-Charting-Student-Math-Misunderstandings"
    MODEL_NAME            = "deberta-v3-large"
    CHECKPOINT            = "/kaggle/input/huggingfacedebertav3variants/deberta-v3-large"
    PRETRAINED_CHECKPOINT = "/kaggle/input/models-for-competitions/.models_for_competitions/MAP-Charting-Student-Math-Misunderstandings/deberta-v3-large_stage2/checkpoint-2295"
    
    TRAINING       = False # True: train models
    PUSH_TO_HUB    = False # True: push models to hub
    PRE_TRAINED    = True  # True: loead and use pre-trained model
    SEED           = 42    # Reproducibility
    VALID_SIZE     = 0.2
    EPOCHS         = 8
    PATIENCE       = 2
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
def get_optimizer(model, lr_head=5e-5, lr_base=1e-5, n_unfrozen=24, custom=True):
    if custom:
        optimizer_grouped_parameters = [
            {
                "params": model.classifier.parameters(),
                "lr": lr_head,  # faster adaptation for the classifier head
            },
            {
                "params": [p for layer in model.deberta.encoder.layer[-n_unfrozen:] for p in layer.parameters()],
                "lr": lr_base,  # slower LR for unfrozen pretrained layers
            },   
        ]
        return AdamW(optimizer_grouped_parameters, weight_decay=0.01)
    else:
        # Let Trainer handle it with default values
        return None     

# Define function to get model info 
def count_trainable_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100 * trainable / total
    print(f"Total params: {total:,} | Trainable: {trainable:,} ({pct:.2f}%)")
    return total, trainable, pct

# MAP@3 METRIC
def compute_map3(eval_pred):
    logits, labels = eval_pred
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    
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
    return {"map@3": map3 / len(labels)}


from transformers import (AutoTokenizer, AutoModelForSequenceClassification, 
                          AutoConfig, DataCollatorWithPadding, EarlyStoppingCallback)

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
        "optimizer"        : lambda model: get_optimizer(model, lr_head=5e-5, lr_base=1e-5, n_unfrozen=4, custom=True),
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
        "optimizer"        : lambda model: get_optimizer(model, lr_head=3e-5, lr_base=8e-6, n_unfrozen=8, custom=True),
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

if CFG.TRAINING:
    for name, cfg in stages.items():
        print(f"\n===== Running {name.upper()} =====\n")

        # Apply freezing strategy
        cfg["freeze_strategy"](model)
        count_trainable_params(model)

        # Define optimizer
        optimizer = cfg["optimizer"](model)
        
        # Initialize trainer
        trainer = WeightedTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        args=cfg["train_args"],
        train_dataset=dataset_encoded["train"],
        eval_dataset=dataset_encoded["valid"],
        compute_metrics=compute_map3,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=CFG.PATIENCE)],
        optimizers=(optimizer, None) if optimizer is not None else (None, None)
        )

        # Start training
        start = time.time()
        
        if name == 'stage_2':
            # trainer.train(resume_from_checkpoint=f"{DIR}_stage1")
            trainer.train()
        else:
            trainer.train()
            trainer.save_model(f"{DIR}_stage1")
        end = time.time()
        print(f"Training time {name}: {(end - start)/60:.2f} min")


if CFG.TRAINING:
    model.save_pretrained(f'.{DIR}')      
    tokenizer.save_pretrained(f'.{DIR}')


from transformers import AutoTokenizer, AutoModelForSequenceClassification
if CFG.PRE_TRAINED:
    model = AutoModelForSequenceClassification.from_pretrained(CFG.PRETRAINED_CHECKPOINT,
                                                              config=config)
    tokenizer = AutoTokenizer.from_pretrained(CFG.PRETRAINED_CHECKPOINT)
    training_args = TrainingArguments(report_to="none")
    trainer   = WeightedTrainer(model=model, tokenizer=tokenizer, data_collator=data_collator, args=training_args)


# =================================================================
# CELL Ä�á»‚ Váº¼ MA TRáº¬N NHáº¦M LáºªN (CONFUSION MATRIX)
# Vá»‹ trÃ­: Ngay sau Ã´ code "if CFG.PRE_TRAINED:..."
# =================================================================

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

print("Ä�ang táº¡o dá»± Ä‘oÃ¡n trÃªn táº­p validation...")
# Ä�áº£m báº£o báº¡n cÃ³ sáºµn 'dataset_encoded["valid"]' tá»« Ã´ code #22 (Ã´ "Split dataset")
val_preds = trainer.predict(dataset_encoded["valid"])

# Láº¥y nhÃ£n tháº­t (y_true) vÃ  nhÃ£n dá»± Ä‘oÃ¡n (y_pred)
y_true = val_preds.label_ids
y_pred = np.argmax(val_preds.predictions, axis=1)

print("Ä�ang táº¡o confusion matrix heatmap...")
# TÃ­nh toÃ¡n ma tráº­n nháº§m láº«n
cm = confusion_matrix(y_true, y_pred)

# Láº¥y sá»‘ lÆ°á»£ng class (Ä‘Ã£ Ä‘á»‹nh nghÄ©a á»Ÿ Ã´ code #11)
n_classes = len(id2label)

# Táº¡o nhÃ£n cho cÃ¡c trá»¥c (chá»‰ hiá»ƒn thá»‹ sá»‘ cháºµn, giá»‘ng áº£nh cá»§a báº¡n)
# Ä�iá»�u chá»‰nh step (vÃ­ dá»¥: % 2 hoáº·c % 4) náº¿u 58 nhÃ£n quÃ¡ dÃ y
tick_step = 2
tick_labels = [i for i in range(n_classes) if i % tick_step == 0]
tick_positions = [i + 0.5 for i in tick_labels] # +0.5 Ä‘á»ƒ cÄƒn giá»¯a Ã´

plt.figure(figsize=(18, 14))
sns.heatmap(
    cm, 
    annot=False,          # <--- YÃªu cáº§u chÃ­nh: Táº¯t hiá»ƒn thá»‹ sá»‘
    fmt='d',              # Ä�á»‹nh dáº¡ng sá»‘ (khÃ´ng quan trá»�ng khi annot=False)
    cmap='coolwarm',      # DÃ¹ng colormap 'coolwarm' (blue-red) giá»‘ng áº£nh cá»§a báº¡n
    xticklabels=tick_labels,
    yticklabels=tick_labels,
    cbar=True             # Hiá»ƒn thá»‹ thanh mÃ u (colorbar)
)

# Ä�áº·t láº¡i vá»‹ trÃ­ cá»§a ticks Ä‘á»ƒ khá»›p vá»›i nhÃ£n
plt.xticks(ticks=tick_positions, labels=tick_labels, rotation=0, fontsize=10)
plt.yticks(ticks=tick_positions, labels=tick_labels, rotation=0, fontsize=10)

plt.title('Confusion Matrix (Heatmap - KhÃ´ng cÃ³ sá»‘)', fontsize=18)
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)
plt.show()


from transformers import pipeline
from tqdm.auto import tqdm

pipe = pipeline(task='text-classification', model=model, tokenizer=tokenizer, device=CFG.DEVICE)
pipe.predict(['This is a test'])


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

