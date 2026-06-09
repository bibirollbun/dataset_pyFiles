import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

VER=1
model_name = "/kaggle/input/huggingfacedebertav3variants/deberta-v3-base"
EPOCHS = 6

DIR = f"ver_{VER}"
os.makedirs(DIR, exist_ok=True)


import pandas as pd, numpy as np
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
train = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')
train.Misconception = train.Misconception.fillna('NA')
train['target'] = train.Category+":"+train.Misconception
train['label'] = le.fit_transform(train['target'])
n_classes = len(le.classes_)
print(f"Train shape: {train.shape} with {n_classes} target classes")
train.head()


import matplotlib.pyplot as plt
import seaborn as sns

# Cáº¥u hÃ¬nh kÃ­ch thÆ°á»›c biá»ƒu Ä‘á»“
plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")

# --- BIá»‚U Ä�á»’ 1: PhÃ¢n phá»‘i cÃ¡c Category chÃ­nh (High-level) ---
plt.subplot(1, 2, 1)
ax = sns.countplot(data=train, y='Category', order=train['Category'].value_counts().index, palette='viridis')
plt.title('PhÃ¢n phá»‘i cÃ¡c Category chÃ­nh', fontsize=14)
plt.xlabel('Sá»‘ lÆ°á»£ng máº«u')
plt.ylabel('Category')
# Hiá»ƒn thá»‹ sá»‘ lÆ°á»£ng trÃªn thanh
for container in ax.containers:
    ax.bar_label(container)

# --- BIá»‚U Ä�á»’ 2: Top 20 Lá»›p Target phá»• biáº¿n nháº¥t (Low-level) ---
# VÃ¬ cÃ³ tá»›i 65 lá»›p, váº½ háº¿t sáº½ ráº¥t rá»‘i, ta chá»‰ váº½ Top 20 lá»›p chiáº¿m Ä‘a sá»‘
plt.subplot(1, 2, 2)
top_20_targets = train['target'].value_counts().head(20)
sns.barplot(x=top_20_targets.values, y=top_20_targets.index, palette='magma')
plt.title('Top 20 NhÃ£n (Target) phá»• biáº¿n nháº¥t', fontsize=14)
plt.xlabel('Sá»‘ lÆ°á»£ng máº«u')
plt.ylabel('Target (Category:Misconception)')

plt.tight_layout()
plt.show()

# --- BIá»‚U Ä�á»’ 3: Biá»ƒu Ä‘á»“ toÃ n cáº£nh (Náº¿u báº¡n muá»‘n xem Ä‘á»™ máº¥t cÃ¢n báº±ng dá»¯ liá»‡u) ---
plt.figure(figsize=(10, 15))
target_counts = train['target'].value_counts().sort_values()
sns.barplot(x=target_counts.values, y=target_counts.index, palette='coolwarm')
plt.title('PhÃ¢n phá»‘i toÃ n bá»™ 65 lá»›p cáº§n phÃ¢n loáº¡i (Tá»« Ã­t nháº¥t Ä‘áº¿n nhiá»�u nháº¥t)', fontsize=14)
plt.xlabel('Sá»‘ lÆ°á»£ng máº«u')
plt.show()

# In thá»‘ng kÃª text
print("\n--- Thá»‘ng kÃª chi tiáº¿t ---")
print(f"Lá»›p phá»• biáº¿n nháº¥t: {target_counts.index[-1]} ({target_counts.values[-1]} máº«u)")
print(f"Lá»›p Ã­t phá»• biáº¿n nháº¥t: {target_counts.index[0]} ({target_counts.values[0]} máº«u)")


idx = train.apply(lambda row: row.Category.split('_')[0],axis=1)=='True'
correct = train.loc[idx].copy()
correct['c'] = correct.groupby(['QuestionId','MC_Answer']).MC_Answer.transform('count')
correct = correct.sort_values('c',ascending=False)
correct = correct.drop_duplicates(['QuestionId'])
correct = correct[['QuestionId','MC_Answer']]
correct['is_correct'] = 1

train = train.merge(correct, on=['QuestionId','MC_Answer'], how='left')
train.is_correct = train.is_correct.fillna(0)


from IPython.display import display, Math, Latex

# GET ANSWER CHOICES
tmp = train.groupby(['QuestionId','MC_Answer']).size().reset_index(name='count')
tmp['rank'] = tmp.groupby('QuestionId')['count'].rank(method='dense', ascending=False).astype(int) - 1
tmp = tmp.drop('count',axis=1)
tmp = tmp.sort_values(['QuestionId','rank'])

# DISPLAY QUESTION AND ANSWER CHOICES
Q = tmp.QuestionId.unique()
for q in Q:
    question = train.loc[train.QuestionId==q].iloc[0].QuestionText
    choices = tmp.loc[tmp.QuestionId==q].MC_Answer.values
    labels="ABCD"
    choice_str = " ".join([f"({labels[i]}) {choice}" for i, choice in enumerate(choices)])
    
    print()
    display(Latex(f"QuestionId {q}: {question}") )
    display(Latex(f"MC Answers: {choice_str}"))


import torch
from transformers import DebertaTokenizer, DebertaForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from datasets import Dataset
import numpy as np

tokenizer = AutoTokenizer.from_pretrained(model_name)
MAX_LEN = 256


# --- Code chuáº©n hÃ³a dá»¯ liá»‡u má»›i thÃªm vÃ o ---
import re

def clean_math_text(text):
    if not isinstance(text, str): return text
    
    # VÃ­ dá»¥ thay tháº¿ cÃ¡c kÃ½ hiá»‡u cÆ¡ báº£n
    text = text.replace('+', ' plus ')
    text = text.replace('-', ' minus ')
    text = text.replace('=', ' equals ')
    text = text.replace('/', ' over ')
    text = text.replace('*', ' times ')
    
    # Xá»­ lÃ½ LaTeX Ä‘Æ¡n giáº£n (tÃ¹y nhu cáº§u)
    text = re.sub(r'\\frac\{(\d+)\}\{(\d+)\}', r'\1 over \2', text) # \frac{1}{2} -> 1 over 2
    
    # XÃ³a cÃ¡c kÃ½ tá»± Ä‘áº·c biá»‡t cÃ²n sÃ³t láº¡i náº¿u muá»‘n
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text) 
    
    # Chuáº©n hÃ³a khoáº£ng tráº¯ng
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# --- Cell 34 CÅ© ---
def format_input(row):
    x = "This answer is correct."
    if not row['is_correct']:
        x = "This is answer is incorrect."
    
    # --- THÃŠM PHáº¦N NÃ€Y: Ã�p dá»¥ng chuáº©n hÃ³a ---
    question = clean_math_text(row['QuestionText'])
    answer = clean_math_text(row['MC_Answer'])
    explanation = clean_math_text(row['StudentExplanation'])
    # --------------------------------------

    return (
        f"Question: {question}\n"  # DÃ¹ng biáº¿n Ä‘Ã£ chuáº©n hÃ³a
        f"Answer: {answer}\n"      # DÃ¹ng biáº¿n Ä‘Ã£ chuáº©n hÃ³a
        f"{x}\n"
        f"Student Explanation: {explanation}" # DÃ¹ng biáº¿n Ä‘Ã£ chuáº©n hÃ³a
    )

train['text'] = train.apply(format_input,axis=1)
# ... pháº§n cÃ²n láº¡i giá»¯ nguyÃªn


lengths = [len(tokenizer.encode(t, truncation=False)) for t in train["text"]]
import matplotlib.pyplot as plt

plt.hist(lengths, bins=50)
plt.title("Token Length Distribution")
plt.xlabel("Number of tokens")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()


L = (np.array(lengths)>MAX_LEN).sum()
print(f"There are {L} train sample(s) with more than {MAX_LEN} tokens")
np.sort( lengths )


import gc
import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold
from datasets import Dataset
from transformers import DebertaV2ForSequenceClassification, TrainingArguments, Trainer
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import set_seed

# --- Cáº¤U HÃŒNH "CLONE" CODE Gá»�C ---
EPOCHS = 6       
BATCH_SIZE = 4   # Giá»¯ má»©c nhá»� Ä‘á»ƒ khÃ´ng OOM
GRAD_ACCUM = 6   # TÃ­ch lÅ©y 6 láº§n => Tá»•ng Batch = 48 (Giá»‘ng há»‡t log gá»‘c)
LR = 5e-5        # LR cao nhÆ° gá»‘c

print(f"ğŸš€ Báº¯t Ä‘áº§u training 5-Fold (Simulating Original Batch=48, LR={LR})...")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(train, train['label'])):
    print(f"\n{'='*20} FOLD {fold+1}/5 {'='*20}")
    
    # Reset Accelerator
    try:
        AcceleratorState._reset_state(reset_partial_state=True)
        dummy = Accelerator()
    except: pass
    gc.collect()
    torch.cuda.empty_cache()
    set_seed(42)

    # 1. Data
    train_df = train.iloc[train_idx].copy()
    val_df = train.iloc[val_idx].copy()
    
    train_ds = Dataset.from_pandas(train_df[['text', 'label']]).map(tokenize, batched=True)
    val_ds = Dataset.from_pandas(val_df[['text', 'label']]).map(tokenize, batched=True)
    
    cols = ['input_ids', 'attention_mask', 'label']
    train_ds.set_format('torch', columns=cols)
    val_ds.set_format('torch', columns=cols)
    
    # 2. Model
    model = DebertaV2ForSequenceClassification.from_pretrained(
        model_name, num_labels=n_classes, ignore_mismatched_sizes=True
    )
    
    # 3. Arguments (Match Code Gá»‘c)
    training_args = TrainingArguments(
        output_dir=f"./{DIR}/fold_{fold}",
        do_train=True, do_eval=True,
        
        eval_strategy="steps", save_strategy="steps",
        eval_steps=200, save_steps=200, logging_steps=50,
        
        num_train_epochs=EPOCHS,
        learning_rate=LR,              # 5e-5
        
        # Giáº£ láº­p Batch 48
        per_device_train_batch_size=BATCH_SIZE, 
        gradient_accumulation_steps=GRAD_ACCUM, 
        
        per_device_eval_batch_size=16,
        
        warmup_ratio=0.1,
        save_total_limit=1,
        metric_for_best_model="map@3",
        greater_is_better=True,
        load_best_model_at_end=True,
        report_to="none",
        fp16=True
    )
    
    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        tokenizer=tokenizer, compute_metrics=compute_map3
    )
    
    trainer.train()
    
    # Score
    best_score = trainer.evaluate()['eval_map@3']
    cv_scores.append(best_score)
    print(f">> Fold {fold+1} Best MAP@3: {best_score:.4f}")
    
    # Save
    trainer.save_model(f"{DIR}/model_fold_{fold}")
    tokenizer.save_pretrained(f"{DIR}/model_fold_{fold}")
    
    del model, trainer
    gc.collect()

print(f"\nâœ… TRAINING DONE! AVG MAP@3: {np.mean(cv_scores):.5f}")


!rm -r /kaggle/working/ver_1/label_encoder.joblib


tokenizer = AutoTokenizer.from_pretrained(f"{DIR}/best")
model = DebertaV2ForSequenceClassification.from_pretrained(
    f"{DIR}/best",
    num_labels=n_classes  
)
training_args = TrainingArguments(report_to="none")
trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args)
le = joblib.load(f"{DIR}/label_encoder.joblib")


# --- CHUáº¨N Bá»Š Dá»® LIá»†U TEST (Giá»¯ nguyÃªn logic cÅ©) ---
test = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')

# Feature Engineering cho Test
test = test.merge(correct, on=['QuestionId','MC_Answer'], how='left')
test.is_correct = test.is_correct.fillna(0)
test['text'] = test.apply(format_input, axis=1)

# Tokenize Test
ds_test = Dataset.from_pandas(test[['text']])
ds_test = ds_test.map(tokenize, batched=True)

# --- INFERENCE ENSEMBLE (Dá»± Ä‘oÃ¡n gá»™p 5 Fold) ---
import numpy as np

# Táº¡o máº£ng chá»©a tá»•ng xÃ¡c suáº¥t: kÃ­ch thÆ°á»›c (sá»‘ máº«u test, sá»‘ classes)
final_probs = np.zeros((len(test), n_classes))

print("Báº¯t Ä‘áº§u dá»± Ä‘oÃ¡n Ensemble...")

for fold in range(N_FOLDS):
    print(f"Ä�ang dá»± Ä‘oÃ¡n vá»›i Model Fold {fold}...")
    
    # Ä�Æ°á»�ng dáº«n model Ä‘Ã£ lÆ°u
    model_path = f"{DIR}/model_fold_{fold}"
    
    # Load model cá»§a fold Ä‘Ã³
    model_fold = DebertaV2ForSequenceClassification.from_pretrained(model_path, num_labels=n_classes)
    
    # Táº¡o Trainer chá»‰ Ä‘á»ƒ predict (khÃ´ng cáº§n training args phá»©c táº¡p)
    trainer_fold = Trainer(
        model=model_fold,
        processing_class=tokenizer
    )
    
    # Predict
    preds_fold = trainer_fold.predict(ds_test)
    
    # Chuyá»ƒn logits thÃ nh probabilities (Softmax)
    # logits lÃ  Ä‘áº§u ra thÃ´, pháº£i qua softmax Ä‘á»ƒ thÃ nh xÃ¡c suáº¥t %
    probs_fold = torch.nn.functional.softmax(torch.tensor(preds_fold.predictions), dim=1).numpy()
    
    # Cá»™ng dá»“n vÃ o káº¿t quáº£ tá»•ng
    final_probs += probs_fold
    
    # Dá»�n dáº¹p
    del model_fold, trainer_fold, preds_fold, probs_fold
    gc.collect()
    torch.cuda.empty_cache()

# Chia trung bÃ¬nh cá»™ng xÃ¡c suáº¥t (Average Blending)
avg_probs = final_probs / N_FOLDS
print("HoÃ n thÃ nh dá»± Ä‘oÃ¡n!")

# --- Táº O SUBMISSION ---

# Láº¥y Top 3 tá»« xÃ¡c suáº¥t trung bÃ¬nh
top3_indices = np.argsort(-avg_probs, axis=1)[:, :3]

# Giáº£i mÃ£ tá»« sá»‘ sang chá»¯
le = joblib.load(f"{DIR}/label_encoder.joblib") # Load láº¡i encoder
flat_top3 = top3_indices.flatten()
decoded_labels = le.inverse_transform(flat_top3)
top3_labels = decoded_labels.reshape(top3_indices.shape)

# GhÃ©p chuá»—i
joined_preds = [" ".join(row) for row in top3_labels]

# LÆ°u file
sub = pd.DataFrame({
    "row_id": test.row_id.values,
    "Category:Misconception": joined_preds
})
sub.to_csv("submission.csv", index=False)
print("Ä�Ã£ lÆ°u file submission.csv")
sub.head()

