# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Membersihkan library lama dan menginstal versi terbaru yang stabil
# untuk memastikan tidak ada konflik lingkungan.
!pip uninstall -y transformers accelerate peft bitsandbytes
!pip install -U transformers accelerate peft bitsandbytes datasets evaluate


# --- 0. Impor Library ---
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    Gemma3ForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model
import evaluate

# --- 1. Kelas Model Custom ---
# Kelas ini sudah benar dan tidak perlu diubah.
class Gemma3ForSequenceClassification(nn.Module):
    def __init__(self, model_path, num_labels):
        super(Gemma3ForSequenceClassification, self).__init__()
        self.num_labels = num_labels
        
        # SINTAKS TERBARU: Menggunakan 'dtype' (terbaru) bukan 'torch_dtype' (usang).
        full_model = Gemma3ForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16
        )
        self.gemma3 = full_model.model
        self.config = self.gemma3.config
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        
        # SOLUSI: Secara eksplisit ubah tipe data lapisan classifier agar sama.
        self.classifier.to(full_model.dtype)

    # Menggunakan **kwargs agar forward pass lebih fleksibel.
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        outputs = self.gemma3(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        
        sequence_lengths = torch.sum(attention_mask, dim=1) - 1
        batch_size = input_ids.shape[0]
        pooled_output = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
            
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
        return (loss, logits) if loss is not None else logits

# --- 2. Persiapan Data (Logika sudah benar) ---
df = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")
df['Misconception'] = df['Misconception'].fillna('NA')
df['text'] = df['QuestionText'] + ' [SEP] ' + df['MC_Answer'] + ' [SEP] ' + df['StudentExplanation']
df['target'] = df['Category'] + ':' + df['Misconception']

label_counts = df['target'].value_counts()
valid_targets = label_counts[label_counts > 1].index
df_filtered = df[df['target'].isin(valid_targets)].copy()

labels_map = {label: i for i, label in enumerate(df_filtered['target'].unique())}
df_filtered['label'] = df_filtered['target'].map(labels_map)

train_df, val_df = train_test_split(
    df_filtered, test_size=0.2, stratify=df_filtered['label'], random_state=42
)

# --- 3. Tokenizer & Data Collator ---
model_id = "/kaggle/input/gemma-3/transformers/gemma-3-1b-pt/1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
def tokenize_fn(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

train_ds = Dataset.from_pandas(train_df).map(tokenize_fn, batched=True)
val_ds = Dataset.from_pandas(val_df).map(tokenize_fn, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# --- 4. Model & LoRA ---
num_labels = len(labels_map)
base_model = Gemma3ForSequenceClassification(model_id, num_labels)
peft_config = LoraConfig(
    task_type="SEQ_CLS", r=16, lora_alpha=32, lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)
model = get_peft_model(base_model, peft_config)
model.print_trainable_parameters()

# --- 5. Training Arguments (dengan Hyperparameter yang Diperbaiki) ---
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5, 
    per_device_train_batch_size=4, 
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    bf16=True, 
    report_to="none"
)

# --- 6. Metrics ---
accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=preds, references=labels)

# --- 7. Trainer ---
# Menggunakan DataCollator, cara modern tanpa argumen 'tokenizer' usang.
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# --- 8. Run Training ---
print("\nMulai fine-tuning dengan LoRA di 2xT4...")
trainer.train()
print("Training selesai!")

# --- 9. Simpan Model Terbaik ---
trainer.save_model('./best_model')
print("Model dan tokenizer terbaik berhasil disimpan di folder './best_model'")





