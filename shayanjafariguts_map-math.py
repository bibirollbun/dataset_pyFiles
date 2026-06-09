# -*- coding: utf-8 -*-
"""
ğŸ�† MAP Competition - Final Inference-Only Notebook
Model: microsoft/deberta-v3-small
Fixes: Correct dataset path, inverse_transform on 2D, no extra files
Output: Only submission.csv
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# --- ØªÙ†Ø¸ÛŒÙ…Ø§Øª ---
MODEL_NAME = "microsoft/deberta-v3-small"
MAX_LENGTH = 256
BATCH_SIZE = 32
FP16 = True  # Ù�Ù‚Ø· Ø¨Ø§ GPU

# --- Ø¨Ø§Ø±Ú¯Ø°Ø§Ø±ÛŒ Ø¯Ø§Ø¯Ù‡â€ŒÙ‡Ø§ (Ù…Ø³ÛŒØ± Ø¯Ø±Ø³Øª!) ---
print("ğŸ”„ Ø¨Ø§Ø±Ú¯Ø°Ø§Ø±ÛŒ Ø¯Ø§Ø¯Ù‡â€ŒÙ‡Ø§...")
train = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')
test = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')

# --- Ù¾Ø± Ú©Ø±Ø¯Ù† NA Ùˆ Ø³Ø§Ø®Øª Ø¨Ø±Ú†Ø³Ø¨ ---
print("ğŸ§¹ Ø¢Ù…Ø§Ø¯Ù‡â€ŒØ³Ø§Ø²ÛŒ Ø¨Ø±Ú†Ø³Ø¨â€ŒÙ‡Ø§...")
train['Misconception'] = train['Misconception'].fillna('NA')
train['label_str'] = train['Category'] + ':' + train['Misconception']

# --- Feature Engineering: is_correct ---
print("ğŸ”§ Ø³Ø§Ø®Øª ÙˆÛŒÚ˜Ú¯ÛŒ is_correct...")
true_mask = train['Category'].str.startswith('True')
true_answers = train[true_mask].copy()

# Ù¾ÛŒØ¯Ø§ Ú©Ø±Ø¯Ù† Ø¬ÙˆØ§Ø¨ Ø¯Ø±Ø³Øª Ø¨Ø±Ø§ÛŒ Ù‡Ø± Ø³Ø¤Ø§Ù„ (Ù¾Ø±Ù�Ø±Ø¬Ù†Ø³â€ŒØªØ±ÛŒÙ† Ø¬ÙˆØ§Ø¨ Ø¯Ø±Ø³Øª)
correct_by_q = true_answers.groupby('QuestionId')['MC_Answer'] \
    .agg(lambda x: x.value_counts().index[0]) \
    .reset_index()
correct_by_q.columns = ['QuestionId', 'CorrectAnswer']

# Ø§Ø¯ØºØ§Ù… Ø¨Ø§ ØªØ³Øª
test = test.merge(correct_by_q, on='QuestionId', how='left')
test['is_correct'] = (test['MC_Answer'] == test['CorrectAnswer']).astype(int)

# --- Ø³Ø§Ø®Øª Ù¾Ø±Ø§Ù…Ù¾Øª Ù‡ÙˆØ´Ù…Ù†Ø¯Ø§Ù†Ù‡ ---
def format_prompt(row):
    correct_text = "This answer is correct." if row['is_correct'] else "This answer is incorrect."
    return (
        f"Question: {row['QuestionText']}\n"
        f"Chosen Answer: {row['MC_Answer']}\n"
        f"{correct_text}\n"
        f"Student Explanation: {row['StudentExplanation']}"
    )

test['text'] = test.apply(format_prompt, axis=1)

# --- Ú©Ø¯Ú¯Ø°Ø§Ø±ÛŒ Ø¨Ø±Ú†Ø³Ø¨â€ŒÙ‡Ø§ ---
print("ğŸ�·ï¸� Ú©Ø¯Ú¯Ø°Ø§Ø±ÛŒ Ø¨Ø±Ú†Ø³Ø¨â€ŒÙ‡Ø§...")
le = LabelEncoder()
train['label'] = le.fit_transform(train['label_str'])
num_labels = len(le.classes_)
print(f"âœ… {num_labels} Ø¨Ø±Ú†Ø³Ø¨ Ù…Ù†Ø­ØµØ± Ø¨Ù‡ Ù�Ø±Ø¯ Ø´Ù†Ø§Ø³Ø§ÛŒÛŒ Ø´Ø¯.")

# --- ØªÙˆÚ©Ù†Ø§ÛŒØ²Ø± Ùˆ Ù…Ø¯Ù„ ---
print("ğŸ¤– Ù„ÙˆØ¯ Ù…Ø¯Ù„ Ùˆ ØªÙˆÚ©Ù†Ø§ÛŒØ²Ø±...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels
).cuda()

if FP16:
    model = model.half()

# --- ØªØ§Ø¨Ø¹ Ù¾ÛŒØ´â€ŒØ¨ÛŒÙ†ÛŒ Ø¯Ø³ØªÙ‡â€ŒØ§ÛŒ ---
def predict_batch(texts):
    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
        return_tensors="pt"
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model(**encoded)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()
    
    return probs

# --- Ù¾ÛŒØ´â€ŒØ¨ÛŒÙ†ÛŒ Ø±ÙˆÛŒ ØªØ³Øª ---
print("ğŸ”® Ø¯Ø± Ø­Ø§Ù„ Ù¾ÛŒØ´â€ŒØ¨ÛŒÙ†ÛŒ Ø±ÙˆÛŒ Ø¯Ø§Ø¯Ù‡ ØªØ³Øª...")
all_probs = []
for i in range(0, len(test), BATCH_SIZE):
    batch_texts = test['text'].iloc[i:i+BATCH_SIZE].tolist()
    probs = predict_batch(batch_texts)
    all_probs.append(probs)

all_probs = np.vstack(all_probs)

# --- Ø±Ù�Ø¹ Ø®Ø·Ø§ÛŒ inverse_transform Ø±ÙˆÛŒ Ø¢Ø±Ø§ÛŒÙ‡ 2D ---
print("ğŸ”§ ØªØ¨Ø¯ÛŒÙ„ Ø§Ø­ØªÙ…Ø§Ù„Ø§Øª Ø¨Ù‡ Ø¨Ø±Ú†Ø³Ø¨â€ŒÙ‡Ø§ÛŒ Ù‚Ø§Ø¨Ù„ Ø®ÙˆØ§Ù†Ø¯Ù†...")
top3_indices = np.argsort(-all_probs, axis=1)[:, :3]  # (n_samples, 3)

# âœ… Ø±Ø§Ù‡â€ŒØ­Ù„: Ù�Ù„Øª Ú©Ø±Ø¯Ù†ØŒ ØªØ¨Ø¯ÛŒÙ„ØŒ Ùˆ Ø¨Ø§Ø²Ø³Ø§Ø²ÛŒ
flat_top3 = top3_indices.flatten()  # (n_samples * 3,)
decoded_flat = le.inverse_transform(flat_top3)
decoded_labels = decoded_flat.reshape(top3_indices.shape)  # Ø¯ÙˆØ¨Ø§Ø±Ù‡ Ø¨Ù‡ (n_samples, 3)

# --- Ø³Ø§Ø®Øª Ø³Ø§Ø¨Ù…ÛŒØ´Ù† ---
print("ğŸ“¥ Ø³Ø§Ø®Øª submission.csv...")
submission_rows = []
for i, row in test.iterrows():
    preds = " ".join(decoded_labels[i])
    submission_rows.append({'row_id': row['row_id'], 'Category:Misconception': preds})

submission_df = pd.DataFrame(submission_rows)
submission_df.to_csv('submission.csv', index=False)

print("\nğŸ�‰ Ù…ÙˆÙ�Ù‚ÛŒØª! Ù�Ù‚Ø· submission.csv Ø°Ø®ÛŒØ±Ù‡ Ø´Ø¯.")
print(submission_df.head())



























































































