import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from tqdm.notebook import tqdm
import torch

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")
test = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")
sample_submission = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv")


sns.countplot(data=train, x='Category')
plt.xticks(rotation=45)
plt.show()


def clean_text(text):
    """
    Verilen metni temel NLP temizleme adÄ±mlarÄ±yla sadeleÅŸtirir.
    Simplifies the input text using basic NLP preprocessing steps.
    """
    text = str(text).lower()  # KÃ¼Ã§Ã¼k harfe Ã§evir / Lowercase
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Harf ve boÅŸluk dÄ±ÅŸÄ±ndakileri kaldÄ±r / Remove non-letter characters
    text = re.sub(r"\s+", " ", text).strip()  # Fazla boÅŸluklarÄ± kaldÄ±r / Remove extra spaces
    return text


train['CleanExplanation'] = train['StudentExplanation'].fillna('').apply(clean_text)
test['CleanExplanation'] = test['StudentExplanation'].fillna('').apply(clean_text)



train[['StudentExplanation', 'CleanExplanation']].head()


train.shape


train_small = train.sample(n=36696, random_state=42).reset_index(drop=True) 
#n=2000 for quickly train in beginings
#and then n=train.shape in next version


# LabelEncoder nesnesini oluÅŸturuyoruz / Creating the encoder
label_encoder = LabelEncoder()

# Sadece train setinde Category var, ona gÃ¶re fit ve transform yapÄ±yoruz
# Fitting and transforming only on the train set (test has no labels)
train_small['label'] = label_encoder.fit_transform(train_small['Category'])

# Kodlama haritasÄ±nÄ± gÃ¶relim / Display the mapping from Category to label
label_map = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Category to Label Mapping:", label_map)



base_path = "/kaggle/input/rubert-tiny-nerel/transformers/rubert-tiny-nerel/1"

# BERT-Tiny iÃ§in tokenizer'Ä± yerel olarak yÃ¼klÃ¼yoruz / Loading tokenizer locally for BERT-Tiny
tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)


# Ä°lk aÃ§Ä±klamayÄ± alalÄ±m / Let's take the first cleaned explanation
sample_text = train_small['CleanExplanation'].iloc[0]

# Tokenizer ile iÅŸleyelim / Process it with tokenizer
encoded = tokenizer(sample_text)

# Ã‡Ä±ktÄ±ya bakalÄ±m / Display the output
print("Original Text:\n", sample_text)
print("\nInput IDs:\n", encoded['input_ids'])
print("\nAttention Mask:\n", encoded['attention_mask'])


# Bu iÅŸlem, modelin tÃ¼m veriyi tek formatta iÅŸlemesini saÄŸlar.
# This ensures that the model processes all examples in a consistent format.
train_encodings = tokenizer(
    list(train_small['CleanExplanation']),
    truncation=True,
    padding=True,
    max_length=128
)

test_encodings = tokenizer(
    list(test['CleanExplanation']),
    truncation=True,
    padding=True,
    max_length=128
)


class MisconceptionDataset(Dataset):
    """
    ğŸ‡¹ğŸ‡· Tokenize edilmiÅŸ metinleri ve etiketleri birlikte tutan Ã¶zel veri sÄ±nÄ±fÄ±.
    ğŸ‡¬ğŸ‡§ Custom dataset class to hold tokenized inputs and labels together.
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # Bir Ã¶rneÄŸi al ve dictionary olarak dÃ¶ndÃ¼r
        # Get one sample and return it as a dictionary
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)



# Train dataset
train_dataset = MisconceptionDataset(train_encodings, train_small['label'].tolist())


model = AutoModelForSequenceClassification.from_pretrained(
    base_path,
    num_labels=len(label_encoder.classes_),  # Kategori sayÄ±sÄ± / Number of classes,
    local_files_only=True
)


training_args = TrainingArguments(
    output_dir='./results',              # SonuÃ§larÄ±n kaydedileceÄŸi klasÃ¶r
    num_train_epochs=1,                  # Epoch sayÄ±sÄ±
    per_device_train_batch_size=16,      # EÄŸitim batch size
    per_device_eval_batch_size=64,       # DeÄŸerlendirme batch size
    warmup_steps=0,                      # IsÄ±nma adÄ±mÄ±
    weight_decay=0.01,                   # AÄŸÄ±rlÄ±k dÃ¼ÅŸÃ¼ÅŸÃ¼
    logging_dir='./logs',                # Log klasÃ¶rÃ¼
    logging_steps=100,                    # Log sÄ±klÄ±ÄŸÄ±
    disable_tqdm=False,                   # âœ… TQDM Ã§ubuÄŸunu aktif et
    report_to="none"
)




model.to("cuda")   #move the model to cuda


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)


trainer.train()



# Modelin tahmin moduna alÄ±nmasÄ± / Set model to eval mode
model.eval()

# Tahminler burada toplanacak / To store predictions
predictions = []

# Batch halinde test verisini iÅŸliyoruz / Process in batches
# Tahmin dÃ¶ngÃ¼sÃ¼ / Prediction loop
with torch.no_grad():
    for i in range(0, len(test_encodings['input_ids']), 16):
        batch_input = {key: torch.tensor(val[i:i+16]).to("cuda") for key, val in test_encodings.items()}
        outputs = model(**batch_input)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        predictions.extend(probs.cpu().numpy())


# En yÃ¼ksek 3 olasÄ±lÄ±ÄŸÄ± al / Get top 3 probabilities
top3_indices = np.argsort(predictions, axis=1)[:, -3:][:, ::-1]  # BÃ¼yÃ¼kten kÃ¼Ã§Ã¼ÄŸe

# SayÄ±sal etiketleri geri Ã§evir / Convert back to text labels
top3_categories = []
for indices in top3_indices:
    cats = [label_encoder.inverse_transform([idx])[0] + ":NA" for idx in indices]
    top3_categories.append(" ".join(cats))


submission = pd.DataFrame({
    'row_id': test['row_id'],
    'Category:Misconception': top3_categories
})

# CSV'ye yaz / Save to CSV
submission.to_csv('submission.csv', index=False)
submission

