# !pip install -U transformers > /dev/null


from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "/kaggle/input/smollm-135-map-classification-65-labels-v1/transformers/default/1"
device = "cuda"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(model_name,).to(device)


id2label = model.config.id2label
label2id = model.config.label2id


from pathlib import Path

dataset_path = Path("/kaggle/input/map-charting-student-math-misunderstandings/")
train_path = dataset_path / "train.csv"
test_path = dataset_path / "test.csv"
sample_submission_path = dataset_path / "sample_submission.csv"


from datasets import load_dataset

ds = load_dataset("csv", data_files=str(test_path))["train"]
ds


def gen_prompt(row):

    prompt = (f"Question: {row['QuestionText']}\n" + 
              f"MC_Answer: {row['MC_Answer']}\n" +
              f"StudentExplanation: {row['StudentExplanation']}"
             )
    return prompt
    
def process_dataset(sample):
    # обработка сообщения
    prompt = gen_prompt(sample)

    return {"prompt": prompt}

ds = ds.map(process_dataset)

row_ids = ds['row_id']
drop_columns = ["row_id",
                "QuestionId",
                "QuestionText",
                "StudentExplanation",
                "MC_Answer"]

ds = ds.remove_columns(drop_columns)


def tokenize(examples):
    return tokenizer(examples["prompt"], truncation=True, padding=True)

ds = ds.map(tokenize, batched=True)
ds = ds.remove_columns("prompt")


from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from tqdm import tqdm
import torch

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
test_dataloader = DataLoader(ds, batch_size=16, collate_fn=data_collator)

all_logits = []

model.eval()
with torch.no_grad():
    for batch in tqdm(test_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        all_logits.append(logits.cpu())

logits = torch.cat(all_logits, dim=0)
probs = torch.nn.functional.softmax(logits, dim=1).numpy()


import numpy as np
import pandas as pd

# Получаем индексы топ-3 предсказаний для каждого примера
top3_indices = np.argsort(-probs, axis=1)[:, :3]

# Преобразуем индексы в метки классов
top3_labels = []
for indices in top3_indices:
    labels = [id2label[idx] for idx in indices]
    top3_labels.append(labels)

formatted_predictions = [' '.join(labels) for labels in top3_labels]

submission = pd.DataFrame({
    'row_id': row_ids,
    'Category:Misconception': formatted_predictions
})

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)

print(f"Submission файл успешно создан: {submission_path}")
print(submission.head())

