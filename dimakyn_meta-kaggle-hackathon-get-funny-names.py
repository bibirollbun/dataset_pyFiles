import os

import numpy as np
import pandas as pd

from tqdm import tqdm


from transformers import pipeline
from datasets import Dataset, concatenate_datasets 


import kagglehub

MK_PATH = kagglehub.dataset_download("kaggle/meta-kaggle")
MKC_PATH = kagglehub.dataset_download("kaggle/meta-kaggle-code")

print("Path to Meta-Kaggle dataset files:", MK_PATH)
print("Path to Meta-Kaggle-Code dataset files:", MKC_PATH)


df_users = pd.read_csv(f"{MK_PATH}/Users.csv") 


MODEL = "facebook/bart-large-mnli"
# MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
# MODEL = "valhalla/distilbart-mnli-12-1"


df_users_1 = df_users[df_users.PerformanceTier > 0][['UserName']].dropna().reset_index(drop=True)

hf_dataset = Dataset.from_pandas(df_users_1)

classifier = pipeline("zero-shot-classification", model=MODEL)

labels = ["funny", "serious", "professional", "personal"]

chunk_size = 100000
total = len(hf_dataset)
num_chunks = (total + chunk_size - 1) // chunk_size

all_results = []

for i in range(num_chunks):
    print(f"[INFO] Processing chunk {i+1}/{num_chunks} ...")
    start = i * chunk_size
    end = min(start + chunk_size, total)
    
    chunk = hf_dataset.select(range(start, end))

    def classify_batch(batch):
        results = classifier(batch["UserName"], candidate_labels=labels)
        return {
            "FunnyScore": [r["scores"][r["labels"].index("funny")] for r in results],
            "TopLabel": [r["labels"][0] for r in results]
        }

    chunk_with_scores = chunk.map(classify_batch, batched=True, batch_size=64)
    all_results.append(chunk_with_scores)

final_dataset = concatenate_datasets(all_results)

df_result = final_dataset.to_pandas()

df_result.to_csv("funny_scores.csv", index=False)




