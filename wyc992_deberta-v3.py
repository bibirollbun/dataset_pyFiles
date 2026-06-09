%%writefile utils.py
import pandas as pd
import re

def url_to_semantics(text: str) -> str:
    if not isinstance(text, str):
        return ""

    url_pattern = r'https?://[^\s/$.?#].[^\s]*'
    urls = re.findall(url_pattern, text)
    
    if not urls:
        return "" 

    all_semantics = []
    seen_semantics = set()

    for url in urls:
        url_lower = url.lower()
        
        domain_match = re.search(r"(?:https?://)?([a-z0-9\-\.]+)\.[a-z]{2,}", url_lower)
        if domain_match:
            full_domain = domain_match.group(1)
            parts = full_domain.split('.')
            for part in parts:
                if part and part not in seen_semantics and len(part) > 3: # Avoid short parts like 'www'
                    all_semantics.append(f"domain:{part}")
                    seen_semantics.add(part)

        # 2. Extract path parts
        path = re.sub(r"^(?:https?://)?[a-z0-9\.-]+\.[a-z]{2,}/?", "", url_lower)
        path_parts = [p for p in re.split(r'[/_.-]+', path) if p and p.isalnum()] # Split by common delimiters

        for part in path_parts:
            # Clean up potential file extensions or query params
            part_clean = re.sub(r"\.(html?|php|asp|jsp)$|#.*|\?.*", "", part)
            if part_clean and part_clean not in seen_semantics and len(part_clean) > 3:
                all_semantics.append(f"path:{part_clean}")
                seen_semantics.add(part_clean)

    if not all_semantics:
        return ""

    return f"\nURL Keywords: {' '.join(all_semantics)}"


def get_dataframe_to_train(data_path):
    train_dataset = pd.read_csv(f"{data_path}/train.csv") 
    test_dataset = pd.read_csv(f"{data_path}/test.csv")

    flatten = []

    flatten.append(train_dataset[["body", "rule", "subreddit","rule_violation"]].copy())

    for violation_type in ["positive", "negative"]:
        for i in range(1, 3):
            col_name = f"{violation_type}_example_{i}"
            
            if col_name in train_dataset.columns:
                sub_dataset = train_dataset[[col_name, "rule", "subreddit"]].copy()
                sub_dataset = sub_dataset.rename(columns={col_name: "body"})
                sub_dataset["rule_violation"] = 1 if violation_type == "positive" else 0
                
                sub_dataset.dropna(subset=['body'], inplace=True)
                sub_dataset = sub_dataset[sub_dataset['body'].str.strip().str.len() > 0]
                
                if not sub_dataset.empty:
                    flatten.append(sub_dataset)
    
    for violation_type in ["positive", "negative"]:
        for i in range(1, 3):
            col_name = f"{violation_type}_example_{i}"
            
            if col_name in test_dataset.columns:
                sub_dataset = test_dataset[[col_name, "rule", "subreddit"]].copy()
                sub_dataset = sub_dataset.rename(columns={col_name: "body"})
                sub_dataset["rule_violation"] = 1 if violation_type == "positive" else 0
                
                sub_dataset.dropna(subset=['body'], inplace=True)
                sub_dataset = sub_dataset[sub_dataset['body'].str.strip().str.len() > 0]
                
                if not sub_dataset.empty:
                    flatten.append(sub_dataset)
    
    dataframe = pd.concat(flatten, axis=0)
    dataframe = dataframe.drop_duplicates(subset=['body', 'rule', 'subreddit'], ignore_index=True)
    dataframe.drop_duplicates(subset=['body','rule'],keep='first',inplace=True)
    
    return dataframe.sample(frac=1, random_state=42).reset_index(drop=True)


%%writefile train_deberta.py

import os
import pandas as pd
import torch
import random
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

from utils import get_dataframe_to_train, url_to_semantics


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CFG:
    model_name_or_path = "/kaggle/input/huggingfacedebertav3variants/deberta-v3-base"
    data_path = "/kaggle/input/jigsaw-agile-community-rules/"
    output_dir = "./deberta_v3_small_final_model"

    EPOCHS = 3
    LEARNING_RATE = 2e-5

    MAX_LENGTH = 512
    BATCH_SIZE = 8


class JigsawDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])


def main():
    seed_everything(42)

    # ========= 1. 读训练数据 =========
    training_data_df = get_dataframe_to_train(CFG.data_path)
    print(f"Training dataset size: {len(training_data_df)}")

    # ========= 2. 读测试数据 =========
    test_df_for_prediction = pd.read_csv(f"{CFG.data_path}/test.csv")

    # ========= 3. 拼接 URL 语义 =========
    training_data_df['body_with_url'] = training_data_df['body'].apply(lambda x: x + url_to_semantics(x))
    test_df_for_prediction['body_with_url'] = test_df_for_prediction['body'].apply(lambda x: x + url_to_semantics(x))

    # ========= 4. Tokenizer =========
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name_or_path)

    collator = DataCollatorWithPadding(tokenizer)

    # 【关键改动】用句对编码，rule = text, body = text_pair
    train_encodings = tokenizer(
        training_data_df["rule"].tolist(),
        training_data_df["body_with_url"].tolist(),
        truncation="only_second",       # 只截断评论
        max_length=CFG.MAX_LENGTH,
        # padding=True
    )
    train_labels = training_data_df['rule_violation'].tolist()
    train_dataset = JigsawDataset(train_encodings, train_labels)

    # ========= 5. 模型 =========
    model = AutoModelForSequenceClassification.from_pretrained(CFG.model_name_or_path, num_labels=2)

    training_args = TrainingArguments(
        output_dir=CFG.output_dir,
        num_train_epochs=CFG.EPOCHS,
        learning_rate=CFG.LEARNING_RATE,
        per_device_train_batch_size=CFG.BATCH_SIZE,
        warmup_ratio=0.1,
        weight_decay=0.01,
        report_to="none",
        save_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,   # ★ 动态 padding
    )

    trainer.train()

    # ========= 6. 推理 =========
    test_encodings = tokenizer(
        test_df_for_prediction["rule"].tolist(),
        test_df_for_prediction["body_with_url"].tolist(),
        truncation="only_second",
        max_length=CFG.MAX_LENGTH,
        # padding=True
    )
    test_dataset = JigsawDataset(test_encodings)

    predictions = trainer.predict(test_dataset)
    probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1)[:, 1].numpy()

    submission_df = pd.DataFrame({
        "row_id": test_df_for_prediction["row_id"],
        "rule_violation": probs
    })
    submission_df.to_csv("submission.csv", index=False)
    print("Saved submission.csv")


if __name__ == "__main__":
    main()



!python train_deberta.py

