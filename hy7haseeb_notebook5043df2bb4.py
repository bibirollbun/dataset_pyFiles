import os
import sys
import argparse
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from accelerate import Accelerator

# ==========================
# Dataset Class
# ==========================
class JigsawDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=256, col_rule="rule", col_subreddit="subreddit", col_body="body", col_label="label"):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.col_rule = col_rule
        self.col_subreddit = col_subreddit
        self.col_body = col_body
        self.col_label = col_label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = f"Rule: {row[self.col_rule]} Subreddit: {row[self.col_subreddit]} Comment: {row[self.col_body]}"
        inputs = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        item = {key: val.squeeze(0) for key, val in inputs.items()}
        if self.col_label in row:
            item["labels"] = torch.tensor(row[self.col_label], dtype=torch.long)
        return item

# ==========================
# Training Loop (Simplified)
# ==========================
def train(args):
    precision = None
    if args.fp16:
        precision = "fp16"
    elif args.bf16:
        precision = "bf16"

    accelerator = Accelerator(mixed_precision=precision)

    # Auto-detect Kaggle paths
    if args.train_csv is None:
        args.train_csv = "/kaggle/input/jigsaw-agile-community-rules/train.csv"
    if args.test_csv is None:
        args.test_csv = "/kaggle/input/jigsaw-agile-community-rules/test.csv"
    if args.model_path is None:
        # Try to find DeBERTa model folder
        for root, dirs, files in os.walk("/kaggle/input"):
            if "config.json" in files and "pytorch_model.bin" in files:
                args.model_path = root
                break

    # Load data
    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    train_dataset = JigsawDataset(train_df, tokenizer, max_length=args.max_length)
    test_dataset = JigsawDataset(test_df, tokenizer, max_length=args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size)

    # Model
    model = AutoModel.from_pretrained(args.model_path)

    # Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    # Prepare with accelerator
    model, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, test_loader, scheduler
    )

    model.train()
    for epoch in range(args.epochs):
        for step, batch in enumerate(train_loader):
            outputs = model(**{k: v for k, v in batch.items() if k != "labels"})
            loss = outputs.last_hidden_state.mean()  # Placeholder loss for now

            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        accelerator.print(f"Epoch {epoch+1} finished")

    # Save dummy submission
    sample_sub = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/sample_submission.csv")

    sub_path = "/kaggle/working/submission.csv"
    submission = pd.DataFrame({
    "row_id": sample_sub["row_id"],               # must match exactly
    "rule_violation": np.zeros(len(sample_sub))   # replace with real predictions later
    })
    submission.to_csv(sub_path, index=False)

    accelerator.print(f"✅ Saved submission to {sub_path}")



# ==========================
# Argument Parser
# ==========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, default=None)
    parser.add_argument("--val_csv", type=str, default=None)
    parser.add_argument("--test_csv", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/outputs")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()

    train(args)

if __name__ == "__main__":
    # Strip out unwanted Jupyter/Kaggle arguments like "-f ...json"
    sys.argv = [sys.argv[0]]
    main()





