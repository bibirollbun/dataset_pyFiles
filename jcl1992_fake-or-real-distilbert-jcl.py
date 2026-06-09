#!pip install -q evaluate


import os
import numpy as np
import pandas as pd

import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
#import evaluate

pd.set_option('display.max_colwidth', 100)


DATA_PATH = "/kaggle/input/fake-or-real-the-impostor-hunt/data/"
N_EPOCHS = 5
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
RANDOM_STATE = 4

USE_TRAINER = True


def read_file(dir_path, filename):
    
    with open(os.path.join(dir_path, filename), 'r', encoding='utf-8') as f:
        return f.read()

def read_data(root_path, split="train"):
    
    dir_path = os.path.join(root_path, split)
    article_names = sorted(os.listdir(dir_path))
    ids = [int(x[-4:]) for x in article_names]
    data = pd.DataFrame(index=ids, columns=["text1", "text2"])
    
    for i, article_name in enumerate(article_names):
        article_path = os.path.join(dir_path, article_name)
        text1 = read_file(article_path, "file_1.txt")
        text2 = read_file(article_path, "file_2.txt")
        data.iloc[i] = [text1, text2]
    
    return data


df_train = read_data(DATA_PATH, split="train")
df_test = read_data(DATA_PATH, split="test")


df = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
df = df.join(df_train, on="id")

df


real_texts1 = df.loc[df["real_text_id"]==1, "text1"]
real_texts2 = df.loc[df["real_text_id"]==2, "text2"]
real_texts = pd.concat([real_texts1, real_texts2], axis=0).sort_index().to_frame("text")
real_texts["id"] = real_texts.index
real_texts["real"] = 1

fake_texts1 = df.loc[df["real_text_id"]==2, "text1"]
fake_texts2 = df.loc[df["real_text_id"]==1, "text2"]
fake_texts = pd.concat([fake_texts1, fake_texts2], axis=0).sort_index().to_frame("text")
fake_texts["id"] = fake_texts.index
fake_texts["real"] = 0

df_texts = pd.concat([real_texts, fake_texts], axis=0).sort_values("id").reset_index(drop=True)

df_texts


df_texts_test = df_test.stack().to_frame("text").reset_index(names=["id", "text_id"])
df_texts_test["text_id"] = df_texts_test["text_id"].str.get(-1)

test_texts = df_texts_test["text"].to_list()

df_texts_test


train_texts, val_texts, train_labels, val_labels = train_test_split(df_texts["text"].to_list(), 
                                                                    df_texts["real"].to_list(), 
                                                                    test_size=0.2, 
                                                                    random_state=RANDOM_STATE)


model_name = "distilbert/distilbert-base-uncased"
#model_name = "microsoft/deberta-v3-base"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)


class TextDataset(Dataset):
    
    def __init__(self, tokenizer, corpus, labels=None):
        self.tokenizer = tokenizer
        self.corpus = corpus
        self.labels = labels

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        texts = self.corpus[idx]
        item = self.tokenizer(texts,
            add_special_tokens=True,
            max_length=512, 
            padding='max_length',
            truncation=True, 
            return_tensors='pt')
        item["input_ids"] = item["input_ids"].squeeze(0)
        item["attention_mask"] = item["attention_mask"].squeeze(0)
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])
        return item


train_dataset = TextDataset(tokenizer, train_texts, train_labels)
val_dataset = TextDataset(tokenizer, val_texts, val_labels)
test_dataset = TextDataset(tokenizer, test_texts)


if USE_TRAINER:

    #accuracy = evaluate.load("accuracy")
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(axis=1)
        return {'accuracy': (predictions == labels).mean()}
        #return accuracy.compute(predictions=predictions, references=labels)
    
    training_args = TrainingArguments(output_dir="./finetuned_model",
                                      per_device_train_batch_size=BATCH_SIZE,
                                      per_device_eval_batch_size=BATCH_SIZE,
                                      num_train_epochs=N_EPOCHS,
                                      learning_rate=LEARNING_RATE,
                                      weight_decay=WEIGHT_DECAY,
                                      eval_strategy='epoch',
                                      save_strategy='epoch',
                                      load_best_model_at_end=True,
                                      metric_for_best_model='accuracy',
                                      report_to='none')
    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=train_dataset,
                      eval_dataset=val_dataset,
                      processing_class=tokenizer,
                      compute_metrics=compute_metrics)


if USE_TRAINER:

    trainer.train()


if USE_TRAINER:

    print(trainer.evaluate())


if USE_TRAINER:

    logits = trainer.predict(test_dataset).predictions[:,1]


if not USE_TRAINER:

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    model.to(device)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


if not USE_TRAINER:

    train_losses = []
    val_losses = []
    val_accuracies = []
    best_acc = 0.0
    
    for epoch in range(1, N_EPOCHS+1):
    
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * input_ids.size(0)
            
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
    
        model.eval()
        val_loss = 0.0
        correct = 0
        
        with torch.no_grad():
            
            for batch in val_loader:
                
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                val_loss += loss.item() * input_ids.size(0)
                preds = outputs.logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                
        val_loss /= len(val_loader.dataset)
        val_acc = correct / len(val_loader.dataset)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best-parameters.pt")
               
        print(f"Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")

    model.load_state_dict(torch.load("best-parameters.pt"))


if not USE_TRAINER:
    
    model.eval()
    logits_list = []
    
    with torch.no_grad():
        
        for batch in test_loader:
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:,1].cpu().numpy()
            logits_list.append(logits)
    
    logits = np.concatenate(logits_list)


df_texts_test["proba_real"] = 1/(1 + np.exp(-logits))

df_texts_test


df_texts_results = df_texts_test.loc[df_texts_test.groupby("id")["proba_real"].idxmax()]
df_texts_results = df_texts_results.rename({"text_id": "real_text_id"}, axis=1).reset_index(drop=True)

df_texts_results


df_submission = df_texts_results[["id", "real_text_id"]]
df_submission.to_csv("submission.csv", index=False)




