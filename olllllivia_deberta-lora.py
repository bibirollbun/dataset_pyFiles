import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
from transformers import AutoTokenizer, DebertaV2Tokenizer, AutoModel, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
# from datasets import  Dataset 
from torch.utils.data import Dataset,DataLoader
from peft import LoraConfig, get_peft_model

from transformers import TrainingArguments, Trainer, EarlyStoppingCallback



torch.cuda.is_available()


class CFG:
    seed = 42
    preset = "/kaggle/input/deberta-v3-base/pytorch/default/1"
    # preset = "microsoft/deberta-v3-base"   # HF model name
    sequence_length = 512
    epochs = 3
    batch_size = 2
    num_labels = 3
    label2name = {0: 'winner_model_a', 1: 'winner_model_b', 2: 'winner_tie'}
    name2label = {v:k for k, v in label2name.items()}
    class_labels = list(label2name.keys())
    class_names = list(label2name.values())

torch.cuda.manual_seed_all(CFG.seed)
np.random.seed(CFG.seed)



BASE_PATH = '/kaggle/input/dataset'

# Load Train Data
df = pd.read_csv(f'{BASE_PATH}/train.csv') 

df["prompt"] = df.prompt.map(lambda x: eval(x.replace("\\/", "/").replace("null", "''"))[0])
df["response_a"] = df.response_a.map(lambda x: eval(x.replace("\\/", "/").replace("null", "''"))[0])
df["response_b"] = df.response_b.map(lambda x: eval(x.replace("\\/", "/").replace("null", "''"))[0])

df["class_name"] = df[["winner_model_a", "winner_model_b" , "winner_tie"]].idxmax(axis=1)
df["class_label"] = df.class_name.map(CFG.name2label)

# Load Test Data
test_df = pd.read_csv(f'{BASE_PATH}/test.csv')

test_df["prompt"] = test_df.prompt.map(lambda x: eval(x.replace("\\/", "/").replace("null", "''"))[0])
test_df["response_a"] = test_df.response_a.map(lambda x: eval(x.replace("\\/", "/").replace("null", "''"))[0])
test_df["response_b"] = test_df.response_b.map(lambda x: eval(x.replace("\\/", "/").replace("null", "''"))[0])

df.head()



# Define a function to create options based on the prompt and choices
def make_pairs(row):
    row["encode_fail"] = False
    try:
        prompt = row.prompt.encode("utf-8", errors='ignore').decode("utf-8")
    except:
        prompt = ""
        row["encode_fail"] = True

    try:
        response_a = row.response_a.encode("utf-8", errors='ignore').decode("utf-8")
    except:
        response_a = ""
        row["encode_fail"] = True

    try:
        response_b = row.response_b.encode("utf-8", errors='ignore').decode("utf-8")
    except:
        response_b = ""
        row["encode_fail"] = True
        
    row['options'] = [f"Prompt: {prompt}\n\nResponse: {response_a}",  # Response from Model A
                      f"Prompt: {prompt}\n\nResponse: {response_b}"  # Response from Model B
                     ]
    return row



df = df.apply(make_pairs, axis=1)
df_test = test_df.apply(make_pairs, axis=1)

# for i, opts in enumerate(df["options"]):
#     print(i, type(opts), [type(o) for o in opts])
#     if not all(isinstance(o, str) for o in opts):
#         print("Found non-string:", opts)

train_df, valid_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["class_label"],
    random_state=CFG.seed
)



df.head()



tokenizer = DebertaV2Tokenizer.from_pretrained(
    "/kaggle/input/deberta-v3-base/pytorch/default/1",
    # "microsoft/deberta-v3-base"
    model_max_length=CFG.sequence_length,
    padding_side="right",
    truncation_side="right",
    # trust_remote_code=True, 
    # use_fast=False 
)



class PairDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.texts = df["options"].tolist()
        
        if "class_label" in df.columns:
            self.labels = df["class_label"].tolist()
            self.has_labels = True
        else:
            self.labels = [0] * len(self.texts)  
            self.has_labels = False

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        pairs = self.texts[idx]
        # print(f"idx={idx}, type(opt)={type(pairs)}, opt={pairs}")
        label = self.labels[idx]

        # text = [str(t) for t in text]  # Ensure each option is a string
        a = str(pairs[0]) 
        b = str(pairs[1])   

        # if torch.rand(1).item() < 0.5:
        #     a, b = b, a
        #     if label == 0:     # A wins
        #         label = 1      # now B wins
        #     elif label == 1:   # B wins
        #         label = 0      # now A wins 

        encodings_a = self.tokenizer(
            a,
            max_length=CFG.sequence_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        ) 

        encodings_b = self.tokenizer(
            b,
            max_length=CFG.sequence_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # input_ids 和 attention_mask shape = (2, seq_len)
        # input_ids = torch.cat([e['input_ids'] for e in encodings], dim=0)
        # attention_mask = torch.cat([e['attention_mask'] for e in encodings], dim=0)
        # # print(input_ids.shape, attention_mask.shape)
        # 
        

        return {
            "input_ids_a": encodings_a["input_ids"].squeeze(0),  # shape: (seq_len)
            "attention_mask_a": encodings_a["attention_mask"].squeeze(0),
            "input_ids_b": encodings_b["input_ids"].squeeze(0),
            "attention_mask_b": encodings_b["attention_mask"].squeeze(0), 
            "labels": torch.tensor(label, dtype=torch.long)
            }




train_dataset = PairDataset(train_df, tokenizer)
valid_dataset = PairDataset(valid_df, tokenizer)

def collatet_fn(batch):
    
    # {
    #     "input_ids": tensor(2, seq_len), 
    #     "attention_mask": tensor(2, seq_len), 
    #     "labels": tensor(1)
    # }
    
    
    input_ids_a = torch.stack([b["input_ids_a"] for b in batch])
    attention_mask_a = torch.stack([b["attention_mask_a"] for b in batch])
    input_ids_b = torch.stack([b["input_ids_b"] for b in batch])
    attention_mask_b = torch.stack([b["attention_mask_b"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    
    return {
        "input_ids_a": input_ids_a,
        "attention_mask_a": attention_mask_a,
        "input_ids_b": input_ids_b,
        "attention_mask_b": attention_mask_b,
        "labels": labels
    }

train_loader = DataLoader(
    train_dataset,
    batch_size=CFG.batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    collate_fn=collatet_fn
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=CFG.batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)




for batch in train_loader:
    print(batch["input_ids_a"].shape)       # (batch_size, 2, seq_len)
    print(batch["input_ids_b"].shape)       # (batch_size, 2, seq_len)
    print(batch["attention_mask_a"].shape)  # (batch_size, 2, seq_len)
    print(batch["attention_mask_b"].shape)  # (batch_size, 2, seq_len)
    print(batch["labels"].shape)          # (batch_size,)
    break


# model_name = "microsoft/deberta-v3-xsmall"
# backbone = AutoModel.from_pretrained(model_name)

# for name, module in backbone.named_modules():
#     if "proj" in name.lower():   # 你也可以換成 "attention"
#         print(name)



class InteractionHead(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        in_dim = hidden * 4      # concat(A, B, diff, prod)

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden * 2),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.out = nn.Linear(hidden // 2, 3)

    def forward(self, x):
        x = self.mlp(x)
        return self.out(x)



class DebertaLORA(nn.Module):
    def __init__(self, model_name=CFG.preset, num_labels=CFG.num_labels):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            model_name,
            num_labels=num_labels
        )

        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["query_proj", "value_proj","key_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="FEATURE_EXTRACTION"
        )
        self.model = get_peft_model(self.backbone, lora_config)

        # self.classifier = nn.Sequential(
        #     nn.Linear(self.model.config.hidden_size*2, 256),
        #     nn.ReLU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(256, num_labels)
        #     )
        self.classifier = InteractionHead(self.model.config.hidden_size) 
        
    def forward(self,input_ids_a, attention_mask_a, input_ids_b, attention_mask_b):
        

        out_a = self.model(input_ids=input_ids_a, attention_mask=attention_mask_a).last_hidden_state
        out_b = self.model(input_ids=input_ids_b, attention_mask=attention_mask_b).last_hidden_state


        emb_a = out_a.mean(dim=1)
        emb_b = out_b.mean(dim=1)

        diff = emb_a - emb_b
        prod = emb_a * emb_b

        concat_emb = torch.cat([emb_a, emb_b, diff, prod], dim=1)
        logits = self.classifier(concat_emb)
        
        return logits


'''
device = "cuda" if torch.cuda.is_available() else "cpu"

model = DebertaLORA().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
num_training_steps = len(train_loader) * CFG.epochs
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*num_training_steps), num_training_steps=num_training_steps)
criterion = nn.CrossEntropyLoss(label_smoothing=0.02)

best_val_loss = float("inf")

for epoch in range(CFG.epochs):
    model.train()
    train_loss = 0.0

    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        
        input_ids_a = batch["input_ids_a"].to(device)
        input_ids_b = batch["input_ids_b"].to(device)
        attention_mask_a = batch["attention_mask_a"].to(device)
        attention_mask_b = batch["attention_mask_b"].to(device)
        labels = batch["labels"].to(device)

        # print(input_ids.shape, attention_mask.shape, labels.shape)
        optimizer.zero_grad()

        logits = model(input_ids_a, input_ids_b, attention_mask_a, attention_mask_b)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss += loss.item() #* input_ids.size(0)

    avg_train_loss = train_loss / len(train_loader.dataset)
    # print(f"Epoch {epoch+1}, Training Loss: {avg_train_loss:.4f}")

    model.eval()
    valid_loss = 0.0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for batch in tqdm(valid_loader, desc=f"Validation Epoch {epoch+1}"):
            input_ids_a = batch["input_ids_a"].to(device)
            input_ids_b = batch["input_ids_b"].to(device)
            attention_mask_a = batch["attention_mask_a"].to(device)
            attention_mask_b = batch["attention_mask_b"].to(device)
            labels = batch["labels"].to(device)


            logits = model(input_ids_a, input_ids_b, attention_mask_a, attention_mask_b)
            loss = criterion(logits, labels)

            valid_loss += loss.item() #* input_ids.size(0)
            all_labels.extend(labels.cpu().numpy())
            
            all_preds.extend(torch.softmax(logits, dim=1).cpu().numpy())

    avg_valid_loss = valid_loss / len(valid_loader.dataset)
    valid_logloss = log_loss(all_labels, all_preds)

    valid_accuracy = accuracy_score(all_labels, np.argmax(all_preds, axis=1))
    print(f"Epoch {epoch+1}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_valid_loss:.4f}, Log Loss: {valid_logloss:.4f}, Accuracy: {valid_accuracy:.4f}")


    # if avg_valid_loss < best_val_loss:
    #     best_val_loss = avg_valid_loss
    #     torch.save(model.state_dict(), f"{BASE_PATH}/deberta_lora_model_epoch{epoch+1}.pth")
    #     print("Saved best model.")
'''      


test_df.head()


df_test.head()


device = "cuda" if torch.cuda.is_available() else "cpu"

# Build Test Dataset
test_dataset = PairDataset(df_test, tokenizer)
# Test DataLoader
test_loader = DataLoader(
    test_dataset,
    batch_size=CFG.batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    collate_fn=collatet_fn
)
model = DebertaLORA().to(device)
# model.load_state_dict(torch.load("/kaggle/input/deberta-lora-model/pytorch/default/1/deberta_lora_model_epoch2.pth", map_location="cpu"))
model.to(device)

model.eval()
all_probs = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Test Inference"):
        input_ids_a = batch["input_ids_a"].to(device)
        input_ids_b = batch["input_ids_b"].to(device)
        attention_mask_a = batch["attention_mask_a"].to(device)
        attention_mask_b = batch["attention_mask_b"].to(device)

        logits = model(input_ids_a, input_ids_b, attention_mask_a, attention_mask_b)
        probs = torch.softmax(logits, dim=-1)  # Convert logits to probabilities
        all_probs.append(probs.cpu().numpy())

# Concatenate all batches
# all_probs = torch.cat(all_probs, dim=0).numpy()
final_probs = np.concatenate(all_probs, axis=0)

assert len(final_probs) == len(test_df), f"長度不符！模型預測了 {len(final_probs)} 筆，但測試集有 {len(test_df)} 筆"
# # Create submission dataframe
# sub_df = test_df[["id"]].copy()
# class_names = ['winner_model_a','winner_model_b','winner_tie']
# sub_df[class_names] = all_probs
# sub_df.to_csv("submission.csv", index=False)
# sub_df.head()

submission = pd.DataFrame({
    'id': test_df['id'], # 確保使用原始 test_df 的 ID
    'winner_model_a': final_probs[:, 0],
    'winner_model_b': final_probs[:, 1],
    'winner_tie': final_probs[:, 2]
})

# submission = submission.fillna(0.333333)

submission.to_csv('submission.csv', index=False)

sub = pd.read_csv('submission.csv')
print(sub.head())
print(sub.shape)

