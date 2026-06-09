!pip install peft


!pip install -U bitsandbytes


import os

path = '/kaggle/input/fake-or-real-the-impostor-hunt/data/'
articles = sorted(os.listdir(f'{path}train'))

file_1, file_2 = [], []
filenames = []
indexes = []

for article in articles:
    file_names=os.listdir(os.path.join(f'{path}train/'+article))
    for file in file_names:
        if file == 'file_1.txt':
            with open(os.path.join(f'{path}train/'+article+'/'+file), 'r') as f:

                content=f.read()

            file_1.append(content)

        elif file == 'file_2.txt':

            with open(os.path.join(f'{path}train/'+article+'/'+file), 'r') as f:
                content=f.read()

            file_2.append(content)


    filenames.append(article)
    indexes.append(article[-2:])



import pandas as pd

df_labels = pd.read_csv('/kaggle/input/fake-or-real-the-impostor-hunt/data/train.csv')

labels = df_labels['real_text_id'].to_list()



df = pd.DataFrame()

df['articles'], df['file_1'], df['file_2'], df['labels'] = filenames, file_1, file_2, labels


df.head()


df_tmp1 = df.copy().drop(columns=['file_2']).rename(columns={'file_1': 'text'})
df_tmp2 = df.copy().drop(columns=['file_1']).rename(columns={'file_2': 'text'})



df_tmp1['labels'] =  df_tmp1['labels'].apply(lambda x: 1 if x==1 else 0)
df_tmp2['labels'] =  df_tmp2['labels'].apply(lambda x: 1 if x==2 else 0)

df_data = pd.concat([df_tmp1, df_tmp2], ignore_index=True)



df_data.head(100)


from transformers import BertTokenizer, BertModel, BitsAndBytesConfig, BertForSequenceClassification

import torch

torch.cuda.empty_cache()  # clears cached but unallocated memory

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load pretrained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
base_model = BertModel.from_pretrained("bert-base-uncased",
                                       quantization_config = BitsAndBytesConfig(load_in_4bit=True))

# base_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)




print(device)


found = False
try:
  try:
    for name, _ in base_model.named_modules():
        if "q_proj" in name or "v_proj" in name:
            found=True

            print("q_proj in name or v_proj in name")
            break

    if not found:
      raise Exception

  except:
    for name, _ in base_model.named_modules():
        if "query" in name or "key" in name:
            found=True

            print("query in name or query in name")
            break
    if not found:
      raise Exception

except Exception as e:
  print('nothing found')







from peft import get_peft_model, LoraConfig, TaskType

# Create a LoRA config
lora_config = LoraConfig(
    r=8,                     # rank
    lora_alpha=32,           # scaling
    target_modules=["query","value"],  # which layers to inject LoRA into
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.FEATURE_EXTRACTION,     # since we want embeddings, not classification
)

# Wrap BERT with LoRA
lora_model = get_peft_model(base_model, lora_config)
lora_model.print_trainable_parameters()




inputs = tokenizer(df_data['text'].to_list(), padding=True,
                    truncation = True, add_special_tokens = True, return_tensors="pt")


# inputs = {k: v.half() for k, v in inputs.items()}

print(inputs['input_ids'].shape, device)


inputs, labels = inputs.to(device), torch.tensor(labels)

# # print(inputs['input_ids'][0], labels[0])
# lora_model = lora_model.to(device)


inputs['input_ids'].shape


lora_model.dtype, inputs['input_ids'][0].dtype  # mixed precisions


# CLS token Strategy

from torch import nn

class Model(nn.Module):

    def __init__(self, lora_model, hidden_dim, new_hidden_dim, num_class):
        super().__init__()
        self.dense = nn.Linear(hidden_dim, new_hidden_dim)
        self.classifier = nn.Linear(new_hidden_dim, num_class)
        self.bert_lora_model = lora_model


    def forward(self, input_ids, attention_mask):
        output_states= self.bert_lora_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = output_states.last_hidden_state[:, 0, :]
        pooled_output = self.dense(cls_embedding.float())
        logits=self.classifier(pooled_output)

        return logits



hidden_dim, new_hidden_dim = 768, 256
main_model = Model(lora_model, hidden_dim, new_hidden_dim, 2)


trainable_params = sum(p.numel() for p in main_model.parameters() if p.requires_grad)
trainable_params


from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split


class MyDataset(Dataset):
    def __init__(self, inputs, labels):
        self.enc = inputs

        self.labels = labels

    def __len__(self):
        return len(self.enc["input_ids"])

    def __getitem__(self, idx):

        return {
            "input_ids": self.enc["input_ids"][idx],
            "attention_mask": self.enc["attention_mask"][idx],
            "labels": self.labels[idx]
        }


dataset = MyDataset(inputs, torch.tensor(df_data['labels'].to_list()))

# let's say 80% train, 20% validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)


import torch.optim as optim
import torch.nn as nn

from torch.cuda.amp import autocast


optimizer = optim.AdamW(main_model.parameters(), lr=1e-5)
# criterion = nn.CrossEntropyLoss() # Remove this line
Epochs = 5
patience = 2  # how many epochs to wait for improvement
best_val_loss = float('inf')
epochs_no_improve = 0

main_model.to(device)


for epoch in range(Epochs):
    total_loss = 0
    main_model.train()
    for batch in train_loader:
        batch_input_ids = batch['input_ids'].to(device)
        batch_attention_mask = batch['attention_mask'].to(device)
        batch_labels = batch['labels'].to(device).long()

        optimizer.zero_grad()

        with autocast():  # mixed precision
            # Pass labels to the model to calculate loss internally
            outputs = main_model(input_ids=batch_input_ids,
                                attention_mask=batch_attention_mask)
            loss = nn.CrossEntropyLoss()(outputs, batch_labels)



        #backprop
        loss.backward()   # THIS CALCULATES THE GRADIETS OF LOSS W.R.T TO ALL TRAINABLE PARAMS
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {avg_train_loss:.4f}")

    # ----- Validation -----
    main_model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:  # <-- create validation dataloader
            batch_input_ids = batch['input_ids'].to(device)
            batch_attention_mask = batch['attention_mask'].to(device)
            batch_labels = batch['labels'].to(device).long()

            # Pass labels to the model to calculate loss internally
            outputs = main_model(input_ids=batch_input_ids,
                              attention_mask=batch_attention_mask)

            loss = nn.CrossEntropyLoss()(outputs, batch_labels)
            val_loss += loss.item()

            # predictions
            # Access logits and get predictions for accuracy calculation
            logits = outputs
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == batch_labels).sum().item()
            total += batch_labels.size(0)

    avg_val_loss = val_loss / len(val_loader)
    accuracy = correct / total

    print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | Val Acc: {accuracy:.4f}")


    # ----- Early Stopping -----
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(main_model.state_dict(), "best_model.pt")  # save best model
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            main_model.load_state_dict(torch.load("best_model.pt"))  # restore best
            break

























