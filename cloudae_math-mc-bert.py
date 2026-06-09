import pandas as pd


train_df = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')


train_df['Category'].value_counts()


train_df['Misconception'].value_counts()


NUM_CLASSES = len(train_df['Misconception'].unique())


train_df.sample(20)


text = []
for question,answer,explanation in zip(train_df['QuestionText'],train_df['MC_Answer'],train_df['StudentExplanation']):
    text.append(f'Question: {question}\nAnswer: {answer}\nExplanation: {explanation}')


train_df['text'] = text


text_train_df = train_df.drop(['QuestionText','MC_Answer','StudentExplanation'],axis=1)


cg_train_df = text_train_df.drop(['Misconception'],axis=1)


cg_train_df


import transformers
from transformers import AutoTokenizer,AutoModelForSequenceClassification


model_path = 'bert-base-uncased'
misconception_model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=NUM_CLASSES)
category_model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=6)
tokenizer = AutoTokenizer.from_pretrained(model_path)


cg_tokenized = tokenizer(
    cg_train_df["text"].tolist(),
    padding=True,
    truncation=True,
    return_tensors="pt",
)


from sklearn.preprocessing import LabelEncoder
cg_le = LabelEncoder()
le = LabelEncoder()


cg_train_df['Category'] = cg_le.fit_transform(cg_train_df['Category'])


labels = cg_train_df['Category'].tolist()


from sklearn.model_selection import train_test_split
train_idx, val_idx = train_test_split(
    range(len(labels)),
    test_size=0.2,
    stratify=labels,
    random_state=42
)


from torch.utils.data import Subset

from torch.utils.data import Dataset
import torch


class textDataset(Dataset):
    def __init__(self,encodings,labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self,idx):
        item = {key: val[idx] for key,val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


cg_train_dataset = textDataset(cg_tokenized,cg_train_df['Category'])


from torch.utils.data import random_split

cg_train_ds = Subset(cg_train_dataset,train_idx)
cg_val_ds = Subset(cg_train_dataset,val_idx)


from torch.utils.data import DataLoader

cg_train_dataloader = DataLoader(
    cg_train_ds, shuffle=True, batch_size=8
)
cg_eval_dataloader = DataLoader(
    cg_val_ds, batch_size=16
)


EPOCHS = 10
total_train_steps = len(cg_train_dataloader) * EPOCHS

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
optimizer = AdamW(category_model.parameters(),lr=5e-5)
scheduler = CosineAnnealingWarmRestarts(optimizer,int(0.1 * total_train_steps))

from tqdm.auto import tqdm


from torch.amp import GradScaler


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
category_model.to(device)
best_val_loss = float('inf')
PATIENCE = 3
scaler = GradScaler()
patience = PATIENCE

for epoch in range(EPOCHS):
    category_model.train()
    train_pbar = tqdm(cg_train_dataloader,desc='Training',leave=False)
    val_pbar = tqdm(cg_eval_dataloader,desc='Evaluating',leave=False)
    
    print(f"Epoch {epoch+1}/{EPOCHS}:")

    train_loss,train_total = 0.0,0.0
    for batch in train_pbar:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            outputs = category_model(**batch)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        optimizer.zero_grad(set_to_none=True)

        train_loss += loss * len(batch['labels'])
        train_total += len(batch['labels'])
        train_pbar.set_postfix(loss=train_loss / train_total)

    print(f'Training Loss: {train_loss/train_total}')

    val_loss,val_total,correct = 0.0,0.0,0.0
    category_model.eval()
    with torch.inference_mode():
        for batch in val_pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = category_model(**batch)
            loss = outputs.loss
    
            pred = torch.argmax(outputs.logits,dim=1)
            label = batch['labels']
    
            correct += (pred == label).sum().cpu().numpy()
            val_total += len(batch['labels'])

            val_loss += loss.cpu().numpy() * len(batch['labels'])

            val_pbar.set_postfix(loss = val_loss/val_total,accuracy = correct/val_total)

        accuracy = correct / val_total
        val_loss /= val_total

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(category_model,'./model_category.pt')
        patience = PATIENCE
    else:
        patience -= 1

    if patience == 0:
        category_model = torch.load('./model_category.pt',weights_only=False)
        print("Early stopping triggered")
        break

    print(f"Validation Accuracy: {accuracy}, Validation Loss: {val_loss}")

torch.save(category_model,'./model_category.pt')


misconception_train_df = text_train_df[text_train_df['Misconception'].notnull()].drop(['Category'],axis=1)


misconception_train_df


misconception_train_df.iloc[107]['text']


misconception_train_df['Misconception'] = le.fit_transform(misconception_train_df['Misconception'])


tokenized = tokenizer(
    misconception_train_df["text"].tolist(),
    padding=True,
    truncation=True,
    return_tensors="pt",
)


labels = torch.tensor(misconception_train_df['Misconception'].tolist())


text_ds = textDataset(tokenized,labels)


len(text_ds)


from torch.utils.data import random_split

train_size = int(0.8 * len(text_ds))
val_size = len(text_ds) - train_size

generator = torch.Generator().manual_seed(42)
train_ds, val_ds = random_split(text_ds, [train_size, val_size], generator=generator)


len(train_ds)


train_dataloader = DataLoader(
    train_ds, shuffle=True, batch_size=8
)
eval_dataloader = DataLoader(
    val_ds, batch_size=16
)


EPOCHS = 10
total_train_steps = len(train_dataloader) * EPOCHS

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
optimizer = AdamW(misconception_model.parameters(),lr=5e-5)
scheduler = CosineAnnealingWarmRestarts(optimizer,int(0.1 * total_train_steps))

from tqdm.auto import tqdm


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
misconception_model.to(device)
best_val_loss = float('inf')
mc_scaler = GradScaler()
PATIENCE = 3
patience = PATIENCE

for epoch in range(EPOCHS):
    misconception_model.train()
    train_pbar = tqdm(train_dataloader,desc='Training',leave=False)
    val_pbar = tqdm(eval_dataloader,desc='Evaluating',leave=False)
    
    print(f"Epoch {epoch+1}/{EPOCHS}:")

    train_loss,train_total = 0.0,0.0
    for batch in train_pbar:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            outputs = misconception_model(**batch)
            loss = outputs.loss

        mc_scaler.scale(loss).backward()
        mc_scaler.step(optimizer)
        mc_scaler.update()
        scheduler.step()

        optimizer.zero_grad(set_to_none=True)

        train_loss += loss * len(batch['labels'])
        train_total += len(batch['labels'])
        train_pbar.set_postfix(loss=train_loss / train_total)

    print(f'Training Loss: {train_loss/train_total}')

    val_loss,val_total,correct = 0.0,0.0,0.0
    misconception_model.eval()
    with torch.inference_mode():
        for batch in val_pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = misconception_model(**batch)
            loss = outputs.loss
    
            pred = torch.argmax(outputs.logits,dim=1)
            label = batch['labels']
    
            correct += (pred == label).sum().cpu().numpy()
            val_total += len(batch['labels'])

            val_loss += loss.cpu().numpy() * len(batch['labels'])

            val_pbar.set_postfix(loss = val_loss/val_total,accuracy = correct/val_total)

        accuracy = correct / val_total
        val_loss /= val_total

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(misconception_model,'./model_misconception.pt')

    print(f"Validation Accuracy: {accuracy}, Validation Loss: {val_loss}")


import joblib

joblib.dump(cg_le,'cg_label_encoder.pkl')
joblib.dump(le,'mc_label_encoder.pkl')

