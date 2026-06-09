from transformers import pipeline
unmasker = pipeline('fill-mask', model='bert-base-uncased')
unmasker("Hello I'm a [MASK] model.")


from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
encoded_input


output = model(**encoded_input)
print(output['last_hidden_state'][0][0].shape)
print(output['last_hidden_state'][0][0])


import torch
import gc

def clear_gpu_memory(device_index=0):
    """
    Очищает память GPU от тензоров и освобождает кэш
    
    Args:
        device_index (int): индекс GPU устройства
    """
    # Очистка кэша CUDA
    torch.cuda.empty_cache()
    
    # Принудительный сбор мусора
    gc.collect()
    
    # Очистка кэша текущего устройства
    with torch.cuda.device(device_index):
        torch.cuda.empty_cache()
        
    print(f"Память GPU {device_index} очищена")
clear_gpu_memory()


TEST_MODE = True


import pandas as pd
import os

data_path = "/kaggle/input/jigsaw-toxic-comment-classification-challenge"
if TEST_MODE:
    train = pd.read_csv(os.path.join(data_path, "train.csv.zip"))[:100]
else:
    train = pd.read_csv(os.path.join(data_path, "train.csv.zip"))[:2000]
test = pd.read_csv(os.path.join(data_path, "test.csv.zip"))
test_labels = pd.read_csv(os.path.join(data_path, "test_labels.csv.zip"))
sample_submission = pd.read_csv(os.path.join(data_path, "sample_submission.csv.zip"))

train.head()


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Bert_classification(nn.Module):
    def __init__(self, num_labels=6):  # 6 классов
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.bert = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=num_labels,
            problem_type="multi_label_classification"
        ).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def forward(self, input):
        encoding = self.tokenizer.batch_encode_plus(
            input,
            padding="max_length",
            max_length=256,
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        out = self.bert(input_ids, attention_mask=attention_mask)

        return out.logits  # (batch, 6)


class dataset_bilder(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], torch.tensor(self.y[idx], dtype=torch.float)


class model_usage():
    def __init__(self, model):
        self.model = model

    def train(self, dataloader, optimizer, loss_func, scheduler, epochs, val):
        self.model.bert.train()
        for epoch in range(epochs):
            for texts, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}", colour="GREEN"):
                labels = labels.to(device)
                optimizer.zero_grad()
                output = self.model(texts)  # (batch, 6)
                loss = loss_func(output, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()

            self.test(val, loss_func)

    def test(self, dataloader, loss_func):
        self.model.eval()
        testloss, all_labels, all_preds = 0, [], []
        num_batches = len(dataloader)

        with torch.no_grad():
            for texts, labels in tqdm(dataloader, desc="Eval", colour="CYAN"):
                labels = labels.to(device)
                output = self.model(texts)  # (batch, 6)
                loss = loss_func(output, labels)
                testloss += loss.item()

                probs = torch.sigmoid(output).cpu().numpy()
                all_preds.append(probs)
                all_labels.append(labels.cpu().numpy())

        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)

        # считаем mean column-wise ROC AUC
        aucs = []
        for i in range(all_labels.shape[1]):
            try:
                auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
                aucs.append(auc)
            except ValueError:  # если в колонке только один класс
                continue

        mean_auc = np.mean(aucs)
        testloss /= num_batches

        print(f"Eval: Mean ROC-AUC = {mean_auc:.4f}, Avg loss = {testloss:.4f}")
        return mean_auc

    def create_result(self, pd_test):
        texts = pd_test['comment_text'].values.tolist()
        model_output = []

        for text in tqdm(texts, desc="Creating submission", colour="CYAN"):
            logits = self.model([text])
            probs = torch.sigmoid(logits).cpu().detach().numpy().tolist()[0]
            model_output.append(probs)

        preds_array = np.array(model_output)

        submission = pd.DataFrame({
            'id': pd_test['id']
        })
        submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = preds_array

        return submission



from sklearn.model_selection import train_test_split

lables = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

X_train, X_val, y_train, y_val = train_test_split(train['comment_text'].values.tolist(), train[lables].values.tolist(), test_size=0.2)

train_dataset = dataset_bilder(X_train, y_train)
val_dataset = dataset_bilder(X_val, y_val)

batch_train_size = 76
batch_val_size = 76

train_dl = DataLoader(
        train_dataset,
        batch_size=batch_train_size,
        num_workers=4,
        shuffle=True,
        collate_fn=None,
    )

val_dl = DataLoader(
        val_dataset,
        batch_size=batch_val_size,
        num_workers=4,
        shuffle=False,
        collate_fn=None,
    )

# Инициализируем 
model = Bert_classification()

epochs = 3
LR = 1e-6
optimizer = optim.Adam(model.bert.parameters(), lr = LR) # почему нельзя SGD? это относится ко всем NLP моделям!
loss_func = nn.BCEWithLogitsLoss()
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=120, num_training_steps = (len(X_train)//batch_train_size + 1))  # 226 - число батчей в датасете


# Обучаем
model_f = model_usage(model)
model_f.train(train_dl, optimizer, loss_func, scheduler, epochs, val_dl)        


# # Сохраняем результат на тест дате
# submission = model_f.create_result(test)
# submission.to_csv('submission.csv', index=False)


TEST_MODE = True


import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import os
os.environ["WANDB_DISABLED"] = "true" # вандб не работает больше

# 1. Загружаем данные
data_path = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/"
if TEST_MODE:
    train_df = pd.read_csv(os.path.join(data_path, "train.csv.zip"))[:100]
else:
    train_df = pd.read_csv(os.path.join(data_path, "train.csv.zip"))[:2000]

test_df = pd.read_csv(data_path + "test.csv.zip")
test_df = test_df.rename(columns = {'comment_text': 'text'})

label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
train_df['labels'] = train_df[label_cols].values.astype(np.float32).tolist()
train_df = train_df.rename(columns = {'comment_text': 'text'})
train_df = train_df[['text', 'labels']]
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)


# 2. Dataset
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
val_dataset = Dataset.from_pandas(val_df)

# 3. Модель и токенайзер
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(label_cols),
    problem_type="multi_label_classification"
)

# 4. DataCollator (сам делает паддинг + токенизацию на лету)
def tokenize_function(batch):
    return tokenizer(batch["text"], truncation=True, max_length=256)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 5. Метрика ROC-AUC
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    labels = labels.astype(int)

    aucs = []
    for i in range(labels.shape[1]):
        try:
            auc = roc_auc_score(labels[:, i], probs[:, i])
            aucs.append(auc)
        except ValueError:
            continue
    return {"mean_auc": np.mean(aucs)}

# 6. TrainingArguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=2,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="mean_auc",
    logging_strategy="steps",  
    logging_steps=10,         
    logging_first_step=True,  
)

# 7. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# 8. Обучение
trainer.train()

# # 9. Сабмит
# preds = trainer.predict(test_dataset.map(tokenize_function, batched=True))
# probs = torch.sigmoid(torch.tensor(preds.predictions)).numpy()

# submission = pd.DataFrame({
#     "id": test_df["id"]
# })
# submission[label_cols] = probs
# submission.to_csv("submission.csv", index=False)





