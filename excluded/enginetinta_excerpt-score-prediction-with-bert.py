import pandas as pd
import torch
from torch import optim, nn
from torch.utils.data import Dataset, random_split, DataLoader
import transformers
from transformers import BertTokenizerFast, BertForSequenceClassification
import timeit
from tqdm import tqdm


transformers.utils.logging.set_verbosity_error()
device="cuda" if torch.cuda.is_available() else "cpu"

MAX_LENGTH= 256
MODEL_PATH="./../input/transformers/bert-base-uncased"
BATCH_SIZE=32
LEARNING_RATE=2e-5
EPOCHS=2


train_data=pd.read_csv("/kaggle/input/commonlitreadabilityprize/train.csv")
train_data.head()


from sklearn.preprocessing import StandardScaler
import pandas as pd

scaler = StandardScaler()
# Fit and transform the 'target' column (if you only want to standardize this column)
train_data['target_scaled'] = scaler.fit_transform(train_data[['target']])

# Now to get back the "real" scores (inverse transform)
X_real = scaler.inverse_transform(train_data[['target_scaled']])

# You can store the real values back into the DataFrame if needed
train_data['target_real'] = X_real

# Display the DataFrame with the real values
print(train_data)



test_data=pd.read_csv("/kaggle/input/commonlitreadabilityprize/test.csv")
test_data.head()


submission_data=pd.read_csv("/kaggle/input/commonlitreadabilityprize/sample_submission.csv")
submission_data.head()


tokenizer=BertTokenizerFast.from_pretrained(MODEL_PATH)
model=BertForSequenceClassification.from_pretrained(MODEL_PATH,num_labels=1).to(device)


class ComplexityDataset(Dataset):
    def __init__(self,sentences,targets,tokenizer):
        self.encodings= tokenizer(sentences, padding=True, truncation=True, max_length=MAX_LENGTH) 
        self.targets= targets

    def __getitem__(self,idx):
        out_dic= {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        out_dic["targets"] = torch.tensor(self.targets[idx], dtype=torch.float)
        return out_dic

    def __len__(self):
        return len(self.targets)


class ComplexitySubmitDataset(Dataset):
    def __init__(self, sentences, tokenizer, ids):
        self.encodings= tokenizer(sentences, padding=True, truncation=True, max_length=MAX_LENGTH) 
        self.ids= ids

    def __getitem__(self,idx):
        out_dic= { key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        out_dic["ids"]=self.ids[idx]
        return out_dic

    def __len__(self):
        return len(self.ids)


dataset=ComplexityDataset(train_data["excerpt"].to_list(),train_data["target"].to_list(),tokenizer)


print(len(dataset))


print(dataset[0])


test_dataset=ComplexitySubmitDataset(test_data["excerpt"].to_list(), tokenizer, test_data["id"].to_list())


print(len(test_dataset))


print(test_dataset[0])


generator= torch.Generator().manual_seed(42)
train_dataset, val_dataset = random_split(dataset, [0.9, 0.1], generator=generator)


train_dataloader=DataLoader(dataset=train_dataset,
                           batch_size=BATCH_SIZE,
                           shuffle=True)
val_dataloader=DataLoader(dataset=val_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=True)
test_dataloader=DataLoader(dataset=test_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=False)


optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

start = timeit.default_timer() 
for epoch in tqdm(range(EPOCHS), position=0, leave=True):
    model.train()
    train_running_loss = 0 
    for idx, sample in enumerate(tqdm(train_dataloader, position=0, leave=True)):
        input_ids = sample['input_ids'].to(device)
        attention_mask = sample['attention_mask'].to(device)
        targets = sample["targets"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=targets)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_running_loss += loss.item()
    train_loss = train_running_loss / (idx + 1)

    model.eval()
    val_running_loss = 0 
    with torch.no_grad():
        for idx, sample in enumerate(tqdm(val_dataloader, position=0, leave=True)):
            input_ids = sample['input_ids'].to(device)
            attention_mask = sample['attention_mask'].to(device)
            targets = sample["targets"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=targets)

            val_running_loss += outputs.loss.item()
        val_loss = val_running_loss / (idx + 1)

    print("-"*30)
    print(f"Train Loss EPOCH {epoch+1}: {train_loss:.4f}")
    print(f"Valid Loss EPOCH {epoch+1}: {val_loss:.4f}")
    print("-"*30)
stop = timeit.default_timer()
print(f"Training Time: {stop-start:.2f}s")


torch.cuda.empty_cache()


preds = []
ids = []
model.eval()
with torch.no_grad():
    for idx, sample in enumerate(tqdm(test_dataloader, position=0, leave=True)):
        input_ids = sample['input_ids'].to(device)
        attention_mask = sample['attention_mask'].to(device)
        ids.extend(sample["ids"])
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds.extend([float(i) for i in outputs["logits"].squeeze()])


submission_df = pd.DataFrame(list(zip(ids, preds)),
               columns =['id', 'target'])
submission_df.to_csv("submission.csv", index=False)

