!pip install transformers


import pandas as pd


data=pd.read_csv('/kaggle/input/ai-mathematical-olympiad-progress-prize-2/reference.csv')
print(len(data))
display(data[0:2].T)


for i in range(len(data)):
    print(data.iloc[i,1])


TEST=pd.read_csv('/kaggle/input/ai-mathematical-olympiad-progress-prize-2/test.csv')
submit=pd.read_csv('/kaggle/input/ai-mathematical-olympiad-progress-prize-2/sample_submission.csv')


for i in range(len(TEST)):
    print(TEST.iloc[i,1])


inputt="Answer the number of 0 - 999: "+data['problem']
labelt=data['answer'].astype(str)


TEST_inputs = "Answer the number of 0 - 999: "+TEST['problem']
TEST_labels = submit['answer'].astype(str) #dummy


from sklearn.model_selection import train_test_split

train_inputs, test_inputs, train_labels, test_labels = train_test_split(
    inputt, labelt, test_size=0.2, random_state=42
)


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")


def tokenize_data(inputs, labels, tokenizer, max_length=128):
    input_encodings = tokenizer(
        list(inputs), max_length=max_length, padding=True, truncation=True, return_tensors="pt"
    )
    label_encodings = tokenizer(
        list(labels), max_length=max_length, padding=True, truncation=True, return_tensors="pt"
    )
    return input_encodings, label_encodings

train_inputs_enc, train_labels_enc = tokenize_data(train_inputs, train_labels, tokenizer)
#test_inputs_enc, test_labels_enc = tokenize_data(test_inputs, test_labels, tokenizer)
TEST_inputs_enc, TEST_labels_enc = tokenize_data(TEST_inputs, TEST_labels, tokenizer)


import torch
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels["input_ids"][idx],
        }

train_dataset = CustomDataset(train_inputs_enc, train_labels_enc)
#test_dataset = CustomDataset(test_inputs_enc, test_labels_enc)
TEST_dataset = CustomDataset(TEST_inputs_enc, TEST_labels_enc)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
#test_loader = DataLoader(test_dataset, batch_size=8)
TEST_loader = DataLoader(TEST_dataset, batch_size=8)


from transformers import AdamW

optimizer = AdamW(model.parameters(), lr=5e-6)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

epochs = 30
for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1} Loss: {loss.item()}")



model.eval()
preds=[]
for batch in TEST_loader:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device) 

    input_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
    true_labels = [tokenizer.decode(label, skip_special_tokens=True) for label in labels]

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=50
    )
    predictions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

    for input_text, true_label, pred in zip(input_texts, true_labels, predictions):
        print("-" * 50)
        print(f"input_txt: {input_text}")
        print(f"true_label: {true_label}")
        print(f"pred_label: {pred}")

    preds+=predictions


submit['answer']=preds
submit.to_csv('submission.csv',index=False)
display(submit)




