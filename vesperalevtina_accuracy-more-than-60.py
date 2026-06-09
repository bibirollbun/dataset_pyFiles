from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
import os
import pandas as pd
torch.cuda.empty_cache()
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Used device: {device}")

import warnings
warnings.filterwarnings("ignore", category=Warning)












# class CustomClassifier(nn.Module):
#     def __init__(self, model_name, num_labels):
#         super().__init__()
#         self.base_model = AutoModel.from_pretrained(model_name)
#         self.dropout = nn.Dropout(0.2)
#         self.fc = nn.Linear(self.base_model.config.hidden_size, num_labels)

#     def forward(self, input_ids, attention_mask=None):
#         outputs = self.base_model(input_ids, attention_mask=attention_mask)
#         cls_output = outputs.last_hidden_state[:, 0, :]
#         pooled_output = self.dropout(cls_output)
#         logits = self.fc(pooled_output)
#         return logits






class CustomClassifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(model_name)
        hidden_size = self.base_model.config.hidden_size
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, num_labels)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(cls_output)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


model_name = "cointegrated/rubert-tiny2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = CustomClassifier(model_name=model_name, num_labels=6)




DATA_PATH = "/kaggle/input/writerextnd/Bradbury.txt"

author_mapping = {
    "Genri": 0,
    "Simak": 1,
    "Bulgakov": 2,
    "Bradbury": 3,
    "Fry": 4,
    "Strugatskie": 5
}



def read_texts(files, path, mapping=None, split_train=True, split_by='\n\n'):
    texts, labels = [], []
    for file in files:
        with open(os.path.join(path, file), 'r', encoding='utf-8') as f:
            content = f.read()
            author = file.replace('.txt', '')
            if mapping and split_train:
                # Split the content into chunks based on the delimiter
                chunks = [chunk.strip() for chunk in content.split(split_by) if chunk.strip()]
                print(f"File: {file}, Author: {author}, Label: {mapping[author]}, Number of parts: {len(chunks)}")
                for chunk in chunks:
                    texts.append(chunk)
                    labels.append(mapping[author])
            else:
                texts.append(content)
                if mapping:
                    labels.append(mapping[author])
    if mapping:
        print(f"Total number of parts: {len(texts)}")
    return texts, labels if mapping else None

train_files = ['Bradbury.txt', 'Bulgakov.txt', 'Fry.txt', 'Genri.txt', 'Simak.txt', 'Strugatskie.txt']
test_files = [f"author{i}.txt" for i in range(1, 22)]

train_texts, train_labels = read_texts(train_files, DATA_PATH, author_mapping, split_train=True)
test_texts, _ = read_texts(test_files, DATA_PATH, split_train=False)






# aug = naw.RandomWordAug(action="swap", aug_p=0.2)
# augmented_texts = []
# augmented_labels = []
# for text, label in zip(train_texts, train_labels):
#     augmented_texts.append(text)
#     augmented_labels.append(label)
#     aug_text = aug.augment(text)[0]
#     augmented_texts.append(aug_text)
#     augmented_labels.append(label)
# train_texts, train_labels = augmented_texts, augmented_labels





class AuthorDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        data = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }
        if self.labels is not None:
            data['labels'] = torch.tensor(self.labels[item], dtype=torch.long)
        return data

train_dataset = AuthorDataset(train_texts, train_labels, tokenizer, max_len=256)
test_dataset = AuthorDataset(test_texts, labels=None, tokenizer=tokenizer, max_len=256)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)







LEARNING_RATE = 2e-5
EPOCHS = 10

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=2, num_training_steps=EPOCHS * len(train_loader))
loss_fn = nn.CrossEntropyLoss()






def train_model(model, train_loader, optimizer, loss_fn, scheduler, epochs=EPOCHS, device='cuda'):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss, total_accuracy = 0, 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            total_accuracy += torch.sum(preds == labels).item()

        avg_loss = total_loss / len(train_loader)
        avg_accuracy = total_accuracy / len(train_loader.dataset)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")








train_model(model, train_loader, optimizer, loss_fn, scheduler, epochs=EPOCHS, device='cuda')





def predict(model, test_loader, device='cuda'):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().numpy())
    return predictions

test_predictions = predict(model, test_loader)
print("Test predictions:", test_predictions)





from collections import Counter
print(Counter(test_predictions))








data = {
    'id': [f'author{i+1}' for i in range(21)],
    'label': [
        0,  # author1: Genri (О. Генри)
        1,  # author2: Simak (Клиффорд Саймак)
        0,  # author3: Genri (О. Генри)
        0,  # author4: Genri (О. Генри)
        2,  # author5: Bulgakov (Михаил Булгаков)
        3,  # author6: Bradbury (Рэй Брэдбери)
        4,  # author7: Fry (Макс Фрай)
        4,  # author8: Fry (Макс Фрай)
        5,  # author9: Strugatskie (Братья Стругацкие)
        3,  # author10: Bradbury (Рэй Брэдбери)
        2,  # author11: Bulgakov (Михаил Булгаков)
        3,  # author12: Bradbury (Рэй Брэдбери)
        0,  # author13: Genri (О. Генри)
        2,  # author14: Bulgakov (Михаил Булгаков)
        1,  # author15: Simak (Клиффорд Саймак)
        1,  # author16: Simak (Клиффорд Саймак)
        4,  # author17: Fry (Макс Фрай)
        5,  # author18: Strugatskie (Братья Стругацкие)
        5,  # author19: Strugatskie (Братья Стругацкие)
        2,  # author20: Bulgakov (Михаил Булгаков)
        1   # author21: Simak (Клиффорд Саймак)
    ]
}
mapping = {
    "Genri": 0,
    "Simak": 1,
    "Bulgakov": 2,
    "Bradbury": 3,
    "Fry": 4,
    "Strugatskie": 5
}
submission_check = pd.DataFrame(data)

print(submission_check)


print(submission_check['label'].value_counts())








for i, pred in enumerate(test_predictions):
    print(f"Test text {test_files[i]} → Predicted author: {list(author_mapping.keys())[pred]}")

submission_df = pd.DataFrame({
    'id': [f'author{i+1}' for i in range(len(test_predictions))],
    'label': test_predictions
})
submission_df.to_csv('submission.csv', index=False)
print("File submission.csv is saved.")






correct = 0
total = 21
for i in range(total):
    if submission_df.label[i] == submission_check.label[i]:
        correct += 1

accuracy = correct / total
print("Accuracy:", accuracy)
    





submission_df.to_csv('submission.csv', index=False)

