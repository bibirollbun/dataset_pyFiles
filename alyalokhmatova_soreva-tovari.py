import pandas as pd
import torch.nn
from matplotlib import pyplot as plt


df = pd.read_csv('../input/shopee-product-matching/train.csv',
                 usecols=['posting_id', 'image', 'title', "label_group"])
df.sample(2)


df["posting_id"] = df["posting_id"].apply(lambda x: x.split("_")[-1])
df.sample(2)


len(df)


from pathlib import Path


img_dir = Path('../input/shopee-product-matching/train_images/')
device = "cpu"


import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
import torch.nn.functional as F

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
vit_model = AutoModel.from_pretrained("google/vit-base-patch16-224").to(device)
vit_model


!pip install faiss-cpu


vit_model.eval()
bert_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
bert_model = AutoModel.from_pretrained("google-bert/bert-base-uncased").to(device)
bert_model.eval()


import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer

class ImageTextDataset(Dataset):
    def __init__(self, csv_df, img_dir, max_length=128):
        self.shape = csv_df.shape
        self.data = csv_df.drop("label_group", axis=1)
        self.target = csv_df['label_group']
        self.image_dir = img_dir
        self.max_length = max_length

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]["image"]
        text = self.data.iloc[idx]["title"]
        ids = self.data.iloc[idx]["posting_id"]
        target = self.target[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path)
        image = image_processor(image, return_tensors="pt")
        image = np.array(image.pixel_values)
        image = torch.tensor(image)
        # inputs = {k: v.to(self.device) for k, v in inputs.items()}
        image_outputs = vit_model(image)
        image_emb = image_outputs.last_hidden_state[:, 0, :]
        encoding = bert_tokenizer(
            text, truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt")
        # encoding = {k: v.to(self.device) for k, v in encoding.items()}
        text_outputs = bert_model(**encoding)
        text_emb = text_outputs.last_hidden_state[:, 0, :]

        return {"embedding": torch.cat((image_emb, text_emb), dim=1).detach().numpy(),
                "label_id": target, "self_id": ids}


dataset_train = ImageTextDataset(df[:15], img_dir)
dataloader = DataLoader(dataset_train, batch_size=8, shuffle=False, num_workers=0)


embeddings = []
metadata = []  # сохраним пути и тексты для вывода результатов

for batch in dataloader:
    emb = batch["embedding"]  # np.array формы (D,)
    for e in emb:
        embeddings.append(e[0])
    for i, j in zip(batch["label_id"],batch["self_id"]):
        metadata.append({
            "label_id": i,
            "self_id": j
        })

X = np.array(embeddings).astype('float32') 


metadata[0]


import faiss
d = X.shape[1]
print(type(X))
faiss.normalize_L2(X)
index = faiss.IndexFlatIP(d)

index.add(X)


def search_top_k(index, query_embedding, metadata, k=30):
    """
    Ищет топ-K ближайших соседей для query_embedding.
    
    Args:
        index: Faiss индекс
        query_embedding: np.array формы (D,) или (1, D)
        metadata: список словарей с метаданными (image_path, text, ...)
        k: количество соседей
    
    Returns:
        список словарей: [{"score": ..., "image_path": ..., "text": ...}, ...]
    """
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    distances, indices = index.search(query_embedding, k)
    
    # distances[0] — расстояния для первого (и единственного) запроса
    # indices[0] — индексы соседей
    
    results = []
    for i, idx in enumerate(indices[0]):
        score = distances[0][i]
        results.append({
            "score": float(score),
            "self_id": metadata[idx]["self_id"]
        })
    return results


import numpy as np

X_pairs = []
y_pairs = []
for i in range(len(metadata)):
    query_embedding = X[i]
    query_label_group = metadata[i]["label_id"]

    results = search_top_k(index, query_embedding, metadata, k=30)
    for res in results:
        j = res["self_id"]  # индекс соседа в исходном датасете
        print(type(j))
        neighbor_embedding = X[j]
        neighbor_label_group = metadata[j]["label_id"]
        combined_pair = np.concatenate([query_embedding, neighbor_embedding], axis=1)  # → [2*D]

        label = 1 if query_label_group == neighbor_label_group else 0

        X_pairs.append(combined_pair)
        y_pairs.append(label)

X_pairs = np.array(X_pairs).astype('float32')
y_pairs = np.array(y_pairs)

print("X_pairs shape:", X_pairs.shape)  # → (N * 30, 2 * D)
print("y_pairs shape:", y_pairs.shape)  # → (N * 30,)
print("Пример меток:", np.bincount(y_pairs))  # сколько 0 и 1


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    loss_function='Logloss',
    eval_metric='f1',
    verbose=100,
    random_seed=42)
model.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),
    use_best_model=True,
    plot=False)

