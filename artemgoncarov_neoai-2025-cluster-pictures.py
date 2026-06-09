from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


import timm
class EmbNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model("tiny_vit_5m_224.dist_in22k_ft_in1k", pretrained=True, num_classes=0)

    def forward(self, image):
        x = self.model(image)
        return x


model = EmbNet()
model.eval();


def generate_submit(pred_cluster):
    import hashlib
    sub = pd.DataFrame()
    sub['id'] = np.arange(len(pred_cluster))
    sub['target'] = pred_cluster
    # hsh = hashlib.sha256(sub.to_csv(index=False).encode('utf-8')).hexdigest()[:8]
    submit_path = f"submit.csv"
    print(f"SUBMIT_NAME: {submit_path}")
    print(sub.head(10))
    sub.to_csv(submit_path, index = None)


X_1 = np.load('/kaggle/input/neoai-2025-cluster-pictures/data_1.npz')
X_1 = X_1.f.arr_0
X_2 = np.load('/kaggle/input/neoai-2025-cluster-pictures/data_2.npz')
X_2 = X_2.f.arr_0

# X = np.concatenate((X_1.reshape((X_1.shape[0], X_1.shape[1] * X_1.shape[2])), X_2.reshape((X_2.shape[0], X_2.shape[1] * X_2.shape[2]))), 1)
X = np.concatenate((X_1, X_2.reshape((3840, 128, 4))), 1).reshape((3840, 1024))


X.shape


X = X.reshape((3840, 1, 32, 32))
X_rgb = np.repeat(X, 3, axis=1)


tensor_images = torch.tensor(X_rgb, dtype=torch.float32)
dataset = TensorDataset(tensor_images)
dataloader = DataLoader(dataset, batch_size=32)


from tqdm import tqdm

embeddings = []
with torch.no_grad():
    for batch in tqdm(dataloader, desc="Processing"):
        images_batch = batch[0]
        embedding = model(images_batch)
        embeddings.append(embedding)

embeddings = torch.cat(embeddings)


embeddings.shape


d = cosine_distances(embeddings)


km = KMeans(32)
pred_cluster = km.fit_predict(d)


generate_submit(pred_cluster)


len(embeddings)




