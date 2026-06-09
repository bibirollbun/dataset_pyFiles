from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import timm
from sklearn.preprocessing import normalize
from torch.utils.data import Dataset, DataLoader
from torch import nn
import matplotlib.pyplot as plt


class EmbNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model("tiny_vit_5m_224.dist_in22k_ft_in1k", pretrained=True, num_classes=0)

    def forward(self, image):
        x = self.model(image)
        return x


def generate_submit(pred_cluster):
    import hashlib
    sub = pd.DataFrame()
    sub['id'] = np.arange(len(pred_cluster))
    sub['target'] = pred_cluster
    hsh = hashlib.sha256(sub.to_csv(index=False).encode('utf-8')).hexdigest()[:8]
    submit_path = f"submit_{hsh}.csv"
    print(f"SUBMIT_NAME: {submit_path}")
    print(sub.head(10))
    sub.to_csv(submit_path, index = None)


X_1 = np.load('/kaggle/input/neoai-2025-cluster-pictures/data_1.npz')
X_1 = X_1.f.arr_0
X_2 = np.load('/kaggle/input/neoai-2025-cluster-pictures/data_2.npz')
X_2 = X_2.f.arr_0


model = EmbNet().to('cuda').eval()


X = np.concatenate((X_1.mean(2), X_2.mean(1)), 1)
X.shape


class ShizaDataset(Dataset):
    def __init__(self, x):
        super().__init__()
        
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return torch.from_numpy(self.x[i]).unsqueeze(0).repeat(3, 1, 1)


ds = ShizaDataset(X_1 @ X_2)
dl = DataLoader(ds, batch_size=64, shuffle=False, drop_last=False)


res = []
with torch.no_grad():
    for x in tqdm(dl):
        emb = model(x.to('cuda'))

        res.append(emb.cpu().detach().numpy())


embeddings = np.concatenate(res, 0)


km = KMeans(32, algorithm='elkan')
pred_cluster = km.fit_predict(embeddings)

generate_submit(pred_cluster)




