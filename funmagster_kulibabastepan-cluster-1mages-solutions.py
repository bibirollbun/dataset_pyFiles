from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import numpy as np
import pandas as pd
from torch import nn
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import hashlib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import hashlib
from urllib.request import urlopen
from PIL import Image
import timm
import numpy as np
from scipy.spatial.distance import cosine


class EmbNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model("tiny_vit_5m_224.dist_in22k_ft_in1k", pretrained=True, num_classes=0)

    def forward(self, image):
        x = self.model(image)
        return x


# def generate_submit(pred_cluster):
#     sub = pd.DataFrame()
#     sub['id'] = np.arange(len(pred_cluster))
#     sub['target'] = pred_cluster
#     hsh = hashlib.sha256(sub.to_csv(index=False).encode('utf-8')).hexdigest()[:8]
#     submit_path = f"submit_{hsh}.csv"
#     print(f"SUBMIT_NAME: {submit_path}")
#     print(sub.head(10))
#     sub.to_csv(submit_path, index = None)


# X_1 = np.load('/kaggle/input/neoai-2025-cluster-pictures/data_1.npz')
# X_1 = X_1.f.arr_0
# X_2 = np.load('/kaggle/input/neoai-2025-cluster-pictures/data_2.npz')
# X_2 = X_2.f.arr_0

# km = KMeans(32)
# X = np.concatenate((X_1.reshape((X_1.shape[0], X_1.shape[1] * X_1.shape[2])), X_2.reshape((X_2.shape[0], X_2.shape[1] * X_2.shape[2]))), 1)
# pred_cluster = km.fit_predict(X)

# generate_submit(pred_cluster)


import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

class EmbNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model("tiny_vit_5m_224.dist_in22k_ft_in1k", pretrained=True, num_classes=0)

    def forward(self, image):
        x = self.model(image)
        return x
        
model = EmbNet()
model.eval()


X_1 = np.load('/kaggle/input/neoai-2025-cluster-pictures/data_1.npz')
X_1 = X_1.f.arr_0
X_2 = np.load('/kaggle/input/neoai-2025-cluster-pictures/data_2.npz')
X_2 = X_2.f.arr_0


tensor_X1 = torch.tensor(X_1, dtype=torch.float32).reshape(-1, 1, 128, 4)
tensor_X2 = torch.tensor(X_2, dtype=torch.float32).reshape(-1, 1, 4, 128)

tensor_X1 = tensor_X1.repeat(1, 3, 1, 1)
tensor_X2 = tensor_X2.repeat(1, 3, 1, 1)

dataset = TensorDataset(tensor_X1, tensor_X2)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

features = []
with torch.no_grad():
    for x1, x2 in dataloader:
        f1 = model(x1)
        f2 = model(x2)
        combined_features = torch.cat((f1, f2), dim=1)
        features.append(combined_features)


features = torch.cat(features).numpy()


def generate_submit(pred_cluster):
    sub = pd.DataFrame()
    sub['id'] = np.arange(len(pred_cluster))
    sub['target'] = pred_cluster
    hsh = hashlib.sha256(sub.to_csv(index=False).encode('utf-8')).hexdigest()[:8]
    submit_path = f"submit_{hsh}.csv"
    print(f"SUBMIT_NAME: {submit_path}")
    print(sub.head(10))
    sub.to_csv(submit_path, index=None)
    return sub


scaler = StandardScaler()
features_normalized = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=32, n_init=10, random_state=42)
kmeans_labels = kmeans.fit_predict(features_normalized)

pred_cluster = kmeans_labels
sub_1 = generate_submit(pred_cluster)

