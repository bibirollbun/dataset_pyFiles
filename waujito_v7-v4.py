from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import numpy as np
import pandas as pd
from torch import nn
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import timm
import torch
from tqdm import tqdm
from PIL import Image

import torch
torch.manual_seed(56)
np.random.seed(56)


class EmbNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model("tiny_vit_5m_224.dist_in22k_ft_in1k", pretrained=True, num_classes=0)

    def forward(self, image):
        x = self.model(image)
        return x


X_1 = np.load('/kaggle/input/neoai-2025-cluster-pictures/data_1.npz')
X_1 = X_1.f.arr_0
X_2 = np.load('/kaggle/input/neoai-2025-cluster-pictures/data_2.npz')
X_2 = X_2.f.arr_0

Xs = list()
# Xtws = list()
for i in range(len(X_1)):
    # a = X_1[i].T.copy()
    # b = X_2[i].copy()
    
    # t = b[1].copy()
    # b[1] = b[2].copy()
    # b[2] = t.copy()
    
    # t = b[2].copy()
    # b[2] = b[3].copy()
    # b[3] = t.copy()
    # a = a.T
    
    Xs.append(np.dot(X_1[i], X_2[i]))
    # Xtws.append((np.dot(X_1[i].T, X_2[i].T)).reshape(-1))
Xs = np.array(Xs)
# Xtws = np.array(Xtws)
Xs = np.repeat(np.expand_dims(Xs, 3), repeats=3, axis=3).astype(np.uint8)


csm = cosine_similarity(X_1[0].T, X_2[0])


sns.heatmap(csm)


xxx = iter(Xs)


plt.imshow(next(xxx).astype(np.uint8))


model = EmbNet().cuda()


from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2

class CustomImageDataset(Dataset):
    def __init__(self, image_array, transform=None):
        self.image_array = image_array
        self.transform = transform

    def __len__(self):
        return len(self.image_array)

    def __getitem__(self, idx):
        image = self.image_array[idx]
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        return image

transform = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToTensor(),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # v2.ColorJitter(brightness=0.3),
    # v2.RandomVerticalFlip(p=0.5),
    # v2.RandomHorizontalFlip(p=0.5)
])

dataset = CustomImageDataset(Xs, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)


embs = list()
for images in tqdm(dataloader):
    with torch.no_grad():
        embeds = model(images.cuda()).detach().cpu().numpy()
    embs.append(embeds)
embs = np.concatenate(embs)


from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2

class CustomImageDataset(Dataset):
    def __init__(self, image_array, transform=None):
        self.image_array = image_array
        self.transform = transform

    def __len__(self):
        return len(self.image_array)

    def __getitem__(self, idx):
        image = self.image_array[idx]
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        return image

transform = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToTensor(),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    v2.ColorJitter(brightness=0.3),
    v2.RandomVerticalFlip(p=0.5),
    v2.RandomHorizontalFlip(p=0.5)
])

dataset = CustomImageDataset(Xs, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)


nembs = list()
for images in tqdm(dataloader):
    with torch.no_grad():
        embeds = model(images.cuda()).detach().cpu().numpy()
    nembs.append(embeds)
nembs = np.concatenate(nembs) 


embs = np.concatenate((embs, nembs), axis=1)


from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2

class CustomImageDataset(Dataset):
    def __init__(self, image_array, transform=None):
        self.image_array = image_array
        self.transform = transform

    def __len__(self):
        return len(self.image_array)

    def __getitem__(self, idx):
        image = self.image_array[idx]
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        return image

transform = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToTensor(),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    v2.ColorJitter(brightness=0.4),
    v2.RandomVerticalFlip(p=0.4),
    v2.RandomHorizontalFlip(p=0.4)
])

dataset = CustomImageDataset(Xs, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

nembs = list()
for images in tqdm(dataloader):
    with torch.no_grad():
        embeds = model(images.cuda()).detach().cpu().numpy()
    nembs.append(embeds)
nembs = np.concatenate(nembs) 


embs = np.concatenate((embs, nembs), axis=1)


nembs


cosims = cosine_similarity(embs)


cosims


# import numpy as np
# from sklearn.cluster import SpectralClustering
# # mat = np.matrix([[1.,.1,.6,.4],[.1,1.,.1,.2],[.6,.1,1.,.7],[.4,.2,.7,1.]])
# clustering = SpectralClustering(32).fit_predict(cosims)





# Xt = np.concatenate((embs, X_1[:,:,0]), axis=1)
# Xt.shape


# from sklearn.preprocessing import normalize


# nembs = np.concatenate((normalize(embs), normalize(X_1[:, :, 0])), axis=1)


# tt = normalize(nembs, axis=1, norm="l2")


# from sklearn.decomposition import TruncatedSVD


# X = Xs[:, :64, :64, 0].reshape((len(Xs), -1))


def degenerate_submit(pred_cluster):
    import hashlib
    sub = pd.DataFrame()
    sub['id'] = np.arange(len(pred_cluster))
    sub['target'] = pred_cluster
    hsh = hashlib.sha256(sub.to_csv(index=False).encode('utf-8')).hexdigest()[:8]
    submit_path = f"submit_{hsh}.csv"
    print(f"SUBMIT_NAME: {submit_path}")
    print(sub.head(10))
    sub.to_csv(submit_path, index = None)


km = KMeans(32, init="k-means++", random_state=56, max_iter=700)
#X = np.concatenate((X_1.reshape((X_1.shape[0], X_1.shape[1] * X_1.shape[2])), X_2.reshape((X_2.shape[0], X_2.shape[1] * X_2.shape[2]))), 1)
#X = Xt
# X = cosine_similarity(embeds, embeds)
X = embs
# X = Xs[:, 0, ]
pred_cluster = km.fit_predict(X)
# pred_cluster=clustering

degenerate_submit(pred_cluster)

