from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import numpy as np
import pandas as pd
from torch import nn


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

km = KMeans(32)
X = np.concatenate((X_1.reshape((X_1.shape[0], X_1.shape[1] * X_1.shape[2])), X_2.reshape((X_2.shape[0], X_2.shape[1] * X_2.shape[2]))), 1)
pred_cluster = km.fit_predict(X)

generate_submit(pred_cluster)


When you make a submit, 
1. Make a Quick Save of the notebook, otherwise we may reject your solution! 
2. Add notebook version to the comment for the submit.

===

ĞŸÑ€Ğ¸ Ğ¾Ñ‚Ğ¿Ñ€Ğ°Ğ²ĞºĞµ Ñ€ĞµÑˆĞµĞ½Ğ¸Ñ�:

1. Ğ¡Ğ´ĞµĞ»Ğ°Ğ¹Ñ‚Ğµ Quick Save Ğ½Ğ¾ÑƒÑ‚Ğ±ÑƒĞºĞ°, Ğ¸Ğ½Ğ°Ñ‡Ğµ Ğ¼Ñ‹ Ğ¼Ğ¾Ğ¶ĞµĞ¼ Ğ¾Ñ‚ĞºĞ»Ğ¾Ğ½Ğ¸Ñ‚ÑŒ Ğ²Ğ°ÑˆĞµ Ñ€ĞµÑˆĞµĞ½Ğ¸Ğµ!
2. Ğ”Ğ¾Ğ±Ğ°Ğ²ÑŒÑ‚Ğµ Ğ²ĞµÑ€Ñ�Ğ¸Ñ� Ğ½Ğ¾ÑƒÑ‚Ğ±ÑƒĞºĞ° Ğ² ĞºĞ¾Ğ¼Ğ¼ĞµĞ½Ñ‚Ğ°Ñ€Ğ¸Ğ¹ Ğº Ğ¾Ñ‚Ğ¿Ñ€Ğ°Ğ²ĞºĞµ.

