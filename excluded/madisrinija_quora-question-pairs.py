!pip install sentence-transformers


import pandas as pd
import numpy as np
import transformers
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer


!unzip /kaggle/input/quora-question-pairs/train.csv.zip -d /kaggle/working
!unzip /kaggle/input/quora-question-pairs/test.csv.zip -d /kaggle/working


train  = pd.read_csv("/kaggle/working/train.csv")
test = pd.read_csv("/kaggle/working/test.csv")


train.head()


test.head()


model =  SentenceTransformer("all-MiniLM-L6-v2")


import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


q1_emb = model.encode(
    train['question1'].tolist(),
    convert_to_tensor=True,
    show_progress_bar=True,
    batch_size=128
)


q2_emb = model.encode(
    train['question2'].tolist(),
    convert_to_tensor=True,
    show_progress_bar=True,
    batch_size=128
)


cos_sim = torch.nn.functional.cosine_similarity(
    q1_emb,
    q2_emb
)


q1_emb_np = q1_emb.cpu().numpy()
q2_emb_np = q2_emb.cpu().numpy()


from sklearn.metrics.pairwise import cosine_similarity

cos_sim = np.array([
    cosine_similarity(q1_emb_np[i].reshape(1, -1),
                      q2_emb_np[i].reshape(1, -1))[0][0]
    for i in range(len(q1_emb_np))
])

abs_diff = np.abs(q1_emb_np - q2_emb_np)
prod = q1_emb_np * q2_emb_np



X = np.hstack([
    cos_sim.reshape(-1, 1),
    abs_diff,
    prod
])



from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

y = train["is_duplicate"].values

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

clf = LogisticRegression(max_iter=1000)
clf.fit(X, y)



test["question1"] = test["question1"].fillna("")
test["question2"] = test["question2"].fillna("")


#q1_test_emb = model.encode(
#    test["question1"].tolist(),
#    batch_size=256,
#    show_progress_bar=True
#)


'''q2_test_emb = model.encode(
    test["question2"].tolist(),
    batch_size=256,
    show_progress_bar=True
)'''


'''from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

cos_sim_test = np.array([
    cosine_similarity(q1_test_emb[i].reshape(1, -1),
                      q2_test_emb[i].reshape(1, -1))[0][0]
    for i in range(len(q1_test_emb))
])

abs_diff_test = np.abs(q1_test_emb - q2_test_emb)
prod_test = q1_test_emb * q2_test_emb

X_test = np.hstack([
    cos_sim_test.reshape(-1, 1),
    abs_diff_test,
    prod_test
])'''


