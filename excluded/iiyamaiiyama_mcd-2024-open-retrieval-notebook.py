import pandas as pd

data_dir = "/kaggle/input/mcd-data-science-competition-2024-open/"

article_path = data_dir + "article.csv"
sample_submission_path = data_dir + "sample_submission.csv"
test_path = data_dir + "test.csv"
train_path = data_dir + "train.csv"


article_df = pd.read_csv(article_path)
sample_submission_df = pd.read_csv(sample_submission_path)
test_df = pd.read_csv(test_path)
train_df = pd.read_csv(train_path)
article_df.shape,sample_submission_df.shape,test_df.shape,train_df.shape


!pip install -q sentence-transformers


from sentence_transformers import SentenceTransformer

model = SentenceTransformer("avsolatorio/GIST-small-Embedding-v0")



embedding = model.encode("Hello!")
embedding.shape


import matplotlib.pyplot as plt

plt.plot(embedding)
plt.grid()


import numpy as np

texts1 = [
    "今日はいい天気ですね",
    "I love cats",
    "MCD Kaggle Competition に参加しています"
]
texts2 = [
    "これはKaggleコンペです",
    "犬が好きです",
    "悪天候です"
]

emb_lis1 = model.encode(texts1)
emb_lis2 = model.encode(texts2)


# dotでコサイン類似度が得られる
mat = np.dot(emb_lis1, emb_lis2.T)
mat


best_ind = np.argmax(mat, axis=1)
best_ind


for i in range(len(best_ind)):
    print(f"`{texts1[i]}` に最も近い文: `{texts2[best_ind[i]]}`")


article_text_lis = article_df.article_text.values
quiz_text_lis = train_df.quiz_text.values

len(article_text_lis), len(quiz_text_lis)


%%time
quiz_text_emb_lis = model.encode(quiz_text_lis)
article_text_emb_lis = model.encode(article_text_lis)


article_text_emb_lis.shape


quiz_text_emb_lis.shape


import numpy as np

mat = np.dot(quiz_text_emb_lis, article_text_emb_lis.T)
best_ind = np.argmax(mat, axis=1)

pred_id_lis = article_df.iloc[best_ind].article_id.values

len(pred_id_lis)


train_df["pred_article_id"] = pred_id_lis
train_df.head(3)


correct_cnt = train_df[train_df['article_id'] == train_df['pred_article_id']].shape[0]
correct_cnt


correct_cnt/len(train_df)




