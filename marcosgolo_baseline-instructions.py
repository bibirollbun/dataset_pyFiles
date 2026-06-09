!gdown 1csTzz_UWpe89KYh11qj_V7fDssTv2FFO
!gdown 1KlZGlcUEVgVVvxV0l4vYwQ1g1pgmsYio


import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('df_social_data_train.csv')
model = SentenceTransformer("all-MiniLM-L6-v2")

df = df.dropna()
df['features'] = list(model.encode(df['content'].tolist(), show_progress_bar=True))
knn = KNeighborsClassifier(n_neighbors=3,metric="cosine")
knn.fit(df.features.to_list(), df.engagement.to_list())


df_test = pd.read_csv('df_social_data_test.csv')
df_test


df_test['content'] = df_test['content'].astype(str)
df_test['features'] = list(model.encode(df_test['content'].tolist(), show_progress_bar=True))
df_test['Engagement'] = knn.predict(df_test.features.to_list())
df_test['ID'] = range(len(df_test))
df_test[['ID','Engagement']].to_csv('df_kaggle.csv', index=False)

