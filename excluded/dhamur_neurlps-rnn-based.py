import os
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt


import torch as t
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from catboost import CatBoostRegressor

from PIL import Image
import pandas as pd
import pickle as pkl
import numpy as np

import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
df.head()


df_d1 = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset1.csv')
df_d1.head()


sm_df = df['SMILES'].tolist()


not_ins = []
for i in df_d1['SMILES'].tolist():
    if i not in sm_df:
        not_ins.append(df_d1[df_d1['SMILES'] == i])

TC_extra = pd.concat(not_ins, ignore_index=True)
print(f"Length before = {len(df_d1)}, After = {len(TC_extra)}")
TC_extra['Tc'] = TC_extra['TC_mean']
TC_extra = TC_extra[['SMILES', 'Tc']]
TC_extra.head()


TC_extra['Tg'] = [np.nan]*len(TC_extra)
TC_extra['FFV'] = [np.nan]*len(TC_extra)
TC_extra['Rg'] = [np.nan]*len(TC_extra)
TC_extra['Density'] = [np.nan]*len(TC_extra)
TC_extra['id'] = [i for i in range(len(TC_extra))]

df = pd.concat([df, TC_extra], ignore_index=True)
print(len(df))
df.head()


df_d3 = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset3.csv')
df_d3.head()


not_ins = []

for i in df_d3['SMILES'].tolist():
    if i not in sm_df:
        not_ins.append(df_d3[df_d3['SMILES'] == i])

Tg_extra = pd.concat(not_ins, ignore_index=True)
print(f"Length before = {len(df_d3)}, After = {len(Tg_extra)}")

Tg_extra.head()


Tg_extra['Tc'] = [np.nan]*len(Tg_extra)
Tg_extra['FFV'] = [np.nan]*len(Tg_extra)
Tg_extra['Rg'] = [np.nan]*len(Tg_extra)
Tg_extra['Density'] = [np.nan]*len(Tg_extra)
Tg_extra['id'] = [i for i in range(129, 129+len(Tg_extra))]


df = pd.concat([df, Tg_extra], ignore_index=True)
print(len(df))
df.head()


df_d4 = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset4.csv')
df_d4.head()


not_ins = []

for i in df_d4['SMILES'].tolist():
    if i not in sm_df:
        not_ins.append(df_d4[df_d4['SMILES'] == i])

FFV_extra = pd.concat(not_ins, ignore_index=True)
print(f"Length before = {len(df_d4)}, After = {len(FFV_extra)}")
FFV_extra.head()


FFV_extra['Tg'] = [np.nan]*len(FFV_extra)
FFV_extra['Tc'] = [np.nan]*len(FFV_extra)
FFV_extra['Rg'] = [np.nan]*len(FFV_extra)
FFV_extra['Density'] = [np.nan]*len(FFV_extra)
FFV_extra['id'] = [i for i in range(175, 175+len(FFV_extra))]

df = pd.concat([df, FFV_extra], ignore_index=True)
print(len(df))
df.head()


df.isna().sum()


df.dtypes


DEVICE = 'cuda' if t.cuda.is_available() else 'cpu'


model = SentenceTransformer('/kaggle/input/polybert/polyBERT-local')

print(f'Model loaded on {next(model.parameters()).device}')


img_path = []
img_embds = []
ids = df['id'].tolist()
smiles = df['SMILES'].tolist()
batch_size = 512
for i in range(0, len(smiles), batch_size):
    emb = model.encode(smiles[i:i+batch_size])
    img_embds.extend(t.from_numpy(emb))
    img_path.extend([str(j) for j in ids[i:i+batch_size]])


# img_embd = [(i - i.min()) / (i.max() - i.min()) for i in img_embds]
# img_embd = [(i - i.mean()) / i.std() for i in img_embds]

img_embds = t.from_numpy(np.array(img_embds))
embd_mean = img_embds.mean()
embd_std = img_embds.std()
img_embd = [(i - embd_mean) / embd_std for i in img_embds]


len(img_path), len(img_embd), img_path[:2]


def get_df(name):
    temp = df[['id', 'SMILES', name]].copy()
    plt.hist(temp[name], bins=15, edgecolor='black')
    plt.show()
    print(f"BEFORE : min = {temp[name].min()}, max = {temp[name].max()}")
    temp_max = temp[name].max()
    temp_min = temp[name].min()
    temp[name] = (temp[name] - temp_min) / (temp_max - temp_min)
    print(f"AFTER : min = {temp[name].min()}, max = {temp[name].max()}")
    
    print(f'Length before dropping duplicates = {len(temp)}')
    temp = temp.dropna()
    print(f'Length after dropping duplicates = {len(temp)}')
    
    temp_lst = temp['id'].to_list()
    filtered_emb = []
    filtered_path = []
    
    for i in tqdm(range(len(temp_lst))):
        _ = str(temp_lst[i])
        idx = img_path.index(_)
        filtered_emb.append(img_embd[idx].numpy())
        filtered_path.append(img_path[idx])
    return temp_max, temp_min, filtered_emb, temp[name].tolist()


def trained_model(names):
    models = dict()
    for name in names:
        maxx, minn, x, y = get_df(name)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=369)
        model = CatBoostRegressor(
            iterations=5000,
            learning_rate=0.01,
            depth=5,
            loss_function='MAE',
            verbose=100,
            eval_metric='MAE',
            od_type='Iter',
            od_wait=300,
            use_best_model=True,
            l2_leaf_reg=0.5,
            random_strength=0.5
        )
        
        model_info = model.fit(x_train, y_train, eval_set=(x_test, y_test), plot=False)
        preds = model.predict(x_test)
        print("Norm = ", (np.abs(preds - np.array(y_test))).mean())
        print("Abs = ", (np.abs((preds * (maxx - minn) + minn) - (np.array(y_test) * (maxx - minn) + minn))).mean())
        print(f"Predicted values = {(preds[:10] * (maxx - minn) + minn)} \n Actual values = {(np.array(y_test[:10]) * (maxx - minn) + minn)}")
        models[name] = [model, minn, maxx]
        print(name + " Completed")
        print("==="*30)
    return models


names = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

models = trained_model(names)


df_sub = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv')
df_test = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
print(df_test.dtypes)
df_sub.head()


df_test.head()


def calc_score(smiles, ids):
    batch_size = 32
    embeddingss = []
    for i in range(0, len(smiles), batch_size):
        embeddingss.append(t.from_numpy(model.encode(smiles[i:i+batch_size])))
    embeddingss = t.cat(embeddingss, 0)

    embeddings = np.array([(i - embd_mean) / embd_std for i in embeddingss])
    
    results = dict()
    for name, m in models.items():
        preds = m[0].predict(embeddings)
        preds = preds.astype(np.float64)
        # print(preds.dtype)
        act_preds = (preds * (m[2] - m[1]) + m[1])
        results[name] = act_preds

    results = pd.DataFrame(results)
    results['id'] = ids
    return results


res = calc_score(df_test['SMILES'].tolist(), df_test['id'].tolist())
res = res[['id', 'Tg', 'FFV', 'Tc', 'Density', 'Rg']]
res.head()


res.dtypes


res.to_csv('submission.csv', index=False)

