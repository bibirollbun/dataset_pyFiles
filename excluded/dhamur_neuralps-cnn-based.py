import os
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import Draw

from transformers import AutoImageProcessor, SwinModel
import torch as t
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor

from PIL import Image
import pandas as pd
import pickle as pkl
import numpy as np

import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
print(len(df))
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


sav_dir = 'Images'
if os.path.exists(sav_dir):
    shutil.rmtree(sav_dir)
os.mkdir(sav_dir)

ids = df['id'].tolist()
chem_names = df['SMILES'].tolist()

for i in tqdm(range(len(ids))):
    mol = Chem.MolFromSmiles(chem_names[i])
    Draw.MolToFile(mol, os.path.join(sav_dir, f'{ids[i]}.png'), size=(384, 384))

print("✅ All SMILES converted to images")


DEVICE = 'cuda' if t.cuda.is_available() else 'cpu'
model_name = '/kaggle/input/swin/pytorch/default/1/swim_model'

image_processor = AutoImageProcessor.from_pretrained(model_name, use_fast=False)
feature_extractor_model = SwinModel.from_pretrained(model_name).to(DEVICE)

img_path = []
for i in os.listdir('/kaggle/working/Images'):
    img_path.append(os.path.join('/kaggle/working/Images', i))

print(len(img_path), img_path[0])


embeddings = []
for batch in tqdm(range(0, len(img_path), 64)):
    pil_img = [Image.open(i).convert('RGB') for i in img_path[batch: batch+64]]
    processed = image_processor(images=pil_img, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in processed.items()}
    with t.inference_mode():
        outputs = feature_extractor_model(**inputs)
        batch_embeddings = F.normalize(outputs.last_hidden_state.mean(dim=1), dim=1, p=2)
        embeddings.extend(batch_embeddings.cpu())

with open('features.pkl', 'wb') as f:
    pkl.dump([img_path, embeddings], f)

with open('features.pkl', 'rb') as f:
    img_path, img_embds = pkl.load(f)


img_embd_tensor = t.from_numpy(np.array(img_embds))
img_embd_mean = img_embd_tensor.mean()
img_embd_std = img_embd_tensor.std()
img_embd = [(i-img_embd_mean)/img_embd_std for i in img_embds]


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
        _ = f"/kaggle/working/Images/{str(temp_lst[i])}.png"
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
            depth=8,
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
        print("Act values = ", (preds[:10] * (maxx - minn) + minn), (np.array(y_test[:10]) * (maxx - minn) + minn))
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
    path = 'pred_imgs'  
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)

    # SMILES to images
    for idx in range(len(smiles)):
        mol = Chem.MolFromSmiles(smiles[idx])      
        Draw.MolToFile(mol, os.path.join(path, f'{ids[idx]}.png'), size=(384, 384))

    # Images to features
    t_img_path = []
    for i in ids:
        t_img_path.append(os.path.join(path, str(i)+'.png'))

    batch_size = 32
    embeddingss = []
    for i in range(0, len(t_img_path), batch_size):
        pil_img = [Image.open(i).convert('RGB') for i in t_img_path[i:i+batch_size]]
        processed = image_processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in processed.items()}
        with t.inference_mode():
            outputs = feature_extractor_model(**inputs)
            batch_embeddings = F.normalize(outputs.last_hidden_state.mean(dim=1).cpu(), dim=1, p=2)
            embeddingss.append(batch_embeddings)
    embeddingss = t.cat(embeddingss, 0)
    embeddings = np.array([(i-img_embd_mean)/img_embd_std for i in embeddingss])
    
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


os.remove('features.pkl')
shutil.rmtree('Images')
shutil.rmtree('pred_imgs')

