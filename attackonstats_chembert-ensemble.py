# !pip install torch tqdm transformers==4.38.2 -qqq


!pip install --no-index --find-links /kaggle/input/custom-packages-wheels torch tqdm transformers==4.38.2


import numpy as np
import pandas as pd

import torch 
from tqdm.notebook import tqdm
from transformers import AutoModel,AutoTokenizer

from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import mean_absolute_error

from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,HistGradientBoostingRegressor,AdaBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import joblib


train_df = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train.csv")
density_df = pd.read_csv("/kaggle/input/self-data/DensityFinal.csv")
tc_df = pd.read_csv("/kaggle/input/self-data/TcFinal.csv")
tg_df = pd.read_csv("/kaggle/input/tg2final/Tg2Final.csv")


density_df["Density"] = pd.to_numeric(density_df['Density'],errors = 'coerce')


train_df.info()


ffv_df = train_df.copy()
ffv_df = ffv_df.drop(columns = ['id','Density','Tc','Rg','Tg'])
ffv_df.dropna(inplace = True)


rg_df = train_df.copy()
rg_df = rg_df.drop(columns = ['id','Density','Tc','FFV','Tg'])
rg_df.dropna(inplace = True)


tc_df.dropna(inplace = True)
density_df.dropna(inplace = True)


print(tg_df.info())
print("-"*100)
print(tc_df.info())
print("-"*100)
print(density_df.info())
print("-"*100)
print(rg_df.info())
print("-"*100)
print(ffv_df.info())


if torch.cuda.is_available():
    device = torch.device('cuda')

else:
    device = torch.device('cpu')


model_name = "/kaggle/input/custom-packages-wheels/DeepChem/ChemBERTa-77M-MTR"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model = model.to(device)
model.eval()


def preprocess(x,model= model,tokenizer = tokenizer):
    
    embeddings = []
    
    with torch.no_grad():
        for smile in tqdm(x):
            inputs = tokenizer(smile,return_tensors = 'pt',padding = True,truncation = False).to(device)
            outputs = model(**inputs)
            cls = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
            embeddings.append(cls)

    return np.array(embeddings)


rg_x = preprocess(rg_df["SMILES"])
tc_x = preprocess(tc_df["SMILES"])
tg_x = preprocess(tg_df["SMILES"])
density_x = preprocess(density_df["SMILES"])
ffv_x = preprocess(ffv_df["SMILES"])


rg_y = np.array(rg_df.iloc[:,-1])
tg_y = np.array(tg_df.iloc[:,-1])
tc_y = np.array(tc_df.iloc[:,-1])
ffv_y = np.array(ffv_df.iloc[:,-1])
density_y = np.array(density_df.iloc[:,-1])


# models = {
#     'CatBoostRegressor': CatBoostRegressor(task_type="GPU", devices='0', verbose=0),
#     'RandomForestRegressor': RandomForestRegressor(verbose=0),  # ❌ No GPU support in sklearn
#     'GradientBoostingRegressor': GradientBoostingRegressor(verbose=0),  # ❌ No GPU support in sklearn
#     'HistGradientBoostingRegressor': HistGradientBoostingRegressor(verbose=0),  # ❌ CPU only
#     'AdaBoostRegressor': AdaBoostRegressor(),  # ❌ CPU only
#     'XGBRegressor': XGBRegressor(tree_method="gpu_hist", predictor="gpu_predictor", gpu_id=0, verbosity=0),
#     'LGBMRegressor': LGBMRegressor(device='gpu',verbose = -1)
# }


# def model_train(x,y,models = models,k = 5):
#     kf = KFold(random_state = 42,n_splits = k,shuffle = True)
#     scores = {}

#     for model_name,model in tqdm(models.items(), desc="Cross-Val Models"):
#         fold_score = []
#         for train_idx,val_idx in kf.split(x):
#             train_x,train_y = x[train_idx],y[train_idx]
#             val_x,val_y = x[val_idx],y[val_idx]
    
#             model.fit(train_x,train_y.ravel())
#             preds = model.predict(val_x)
#             score = mean_absolute_error(val_y,preds)
#             fold_score.append(score)
            
#         scores[model_name] = np.mean(fold_score)

#     return scores


model_map = {
    'RG': CatBoostRegressor(verbose=0),
    'TG': CatBoostRegressor(verbose=0),
    'TC': LGBMRegressor(verbose = -1),
    'Density': CatBoostRegressor(verbose=0),
    'FFV': LGBMRegressor(verbose = -1)
}


train_data_map = {
    'RG': (rg_x, rg_y),
    'TG': (tg_x, tg_y),
    'TC': (tc_x, tc_y),
    'Density': (density_x, density_y),
    'FFV': (ffv_x, ffv_y)
}


def train_and_save_all(data_map, model_map):
    for target, (X, y) in data_map.items():
        model = model_map[target]
        print(f"Training model for {target}...")
        model.fit(X, y)
        joblib.dump(model, f"{target}.pkl")
        print(f"Saved {target}.pkl ✅")


train_and_save_all(train_data_map, model_map)


test_df = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/test.csv")
test_x = preprocess(test_df["SMILES"])


model_files = {
    "Tg": "TG.pkl",
    "FFV": "FFV.pkl",
    "Tc": "TC.pkl",
    "Density": "Density.pkl",
    "Rg": "RG.pkl"
}


predictions = {}

for target_name, model_file in model_files.items():
    model = joblib.load(model_file)
    predictions[target_name] = model.predict(test_x)


submission = pd.DataFrame({
    "id": test_df["id"],
    "Tg": predictions["Tg"],
    "FFV": predictions["FFV"],
    "Tc": predictions["Tc"],
    "Density": predictions["Density"],
    "Rg": predictions["Rg"]
})



submission.to_csv("submission.csv", index=False)





