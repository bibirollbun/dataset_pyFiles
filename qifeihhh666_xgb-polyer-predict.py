# ! pip install rdkit
!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl
print("ok")


from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder
from pathlib import Path
from sklearn import preprocessing
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
from catboost import CatBoostRegressor
import lightgbm as lgb
from sklearn.base import clone

from rdkit import Chem
from rdkit.Chem import Descriptors
 
import joblib 
import warnings
warnings.filterwarnings("ignore")


import matplotlib.pyplot as plt
print("ok")


train_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
test_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
print(train_df.shape)
print(test_df.shape)
print("ok")



def count_characters_in_smiles(smiles):
    if pd.isna(smiles):
        return {
            'SMILES_len': 0,
            'C_count': 0, 'N_count': 0, 'O_count': 0, 'F_count': 0,
            'S_count': 0, 'Cl_count': 0, 'Br_count': 0, 'I_count': 0,
            'P_count': 0, 'equal_count': 0, 'hash_count': 0, 'ring_count': 0
        }
    
    features = {}
    features['SMILES_len'] = len(smiles)
    features['C_count'] = smiles.count('C') + smiles.count('c')  # 包括大小写的碳
    features['N_count'] = smiles.count('N') + smiles.count('n')
    features['O_count'] = smiles.count('O') + smiles.count('o')
    features['F_count'] = smiles.count('F')
    features['S_count'] = smiles.count('S') + smiles.count('s')
    features['Cl_count'] = smiles.count('Cl')
    features['Br_count'] = smiles.count('Br')
    features['I_count'] = smiles.count('I')
    features['P_count'] = smiles.count('P')
    features['equal_count'] = smiles.count('=')
    features['hash_count'] = smiles.count('#')
    features['ring_count'] = smiles.count('c') + smiles.count('1') + smiles.count('2')  # 简单环计数

    return features

def add_smiles_features(df, smiles_col='SMILES'):
    features_df = df[smiles_col].apply(count_characters_in_smiles).apply(pd.Series)
    df = pd.concat([df, features_df], axis=1)
    return df

def compute_all_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [None] * len(desc_names)
    return [desc[1](mol) for desc in Descriptors.descList]

# 计算所有描述符的名称
desc_names = [desc[0] for desc in Descriptors.descList]

# 假设 train_df 和 test_df 是你的 DataFrame
train_df_with_features = add_smiles_features(train_df)
test_df_with_features = add_smiles_features(test_df)

# 计算分子描述符
#descriptors = [compute_all_descriptors(smi) for smi in train_df['SMILES'].to_list()]
#descriptors_df = pd.DataFrame(descriptors, columns=desc_names)
#train_df_with_features = pd.concat([train_df, descriptors_df], axis=1)
#
#descriptors = [compute_all_descriptors(smi) for smi in test_df['SMILES'].to_list()]
#descriptors_df = pd.DataFrame(descriptors, columns=desc_names)
#test_df_with_features = pd.concat([test_df, descriptors_df], axis=1)

print(train_df_with_features.shape)
print(test_df_with_features.shape)
print("ok")


null_counts = train_df_with_features.isnull().sum()
columns_with_null = null_counts[null_counts > 0].index.tolist()
train_df_with_features[columns_with_null] = train_df_with_features[columns_with_null].fillna(0)


null_counts = test_df_with_features.isnull().sum()
columns_with_null = null_counts[null_counts > 0].index.tolist()
test_df_with_features[columns_with_null] = test_df_with_features[columns_with_null].fillna(0)
print("ok")


test_ids = test_df_with_features["id"]
submission = pd.DataFrame({
        "id": test_ids,
    })

X_fit = test_df_with_features.drop(['id','SMILES'], axis=1)
print("ok")


num_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

X = train_df_with_features.drop(['id','SMILES','Tg','FFV','Tc','Density','Rg'], axis=1)
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(0, inplace=True)  # 或者使用其他合适的填充值


print("检查数据中的 inf 和 NaN 值:")
print("NaN 值数量:", X.isna().sum().sum())
print("inf 值数量:", np.isinf(X.select_dtypes(include=[np.number])).sum().sum())
 # 检查数据类型

# 检查数据范围
print("数据范围:")


from sklearn.preprocessing import StandardScaler
# 初始化标准化器
scaler = StandardScaler()
# 对数值数据进行标准化
numerical_cols = X.select_dtypes(include=[np.number]).columns
print(X.shape)
print(X_fit.shape)
print(len(numerical_cols))
#X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
#X_fit[numerical_cols] = scaler.transform(X_fit[numerical_cols])
print("ok")




for k in num_cols:
    y = train_df_with_features[k]
    y = y.fillna(y.mean())
    scaler = StandardScaler()
    #y_scaled = scaler.fit_transform(y.values.reshape(-1, 1)).flatten()
    
    # 4. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Missing values in X_train:", X_train.isna().sum().sum())
    #print("Missing values in y_train:", y_train.isna().sum())
    params_xgb = {
            'learning_rate': 0.1,            
            'max_depth': 6,                  
            #'subsample': 0.8,               
            'n_estimators': 1000,  
            'early_stopping_rounds':200,
            'random_state': 42               
    
        }
    model = xgb.XGBRegressor(**params_xgb,missing=np.nan)
    model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=100                  # 每 100 轮输出一次日志
             )
    
    
    X_fit_pred = model.predict(X_fit)
    #y_pred = scaler.inverse_transform(X_fit_pred.reshape(-1, 1)).flatten()
    print(X_fit_pred)
    submission[k]=X_fit_pred


submission.to_csv("submission.csv", index=False)
print("Submission saved successfully")
print(submission.head())

