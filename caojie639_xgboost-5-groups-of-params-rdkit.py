!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl


import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold,KFold,StratifiedGroupKFold,GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD,PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import os
from rdkit import Chem
from rdkit.Chem import AllChem
# from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit import DataStructs
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
os.environ["TOKENIZERS_PARALLELISM"] = "false"


train = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
test = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']


train.isnull().sum()


train.describe()


def compute_all_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [None] * len(desc_names)
    return [desc[1](mol) for desc in Descriptors.descList]

desc_names = [desc[0] for desc in Descriptors.descList]


descriptors = [compute_all_descriptors(smi) for smi in train['SMILES'].to_list()]
descriptors = pd.DataFrame(descriptors, columns=desc_names)

train = pd.concat([train,descriptors],axis=1)


descriptors = [compute_all_descriptors(smi) for smi in test['SMILES'].to_list()]
descriptors = pd.DataFrame(descriptors, columns=desc_names)
test = pd.concat([test,descriptors],axis=1)


train.head()


features = train.columns.values[7:]
features = np.append(['id'], features)
print(len(features))
features


test


def xgb_train_predict(train_df, test_df, target, features):
    params = {
        "Tg":
             {"objective":"reg:squarederror",
              "booster":"gbtree",
              'colsample_bynode': 0.55,
              'colsample_bytree': 0.9,
              'eta': 0.12,
              'gamma': 500.0,
              'lambda': 60.0,
              'max_depth': 9,
              'min_child_weight': 12.0,
              'num_boost_round': 150,
              'subsample': 0.75,
              'seed': 123},
        "FFV":
             {"objective":"reg:absoluteerror",
              "booster":"gbtree",
              'colsample_bynode': 0.3,
              'colsample_bytree': 0.7,
              'eta': 0.05,
              'gamma': 0.45,
              'lambda': 4.2,
              'max_depth': int(16),  # 10 + 6
              'min_child_weight': 6.0,
              'num_boost_round': 280,
              'subsample': 1.0,
              'seed': 123},
        "Tc":
             {"objective":"reg:absoluteerror",
              "booster":"gbtree",
              'colsample_bynode': 0.73,
              'colsample_bytree': 0.96,
              'eta': 0.18,
              'gamma': 0.27,
              'lambda': 56.5,
              'max_depth': int(18),  # 10 + 6
              'min_child_weight': 36,
              'num_boost_round': 208,
              'subsample': 0.97,
              'seed': 123},
        "Density":
             {"objective":"reg:absoluteerror",
              "booster":"gbtree",
              'colsample_bynode': 0.70,
              'colsample_bytree': 0.45,
              'eta': 0.075,
              'gamma': 0.36,
              'lambda': 1.42,
              'max_depth': int(17),
              'min_child_weight': 7,
              'num_boost_round': 196,
              'subsample': 0.98,
              'seed': 123},
        "Rg":
             {"objective":"reg:squarederror",
              "booster":"gbtree",
              'colsample_bynode': 0.68,
              'colsample_bytree': 0.78,
              'eta': 0.07,
              'gamma': 0.6,
              'lambda': 1.8,
              'max_depth': int(13),
              'min_child_weight': 4,
              'num_boost_round': 172,
              'subsample': 0.78,
              'seed': 123},
    }
    # drop_columns = ['SMILES', 'Tg', 'FFV', 'Tc', 'Density', 'Rg']
    # X = train_df.drop(columns=drop_columns)
    X = train_df[features]
    y = train_df[target]
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    data_xgb = xgb.DMatrix(X, y, missing=np.inf)  # 处理缺失值
    model = xgb.train(params[target], data_xgb, num_boost_round=int(params[target]['num_boost_round']))

    # X_test = test_df.drop(columns="SMILES")
    X_test = test_df[features]
    X_test = X_test.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.fillna(0)
    test_xgb = xgb.DMatrix(X_test, missing=np.inf)  # 处理缺失值
    y_pred = model.predict(test_xgb)
    return y_pred


%%time
for target in targets:
    train_df = train[train[target].notnull()]
    test[target] = xgb_train_predict(train_df, test, target, features)


test


test[['id','Tg', 'FFV', 'Tc', 'Density', 'Rg']].to_csv('submission.csv',index=False)

