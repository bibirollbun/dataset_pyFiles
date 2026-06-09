# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install rdkit


import duckdb
import pandas as pd
from tqdm import tqdm
import numpy as np # linear algebra
from rdkit import Chem
from rdkit.Chem import AllChem



# è¨­å®šæª”æ¡ˆè·¯å¾‘
train_path = '/kaggle/input/leash-BELKA/train.parquet'

# å»ºç«‹ DuckDB é€£ç·š
con = duckdb.connect()

# ä½¿ç”¨é€²åº¦æ¢�ä¾†é¡¯ç¤ºé€²åº¦
with tqdm(total=2, desc="Processing Data") as pbar:
    # æŸ¥è©¢ç¬¬ä¸€éƒ¨åˆ†æ•¸æ“š
    df_part1 = con.query(f"""SELECT *
                              FROM parquet_scan('{train_path}')
                              WHERE binds = 0
                              ORDER BY random()
                              LIMIT 180000""").df()
    pbar.update(1)  # æ›´æ–°é€²åº¦æ¢�

    # æŸ¥è©¢ç¬¬äºŒéƒ¨åˆ†æ•¸æ“š
    df_part2 = con.query(f"""SELECT *
                              FROM parquet_scan('{train_path}')
                              WHERE binds = 1
                              ORDER BY random()
                              LIMIT 20000""").df()
    pbar.update(1)  # æ›´æ–°é€²åº¦æ¢�

# å�ˆä½µå…©éƒ¨åˆ†æ•¸æ“š
df = pd.concat([df_part1, df_part2], ignore_index=True)

# éš¨æ©Ÿæ´—ç‰Œæ•¸æ“šï¼ˆfrac=1 è¡¨ç¤ºä¿�æŒ�å�Ÿå§‹å¤§å°�ï¼Œshuffle æ•´å€‹ DataFrameï¼‰
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# é—œé–‰é€£ç·š
con.close()


# # ç¢ºèª�æ•¸æ“š
print(df.head())


def smiles_to_topological_torsion_fingerprint(smiles, n_bits=2048):
    """
    å°‡ SMILES åˆ†å­�çµ�æ§‹è½‰æ�›ç‚ºæ‹“æ¨¸æ‰­è½‰æŒ‡ç´‹ã€‚
    :param smiles: åˆ†å­�çš„ SMILES è¡¨ç¤ºæ³•
    :param n_bits: æŒ‡ç´‹çš„ä½�å…ƒæ•¸ (é»˜èª�ç‚º 2048)
    :return: ä¸€å€‹ numpy æ•¸çµ„ï¼Œè¡¨ç¤ºæ‹“æ¨¸æ‰­è½‰æŒ‡ç´‹
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=int)
    else:
        generator = AllChem.GetTopologicalTorsionGenerator(fpSize=n_bits)
        return np.array(generator.GetFingerprint(mol), dtype=int)

# ä½¿ç”¨é€²åº¦æ¢�è½‰æ�› "molecule_smiles" æ¬„ä½�ç‚ºæ‹“æ¨¸æ‰­è½‰æŒ‡ç´‹
tqdm.pandas(desc="Transforming molecule_smiles to Topological Torsion Fingerprint")
df["molecule_smiles"] = df["molecule_smiles"].progress_apply(lambda x: smiles_to_topological_torsion_fingerprint(x))




# å°� protein æ¬„ä½�é€²è¡Œ One-Hot Encoding
protein_one_hot = pd.get_dummies(df["protein_name"], prefix="protein").astype(int)

# å�ˆä½µ One-Hot çµ�æ�œ
df_one_hot = pd.concat([df, protein_one_hot], axis=1)

# å�ˆä½µéœ€è¦�çš„æ¬„ä½�ï¼šmolecule_smiles, binds, å’Œç¶“é�� One-Hot Encoding çš„ protein
df_one_hot = pd.concat([df_one_hot[["id", "molecule_smiles", "binds"]], protein_one_hot], axis=1)


df_one_hot


# # å�¯é�¸ï¼šç§»é™¤å�Ÿå§‹ protein æ¬„ä½�
# df1_one_hot.drop("protein_name", axis=1, inplace=True)

# # æª¢è¦–è™•ç�†å¾Œæ•¸æ“š
# print(df_one_hot.head())

# åƒ…ä¿�ç•™éœ€è¦�çš„æ¬„ä½�
columns_to_keep = ["id", "molecule_smiles", "binds"] + protein_one_hot.columns.tolist()
df_filtered = df_one_hot[columns_to_keep]

# å„²å­˜è™•ç�†å¾Œçš„æ•¸æ“š
output_filename = "test_transformed_topological(180k,20k).parquet"
df_filtered.to_parquet(output_filename, index=False)



topolfile = '/kaggle/working/test_transformed_topological(180k,20k).parquet'
topol = pd.read_parquet(topolfile)
topol


# è¨­å®šæª”æ¡ˆè·¯å¾‘
train_path = '/kaggle/input/leash-BELKA/train.parquet'

# å»ºç«‹ DuckDB é€£ç·š
con = duckdb.connect()

# ä½¿ç”¨é€²åº¦æ¢�ä¾†é¡¯ç¤ºé€²åº¦
with tqdm(total=2, desc="Processing Data") as pbar:
    # æŸ¥è©¢ç¬¬ä¸€éƒ¨åˆ†æ•¸æ“š
    df_part1 = con.query(f"""SELECT *
                              FROM parquet_scan('{train_path}')
                              WHERE binds = 0
                              ORDER BY random()
                              LIMIT 100000""").df()
    pbar.update(1)  # æ›´æ–°é€²åº¦æ¢�

    # æŸ¥è©¢ç¬¬äºŒéƒ¨åˆ†æ•¸æ“š
    df_part2 = con.query(f"""SELECT *
                              FROM parquet_scan('{train_path}')
                              WHERE binds = 1
                              ORDER BY random()
                              LIMIT 100000""").df()
    pbar.update(1)  # æ›´æ–°é€²åº¦æ¢�

# å�ˆä½µå…©éƒ¨åˆ†æ•¸æ“š
df = pd.concat([df_part1, df_part2], ignore_index=True)

# éš¨æ©Ÿæ´—ç‰Œæ•¸æ“šï¼ˆfrac=1 è¡¨ç¤ºä¿�æŒ�å�Ÿå§‹å¤§å°�ï¼Œshuffle æ•´å€‹ DataFrameï¼‰
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# é—œé–‰é€£ç·š
con.close()


# # ç¢ºèª�æ•¸æ“š
print(df.head())


def smiles_to_topological_torsion_fingerprint(smiles, n_bits=2048):
    """
    å°‡ SMILES åˆ†å­�çµ�æ§‹è½‰æ�›ç‚ºæ‹“æ¨¸æ‰­è½‰æŒ‡ç´‹ã€‚
    :param smiles: åˆ†å­�çš„ SMILES è¡¨ç¤ºæ³•
    :param n_bits: æŒ‡ç´‹çš„ä½�å…ƒæ•¸ (é»˜èª�ç‚º 2048)
    :return: ä¸€å€‹ numpy æ•¸çµ„ï¼Œè¡¨ç¤ºæ‹“æ¨¸æ‰­è½‰æŒ‡ç´‹
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=int)
    else:
        generator = AllChem.GetTopologicalTorsionGenerator(fpSize=n_bits)
        return np.array(generator.GetFingerprint(mol), dtype=int)

# ä½¿ç”¨é€²åº¦æ¢�è½‰æ�› "molecule_smiles" æ¬„ä½�ç‚ºæ‹“æ¨¸æ‰­è½‰æŒ‡ç´‹
tqdm.pandas(desc="Transforming molecule_smiles to Topological Torsion Fingerprint")
df["molecule_smiles"] = df["molecule_smiles"].progress_apply(lambda x: smiles_to_topological_torsion_fingerprint(x))




# å°� protein æ¬„ä½�é€²è¡Œ One-Hot Encoding
protein_one_hot = pd.get_dummies(df["protein_name"], prefix="protein").astype(int)

# å�ˆä½µ One-Hot çµ�æ�œ
df_one_hot = pd.concat([df, protein_one_hot], axis=1)

# å�ˆä½µéœ€è¦�çš„æ¬„ä½�ï¼šmolecule_smiles, binds, å’Œç¶“é�� One-Hot Encoding çš„ protein
df_one_hot = pd.concat([df_one_hot[["id", "molecule_smiles", "binds"]], protein_one_hot], axis=1)


df_one_hot


# # å�¯é�¸ï¼šç§»é™¤å�Ÿå§‹ protein æ¬„ä½�
# df1_one_hot.drop("protein_name", axis=1, inplace=True)

# # æª¢è¦–è™•ç�†å¾Œæ•¸æ“š
# print(df_one_hot.head())

# åƒ…ä¿�ç•™éœ€è¦�çš„æ¬„ä½�
columns_to_keep = ["id", "molecule_smiles", "binds"] + protein_one_hot.columns.tolist()
df_filtered = df_one_hot[columns_to_keep]

# å„²å­˜è™•ç�†å¾Œçš„æ•¸æ“š
output_filename = "test_transformed_topological(100k,100k).parquet"
df_filtered.to_parquet(output_filename, index=False)



topolfile = '/kaggle/working/test_transformed_topological(100k,100k).parquet'
topol = pd.read_parquet(topolfile)
topol


#TRAINING


import numpy as np 
import pandas as pd 


data = pd.read_parquet('/kaggle/input/data-train/train_transformed_topological(150k50k).parquet')
data.head()


# è½‰æ�›æˆ� DataFrame
X_fingerprints_df = pd.DataFrame(data['molecule_smiles'].to_list())
X_fingerprints_df


# å�ˆä½µæ•¸å€¼ç‰¹å¾µ
X = pd.concat([X_fingerprints_df, data[['protein_BRD4', 'protein_HSA', 'protein_sEH']]], axis=1)
X


# è½‰æ�›æ¬„ä½�å��ç¨±éƒ½æ˜¯str
X.columns = X.columns.astype(str)


# è½‰æˆ�int8
int_cols = X.select_dtypes(include=['int64']).columns
for col in int_cols:
    X[col] = X[col].astype(np.int8)

X.dtypes


# è½‰æˆ�int8
data['binds'] = data['binds'].astype(np.int8)
data['binds'].dtypes


y = data['binds']
y


# åˆ†å‰²è³‡æ–™æˆ� train, validation, test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)  # 90% train+validation & 10% test


# -------------------- XGBoost --------------------
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from scipy.stats import randint

# å®šç¾©å�ƒæ•¸ç¯„åœ�
param_dist = {
    'n_estimators': [190, 200, 210],  # æ¨¹çš„æ•¸é‡�
    'max_depth': [7, 8, 9],  # æ¨¹çš„æœ€å¤§æ·±åº¦
    'learning_rate': [0.3, 0.5],  # å­¸ç¿’ç�‡
    'subsample': [1.0],  # è¨“ç·´é›†çš„éš¨æ©ŸæŠ½æ¨£æ¯”ä¾‹
    'colsample_bytree': [0.6, 0.7, 0.8],  # æ¯�æ£µæ¨¹çš„éš¨æ©ŸæŠ½æ¨£æ¯”ä¾‹
    'gamma': [0.1, 0.3],  # è¨­ç½®åˆ†è£‚çš„æœ€å°�æ��å¤±å‡½æ•¸
    'reg_alpha': [0, 0.01],  # L1æ­£å‰‡åŒ–
    'reg_lambda': [10],  # L2æ­£å‰‡åŒ–
}

# åˆ�å§‹åŒ– XGBoost æ¨¡å�‹
xgb = XGBClassifier(random_state=42)

# éš¨æ©Ÿæ�œå°‹
random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=70,  # é€²è¡Œ100æ¬¡éš¨æ©Ÿè©¦é©—
    cv=5,  # 5æŠ˜äº¤å�‰é©—è­‰
    scoring='f1',  # ä½¿ç”¨ AUC è©•ä¼°æ¨¡å�‹
    n_jobs=-1,  # ä½¿ç”¨æ‰€æœ‰å�¯ç”¨çš„è™•ç�†å™¨
    verbose=1,  # é¡¯ç¤ºè©³ç´°ä¿¡æ�¯
    random_state=42,
    refit=True  # ä½¿ç”¨æœ€ä½³å�ƒæ•¸é‡�æ–°è¨“ç·´
)

# è¨“ç·´æ¨¡å�‹
random_search.fit(X_train, y_train)

# è¼¸å‡ºæœ€ä½³å�ƒæ•¸å’Œ F1 åˆ†æ•¸
best_params = random_search.best_params_
best_score = random_search.best_score_

print(f"Best Parameters: {best_params}")
print(f"Best F1 Score: {best_score}")


# æœ€ä½³ param
from xgboost import XGBClassifier
# ä½¿ç”¨ **best_params è§£åŒ…å­—å…¸ä¸¦å‚³å…¥ XGBClassifier
xgb_model = XGBClassifier(**best_params)

# è¨“ç·´æ¨¡å�‹
xgb_model.fit(X_train, y_train)
'''
xgb_model = XGBClassifier(colsample_bytree=, gamma=, learning_rate=,
                          max_depth=, n_estimators=, reg_alpha=, reg_lambda=,
                          subsample=)
xgb_model.fit(X_train, y_train)
'''


#å„²å­˜æ¨¡å�‹
import pickle
with open("/kaggle/working/xgb_model_tp1505.bin", "wb") as f:
    pickle.dump(xgb_model, f)


# -------------------- LightGBM --------------------
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

# å»ºç«‹ LightGBM æ¨¡å�‹
lgb_clf = lgb.LGBMClassifier()

# å®šç¾©å�ƒæ•¸ç¯„åœ�
param_dist = {
    'num_leaves': [20, 31, 50],
    'learning_rate': [ 0.05, 0.1, 0.2],
    'n_estimators': [50, 100, 200],
    'boosting_type': ['gbdt', 'dart'],
    # 'feature_fraction': [0.6, 0.7, 0.8, 0.9],
    # 'bagging_fraction': [0.6, 0.7, 0.8],
    # 'bagging_freq': [1, 3, 5, 7]
}

# è¨­å®š RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=lgb_clf,
    param_distributions=param_dist,
    n_iter=50,  # éš¨æ©Ÿæ�œå°‹æ¬¡æ•¸
    scoring='f1',
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# åŸ·è¡Œå�ƒæ•¸æ�œå°‹
random_search.fit(X_train, y_train)

# è¼¸å‡ºæœ€ä½³å�ƒæ•¸å’Œ F1 åˆ†æ•¸
best_params = random_search.best_params_
best_score = random_search.best_score_

print(f"Best Parameters: {best_params}")
print(f"Best F1 Score: {best_score}")


# æœ€ä½³ param
#lgbm_model = lgb.LGBMClassifier(num_leaves=, n_estimators=, learning_rate=)
lgbm_model = lgb.LGBMClassifier(**best_params)
lgbm_model.fit(X_train, y_train)


#å„²å­˜æ¨¡å�‹
import pickle
with open("/kaggle/working/lgbm_model_tp1505.bin", "wb") as f:
    pickle.dump(lgbm_model, f)


# -------------------- Random Forest --------------------
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV

# å®šç¾©å�ƒæ•¸ç¯„åœ�
param_dist = {
    'n_estimators': randint(50, 200),  # éš¨æ©Ÿæ£®æ�—ä¸­çš„æ¨¹çš„æ•¸é‡�ï¼Œå¾�50åˆ°200
    'max_depth': [None, 10, 20],  # æ¯�æ£µæ¨¹çš„æœ€å¤§æ·±åº¦
    'min_samples_leaf': randint(1, 10),  # æ¯�æ£µæ¨¹çš„è‘‰å­�ç¯€é»�æ‰€éœ€çš„æœ€å°‘æ¨£æœ¬æ•¸
}

# å»ºç«‹éš¨æ©Ÿæ£®æ�—
rf = RandomForestClassifier(random_state=42)

# éš¨æ©Ÿæ�œç´¢
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,  # ä½¿ç”¨ä¸Šé�¢å®šç¾©çš„éš¨æ©Ÿå�ƒæ•¸ç¯„åœ�
    n_iter=50,  # è¨­å®šé€²è¡Œ100æ¬¡éš¨æ©Ÿè©¦é©—
    cv=5,  # ä½¿ç”¨5æŠ˜äº¤å�‰é©—è­‰
    scoring='f1',  # è©•ä¼°æŒ‡æ¨™ä½¿ç”¨F1åˆ†æ•¸
    n_jobs=-1,  # ä½¿ç”¨æ‰€æœ‰å�¯ç”¨çš„è™•ç�†å™¨é€²è¡Œä¸¦è¡Œé�‹ç®—
    verbose=1,   # é¡¯ç¤ºé�‹è¡Œé��ç¨‹ä¸­çš„è©³ç´°ä¿¡æ�¯
    random_state=42,
    refit=True  # ä½¿ç”¨æœ€ä½³å�ƒæ•¸é‡�è¨“ç·´æ¨¡å�‹
)

# è¨“ç·´æ¨¡å�‹
random_search.fit(X_train, y_train)

# è¼¸å‡ºæœ€ä½³å�ƒæ•¸å’Œå°�æ‡‰çš„ F1 åˆ†æ•¸
best_params = random_search.best_params_
best_score = random_search.best_score_

print(f"Best Parameters: {best_params}")
print(f"Best F1 Score: {best_score}")


# æœ€ä½³ param
rf_model = RandomForestClassifier(**best_params)
rf_model.fit(X_train, y_train)


#å„²å­˜æ¨¡å�‹
import pickle
with open("/kaggle/working/rf_model_tp1505.bin", "wb") as f:
    pickle.dump(rf_model, f)


# è¼‰å…¥æ¨¡å�‹
import pickle
l =  open("/kaggle/input/lgbm_model_tp1505.bin", "rb")
lgbm_model =  pickle.load(l)
r =  open("/kaggle/input/rf_model_tp1505.bin", "rb")
rf_model =  pickle.load(r)
xg =  open("/kaggle/input/xgb_model_tp1505.bin", "rb")
xgb_model =  pickle.load(xg)


# -------------------- Voting --------------------
from sklearn.ensemble import VotingClassifier

# å»ºç«‹ VotingClassifier
voting_clf = VotingClassifier(
    estimators=[('rf', rf_model), ('lgb', lgbm_model), ('xgb', xgb_model)],
    voting='soft'  # ä½¿ç”¨æ©Ÿç�‡åŠ æ¬ŠæŠ•ç¥¨
)

# votingï¼š
# 'hard'ï¼šæ ¹æ“šæœ€å¤šæ•¸çš„é �æ¸¬é¡�åˆ¥ä¾†æ±ºå®šæœ€çµ‚çµ�æ�œï¼ˆé�©ç”¨æ–¼åˆ†é¡�å™¨ç„¡æ©Ÿç�‡è¼¸å‡ºæ™‚ï¼‰ã€‚
# 'soft'ï¼šæ ¹æ“šæ‰€æœ‰åˆ†é¡�å™¨çš„æ©Ÿç�‡å¹³å�‡ä¾†æ±ºå®šæœ€çµ‚çµ�æ�œï¼ˆéœ€è¦�åˆ†é¡�å™¨æ”¯æ�´ predict_proba()ï¼‰ã€‚

# è¨“ç·´æ¨¡å�‹
voting_clf.fit(X_train, y_train)

# é �æ¸¬
y_pred = voting_clf.predict(X_test)

# è¨ˆç®—æº–ç¢ºåº¦
print("Voting accuracy (training)ï¼š", voting_clf.score(X_train, y_train))
print("Voting accuracy (test)ï¼š", voting_clf.score(X_test, y_test))


#å„²å­˜æ¨¡å�‹
import pickle
with open("/kaggle/working/voting_model_tp1505.bin", "wb") as f:
    pickle.dump(voting_clf, f)


from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
# LightGBM é �æ¸¬
y_pred_lgb = lgbm_model.predict(X_test)
y_pred_lgb_binary = (y_pred_lgb >= 0.5).astype(int)
y_proba_lgb = lgbm_model.predict_proba(X_test)[:, 1]

# XGBoost é �æ¸¬
y_pred_xgb = xgb_model.predict(X_test)
y_pred_xgb_binary = (y_pred_xgb >= 0.5).astype(int)
y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

# Random Forest é �æ¸¬
y_pred_rf = rf_model.predict(X_test)
y_pred_rf_binary = (y_pred_rf >= 0.5).astype(int)
y_proba_rf = rf_model.predict_proba(X_test)[:, 1]


# è©•ä¼°å‡½å¼�
def evaluate_model(name, y_test, y_pred_binary, y_proba):
    print(f"ğŸ”¹ {name} Scores")
    print(f"F1 Score: {f1_score(y_test, y_pred_binary):.4f}")
    print(f"AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred_binary):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred_binary):.4f}")
    print("="*50)

# è¨ˆç®—ä¸‰å€‹æ¨¡å�‹çš„æŒ‡æ¨™
evaluate_model("LightGBM", y_test, y_pred_lgb_binary, y_proba_lgb)
evaluate_model("Random Forest", y_test, y_pred_rf_binary, y_proba_rf)
evaluate_model("XGBoost", y_test, y_pred_xgb_binary, y_proba_xgb)


# Voting é �æ¸¬
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

# é �æ¸¬æ©Ÿç�‡ (å�ªé�©ç”¨æ–¼ 'soft' voting)
y_proba = voting_clf.predict_proba(X_test)[:, 1]  # å�–å‡ºæ­£é¡�åˆ¥çš„æ©Ÿç�‡

# è½‰æ�›æˆ� 0/1 é �æ¸¬æ¨™ç±¤ï¼ˆæ ¹æ“š 0.5 é–¾å€¼ï¼‰
y_pred_binary = (y_proba >= 0.5).astype(int)

# è¨ˆç®—è©•ä¼°æŒ‡æ¨™
print("F1 Score:", f1_score(y_test, y_pred_binary))
print("AUC Score:", roc_auc_score(y_test, y_proba))  # AUC éœ€è¦�æ©Ÿç�‡è¼¸å‡º
print("Recall:", recall_score(y_test, y_pred_binary))
print("Precision:", precision_score(y_test, y_pred_binary))

