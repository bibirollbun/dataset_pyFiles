import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
# import seaborn as sns
import math
from typing import Optional
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
# import kaggle
import pandas as pd
import zipfile
import io
import os


def get_train_test_data():
    # api = kaggle.KaggleApi()
    # api.authenticate()

    # competition = "playground-series-s5e8"
    # download_path = "./temp_data"
    # os.makedirs(download_path, exist_ok=True)

    # api.competition_download_files(competition, path=download_path)
    # with zipfile.ZipFile(f"{download_path}/{competition}.zip", "r") as zip_file:
    #     with zip_file.open("train.csv") as f:
    #         train = pd.read_csv(f)
    #     with zip_file.open("test.csv") as f:
    #         test = pd.read_csv(f)

    # dataset = "sushant097/bank-marketing-dataset-full"
    # api.dataset_download_files(dataset, path=download_path)
    # with zipfile.ZipFile(f"{download_path}/bank-marketing-dataset-full.zip", "r") as zip_file:
    #     with zip_file.open("bank-full.csv") as f:
    #         original = pd.read_csv(f, sep=';')
    train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
    test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
    original = pd.read_csv('/kaggle/input/bank-marketing-dataset-full/bank-full.csv', sep = ';')

    print("OG Train shape:", train.shape)
    print("Test shape:", test.shape)

    original["y"] = original["y"].map({"no": 0, "yes": 1})
    train.drop(columns="id", inplace=True)
    original = original[train.columns]
    train = pd.concat([train, original], ignore_index=True)

    print("Merged shape:", train.shape)
    print("Merged target distribution:\n", train["y"].value_counts())

    return train, test



train, test = get_train_test_data()


train.isna().sum()


# X = train.drop(columns=["y"])
# y = train["y"]
# X_test = test.drop(columns=["id"])

# categorical_features = X.select_dtypes(include="object").columns.tolist()
# cat_indices = [X.columns.get_loc(col) for col in categorical_features]

# model = CatBoostClassifier(
#     iterations=1000,
#     learning_rate=0.3,
#     depth=10,
#     eval_metric='AUC',
#     cat_features=cat_indices,
#     verbose=100,
#     random_seed=42
# )

# model.fit(X, y)


# y_pred_proba = model.predict_proba(X_test)[:, 1] 

# submission = pd.DataFrame({
#     "id": test["id"],
#     "target": y_pred_proba
# })

# submission.to_csv("submission_base.csv", index=False)


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import StackingClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.base import clone


def get_cat_indices(X):
    return [i for i, col in enumerate(X.columns) if pd.api.types.is_categorical_dtype(X[col])]

def convert_object_to_category(df):
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype('category')
    return df

def preprocess(X):
    X = convert_object_to_category(X)
    return X



RANDOM_STATE = 42
N_SPLITS = 5


X = train.drop(columns='y')
y = train['y']
X = preprocess(X)
X_base, X_final_val, y_base, y_final_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
cat_indices = get_cat_indices(X_base)


oof_cat = np.zeros(len(X_base))
oof_xgb = np.zeros(len(X_base))
oof_lgb = np.zeros(len(X_base))
val_preds_cat = np.zeros(len(X_final_val))
val_preds_xgb = np.zeros(len(X_final_val))
val_preds_lgb = np.zeros(len(X_final_val))


kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
for fold, (train_idx, val_idx) in enumerate(kf.split(X_base, y_base)):
    print(f"\n--- Fold {fold + 1} ---")
    X_tr, y_tr = X_base.iloc[train_idx], y_base.iloc[train_idx]
    X_val, y_val = X_base.iloc[val_idx], y_base.iloc[val_idx]
    
    cat_model = CatBoostClassifier(
        iterations=300, learning_rate=0.3, depth=10,
        class_weights=[1, 6.7],
        eval_metric='AUC',
        cat_features=cat_indices,
        verbose=0,
        random_seed=RANDOM_STATE
    )
    cat_model.fit(X_tr, y_tr)
    oof_cat[val_idx] = cat_model.predict_proba(X_val)[:, 1]
    val_preds_cat += cat_model.predict_proba(X_final_val)[:, 1] / N_SPLITS
    
    xgb_model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        tree_method="hist",
        enable_categorical=True,
        max_depth=6,
        objective='binary:logistic',
        eval_metric='auc',
        scale_pos_weight=6.7,
        use_label_encoder=False,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    xgb_model.fit(X_tr, y_tr)
    oof_xgb[val_idx] = xgb_model.predict_proba(X_val)[:, 1]
    val_preds_xgb += xgb_model.predict_proba(X_final_val)[:, 1] / N_SPLITS
    
    from lightgbm import LGBMClassifier
    lgb_model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        objective='binary',
        metric='auc',
        class_weight='balanced',
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        verbose=-1,
        random_state=RANDOM_STATE
    )
    lgb_model.fit(X_tr, y_tr)
    oof_lgb[val_idx] = lgb_model.predict_proba(X_val)[:, 1]
    val_preds_lgb += lgb_model.predict_proba(X_final_val)[:, 1] / N_SPLITS

stacked_X_train = np.vstack([oof_cat, oof_xgb, oof_lgb]).T
stacked_X_val = np.vstack([val_preds_cat, val_preds_xgb, val_preds_lgb]).T


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
stacked_X_train_scaled = scaler.fit_transform(stacked_X_train)
stacked_X_val_scaled = scaler.transform(stacked_X_val)


from sklearn.neural_network import MLPClassifier
meta_model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    alpha=0.001,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=1000,
    early_stopping=True,
    validation_fraction=0.2,
    n_iter_no_change=20,
    random_state=RANDOM_STATE
)

from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y_base), y=y_base)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

meta_model.fit(stacked_X_train_scaled, y_base)


from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
meta_val_preds = meta_model.predict_proba(stacked_X_val_scaled)[:, 1]
roc_auc = roc_auc_score(y_final_val, meta_val_preds)
prauc = average_precision_score(y_final_val, meta_val_preds)
print(f"\nMeta ROC AUC: {roc_auc:.5f}")
print(f"Meta PR AUC:  {prauc:.5f}")

cat_auc = roc_auc_score(y_final_val, val_preds_cat)
xgb_auc = roc_auc_score(y_final_val, val_preds_xgb)
lgb_auc = roc_auc_score(y_final_val, val_preds_lgb)
print(f"\nIndividual Model AUCs:")
print(f"CatBoost: {cat_auc:.5f}")
print(f"XGBoost:  {xgb_auc:.5f}")
print(f"LightGBM: {lgb_auc:.5f}")

y_val_bin = (meta_val_preds >= 0.5).astype(int)
print("\nClassification Report:\n", classification_report(y_final_val, y_val_bin))


X_full = train.drop(columns='y')
y_full = train['y']
X_full = preprocess(X_full)
test = preprocess(test)

for col in X_full.select_dtypes(include='category').columns:
    test[col] = test[col].astype('category')

test = test[X_full.columns]

cat_indices = get_cat_indices(X_full)

test_preds_cat = np.zeros(len(test))
test_preds_xgb = np.zeros(len(test))
test_preds_lgb = np.zeros(len(test))

kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_full, y_full)):
    print(f"\n--- Full Training Fold {fold + 1} ---")
    X_tr, y_tr = X_full.iloc[train_idx], y_full.iloc[train_idx]

    cat_model = CatBoostClassifier(
        iterations=300, learning_rate=0.3, depth=10,
        class_weights=[1, 6.7],
        eval_metric='AUC',
        cat_features=cat_indices,
        verbose=0,
        random_seed=RANDOM_STATE
    )
    cat_model.fit(X_tr, y_tr)
    test_preds_cat += cat_model.predict_proba(test)[:, 1] / N_SPLITS

    xgb_model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        tree_method="hist",
        enable_categorical=True,
        max_depth=6,
        objective='binary:logistic',
        eval_metric='auc',
        scale_pos_weight=6.7,
        use_label_encoder=False,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    xgb_model.fit(X_tr, y_tr)
    test_preds_xgb += xgb_model.predict_proba(test)[:, 1] / N_SPLITS

    lgb_model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        objective='binary',
        metric='auc',
        class_weight='balanced',
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        verbose=-1,
        random_state=RANDOM_STATE
    )
    lgb_model.fit(X_tr, y_tr)
    test_preds_lgb += lgb_model.predict_proba(test)[:, 1] / N_SPLITS

stacked_test = np.vstack([test_preds_cat, test_preds_xgb, test_preds_lgb]).T

stacked_test_scaled = scaler.transform(stacked_test)

final_test_preds = meta_model.predict_proba(stacked_test_scaled)[:, 1]



test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
ids = test['id']


submission = pd.DataFrame({
    'id': ids,
    'y': final_test_preds
})
submission.to_csv('submission.csv', index=False)

