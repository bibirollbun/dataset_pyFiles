# %load_ext cudf.pandas

# import cudf
# import cupy as cp
# from cuml.preprocessing.TargetEncoder import TargetEncoder
from tqdm import tqdm
from itertools import combinations
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
import os
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
original = pd.read_csv("/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")


train = pd.concat([train, original], axis=0, ignore_index=True)
train.head(3)


def rename_temperature_column(df):
    df = df.rename(columns={'Temparature': 'Temperature'})
    print("Column name corrected from 'Temparature' to 'Temperature'")
    return df
    
train = rename_temperature_column(train)
test = rename_temperature_column(test)


cat_cols = [col for col in train.select_dtypes(include=['object', 'category']).columns 
            if col != "Fertilizer Name"]

for i in cat_cols:
    label_enc = LabelEncoder()
    train[i] = label_enc.fit_transform(train[i])
    test[i] = label_enc.transform(test[i])

fer_label_enc = LabelEncoder()
train["Fertilizer Name"] = fer_label_enc.fit_transform(train["Fertilizer Name"])

for col in cat_cols:
    train[col] = train[col].astype("category")
    test[col] = test[col].astype("category")

##############################################Dataset#########################################

X = train.drop(columns=["id", "Fertilizer Name"])
y = train["Fertilizer Name"]
X_test = test.drop(columns=["id"])


from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold

def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        p = p[:k]
        score = 0.0
        hits = 0
        seen = set()
        for i, pred in enumerate(p):
            if pred in a and pred not in seen:
                hits += 1
                score += hits / (i + 1.0)
                seen.add(pred)
        return score / min(len(a), k)
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

FOLDS = 5
#skf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof = np.zeros(shape = (len(train) ,y.nunique()))
pred_prob = np.zeros(shape = (len(test),y.nunique()))

xgb_model = XGBClassifier(
    max_depth=12,
    colsample_bytree=0.467,
    subsample=0.86,
    n_estimators=4000,
    learning_rate=0.03,
    gamma=0.26,
    max_delta_step=4,
    reg_alpha=2.7,
    reg_lambda=1.4,
    early_stopping_rounds=100,
    objective='multi:softprob',
    random_state=13,
    enable_categorical=True,
    tree_method='hist',     
    device='cuda'           
)

for i, (train_idx, valid_idx) in enumerate(skf.split(X,y)):
    print('#' * 15, i+1, '#' *15)
    x_train, x_valid = X.iloc[train_idx],X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx],y.iloc[valid_idx]

    xgb_model.fit(x_train,y_train, eval_set=[(x_valid,y_valid)],verbose = 0)
    oof[valid_idx] = xgb_model.predict_proba(x_valid)
    pred_prob +=xgb_model.predict_proba(X_test)

    top_3_preds = np.argsort(oof[valid_idx], axis=1)[:, -3:][:, ::-1]  
    actual = [[label] for label in y_valid]
    map3_score = mapk(actual, top_3_preds)
    print(f"✅ FOLD {i+1}: MAP@3 Score: {map3_score:.5f}")


top_3_preds = np.argsort(pred_prob, axis=1)[:, -3:][:, ::-1]
top_3_labels = fer_label_enc.inverse_transform(top_3_preds.ravel()).reshape(top_3_preds.shape)
df_sub = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")
submission = pd.DataFrame({
    'id': df_sub['id'],
    'Fertilizer Name': [' '.join(row) for row in top_3_labels]
})
submission.to_csv('submission.csv', index=False)
print("✅ Submission file saved as 'submission.csv'")

