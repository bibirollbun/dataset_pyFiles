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


from tqdm import tqdm
from itertools import combinations
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
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
original = pd.read_csv("//kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")


original_copy = original.copy()
for k in range(6):
    original = pd.concat([original,original_copy],axis=0)


def feature_eng(df):
    numerical_features = [col for col in df.select_dtypes(include=['int64', 'float64']).columns 
                      if col != 'id']
    for col in numerical_features:
        df[f'{col}_Binned'] = df[col].astype(str).astype('category')
    return df

train = feature_eng(train)
test = feature_eng(test)
original = feature_eng(original)


def rename_temperature_column(df):
    df = df.rename(columns={'Temparature': 'Temperature'})
    print("Column name corrected from 'Temparature' to 'Temperature'")
    return df
    
train = rename_temperature_column(train)
test = rename_temperature_column(test)
original = rename_temperature_column(original)


cat_cols = [col for col in train.select_dtypes(include=['object', 'category']).columns 
            if col != "Fertilizer Name"]

for i in cat_cols:
    label_enc = LabelEncoder()
    train[i] = label_enc.fit_transform(train[i])
    original[i] = label_enc.fit_transform(original[i])
    test[i] = label_enc.transform(test[i])

fer_label_enc = LabelEncoder()
train["Fertilizer Name"] = fer_label_enc.fit_transform(train["Fertilizer Name"])
original["Fertilizer Name"] = fer_label_enc.fit_transform(original["Fertilizer Name"])
for col in cat_cols:
    train[col] = train[col].astype("category")
    test[col] = test[col].astype("category")
    original[col] = original[col].astype("category")


X = train.drop(columns=["id", "Fertilizer Name"])
y = train["Fertilizer Name"]
X_test = test.drop(columns=["id"])

X_original = original.drop(columns=["Fertilizer Name"])
y_original = original["Fertilizer Name"]


params = {
        'objective': 'multi:softprob',  
        'num_class': len(np.unique(y)), 
        'max_depth': 10,
        'learning_rate': 0.03,
        'subsample': 0.8,
        'max_bin': 128,
        'colsample_bytree': 0.3, 
        'colsample_bylevel': 1,  
        'colsample_bynode': 1,  
        'tree_method': 'hist',  
        'random_state': 42,
        'eval_metric': 'mlogloss',
        'device': "cuda",
        'enable_categorical':True,
        'n_estimators':10000,
        'early_stopping_rounds':50,
    }


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
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof = np.zeros(shape = (len(train) ,y.nunique()))
pred_prob = np.zeros(shape = (len(test),y.nunique()))

xgb_model = XGBClassifier(**params)

map3_scores = []

for i, (train_idx, valid_idx) in enumerate(skf.split(X,y)):
    print('#' * 15, i+1, '#' *15)
    x_train, x_valid = X.iloc[train_idx],X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx],y.iloc[valid_idx]

    x_train = pd.concat([x_train,X_original], axis=0, ignore_index=True)
    y_train = pd.concat([y_train,y_original], axis=0, ignore_index=True)
    
    xgb_model.fit(
        x_train, 
        y_train, 
        eval_set=[(x_train, y_train), (x_valid, y_valid)], 
        verbose=100,
    )
    oof[valid_idx] = xgb_model.predict_proba(x_valid)
    pred_prob += xgb_model.predict_proba(X_test)  
    top_3_preds = np.argsort(oof[valid_idx], axis=1)[:, -3:][:, ::-1]  
    actual = [[label] for label in y_valid]
    map3_score = mapk(actual, top_3_preds)
    map3_scores.append(map3_score)  # Store the score
    print(f"âœ… FOLD {i+1}: MAP@3 Score: {map3_score:.5f}")

avg_map3 = np.mean(map3_scores)
print(f"\nðŸŽ¯ Average MAP@3 Score across all folds: {avg_map3:.5f}")


top_3_preds = np.argsort(pred_prob, axis=1)[:, -3:][:, ::-1]
top_3_labels = fer_label_enc.inverse_transform(top_3_preds.ravel()).reshape(top_3_preds.shape)
df_sub = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")
submission = pd.DataFrame({
    'id': df_sub['id'],
    'Fertilizer Name': [' '.join(row) for row in top_3_labels]
})
submission.to_csv('submission.csv', index=False)
print("âœ… Submission file saved as 'submission.csv'")

