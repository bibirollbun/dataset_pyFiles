import pandas as pd 
import numpy as np 
import os 
import time
import logging 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from category_encoders import TargetEncoder

from tqdm.auto import tqdm
from itertools import combinations
import warnings
warnings.simplefilter('ignore')


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")
all_data = pd.concat([train, test], axis=0)
all_data = all_data.reset_index(level=0, drop=True)


# æ•°æ�®ç¼–ç �
all_data['Sex'] = all_data['Sex'].map({'male': -1, 'female': 1})

features = ['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
numerical_features = ["Age","Height","Weight","Duration","Heart_Rate","Body_Temp"]

from sklearn.preprocessing import PolynomialFeatures

# 1.å¤šé¡¹å¼�ç‰¹å¾�
poly_data = all_data[features]
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_transformed = poly.fit_transform(poly_data)
poly_feature_names = poly.get_feature_names_out(features)
poly_feature_names = [name.replace(' ', '_') for name in poly_feature_names]
poly_df = pd.DataFrame(poly_transformed, columns=poly_feature_names, index=all_data.index)
poly_df = poly_df.drop(columns=features) # åˆ é™¤é‡�å¤�å�Ÿå§‹ç‰¹å¾�
all_data = pd.concat([all_data, poly_df], axis=1)

# 2.Durationä¸�å…¶ä»–ç‰¹å¾�äº¤äº’ä»¥å�Šä¸‰æ¬¡æ–¹
non_original_features = [col for col in all_data.columns if col not in features and col != 'Calories']
for feature in non_original_features:
    interaction_feature_name = f'Duration_{feature}'
    all_data[interaction_feature_name] = all_data['Duration'] * all_data[feature]
    
# 3.æ¯”ä¾‹ç‰¹å¾�
for i, feature1 in enumerate(numerical_features):
    for feature2 in numerical_features[i+1:]:
        ratio_feature_name = f"{feature1}_to_{feature2}_ratio"
        all_data[ratio_feature_name] = all_data[feature1] / all_data[feature2]

# æ£€æŸ¥ã€�è¾“å‡ºå¹¶åˆ é™¤å­˜åœ¨infçš„ç‰¹å¾�
inf_columns = all_data.columns[all_data.isin([np.inf, -np.inf]).any()]
if len(inf_columns) > 0:
    print("å­˜åœ¨infçš„ç‰¹å¾�:")
    for col in inf_columns:
        print(col)
    all_data = all_data.drop(columns=inf_columns)
    print("å·²åˆ é™¤å­˜åœ¨infçš„ç‰¹å¾�")

train = all_data[all_data.index < 750000]
test = all_data[all_data.index >= 750000]


X = train.drop(columns=["id", "Calories"])
y = np.log1p(train["Calories"])
X_test = test.drop(columns=["id", "Calories"])


train.describe()


FOLDS = 5
FEATURES = X.columns.tolist()

# KFold setup
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

# Arrays to store predictions
oof = np.zeros(len(train))
pred = np.zeros(len(test))

# Start CV loop
for i, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
    print(f"\n{'#'*10} Fold {i+1} {'#'*10}")
    
    x_train = X.iloc[train_idx].copy()
    y_train = y.iloc[train_idx]
    x_valid = X.iloc[valid_idx].copy()
    y_valid = y.iloc[valid_idx]
    x_test = X_test.copy()

    # No categorical target encoding in this dataset, but you can add if needed
    
    start = time.time()

    # Train model
    model = XGBRegressor(
        device="cuda" if XGBRegressor().get_params().get("device") == "cuda" else "cpu",
        max_depth=10,
        colsample_bytree=0.7,
        subsample=0.9,
        n_estimators=2000,
        learning_rate=0.02,
        gamma=0.01, 
        max_delta_step=2,
        early_stopping_rounds=100,
        eval_metric="rmse",
        enable_categorical=True,
        n_jobs=-1
    )

    model.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],
        verbose=100
    )

    # Predict OOF and test
    oof[valid_idx] = model.predict(x_valid)
    pred += model.predict(x_test)

    rmse = np.sqrt(mean_squared_error(y_valid, oof[valid_idx]))
    print(f"Fold {i+1} RMSE: {rmse:.4f}")
    print(f"Feature engineering & training time: {time.time() - start:.1f} sec")

# Average test predictions
pred /= FOLDS

# Final RMSE
full_rmse = np.sqrt(mean_squared_error(y, oof))
print(f"\nFinal CV RMSE: {full_rmse:.4f}")


y_preds = np.expm1(pred)
print('predict mean :',y_preds.mean())
print('predict median :',np.median(y_preds))

y_preds = np.clip(y_preds,1,314)
print('predict mean after clip:',y_preds.mean())
print('predict median after clip:',np.median(y_preds))

submission["Calories"] = y_preds
submission.to_csv("submission.csv", index=False)
submission.head()

