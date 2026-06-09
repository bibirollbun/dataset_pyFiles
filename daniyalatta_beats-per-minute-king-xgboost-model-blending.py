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


train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")


BeatsPerMinute_global_avg = train['BeatsPerMinute'].mean()
print(BeatsPerMinute_global_avg)


numerical_features = ['RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality',
       'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore',
       'TrackDurationMs', 'Energy']

def add_feature_cross_terms(df, numerical_features):
    df_new = df.copy()
    
    for i in range(len(numerical_features)):
        for j in range(i + 1, len(numerical_features)):  
            feature1 = numerical_features[i]
            feature2 = numerical_features[j]
            cross_term_name = f"{feature1}_x_{feature2}"
            df_new[cross_term_name] = df_new[feature1] * df_new[feature2]

    return df_new

train = add_feature_cross_terms(train, numerical_features)
test = add_feature_cross_terms(test, numerical_features)



num_features = train.select_dtypes(include='number')


X = train.drop(columns=["id", "BeatsPerMinute"])
y = train["BeatsPerMinute"]-BeatsPerMinute_global_avg
X_test = test.drop(columns=["id"])


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
        max_depth=0,
        colsample_bytree=0.9,
        subsample=0.9,
        n_estimators=5000,
        learning_rate=0.01,
        gamma=10.0, 
        max_delta_step=2,
        early_stopping_rounds=100,
        eval_metric="rmse",
        enable_categorical=True
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


pred1 = pred + BeatsPerMinute_global_avg
print('predict mean :',pred1.mean())
print('predict median :',np.median(pred1))

y_pred_after = np.clip(pred1, 46.718, 206.037)
print('predict mean after clip:',y_pred_after.mean())
print('predict median after clip:',np.median(y_pred_after))

submission["BeatsPerMinute"] = y_pred_after
submission.to_csv("submissionTest.csv", index=False)
submission.head()


catsub=pd.read_csv('/kaggle/input/musicsubmission2-0/CatSubmission.csv')
catsub
BeatsPerMinute_global_avg = train['BeatsPerMinute'].mean()
print(BeatsPerMinute_global_avg)
pred1 = (pred + BeatsPerMinute_global_avg)/2
pred2 = 0.7 * catsub["BeatsPerMinute"] + 0.3 * pred1
display(pred2)
print('predict mean :',pred2.mean())
print('predict median :',np.median(pred2))

y_pred_after = np.clip(pred2, 46.718, 206.037)
print('predict mean after clip:',y_pred_after.mean())
print('predict median after clip:',np.median(y_pred_after))
submission["BeatsPerMinute"] = y_pred_after
submission.to_csv("submission.csv", index=False)
submission.head()




