
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import xgboost as xgb
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from scipy.stats import zscore
from catboost import CatBoostRegressor
from tqdm import tqdm
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv").set_index("id")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv").set_index("id")
sex_mean = df.groupby('Sex')['Calories'].mean()
df['Sex'] = df['Sex'].map(sex_mean)
test_df['Sex'] = test_df['Sex'].map(sex_mean)

# Remove outliers
features = df.drop(columns=["Calories"]).select_dtypes(include=[np.number])
z_scores = np.abs((features - features.mean()) / features.std())
outliers = (z_scores > 3).any(axis=1)
Q1, Q3 = df["Calories"].quantile([0.25, 0.75])
IQR = Q3 - Q1
outliers |= (df["Calories"] < (Q1 - 1.5 * IQR)) | (df["Calories"] > (Q3 + 1.5 * IQR))
df = df[~outliers]
X = df.drop("Calories", axis=1)
y = np.log1p(df["Calories"])  # log-transform
dtest = xgb.DMatrix(test_df)



n_models = 10 
base_models = []
meta_data = []

print("ðŸš€ Training base models & collecting OOB predictions...")
for i in tqdm(range(n_models)):
    # Bootstrap sample
    X_sample, y_sample = resample(X, y, replace=True, n_samples=len(X), random_state=i)
    selected_idx = set(X_sample.index)
    oob_idx = list(set(X.index) - selected_idx)

    model = xgb.train(params = {
    'max_depth': 10,
    'learning_rate': 0.0049,
    'subsample': 0.91,
    'colsample_bytree': 0.86,
    'gamma': 0.0014,
    'reg_alpha': 0.025,
    'reg_lambda': 0.0106,
    'min_child_weight': 7,
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'tree_method': 'gpu_hist',   
    'gpu_id': 0                  
    }
, dtrain=xgb.DMatrix(X_sample, y_sample), num_boost_round=2000)

    base_models.append(model)

    # Predict on OOB samples  ->for meta model  data  
    if oob_idx:
        X_oob = X.loc[oob_idx]
        y_oob = y.loc[oob_idx]
        preds_oob = model.predict(xgb.DMatrix(X_oob))

        meta_data.append(pd.DataFrame({
            "id": X_oob.index,
            f"pred_model_{i}": preds_oob
        }).set_index("id"))

# Merge all OOB predictions by outer join (union)  
print("ðŸ”— Merging OOB predictions...")
meta_df = pd.concat(meta_data, axis=1)
meta_df = meta_df.groupby(meta_df.index).first() 
X_meta = X.loc[meta_df.index].copy()  
y_meta = y.loc[meta_df.index]       

# Combine original features + predictions from base models
X_meta_final = pd.concat([X_meta, meta_df], axis=1).fillna(0)




# Train meta-model
print("ðŸŽ¯ Training meta-model (CatBoost)...")
meta_model = CatBoostRegressor(
    iterations=2000,
    learning_rate=0.05,
    depth=6,
    random_state=42,
    task_type="GPU", 
    verbose=100
)
meta_model.fit(X_meta_final, y_meta)



# Predict on test set using stacking
print("ðŸ“¦ Predicting on test set...")
X_test_meta = test_df.copy()

for i, model in enumerate(base_models):
    X_test_meta[f"pred_model_{i}"] = model.predict(dtest)

final_log_preds = meta_model.predict(X_test_meta)
final_preds = np.expm1(final_log_preds)
submission = pd.DataFrame({
    "id": test_df.index,
    "Calories": final_preds
})
submission.to_csv("submission.csv", index=False)
print(" Submission saved ")





