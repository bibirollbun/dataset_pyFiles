import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


ensemble1 = pd.read_csv('/kaggle/input/models/ensemble-1.csv')
ensemble2 = pd.read_csv('/kaggle/input/models/ensemble-2.csv')
xgb = pd.read_csv('/kaggle/input/models/xgb.csv')
light_catboost = pd.read_csv('/kaggle/input/models/catboost_submission.csv')
lightautoml = pd.read_csv('/kaggle/input/models/good.csv')



# Assume the target column is named 'Calories'
target_col = 'Calories'  # Change this if your target column has a different name
id_col = ensemble1.columns[0]  # Assuming first column is ID


models = {
    'Ensemble1': ensemble1[target_col],
    'Ensemble2': ensemble2[target_col],
    'XGBoost': xgb[target_col],
    'Light_CatBoost': light_catboost[target_col],
    'LightAutoML': lightautoml[target_col]
}


pred_df = pd.DataFrame(models)
corr = pred_df.corr()
print("Correlation between models:")
print(corr)


weights = {
    'Ensemble1': 0.25,
    'Ensemble2': 0.25,
    'XGBoost': 0.15,
    'Light_CatBoost': 0.20,
    'LightAutoML': 0.15
}


# Make sure weights sum to 1
weight_sum = sum(weights.values())
weights = {k: v/weight_sum for k, v in weights.items()}


# Calculate weighted average prediction
final_submission = ensemble1[[id_col]].copy()
final_submission[target_col] = 0


for model_name, predictions in models.items():
    final_submission[target_col] += predictions * weights[model_name]



# 4. Save final submission
final_submission.to_csv('submission.csv', index=False)
print("Final submission created successfully!")




