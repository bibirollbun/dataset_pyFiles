# Go to Add-ons > Install Dependencies to install this into the environment
!pip install -q autogluon


import os
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from autogluon.tabular import TabularPredictor
import polars as pl
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split


# Autogluon configuration. Automatically detects if we are using an interactive notebook, and use lower defaults when debugging

def is_interactive_session():
    return os.environ.get('KAGGLE_KERNEL_RUN_TYPE','') == 'Interactive'

is_interactive_session()

config = {
    "autogluon_time": 60*60*0.2,
    "autogluon_presets": "best_quality",
    #"reduce_features": 0, # Set to >0 to use only the first n features
    "tail_rows": 0 # Set to >0 to use only the last n rows in the file
    
}

if is_interactive_session():
    print("Interactive session")
    config["autogluon_time"] = 100
    #config["reduce_features"] = 200
    config["autogluon_presets"] = "medium_quality"
    config["tail_rows"] = 2000
    print(config)
else:
    print("running as job")
    print(config)


train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
sample_sub = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')



train.head()


from autogluon.tabular import TabularPredictor



# ğŸ§¹ Step 2: Custom Feature Engineering Transformer
class FertilizerFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.feature_names_ = list(X.columns) + [
            'N_P_ratio', 'N_K_ratio', 'P_K_ratio',
            'Temp_Humidity', 'Soil_Crop'
        ]
        return self

    def transform(self, X):
        X = X.copy()
        X['N_P_ratio'] = X['Nitrogen'] / (X['Phosphorous'] + 1)
        X['N_K_ratio'] = X['Nitrogen'] / (X['Potassium'] + 1)
        X['P_K_ratio'] = X['Phosphorous'] / (X['Potassium'] + 1)
        X['Temp_Humidity'] = X['Temparature'] * X['Humidity']
        X['Soil_Crop'] = X['Soil Type'] + '_' + X['Crop Type']
        return X[self.feature_names_]


# ğŸ�—ï¸� Step 4: Build sklearn-style pipeline
feature_engineering = ColumnTransformer(transformers=[
    ('fertilizer_features', FertilizerFeatureEngineer(),  train.drop(columns='Fertilizer Name').columns),
])

pipeline = Pipeline(steps=[
    ('feature_eng', feature_engineering)
])



# Use it directly in pipeline
pipeline = Pipeline([
    ('fertilizer_features', FertilizerFeatureEngineer())
])

# Fit and transform
X_train = pipeline.fit_transform(train.drop(columns='Fertilizer Name'))
X_train['Fertilizer Name'] = train['Fertilizer Name']

# Same for test
X_test = pipeline.transform(test)
X_train


X_test


label = 'Fertilizer Name'
predictor = TabularPredictor(label="Fertilizer Name", 
                            eval_metric="log_loss").\
                fit(
                            X_train,
                            presets=config["autogluon_presets"],
                            time_limit=config["autogluon_time"])



# ğŸ”® Step 7: Predict top 3 fertilizers
probs = predictor.predict_proba(X_test)

top3 = probs.apply(lambda row: ' '.join(row.nlargest(3).index), axis=1)


# from sklearn.metrics import label_ranking_average_precision_score
# import numpy as np

# # Ground truth as binary indicator matrix
# y_true = pd.get_dummies(test['Fertilizer Name']).values
# y_score = probs[test.columns[1:]]  # drop 'id'

# map3 = label_ranking_average_precision_score(y_true, y_score)
# print(f'MAP@3: {map3:.4f}')



import matplotlib.pyplot as plt
import seaborn as sns

# Get leaderboard with scores
lb = predictor.leaderboard(silent=True)

# Filter only models with valid CV scores
lb = lb[~lb['score_val'].isna()]

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(data=lb, x='score_val', y='model', palette='viridis')
plt.xlabel('CV Score (MAP@3)')
plt.ylabel('Model')
plt.title('Cross-Validation Scores for AutoGluon Models')
plt.tight_layout()
plt.show()



# from sklearn.model_selection import train_test_split

# # Train/val split
# train_part, val_part = train_test_split(train, test_size=0.2, stratify=train['Fertilizer Name'], random_state=42)
# X_train = pipeline.fit_transform(train_part.drop(columns='Fertilizer Name'))
# X_train['Fertilizer Name'] = train_part['Fertilizer Name']
# X_val = pipeline.transform(val_part)

# # Retrain
# predictor = TabularPredictor(label='Fertilizer Name', eval_metric='log_loss').fit(X_train,
#                                                                                 presets=config["autogluon_presets"],
#                                                                                 time_limit=config["autogluon_time"])
# probs_val = predictor.predict_proba(X_val)
# top3_val = probs_val.apply(lambda row: row.nlargest(3).index.tolist(), axis=1)

# # MAP@3 metric
# def mapk(y_true, y_pred, k=3):
#     def apk(actual, predicted, k):
#         predicted = predicted[:k]
#         score = 0.0
#         for i, p in enumerate(predicted):
#             if p == actual:
#                 score = 1.0 / (i + 1)
#                 break
#         return score
#     return np.mean([apk(a, p, k) for a, p in zip(y_true, y_pred)])

# y_true = val_part['Fertilizer Name'].values
# print(f'MAP@3: {mapk(y_true, top3_val):.4f}')



# ğŸ“¤ Step 8: Prepare submission
submission = pd.DataFrame({'id': test['id'], 'Fertilizer Name': top3})
submission.to_csv('submission.csv', index=False)

