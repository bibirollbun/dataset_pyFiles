# 1. Install & Import Libraries

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import VotingClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



# 2. Load & Encode Data
train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')


le_soil = LabelEncoder()
le_crop = LabelEncoder()
le_target = LabelEncoder()

train['Soil Type'] = le_soil.fit_transform(train['Soil Type'])
train['Crop Type'] = le_crop.fit_transform(train['Crop Type'])
train['Fertilizer Name'] = le_target.fit_transform(train['Fertilizer Name'])

test['Soil Type'] = le_soil.transform(test['Soil Type'])
test['Crop Type'] = le_crop.transform(test['Crop Type'])

X = train.drop(['id', 'Fertilizer Name'], axis=1)
y = train['Fertilizer Name']


# 3. EDA: Target Distribution
plt.figure(figsize=(8,4))
train['Fertilizer Name'].value_counts().plot(kind='bar')
plt.title('Target Distribution')
plt.show()


# 4. SHAP Feature Importance
model_shap = LGBMClassifier(random_state=42)
model_shap.fit(X, y)
explainer = shap.TreeExplainer(model_shap)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X, plot_type="bar")



# 5. Ensemble Model: LGBM + CatBoost
lgb_params = {
    'learning_rate': 0.2976,
    'num_leaves': 68,
    'max_depth': 6,
    'min_child_samples': 50,
    'subsample': 0.5126,
    'colsample_bytree': 0.6306,
    'n_estimators': 500,
    'random_state': 42
}

lgbm = LGBMClassifier(**lgb_params)
catboost = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    random_seed=42,
    verbose=0
)

ensemble = VotingClassifier(
    estimators=[('lgbm', lgbm), ('catboost', catboost)],
    voting='soft'
)
ensemble.fit(X, y)


# 6. Predict Test Set & Submission
test_features = test.drop(['id'], axis=1)
probs = ensemble.predict_proba(test_features)
top_3 = np.argsort(probs, axis=1)[:, -3:][:, ::-1]

submission['Fertilizer Name'] = [' '.join(le_target.inverse_transform(row.astype(int))) for row in top_3]
submission.to_csv('submission.csv', index=False)
submission.head()

