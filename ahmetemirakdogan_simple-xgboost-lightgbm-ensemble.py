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


# Basic utilities
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt

# Machine Learning libraries
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import lightgbm as lgb



train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')


train


test


submission


# Check for missing values
print("Missing values in train data:\n", train.isnull().sum())


train.dtypes


plt.figure(figsize=(8, 4))
sns.countplot(x="Fertilizer Name", data=train)
plt.title("Target Variable Distribution")
plt.xlabel("Fertilizer Name")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# Encode target
fertilizer_le = LabelEncoder()
train['Fertilizer_Label'] = fertilizer_le.fit_transform(train['Fertilizer Name'])

# Encode Soil and Crop
soil_le = LabelEncoder()
crop_le = LabelEncoder()

train['Soil_Type_Label'] = soil_le.fit_transform(train['Soil Type'])
train['Crop_Type_Label'] = crop_le.fit_transform(train['Crop Type'])

test['Soil_Type_Label'] = soil_le.transform(test['Soil Type'])
test['Crop_Type_Label'] = crop_le.transform(test['Crop Type'])



feature_cols = [
    'Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Phosphorous', 'Potassium',
    'Soil_Type_Label', 'Crop_Type_Label'
]
X = train[feature_cols]
X_test = test[feature_cols]
y = train['Fertilizer_Label']



# Train XGBoost model
model_xgb = XGBClassifier(
    n_estimators=129,
    max_depth=10,
    learning_rate=0.2002,
    subsample=0.9347,
    colsample_bytree=0.5184,
    gamma=0.027,
    min_child_weight=9,
    reg_alpha=1.31e-5,
    reg_lambda=0.1834,
    objective='multi:softprob',
    num_class=7,
    eval_metric='mlogloss',
    tree_method="gpu_hist",
    predictor="gpu_predictor",
    use_label_encoder=False,
    random_state=42
)
model_xgb.fit(X, y)
pred_xgb = model_xgb.predict_proba(X_test)

# Train LightGBM model
model_lgb = LGBMClassifier(
    device='gpu',
    objective='multiclass',
    num_class=7,
    learning_rate=0.1283,
    max_depth=11,
    num_leaves=140,
    min_data_in_leaf=20,
    feature_fraction=0.5422,
    bagging_fraction=0.7855,
    bagging_freq=5,
    lambda_l1=4.3517,
    lambda_l2=4.4707,
    random_state=42
)
model_lgb.fit(X, y)
pred_lgb = model_lgb.predict_proba(X_test)

# average of predicted probabilities
pred_ensemble = (pred_xgb + pred_lgb) / 2



top_3_preds = np.argsort(pred_ensemble, axis=1)[:, -3:][:, ::-1]

top_3_names = [
    ' '.join(fertilizer_le.inverse_transform(row)) for row in top_3_preds
]

submission = pd.DataFrame({
    "id": test["id"],
    "Fertilizer Name": top_3_names
})

submission.to_csv("submission.csv", index=False)

