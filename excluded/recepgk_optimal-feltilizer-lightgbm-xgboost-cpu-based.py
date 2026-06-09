import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import lightgbm as lgb


train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')


train.head(10)


test.head(10)


submission.head(10)


print(train.isnull().sum())


train.info()


plt.figure(figsize=(12, 6))
sns.countplot(x="Fertilizer Name", data=train)
plt.title("Target Variable Distribution")
plt.xlabel("Fertilizer Name")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


target_le = LabelEncoder()
train['Fertilizer_Label'] = target_le.fit_transform(train['Fertilizer Name'])

soil_le = LabelEncoder()
crop_le = LabelEncoder()


train['Soil_Type_Label'] = soil_le.fit_transform(train['Soil Type'])
train['Crop_Type_Label'] = crop_le.fit_transform(train['Crop Type'])

test['Soil_Type_Label'] = soil_le.transform(test['Soil Type'])
test['Crop_Type_Label'] = crop_le.transform(test['Crop Type'])


feature_cols = [
    'Temparature', 'Humidity', 'Moisture',
    'Nitrogen', 'Potassium', 'Phosphorous',
    'Soil_Type_Label', 'Crop_Type_Label'
]

X = train[feature_cols]
y = train['Fertilizer_Label']
X_test = test[feature_cols]


model_xgb = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method='hist', 
    random_state=42
)

model_xgb.fit(X, y)

pred_xgb = model_xgb.predict_proba(X_test)


model_lgb = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    device='cpu',        
    random_state=42
)

model_lgb.fit(X, y)

pred_lgb = model_lgb.predict_proba(X_test)



pred_ensemble = (pred_xgb + pred_lgb) / 2


top_3_preds = np.argsort(pred_ensemble, axis=1)[:, -3:][:, ::-1]

top_3_names = [
    ' '.join(target_le.inverse_transform(row)) for row in top_3_preds
]

submission = pd.DataFrame({
    "id": test["id"],
    "Fertilizer Name": top_3_names
})

submission.to_csv("submission.csv", index=False)

