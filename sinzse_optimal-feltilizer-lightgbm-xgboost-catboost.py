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


train.head(5)


test.head(3)


submission.head(5)


print(train.isnull().sum())


train.info()


# plotting different types of fertilizers (our target)
plt.figure(figsize=(12, 6))
sns.countplot(x="Fertilizer Name", data=train)
plt.title("Target Variable Distribution")
plt.xlabel("Fertilizer Name")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


from sklearn.preprocessing import LabelEncoder

# Create an instance of LabelEncoder to encode the target variable 'Fertilizer Name'
target_le = LabelEncoder()

# Use the LabelEncoder to transform the 'Fertilizer Name' column in the 'train' dataset into numerical labels
# This is done to prepare the data for machine learning models, which often require numerical input
# The 'fit_transform' method fits the LabelEncoder to the data and transforms it in one step
train['Fertilizer_Label'] = target_le.fit_transform(train['Fertilizer Name'])

# Create separate instances of LabelEncoder to encode the 'Soil' and 'Crop' variables (not used in this snippet)
soil_le = LabelEncoder()  # Will be used to encode 'Soil' categories
crop_le = LabelEncoder()  # Will be used to encode 'Crop' categories


train['Soil_Type_Label'] = soil_le.fit_transform(train['Soil Type'])
train['Crop_Type_Label'] = crop_le.fit_transform(train['Crop Type'])

test['Soil_Type_Label'] = soil_le.transform(test['Soil Type'])
test['Crop_Type_Label'] = crop_le.transform(test['Crop Type'])


train.head(3)


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
    random_state=42,
    verbose=-1
)

model_lgb.fit(X, y)

pred_lgb = model_lgb.predict_proba(X_test)



train.info()


feature_cols = [
    'Temparature', 'Humidity', 'Moisture',
    'Nitrogen', 'Potassium', 'Phosphorous',
    'Soil_Type_Label', 'Crop_Type_Label'
]

X = train[feature_cols]
y = train['Fertilizer_Label']
X_test = test[feature_cols]


train.head(3)


from catboost import CatBoostClassifier

# Define categorical feature columns
categorical_features = ['Soil Type', 'Crop Type']

# Convert categorical columns to strings (or pandas 'category' type)
for col in categorical_features:
    train[col] = train[col].astype(str)
    test[col] = test[col].astype(str)

# Features and target
feature_cols_cat = [
    'Temparature', 'Humidity', 'Moisture',
    'Nitrogen', 'Potassium', 'Phosphorous',
    'Soil Type', 'Crop Type'
]

X_cat = train[feature_cols_cat]
y = train['Fertilizer Name']
X_test_cat = test[feature_cols_cat]

# Instantiate and train CatBoost model
model_cat = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=3,
    random_seed=42,
    verbose=0,
    cat_features=categorical_features,  # Note: still column names
    task_type='CPU'
)

# Fit model
model_cat.fit(X_cat, y)

# Predict probabilities
pred_cat = model_cat.predict_proba(X_test_cat)



pred_cat


pred_ensemble = (pred_xgb + pred_lgb + pred_cat) / 3


top_3_preds = np.argsort(pred_ensemble, axis=1)[:, -3:][:, ::-1]

top_3_names = [
    ' '.join(target_le.inverse_transform(row)) for row in top_3_preds
]

submission = pd.DataFrame({
    "id": test["id"],
    "Fertilizer Name": top_3_names
})

submission.to_csv("submission.csv", index=False)

