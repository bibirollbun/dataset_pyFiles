import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import lightgbm as lgb
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import OrdinalEncoder
%matplotlib inline


train_df=pd.read_csv("/kaggle/input/medical-insurance-cost-dataset/train.csv")
test_df=pd.read_csv("/kaggle/input/medical-insurance-cost-dataset/test.csv")
test_ids = test_df['id']
train_df.drop(columns=["id"],axis=1,inplace=True)
test_df.drop(columns=['id'], axis=1, inplace=True)
train_df.isnull().sum()
train_df.head()


train_df['bmi_smoker'] = train_df['bmi'] * (train_df['smoker'] == 'yes').astype(int)
test_df['bmi_smoker'] = test_df['bmi'] * (test_df['smoker'] == 'yes').astype(int)
train_df['age_bin'] = pd.cut(train_df['age'], bins=[0, 25, 40, 55, 100], labels=['young', 'adult', 'middle_age', 'senior'])
test_df['age_bin'] = pd.cut(test_df['age'], bins=[0, 25, 40, 55, 100], labels=['young', 'adult', 'middle_age', 'senior'])
train_df['bmi_bin'] = pd.cut(train_df['bmi'], bins=[0, 18.5, 25, 30, 100], labels=['underweight', 'normal', 'overweight', 'obese'])
test_df['bmi_bin'] = pd.cut(test_df['bmi'], bins=[0, 18.5, 25, 30, 100], labels=['underweight', 'normal', 'overweight', 'obese'])


# Define features and target
X = train_df.drop(columns=['charges'])
y = np.log1p(train_df['charges'])  # Log transform target for RMSLE
X_test = test_df

categorical_cols = ['sex', 'smoker', 'region', 'age_bin', 'bmi_bin']
numerical_cols = ['age', 'bmi', 'children', 'bmi_smoker']

for col in categorical_cols:
    X[col] = X[col].astype('category')
    X_test[col] = X_test[col].astype('category')



preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OrdinalEncoder(), categorical_cols)  # Use OrdinalEncoder instead of passthrough
    ])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Define LightGBM model
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 31,
    'learning_rate': 0.06,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'categorical_feature': [X_train.columns.get_loc(col) for col in categorical_cols]  # Specify categorical columns
}
lgb_model = lgb.LGBMRegressor(**lgb_params)

# Create pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', lgb_model)])
# Train and evaluate
pipeline.fit(X_train, y_train)
val_pred = pipeline.predict(X_val)
val_pred = np.expm1(val_pred)  # Revert log transformation
y_val_orig = np.expm1(y_val)   # Revert log transformation for true values
rmsle = np.sqrt(mean_squared_log_error(y_val_orig, np.clip(val_pred, 0, None)))
print(f"Validation RMSLE: {rmsle:.4f}")


pipeline.fit(X, y)
test_pred = pipeline.predict(X_test)
test_pred = np.expm1(test_pred)  # Revert log transformation

# Create submission file
submission = pd.DataFrame({'id': test_ids, 'charges': np.clip(test_pred, 0, None)})
submission.to_csv('submission.csv', index=False)
print("Submission file created: submission.csv")



df=pd.read_csv("/kaggle/input/medical-insurance-cost-dataset/train.csv")
print(f"Data shape: {df.shape}")
print(f"check null values: {df.isnull().sum()}")
print(f"Data info: {df.info()}")
df.drop(columns=["id"],axis=1,inplace=True)
df.head()


sex_map={"female":0,"male":1}
df["sex"]=df["sex"].map(sex_map)
smoker_map={"no":0,"yes":1}
df["smoker"]=df["smoker"].map(smoker_map)
region_map={"northeast":0,"northwest":1,"southwest":2,"southeast":3}
df["region"]=df["region"].map(region_map)


test=pd.read_csv("/kaggle/input/medical-insurance-cost-dataset/test.csv")
Id=test.id
test.drop(columns=["id"],axis=1,inplace=True)
sex_map={"female":0,"male":1}
test["sex"]=test["sex"].map(sex_map)
smoker_map={"no":0,"yes":1}
test["smoker"]=test["smoker"].map(smoker_map)
region_map={"northeast":0,"northwest":1,"southwest":2,"southeast":3}
test["region"]=test["region"].map(region_map)
test.head()


X=df.drop(columns=["charges"],axis=1)
y=np.log1p(df["charges"])
from catboost import CatBoostRegressor
model = CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=6, verbose=300)
model.fit(X, y)
val_pred=model.predict(X)
val_pred = np.expm1(val_pred)  # Revert log transformation
y_val_orig = np.expm1(y)   # Revert log transformation for true values
rmsle = np.sqrt(mean_squared_log_error(y_val_orig, np.clip(val_pred, 0, None)))
print(f"Validation RMSLE: {rmsle:.4f}")


y_pred=model.predict(test)
y_pred = np.expm1(y_pred)  #
submission = pd.DataFrame({'id': Id, 'charges': np.clip(y_pred, 0, None)})
submission.to_csv('cat_submission.csv', index=False)
print("Submission file created: submission.csv")





