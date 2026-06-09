import os
os.listdir('../input')



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/zillow-price-prediction'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error




prop_2016 = pd.read_csv("/kaggle/input/zillow-prize-1/properties_2016.csv")
train_2016 = pd.read_csv("/kaggle/input/zillow-prize-1/train_2016_v2.csv")

prop_2017 = pd.read_csv("/kaggle/input/zillow-prize-1/properties_2017.csv")
train_2017 = pd.read_csv("/kaggle/input/zillow-prize-1/train_2017.csv")

sample = pd.read_csv("/kaggle/input/zillow-prize-1/sample_submission.csv")


df_2016 = train_2016.merge(prop_2016, on="parcelid", how="left")
df_2017 = train_2017.merge(prop_2017, on="parcelid", how="left")

df = pd.concat([df_2016, df_2017], axis=0).reset_index(drop=True)

print("Merged dataset shape:", df.shape)



df.info()


numeric_df = df.select_dtypes(include=['int64', 'float64'])

# Compute correlation matrix
corr_matrix = numeric_df.corr()

# Plot heatmap
plt.figure(figsize=(18, 12))
sns.heatmap(corr_matrix, cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap of Numeric Features", fontsize=18)
plt.show()


corr_target = corr_matrix['logerror'].abs().sort_values(ascending=False)

# Take top 20 most correlated features
top_features = corr_target.head(20).index

plt.figure(figsize=(14, 10))
sns.heatmap(df[top_features].corr(), cmap="coolwarm", annot=True)
plt.title("Top 20 Feature Correlation Heatmap", fontsize=18)
plt.show()


target = "logerror"
ignore_cols = ["logerror", "transactiondate", "parcelid"]

features = [col for col in df.columns if col not in ignore_cols]

X = df[features]
y = df[target]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train Shape:", X_train.shape)
print("Test Shape:", X_test.shape)


print("\nMissing Values (Top 20)")
print(X_train.isnull().sum().sort_values(ascending=False).head(20))

plt.figure(figsize=(10,5))
plt.hist(y_train, bins=40)
plt.title("Logerror Distribution")
plt.show()



# 1. Split numeric & categorical columns
num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X_train.select_dtypes(include=['object']).columns

# 2. Compute median only for numeric columns
train_median = X_train[num_cols].median()

# 3. Fill numeric missing values
X_train[num_cols] = X_train[num_cols].fillna(train_median)
X_test[num_cols]  = X_test[num_cols].fillna(train_median)

# 4. Fill categorical missing values with a placeholder
X_train[cat_cols] = X_train[cat_cols].fillna("Unknown")
X_test[cat_cols]  = X_test[cat_cols].fillna("Unknown")



for col in X_train.columns:
    if X_train[col].dtype == "object":
        X_train[col] = X_train[col].astype("category")
        X_test[col] = X_test[col].astype("category")



dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
dvalid = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)

params = {
    "objective": "reg:squarederror",
    "eval_metric": "mae",
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}

model = xgb.train(
    params,
    dtrain,
    num_boost_round=500,
    evals=[(dtrain, "train"), (dvalid, "valid")],
    early_stopping_rounds=30,
    verbose_eval=50
)


pred_valid = model.predict(dvalid)
mae = mean_absolute_error(y_test, pred_valid)
print("\nValidation MAE:", mae)



test_df = prop_2016.copy()

parcelids = test_df["parcelid"].copy()
test_df = test_df[features]


test_df = test_df.fillna(train_median)

# Convert object → category
for col in test_df.columns:
    if test_df[col].dtype == "object":
        test_df[col] = test_df[col].astype("category")

dsubmit = xgb.DMatrix(test_df, enable_categorical=True)

# Predict logerror
predictions = model.predict(dsubmit)
predictions = np.round(predictions, 4)


submit = pd.DataFrame({
    "ParcelId": parcelids,
    "201610": predictions,
    "201611": predictions,
    "201612": predictions,
    "201710": predictions,
    "201711": predictions,
    "201712": predictions,
})

submit.to_csv("submission.csv", index=False)
print("Submission File Created: submission.csv")


from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error



X_train_lr = pd.get_dummies(X_train, drop_first=True)
X_test_lr = pd.get_dummies(X_test, drop_first=True)

# Align train & test columns
X_train_lr, X_test_lr = X_train_lr.align(X_test_lr, join='left', axis=1, fill_value=0)

print("Shape after encoding:", X_train_lr.shape)



cat_cols = X_train.select_dtypes(include=["object", "category"]).columns
num_cols = X_train.select_dtypes(exclude=["object", "category"]).columns

print("Categorical columns:", len(cat_cols))
print("Numeric columns:", len(num_cols))


from category_encoders import TargetEncoder
#
te = TargetEncoder(cols=cat_cols)
X_train_enc = te.fit_transform(X_train, y_train)
X_test_enc = te.transform(X_test)



lasso = Lasso(alpha=0.0001, max_iter=5000)
lasso.fit(X_train_enc, y_train)
pred_lasso = lasso.predict(X_test_enc)
print("Lasso MAE:", mean_absolute_error(y_test, pred_lasso))


ridge = Ridge(alpha=1.0, solver="sag")   # FAST solver
ridge.fit(X_train_enc, y_train)
pred_ridge = ridge.predict(X_test_enc)
print("Ridge MAE:", mean_absolute_error(y_test, pred_ridge))



lr = LinearRegression(n_jobs=-1)
lr.fit(X_train_enc, y_train)
pred_lr = lr.predict(X_test_enc)
print("Linear Regression MAE:", mean_absolute_error(y_test, pred_lr))




