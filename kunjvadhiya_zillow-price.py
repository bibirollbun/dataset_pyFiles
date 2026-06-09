# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/zillow-prize-1'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler


train_2016 = pd.read_csv("/kaggle/input/zillow-prize-1/train_2016_v2.csv")
train_2017 = pd.read_csv("/kaggle/input/zillow-prize-1/train_2017.csv")
prop_2016 = pd.read_csv("/kaggle/input/zillow-prize-1/properties_2016.csv")
prop_2017 = pd.read_csv("/kaggle/input/zillow-prize-1/properties_2017.csv")
sample_submission = pd.read_csv("/kaggle/input/zillow-prize-1/sample_submission.csv")


df_2016 = train_2016.merge(prop_2016, on="parcelid", how="left")
df_2017 = train_2017.merge(prop_2017, on="parcelid", how="left")
df = pd.concat([df_2016, df_2017], axis=0).reset_index(drop=True)  # merging


print(df.info(verbose=True))
print(df.describe().T)  


missing_ratio = df.isnull().mean().sort_values(ascending=False)
print("Top 30 columns by missing ratio:")
print(missing_ratio.head(30))


plt.figure(figsize=(10,5))
sns.boxplot(x=df["logerror"], color="skyblue")
plt.title("Logerror Boxplot (raw)")
plt.show()


plt.figure(figsize=(12,6))
sns.histplot(df["logerror"], bins=80, kde=True)
plt.title("Logerror Distribution (raw)")
plt.show()


df = df[(df["logerror"] > -3) & (df["logerror"] < 3)].reset_index(drop=True)


plt.figure(figsize=(10,5))
sns.boxplot(x=df["logerror"], color="lightgreen")
plt.title("Logerror Boxplot (trimmed)")
plt.show()


# Drop columns with too many NaNs
missing_ratio = df.isnull().mean()
drop_cols = missing_ratio[missing_ratio > 0.8].index
df = df.drop(columns=drop_cols)


df["missing_count"] = df.isnull().sum(axis=1)


# Clip heavy-tailed numeric columns (reduce long tails)
clip_cols = ["taxvaluedollarcnt", "calculatedfinishedsquarefeet", "lotsizesquarefeet",
             "structuretaxvaluedollarcnt", "landtaxvaluedollarcnt"]
for col in clip_cols:
    if col in df.columns:
        lower = df[col].quantile(0.01)
        upper = df[col].quantile(0.99)
        df[col] = df[col].clip(lower, upper)


# Fill NaNs 
for col in df.columns:
    if df[col].dtype != "object" and col != "transactiondate":
        df[col] = df[col].fillna(df[col].median())
    elif df[col].dtype == "object":
        df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "")


# Convert transactiondate and basic time features
df['transactiondate'] = pd.to_datetime(df['transactiondate'])
df['trans_year'] = df['transactiondate'].dt.year
df['trans_month'] = df['transactiondate'].dt.month
df['trans_quarter'] = df['transactiondate'].dt.quarter
df.drop(columns=['transactiondate'], inplace=True)


# Feature engineering

df['bath_per_bed'] = df['bathroomcnt'] / (df['bedroomcnt'] + 1)
df['room_sum'] = df['bathroomcnt'] + df['bedroomcnt']            
df['living_area_ratio'] = df['calculatedfinishedsquarefeet'] / (df['lotsizesquarefeet'] + 1)
df['value_per_sqft'] = df['taxvaluedollarcnt'] / (df['calculatedfinishedsquarefeet'] + 1)


# Additional derived features
if 'structuretaxvaluedollarcnt' in df.columns:
    df['tax_structure_ratio'] = df['structuretaxvaluedollarcnt'] / (df['taxvaluedollarcnt'] + 1)
if 'landtaxvaluedollarcnt' in df.columns:
    df['tax_land_ratio'] = df['landtaxvaluollarcnt'] if 'taxvaluedollarcnt' not in df.columns else df['landtaxvaluedollarcnt'] / (df['taxvaluedollarcnt'] + 1)
df['bed_bath_ratio'] = df['bedroomcnt'] / (df['bathroomcnt'] + 1)
df['sqft_per_room'] = df['calculatedfinishedsquarefeet'] / (df['room_sum'] + 1)
df['area_per_bed'] = df['calculatedfinishedsquarefeet'] / (df['bedroomcnt'] + 1)
df['area_per_bath'] = df['calculatedfinishedsquarefeet'] / (df['bathroomcnt'] + 1)


# Log-transform some remaining skewed features (if present)
log_cols = ["structuretaxvaluedollarcnt","landtaxvaluedollarcnt","finishedsquarefeet12","taxamount"]
for col in log_cols:
    if col in df.columns:
        df[col] = np.log1p(df[col])


# Replace infs if any and final fill
df = df.replace([np.inf, -np.inf], np.nan).fillna(0)


# Encode object dtype columns to category codes 
cat_cols = df.select_dtypes(include="object").columns.tolist()
cat_mappings = {}
for col in cat_cols:
    df[col] = df[col].astype('category')
    cat_mappings[col] = dict(enumerate(df[col].cat.categories))  
    
    cat_to_code = {cat:code for code,cat in cat_mappings[col].items()}
    df[col] = df[col].map(cat_to_code).astype('int32')


X = df.drop(columns=["logerror"])
y = df["logerror"].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Shapes:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_scaled_full = scaler.fit_transform(X) 


y_mean = y_train.mean()
y_std = y_train.std() if y_train.std() != 0 else 1.0
y_train_scaled = (y_train - y_mean) / y_std
y_test_scaled = (y_test - y_mean) / y_std


results = {}


# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train_scaled)
lin_preds_scaled = lin_reg.predict(X_test_scaled)
lin_preds = lin_preds_scaled * y_std + y_mean
lr_mae = mean_absolute_error(y_test, lin_preds)
results["Linear Regression"] = lr_mae
print("Linear Regression MAE :", lr_mae)


# Ridge
ridge_params = {"alpha": np.logspace(-3, 2, 20)}
ridge = Ridge()
ridge_cv = GridSearchCV(ridge, ridge_params, scoring="neg_mean_absolute_error", cv=5, n_jobs=-1)
ridge_cv.fit(X_train_scaled, y_train_scaled)
ridge_preds_scaled = ridge_cv.predict(X_test_scaled)
ridge_preds = ridge_preds_scaled * y_std + y_mean
ridge_mae = mean_absolute_error(y_test, ridge_preds)
results["Ridge"] = ridge_mae
print("Ridge MAE :", ridge_mae, "with alpha =", ridge_cv.best_params_["alpha"])


# Lasso
lasso_params = {"alpha": np.logspace(-4, 1, 20)}
lasso = Lasso(max_iter=20000)
lasso_cv = GridSearchCV(lasso, lasso_params, scoring="neg_mean_absolute_error", cv=5, n_jobs=-1)
lasso_cv.fit(X_train_scaled, y_train_scaled)
lasso_preds_scaled = lasso_cv.predict(X_test_scaled)
lasso_preds = lasso_preds_scaled * y_std + y_mean
lasso_mae = mean_absolute_error(y_test, lasso_preds)
results["Lasso"] = lasso_mae
print("Lasso MAE :", lasso_mae, "with alpha =", lasso_cv.best_params_["alpha"])


# ElasticNet
elastic_params = {
    "alpha": np.logspace(-4, 1, 12),
    "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]
}
elastic = ElasticNet(max_iter=20000)
elastic_cv = GridSearchCV(elastic, elastic_params, scoring="neg_mean_absolute_error", cv=5, n_jobs=-1)
elastic_cv.fit(X_train_scaled, y_train_scaled)
elastic_preds_scaled = elastic_cv.predict(X_test_scaled)
elastic_preds = elastic_preds_scaled * y_std + y_mean
elastic_mae = mean_absolute_error(y_test, elastic_preds)
results["ElasticNet"] = elastic_mae
print("ElasticNet MAE :", elastic_mae, "with params =", elastic_cv.best_params_)


print("\nMAE Scores:")
for k,v in results.items():
    print(f"{k}: {v:.6f}")

plt.figure(figsize=(10,6))
sns.barplot(x=list(results.keys()), y=list(results.values()))
plt.ylabel("MAE")
plt.title("Model Comparison (Lower is better)")
plt.show()


best_model_name = min(results, key=results.get)
print("\nBest Model :", best_model_name, "with MAE =", results[best_model_name])

if best_model_name == "Linear Regression":
    best_model_fitted = lin_reg
elif best_model_name == "Ridge":
    best_model_fitted = ridge_cv.best_estimator_
elif best_model_name == "Lasso":
    best_model_fitted = lasso_cv.best_estimator_
else:
    best_model_fitted = elastic_cv.best_estimator_


best_preds_scaled = best_model_fitted.predict(X_test_scaled)
best_preds = best_preds_scaled * y_std + y_mean

plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test, y=best_preds, alpha=0.3)
plt.xlabel("Actual Logerror")
plt.ylabel("Predicted Logerror")
plt.title(f"{best_model_name}: Actual vs Predicted")
plt.plot([-0.5,0.5], [-0.5,0.5], color='red', linestyle='--',label="best fit")
plt.legend()
plt.show()


X_submit = prop_2017.copy()
parcel_ids = X_submit["parcelid"].values


X_submit = X_submit.drop(columns=drop_cols, errors="ignore")


for col in ["bathroomcnt", "bedroomcnt", "calculatedfinishedsquarefeet", "lotsizesquarefeet", "taxvaluedollarcnt"]:
    if col not in X_submit.columns:
        X_submit[col] = 0


X_submit['bath_per_bed'] = X_submit['bathroomcnt'] / (X_submit['bedroomcnt'] + 1)
X_submit['room_sum'] = X_submit['bathroomcnt'] + X_submit['bedroomcnt']   # FIXED
X_submit['living_area_ratio'] = X_submit['calculatedfinishedsquarefeet'] / (X_submit['lotsizesquarefeet'] + 1)
X_submit['value_per_sqft'] = X_submit['taxvaluedollarcnt'] / (X_submit['calculatedfinishedsquarefeet'] + 1)


if 'structuretaxvaluedollarcnt' in X_submit.columns and 'taxvaluedollarcnt' in X_submit.columns:
    X_submit['tax_structure_ratio'] = X_submit['structuretaxvaluedollarcnt'] / (X_submit['taxvaluedollarcnt'] + 1)
if 'landtaxvaluedollarcnt' in X_submit.columns and 'taxvaluedollarcnt' in X_submit.columns:
    X_submit['tax_land_ratio'] = X_submit['landtaxvaluedollarcnt'] / (X_submit['taxvaluedollarcnt'] + 1)

X_submit['bed_bath_ratio'] = X_submit['bedroomcnt'] / (X_submit['bathroomcnt'] + 1)
X_submit['sqft_per_room'] = X_submit['calculatedfinishedsquarefeet'] / (X_submit['room_sum'] + 1)
X_submit['area_per_bed'] = X_submit['calculatedfinishedsquarefeet'] / (X_submit['bedroomcnt'] + 1)
X_submit['area_per_bath'] = X_submit['calculatedfinishedsquarefeet'] / (X_submit['bathroomcnt'] + 1)


for col in log_cols:
    if col in X_submit.columns:
        X_submit[col] = np.log1p(X_submit[col])


for col in cat_cols:
    if col in X_submit.columns:
        # Use training categories if available; unknown categories = -1
        train_cats = list(cat_mappings[col].values())
        X_submit[col] = pd.Categorical(X_submit[col], categories=train_cats)
        X_submit[col] = X_submit[col].cat.codes.fillna(-1).astype('int32')
    else:
        # create missing column as zeros if absent in submission
        X_submit[col] = 0


# Drop parcelid and align columns with training X
X_submit = X_submit.drop(columns=["parcelid"], errors="ignore")
X_submit = X_submit.reindex(columns=X.columns, fill_value=0)


# Fill any remaining NaNs with training medians or 0
for col in X_submit.columns:
    if X_submit[col].dtype in [np.float64, np.int64]:
        if col in df.columns:
            X_submit[col] = X_submit[col].fillna(df[col].median())
        else:
            X_submit[col] = X_submit[col].fillna(0)
    else:
        X_submit[col] = X_submit[col].fillna(0)


# Scale submission features using the same scaler
X_submit_scaled = scaler.transform(X_submit)


# Select final estimator (same selection logic as before)
final_model = best_model_fitted


# Predict (remember model was trained on scaled target)
submit_preds_scaled = final_model.predict(X_submit_scaled)
submit_preds = submit_preds_scaled * y_std + y_mean


# Fill sample_submission columns with predictions (same as before)
for col in sample_submission.columns[1:]:
    sample_submission[col] = submit_preds


# Save
sample_submission.to_csv("zillow_submission_new1.csv", index=False)
print("Submission file saved as zillow_submission_new2.csv")

