# Cell 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# setup hiển thị
pd.set_option("display.max_columns", 100)
sns.set(style="whitegrid")


# Cell 2: Load train/test datasets
train = pd.read_csv("/kaggle/input/premiumpulse-risk-modeling/train.csv")
test  = pd.read_csv("/kaggle/input/premiumpulse-risk-modeling/test.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)


# Cell 3: Basic info
train.info()


# Cell 4: Distribution of target
plt.figure(figsize=(10,5))
sns.histplot(train["Premium Amount"], bins=50, kde=True)
plt.title("Distribution of Premium Amount (Target)")
plt.show()

# check skewness
print("Skewness:", train["Premium Amount"].skew())


# Cell 5: Missing values
missing = train.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
missing



# Cell 6: Identify categorical vs numerical features
categorical = train.select_dtypes(include=["object", "category"]).columns.tolist()
numerical = train.select_dtypes(include=[np.number]).columns.tolist()

# loại bỏ id và target khỏi numerical
numerical = [col for col in numerical if col not in ["id", "Premium Amount"]]

print("Categorical features:", categorical)
print("Numerical features:", numerical)



from sklearn.impute import SimpleImputer

# copy dataset để xử lý
df = train.copy()

# 1. Numeric: impute median
num_cols = ['Age','Annual Income','Number of Dependents','Health Score',
            'Previous Claims','Vehicle Age','Credit Score','Insurance Duration']

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# 2. Categorical: fill "Unknown"
cat_cols = ['Gender','Marital Status','Education Level','Occupation','Location',
            'Policy Type','Customer Feedback','Smoking Status',
            'Exercise Frequency','Property Type']

for col in cat_cols:
    df[col] = df[col].fillna("Unknown")



# ép sang datetime
df["Policy Start Date"] = pd.to_datetime(df["Policy Start Date"], errors="coerce")

# Extract Year, Month
df["Policy_Year"] = df["Policy Start Date"].dt.year
df["Policy_Month"] = df["Policy Start Date"].dt.month

# Policy Age = số tháng từ ngày policy đến thời điểm max
max_date = df["Policy Start Date"].max()
df["Policy_Age_Months"] = ((max_date - df["Policy Start Date"]).dt.days / 30).astype(int)

# drop cột gốc
df = df.drop(columns=["Policy Start Date"])


# encode categorical features
from sklearn.preprocessing import LabelEncoder

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])


# log transform target
df["Premium_log"] = np.log1p(df["Premium Amount"])  # log(1+y)


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def feature_engineering_v2(train, test, target="Premium Amount", idcol="id", datecol="Policy Start Date"):
    df_tr = train.copy()
    df_te = test.copy()
    
    # --- 1. Date features ---
    for df in [df_tr, df_te]:
        df[datecol] = pd.to_datetime(df[datecol], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
        df["Policy_Year"] = df[datecol].dt.year
        max_date = df[datecol].max()
        df["Policy_Age_Months"] = ((max_date - df[datecol]).dt.days / 30).astype(float)
        df.drop(columns=[datecol], inplace=True)
    
    # --- 2. Handle missing values ---
    num_cols = df_tr.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in [idcol, target]]
    cat_cols = df_tr.select_dtypes(include=["object","category"]).columns.tolist()
    
    for df in [df_tr, df_te]:
        for c in num_cols:
            if df[c].isna().any():
                df[f"{c}_isna"] = df[c].isna().astype(int)
                df[c] = df[c].fillna(df[c].median())
        for c in cat_cols:
            df[c] = df[c].fillna("Unknown").astype(str)
    
    # --- 3. Keep only strong FE ---
    for col in ["Annual Income","Previous Claims"]:
        if col in df_tr.columns:
            low, high = df_tr[col].quantile([0.01,0.99])
            for df in [df_tr, df_te]:
                df[col] = df[col].clip(low, high)
                df[f"{col}_log1p"] = np.log1p(df[col].clip(lower=0))
    
    # --- 4. Strong interactions ---
    for df in [df_tr, df_te]:
        if {"Health Score","Previous Claims"}.issubset(df.columns):
            df["Health_x_PrevClaims"] = df["Health Score"] * df["Previous Claims"]
    
    # --- 5. Label Encoding for categoricals ---
    for c in cat_cols:
        le = LabelEncoder()
        le.fit(df_tr[c].astype(str).tolist() + df_te[c].astype(str).tolist())
        df_tr[c] = le.transform(df_tr[c].astype(str))
        df_te[c] = le.transform(df_te[c].astype(str))
    
    # --- 6. Final target log1p ---
    df_tr["Premium_log"] = np.log1p(df_tr[target])
    
    # Feature list
    drop_cols = [idcol, target, "Premium_log"]
    X_cols = [c for c in df_tr.columns if c not in drop_cols]
    
    return df_tr, df_te, X_cols

# chạy lại
df_tr2, df_te2, X_cols2 = feature_engineering_v2(train, test)
X2 = df_tr2[X_cols2]
y2 = df_tr2["Premium_log"]
X_test2 = df_te2[X_cols2]

print("X shape:", X2.shape, "| X_test shape:", X_test2.shape)



import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
import numpy as np

# --- 1. Train/valid split ---
X_train, X_valid, y_train, y_valid = train_test_split(
    X2, y2, test_size=0.2, random_state=42
)

# --- 2. Dataset LightGBM ---
train_set = lgb.Dataset(X_train, label=y_train)
valid_set = lgb.Dataset(X_valid, label=y_valid)

# --- 3. Params baseline ---
params = {
    "objective": "regression",
    "metric": "rmse",  # chỉ để theo dõi, ta sẽ tự tính RMSLE
    "learning_rate": 0.05,
    "num_leaves": 64,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "seed": 42
}

# --- 4. Train với early stopping ---
model = lgb.train(
    params,
    train_set,
    valid_sets=[valid_set],
    num_boost_round=500,
    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(50)]
)

# --- 5. Predict + RMSLE ---
y_pred_log = model.predict(X_valid, num_iteration=model.best_iteration)
y_pred = np.expm1(y_pred_log)   # inverse log1p
y_true = np.expm1(y_valid)

rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred))
print("Validation RMSLE:", rmsle)

# --- 6. Feature importance ---
feat_imp = pd.DataFrame({
    "feature": X2.columns,
    "importance": model.feature_importance(importance_type="gain")
}).sort_values("importance", ascending=False)

print(feat_imp.head(15))



# --- 1. Train final model ---
final_params = {
    "objective": "regression",
    "metric": "rmse",
    "verbosity": -1,
    "boosting_type": "gbdt",
    "learning_rate": 0.013088965389284437,  # từ Optuna
    "num_leaves": 173,
    "max_depth": 12,
    "min_data_in_leaf": 34,
    "feature_fraction": 0.7094411071379936,
    "bagging_fraction": 0.7421062273717826,
    "bagging_freq": 1,
    "lambda_l1": 4.249897442914641,
    "lambda_l2": 1.4108308392691358,
    "seed": 42,
    "n_estimators": 1000
}

final_model = lgb.LGBMRegressor(**final_params)
final_model.fit(X2, y2)

# --- 2. Predict test ---
y_test_pred_log = final_model.predict(X_test2)
y_test_pred = np.expm1(y_test_pred_log)

# --- 3. Submission ---
submission = pd.DataFrame({
    "id": df_te2["id"],
    "Premium Amount": y_test_pred
})
submission.to_csv("/kaggle/working/submission.csv", index=False)
print(submission.head())
print(submission.describe())

