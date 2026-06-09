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


# load data 
df_train = pd.read_csv('/kaggle/input/california-homelessness-prediction-challenge/train.csv')
df_test = pd.read_csv('/kaggle/input/california-homelessness-prediction-challenge/test.csv')

# Basic info
print("\n=== Train info ===")
print(df_train.info())

print("\n=== Train missing values ===")
print(df_train.isnull().sum())

print("\n=== Summary Stat for Train ===")
print(df_train.describe().T)

print("\n=== Test info ===")
print(df_test.info())

print("\n=== Test missing values ===")
print(df_test.isnull().sum())


import matplotlib.pyplot as plt
import seaborn as sns

# distribution of HOMELESS_RATE
plt.figure(figsize=(8,4))
sns.histplot(df_train["HOMELESS_RATE"], bins=30, kde=True)
plt.title("Distribution of HOMELESS_RATE")
plt.xlabel("HOMELESS_RATE")
plt.ylabel("Count")
    
plt.tight_layout()
plt.show()


threshold = 0.02 
high_homeless = df_train[df_train["HOMELESS_RATE"] > threshold].copy()

print("Number of regions with HOMELESS_RATE > 1%:", len(high_homeless))
display(high_homeless[["ID", "HOMELESS_RATE"]].sort_values("HOMELESS_RATE", ascending=False))


# Distribution after log1p - Prepare for the subsequent modeling process
plt.figure(figsize=(8,4))
sns.histplot(np.log1p(df_train["HOMELESS_RATE"]), bins=30, kde=True)
plt.title("Distribution of log1p(HOMELESS_RATE)")
plt.xlabel("log1p(HOMELESS_RATE)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()


#corr
numeric_cols = df_train.columns.drop(["ID"]) # only numeric data
corr = df_train[numeric_cols].corr()["HOMELESS_RATE"].sort_values(ascending=False)

print("\n=== Correlation with HOMELESS_RATE (top 15) ===")
print(corr.head(15))

print("\n=== Correlation with HOMELESS_RATE (bottom 15) ===")
print(corr.tail(15))


# heatmap for all 
plt.figure(figsize=(12,10))
sns.heatmap(df_train[numeric_cols].corr(),
            cmap="coolwarm",
            center=0,
            square=False,
            cbar_kws={"shrink": .6})
plt.title("Correlation Matrix (train numeric features)")
plt.tight_layout()
plt.show()


def preprocess_data(df):
    df = df.copy()
    df['State'] = df['ID'].apply(lambda x: x.split('_')[0])
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])
    df = pd.get_dummies(df, columns=['State'], drop_first=True)
    return df

train_proc = preprocess_data(df_train)


# build X and y
y = train_proc['HOMELESS_RATE']
X = train_proc.drop(columns=['HOMELESS_RATE'])


from sklearn.feature_selection import mutual_info_regression

mi_scores = mutual_info_regression(X, y, random_state=42)
mi_series = pd.Series(mi_scores, index=X.columns)
mi_series = mi_series.sort_values(ascending=False)

print("\n======== Mutual Information Top 20 Features ========")
print(mi_series.head(20))


# plot for Top 20 Features
plt.figure(figsize=(10, 8))
sns.barplot(x=mi_series.head(20).values, y=mi_series.head(20).index)
plt.title('Top 20 Features by Mutual Information Score')
plt.xlabel('MI Score (Information Content)')
plt.ylabel('Features')
plt.show()


#Keep all the features with a MI score greater than 0.05
selected_features_mi = mi_series[mi_series > 0.05].index.tolist()
selected_features_mi


final_features = list(set(selected_features_mi))
print(f"all frature used to training: {len(final_features)}")


print(f"Features used for training ({len(final_features)}): {final_features}")
X_final = X[final_features]


from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, ElasticNet,Lasso
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor


#define models
models = {
    "Ridge (alpha=1.0)": Ridge(alpha=1.0, random_state=42),

    "Lasso (alpha=0.001)": Lasso(alpha=0.001, random_state=42, max_iter=10000),
    
    "ElasticNet (alpha=0.001)": ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=42),
    
    "XGBoost": xgb.XGBRegressor(
        n_estimators=1000, 
        learning_rate=0.05, 
        max_depth=6, 
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1, 
        random_state=42
    ),
    
    "LightGBM": lgb.LGBMRegressor(
        n_estimators=1000, 
        learning_rate=0.05, 
        max_depth=-1, 
        num_leaves=31,
        n_jobs=-1, 
        random_state=42, 
        verbose=-1
    ),
    
    "CatBoost": CatBoostRegressor(
        iterations=1000, 
        learning_rate=0.05, 
        depth=6, 
        loss_function='RMSE',
        verbose=0, 
        random_seed=42
    )
}


from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

kf = KFold(n_splits=5, shuffle=True, random_state=42)

results_list = []
rmse_dict = {}  

y_log = np.log1p(y)

for name, model in models.items():
    print(f"Running CV for {name}...")

    # Pipeline
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    # CV scores (neg RMSE)
    cv_scores = cross_val_score(pipe, X_final, y_log,
                                cv=kf,
                                scoring="neg_root_mean_squared_error")

    rmse_scores = -cv_scores  
    rmse_dict[name] = rmse_scores

    mean_rmse = rmse_scores.mean()
    std_rmse = rmse_scores.std()

    results_list.append({
        "model": name,
        "mean_rmse": mean_rmse,
        "std_rmse": std_rmse
    })

# transfrom to DataFrame
results_df = pd.DataFrame(results_list).sort_values("mean_rmse")
results_df.reset_index(drop=True, inplace=True)

print("\n===== Reconstructed results_df =====")
print(results_df)


plt.figure(figsize=(12, 6))
plt.boxplot(rmse_dict.values(), labels=rmse_dict.keys(), patch_artist=True)

plt.title('Model Comparison: 5-Fold RMSE (Log Scale)')
plt.ylabel('RMSE (Lower is Better)')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()


from sklearn.preprocessing import StandardScaler

# Preprocess BEFORE Ensemble

X_train = X_final.copy()          # X_final 
y_train = y_log.copy()            # y_log is HOMELESS_RATE after log1p

# make test data consistent to training data
df_test_proc = preprocess_data(df_test)

# make the feature of test date consistent to column of train
for col in X_final.columns:
    if col not in df_test_proc.columns:
        df_test_proc[col] = 0

X_test = df_test_proc[X_final.columns]
X_test = X_test.fillna(X_train.mean())


# standardization（所有线性模型都需要）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("Scaling finished. You can now run the ensemble code.")


from sklearn.metrics import mean_squared_error, r2_score

print("\nTraining final models for ensemble...")

# Choose 3 selected: model:Ridge / ElasticNet / CatBoost
# =====================================
selected_models = [
    "ElasticNet (alpha=0.001)",
    "Lasso (alpha=0.001)",
    "CatBoost"
]

predictions = []
model_performances = []

print("\nSelected models for ensemble:")
print(selected_models)


# Train each model & predict

for name in selected_models:
    print(f"\nTraining {name}...")
    model = models[name]          
    model.fit(X_train_scaled, y_train)

    # Train Performance 
    train_pred = model.predict(X_train_scaled)
    rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    r2 = r2_score(y_train, train_pred)

    model_performances.append({
        "model": name,
        "train_rmse": rmse,
        "train_r2": r2
    })

    # Test Prediction 
    test_pred = model.predict(X_test_scaled)
    predictions.append(test_pred)


# Ensemble weights (1 / CV RMSE)

weights = []
for name in selected_models:
    row = results_df[results_df["model"] == name].iloc[0]
    weight = 1 / row["mean_rmse"]
    weights.append(weight)

weights = np.array(weights)
weights = weights / weights.sum()   # normalized

print("\nEnsemble Weights:")
for i, name in enumerate(selected_models):
    print(f"{name}: {weights[i]:.4f}")



# Final Ensemble Prediction
final_predictions = np.average(predictions, axis=0, weights=weights)
final_predictions = np.clip(final_predictions, a_min=0, a_max=None)
print("\nFinal ensemble predictions generated!")



# Show train performance summary
print("\n===== Train Performance (ElasticNet + Lasso + CatBoost) =====")
perf_df = pd.DataFrame(model_performances)
print(perf_df)


submission = pd.DataFrame({
    "ID": df_test["ID"],
    "HOMELESS_RATE": final_predictions
})

submission.to_csv("submission_ensemble_ver2.csv", index=False)
print("\nSaved: submission_ensemble_ver2.csv")


train_true = df_train['HOMELESS_RATE']

# Load ensemble predictions
ens = pd.read_csv('/kaggle/working/submission_ensemble_ver2.csv')

# Clip negative values (important!)
ens['HOMELESS_RATE'] = ens['HOMELESS_RATE'].clip(lower=0)

ens_pred = ens['HOMELESS_RATE']


# Plot histogram comparison
plt.figure(figsize=(10,5))
plt.hist(train_true, bins=20, alpha=0.5, label='Train TRUE', color='goldenrod')
plt.hist(ens_pred, bins=20, alpha=0.5, label='Ensemble PRED', color='skyblue')
plt.xlabel('HOMELESS_RATE')
plt.ylabel('Frequency')
plt.title('Histogram Comparison: Train TRUE vs Ensemble Prediction')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


print("\n=== Ensemble info ===")
print(ens.info())

print("\n=== Summary Stat for Ensemble ===")
print(ens.describe().T)

