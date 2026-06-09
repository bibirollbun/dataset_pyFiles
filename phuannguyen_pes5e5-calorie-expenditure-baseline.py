%%capture
!pip install -U xgboost
!pip install -U lightgbm
!pip install -U catboost


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import itertools
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler

import warnings 
warnings.simplefilter(action='ignore')

train_df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
submission_df = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")


train_df.head()


train_df.columns


submission_df.head()


# THERE IS NO NAN VALUES
# print(train_df.isnull().sum())
# print(test_df.isnull().sum())


plt.figure(figsize=(8, 5))

# Boxplot
sns.boxplot(y=train_df['Calories'])
plt.title('Boxplot of Calories')

plt.tight_layout()
plt.show()


numerical_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']

def add_cross_features(df, features):
    df = df.copy()
    df = df.loc[:, ~df.columns.duplicated()]
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            f1 = features[i]
            f2 = features[j]
            df[f"{f1}_{f2}"] = df[f1] * df[f2]
    return df

def add_interaction_features(df, features):
    df_new = df.copy()
    for f1, f2 in itertools.combinations(features, 2):
        df_new[f"{f1}_plus_{f2}"] = df_new[f1] + df_new[f2]
        df_new[f"{f1}_minus_{f2}"] = df_new[f1] - df_new[f2]
        df_new[f"{f2}_minus_{f1}"] = df_new[f2] - df_new[f1]
        df_new[f"{f1}_div_{f2}"] = df_new[f1] / (df_new[f2] + 1e-5)
        df_new[f"{f2}_div_{f1}"] = df_new[f2] / (df_new[f1] + 1e-5)
    return df_new

def add_statistical_features(df, features):
    df_new = df.copy()
    df_new["row_mean"] = df[features].mean(axis=1)
    df_new["row_std"] = df[features].std(axis=1)
    df_new["row_max"] = df[features].max(axis=1)
    df_new["row_min"] = df[features].min(axis=1)
    df_new["row_median"] = df[features].median(axis=1)
    df_new["row_sum"] = df[features].sum(axis=1)
    return df_new

# train_df = add_cross_features(train_df, numerical_features)
train_df = add_interaction_features(train_df, numerical_features)
train_df = add_statistical_features(train_df, numerical_features)

# test_df = add_cross_features(test_df, numerical_features)
test_df = add_interaction_features(test_df, numerical_features)
test_df = add_statistical_features(test_df, numerical_features)


le = LabelEncoder()
train_df["Sex"] = le.fit_transform(train_df["Sex"])
test_df["Sex"] = le.transform(test_df["Sex"])

train_df["Sex"] = train_df["Sex"].astype("category")   
test_df["Sex"] = test_df["Sex"].astype("category")

poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
poly_train = poly.fit_transform(train_df[numerical_features])
poly_test = poly.transform(test_df[numerical_features])

poly_feature_names = poly.get_feature_names_out(numerical_features)

poly_train_df = pd.DataFrame(poly_train, columns=poly_feature_names)    
poly_test_df = pd.DataFrame(poly_test, columns=poly_feature_names)

train_df = pd.concat([train_df.reset_index(drop=True), poly_train_df], axis=1)
test_df = pd.concat([test_df.reset_index(drop=True), poly_test_df], axis=1)

X = train_df.drop(columns=['id', 'Calories'])
y = np.log1p(train_df['Calories'])
X_test = test_df.drop(columns=['id'])


X


X['Age Height'][0]
# X['Age_Height'][0]


# Pearson correlation with the raw target
corr_raw = train_df.corr()['Calories'].abs().sort_values(ascending=False)
print("Top 20 features by |corr| with Calories:")
print(corr_raw.iloc[0:20])


y_log = np.log1p(train_df['Calories'])
df_corr = pd.concat([train_df.drop(columns=['id','Calories']), y_log.rename('log_Calories')], axis=1)
corr_log = df_corr.corr()['log_Calories'].abs().sort_values(ascending=False)
print("\nTop 10 features by |corr| with log(Calories+1):")
print(corr_log.iloc[0:25])


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import time

FOLDS = 7
SEED = 42

kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

models = {
    "CatBoost": CatBoostRegressor(
        iterations=1000,
        verbose=100,
        random_state=SEED,
        cat_features=["Sex"],
        early_stopping_rounds=100
    ),
    "XGBoost": XGBRegressor(
        n_estimators=2000,
        learnning_rate=0.02,
        max_depth=10,
        colsample_bytree=0.7,
        subsample=0.9,
        gamma=0.01,
        max_delta_step=2,
        early_stopping_rounds=100,
        eval_metric="rmse",
        enable_categorical=True,
        random_state=SEED
    ),
    "LightGBM": LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.02,
        max_depth=10,
        colsample_bytree=0.7,
        subsample=0.9,
        random_state=SEED,
        verbose=-1
    )
}

results = {
    name: {
        'oof': np.zeros(len(train_df)),       # lưu dự đoán out-of-fold
        'pred': np.zeros(len(test_df)),       # lưu dự đoán trung bình trên test
        'rmsle': []                           # lưu từng RMSLE trên mỗi fold
    }
    for name in models
}


for name, model in models.items():
    print(f"Training {name}...")
    for i, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
        print(f"\nFold {i+1}")
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_valid, y_valid = X.iloc[valid_idx], y[valid_idx]

        # remove duplicated columns (if any)
        X_train = X_train.loc[:, ~X_train.columns.duplicated()]
        X_valid = X_valid.loc[:, ~X_valid.columns.duplicated()]
        X_test = X_test.loc[:, ~X_test.columns.duplicated()]

        start = time.time()

        if name == "XGBoost":
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=100)
        elif name  == "CatBoost":
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid))
        else:
            model.fit(X_train, y_train)

        oof_pred = model.predict(X_valid)
        test_pred = model.predict(X_test)

        results[name]['oof'][valid_idx] = oof_pred
        results[name]['pred'] += test_pred / FOLDS

        # Compute the RMSLE error 
        rmsle = np.sqrt(
            mean_squared_log_error(
                np.expm1(y_valid),      # trả về y gốc
                np.expm1(oof_pred)
            )
        )
        results[name]['rmsle'].append(rmsle)

        print(f"Fold {i+1} RMSLE: {rmsle:.4f}")
        print(f"Training time: {time.time() - start:.1f} sec")

print("\n=== Model Comparison ===")
for name in models:
    mean_rmsle = np.mean(results[name]['rmsle'])
    std_rmsle = np.std(results[name]['rmsle'])
    print(f"{name} - Mean RMSLE: {mean_rmsle:.4f} ± {std_rmsle:.4f}")


from scipy.optimize import minimize
from sklearn.metrics import mean_squared_log_error

oof_preds = {name: np.expm1(results[name]['oof']) for name in results}
test_preds = {name: np.expm1(results[name]['pred']) for name in results}
y_true = np.expm1(y)

def rmsle_loss(weights):
    blended = (
        weights[0] * oof_preds['CatBoost'] +
        weights[1] * oof_preds['XGBoost'] +
        weights[2] * oof_preds['LightGBM']
    )
    return np.sqrt(mean_squared_log_error(y_true, blended))

initial_weights = [1/3, 1/3, 1/3]
constraints = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
bounds = [(0, 1)] * 3

res = minimize(rmsle_loss, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
best_weights = res.x

print(f"\n✅ Optimized Weights:")
print(f"CatBoost = {best_weights[0]:.4f}")
print(f"XGBoost  = {best_weights[1]:.4f}")
print(f"LightGBM = {best_weights[2]:.4f}")

blended_preds = (
    best_weights[0] * test_preds['CatBoost'] +
    best_weights[1] * test_preds['XGBoost'] +
    best_weights[2] * test_preds['LightGBM']
)

blended_preds = np.clip(blended_preds, 1, 314)

submission_df['Calories'] = blended_preds
submission_df.to_csv('submission.csv', index=False)

print("\nSubmission Head:")
print(submission_df.head())

print(f"\nPredict Mean: {blended_preds.mean():.2f}")
print(f"Predict Median: {np.median(blended_preds):.2f}")

