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


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings as w
w.filterwarnings('ignore')

df = pd.read_csv('/kaggle/input/crop-yield-prediction-challenge/crop_yield_train.csv', index_col='id')
df_test = pd.read_csv('/kaggle/input/crop-yield-prediction-challenge/crop_yield_test.csv', index_col='id')
sample_sub = pd.read_csv('/kaggle/input/crop-yield-prediction-challenge/sample_submission.csv')

print('Train_Shape:', df.shape)
print('Test_Shape:', df_test.shape)
df.head()


df.drop('field_id', axis=1, inplace=True)


target = 'yield_tpha'
numeric_cols = df.select_dtypes(include='number').drop(target, axis=1)
cat_cols = df.select_dtypes(include='object')
print(len(numeric_cols.columns))
print(len(cat_cols.columns))


df.isnull().sum()


df.describe().T


corr = df.select_dtypes(include='number').corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, cmap='viridis')
plt.title('Correlation Hestmap')
plt.show()


sns.scatterplot(data=df, x='fertilizer_amount', y='yield_tpha', alpha=0.4)


sns.histplot(x=df[target], kde=True, bins=120)


# CatBoost training (K-Fold CV) 

from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import joblib
import os

TARGET = 'yield_tpha'
SEED = 42
N_FOLDS = 5



# recompute features robustly in case cell order changed
features = [c for c in df.columns if c != TARGET]
# detect categorical features (object / category dtype)
cat_features = [c for c in features if df[c].dtype == 'object' or str(df[c].dtype).startswith('category')]
cat_feature_indices = [features.index(c) for c in cat_features]

print(f"num features: {len(features)}, cat features: {len(cat_features)} -> {cat_features}")


# CatBoost can handle some missing values, but we'll fill numeric missing with median for stability
num_cols = [c for c in features if c not in cat_features]
for c in num_cols:
    if df[c].isnull().any():
        df[c] = df[c].fillna(df[c].median())

for c in cat_features:
    df[c] = df[c].astype(str).fillna('NA')

# If df_test exists and has same features, prepare it similarly
if 'df_test' in globals():
    for c in num_cols:
        if c in df_test.columns:
            df_test[c] = df_test[c].fillna(df[c].median())  
    for c in cat_features:
        if c in df_test.columns:
            df_test[c] = df_test[c].astype(str).fillna('NA')

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof = np.zeros(len(df))
models = []
fold = 0
cv_scores = []

X = df[features].copy()
y = df[TARGET].values

for train_idx, val_idx in kf.split(X, y):
    fold += 1
    print(f"\n--- Fold {fold} ---")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    train_pool = Pool(X_train, label=y_train, cat_features=cat_feature_indices)
    val_pool = Pool(X_val, label=y_val, cat_features=cat_feature_indices)

    model = CatBoostRegressor(
        iterations=5000,
        learning_rate=0.03,
        depth=6,
        eval_metric='RMSE',
        random_seed=SEED,
        od_type='Iter',
        od_wait=200,
        verbose=200,
        use_best_model=True
    )

    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=200)
    val_pred = model.predict(X_val)
    rmse = mean_squared_error(y_val, val_pred, squared=False)
    print(f"Fold {fold} RMSE: {rmse:.5f}")

    oof[val_idx] = val_pred
    cv_scores.append(rmse)
    models.append(model)

    os.makedirs('models', exist_ok=True)
    joblib.dump(model, f'models/catboost_fold{fold}.pkl')

print("\nCV RMSE per fold:", cv_scores)
print("Mean CV RMSE: {:.5f} +- {:.5f}".format(np.mean(cv_scores), np.std(cv_scores)))

oof_rmse = mean_squared_error(y, oof, squared=False)
print("OOF RMSE: {:.5f}".format(oof_rmse))
df_oof = df.copy()
df_oof['oof_pred'] = oof
# Quick plot (residuals)
plt.figure(figsize=(6,4))
plt.scatter(df_oof['oof_pred'], df_oof[TARGET] - df_oof['oof_pred'], alpha=0.4)
plt.axhline(0, color='k', linestyle='--')
plt.xlabel('OOF Prediction')
plt.ylabel('Residual (true - pred)')
plt.title('OOF Residuals')
plt.show()


from sklearn.metrics import mean_squared_error

print("========== TEXT ERROR ANALYSIS ==========\n")

df_err = df.copy()
df_err["oof_pred"] = oof
df_err["residual"] = df_err[TARGET] - df_err["oof_pred"]
df_err["abs_error"] = df_err["residual"].abs()

rmse = mean_squared_error(df_err[TARGET], df_err["oof_pred"], squared=False)
mae = df_err["abs_error"].mean()
medae = df_err["abs_error"].median()

print(f"Overall RMSE: {rmse:.6f}")
print(f"Mean Absolute Error: {mae:.6f}")
print(f"Median Absolute Error: {medae:.6f}\n")

print("Residual Summary:")
print(df_err["residual"].describe())
print("\nAbsolute Error Summary:")
print(df_err["abs_error"].describe())
print("\n")

df_err["quantile"] = pd.qcut(df_err[TARGET], 10, labels=False)
quantile_rmse = df_err.groupby("quantile").apply(
    lambda x: mean_squared_error(x[TARGET], x["oof_pred"], squared=False)
)

print("RMSE for each target quantile (0=lowest target values, 9=highest):")
print(quantile_rmse)
print("\n")

importances = np.mean([m.get_feature_importance() for m in models], axis=0)
fi = pd.DataFrame({"feature": features, "importance": importances})
fi = fi.sort_values("importance", ascending=False)

print("Top 20 Most Important Features:")
print(fi.head(20))
print("\n")

for cat in cat_features[:5]:  
    print(f"Error by category for: {cat}")
    cat_err = df_err.groupby(cat)["abs_error"].mean().sort_values(ascending=False)
    print(cat_err)
    print("\n")

for num in num_cols[:10]:
    corr = df_err[[num, "abs_error"]].corr().iloc[0,1]
    print(f"Correlation(abs_error, {num}): {corr:.4f}")

print("\n")


print("Top 20 Worst Predictions (Highest Absolute Error):")
worst = df_err.sort_values("abs_error", ascending=False).head(20)
print(worst[[TARGET, "oof_pred", "abs_error"]])
print("\n")

print("========== END OF TEXT ERROR ANALYSIS ==========")


# ============================================================
# 1) CONFIG
# ============================================================
TARGET = "yield_tpha"
SEED = 42
N_FOLDS = 5

from dataclasses import dataclass
import numpy as np
import pandas as pd
import joblib, os
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor, Pool


# ============================================================
# 2) FEATURE STORE
# ============================================================
@dataclass
class FeatureStore:
    df: pd.DataFrame
    target: str

    def get_features(self):
        return [c for c in self.df.columns if c != self.target]

    def get_cat_features(self, features):
        cat_feats = []
        for c in features:
            if self.df[c].dtype == "object" or str(self.df[c].dtype).startswith("category"):
                cat_feats.append(c)
        return cat_feats

    def get_num_features(self, features, cat_features):
        return [c for c in features if c not in cat_features]


# ============================================================
# 3) PREPROCESSOR
# ============================================================
class Preprocessor:
    def __init__(self, df, features, cat_features):
        self.df = df
        self.features = features
        self.cat_features = cat_features
        self.num_features = [c for c in features if c not in cat_features]

    def process(self):
        # numeric → fill median
        for c in self.num_features:
            if self.df[c].isnull().any():
                self.df[c] = self.df[c].fillna(self.df[c].median())

        # categorical → cast to str, fill NA
        for c in self.cat_features:
            self.df[c] = self.df[c].astype(str).fillna("NA")

        return self.df


# ============================================================
# HIGH-YIELD SPECIALIST TRAINER
# ============================================================

def train_high_yield_specialist(df, features, cat_features, target='yield_tpha', top_quantile=0.9):
    """
    Train a separate CatBoost model on top quantile samples (high-yield specialist)
    
    Returns:
    - specialist_model: trained CatBoost model
    - threshold: value of top quantile
    """

    # Compute threshold
    threshold = df[target].quantile(top_quantile)
    high_yield_df = df[df[target] >= threshold].copy()

    print(f"Training High-Yield Specialist on {len(high_yield_df)} samples (>= {threshold:.2f})")

    # Features and cat indices
    cat_idx = [features.index(c) for c in cat_features]

    X = high_yield_df[features]
    y = high_yield_df[target]

    specialist_model = CatBoostRegressor(
        iterations=5000,
        learning_rate=0.03,
        depth=6,
        eval_metric='RMSE',
        od_type='Iter',
        od_wait=200,
        random_seed=SEED,
        verbose=200,
        use_best_model=True
    )
    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    
    specialist_model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        cat_features=cat_idx,
        use_best_model=True
    )
    return specialist_model, threshold


def blend_high_yield_preds(df, features, cat_features, main_models, specialist_model, threshold):
    """
    df: df with features (train OOF or test)
    main_models: list of K-Fold models
    specialist_model: high-yield model
    threshold: value above which to use specialist blending
    """

    cat_idx = [features.index(c) for c in cat_features]

    # Main model ensemble
    main_preds = np.zeros(len(df))
    for model in main_models:
        pool = Pool(df[features], cat_features=cat_idx)
        main_preds += model.predict(pool)
    main_preds /= len(main_models)

    # Specialist predictions
    specialist_pool = Pool(df[features], cat_features=cat_idx)
    specialist_preds = specialist_model.predict(specialist_pool)

    # Blend
    blended = main_preds.copy()
    high_mask = df['yield_tpha'] >= threshold if 'yield_tpha' in df.columns else main_preds >= threshold
    blended[high_mask] = (main_preds[high_mask] + specialist_preds[high_mask]) / 2

    return blended


# ============================================================
# 4) MODEL FACTORY
# ============================================================
def build_catboost():
    return CatBoostRegressor(
        iterations=5000,
        learning_rate=0.03,
        depth=6,
        eval_metric="RMSE",
        od_type="Iter",
        od_wait=200,
        random_seed=SEED,
        verbose=False,
        use_best_model=True,
    )


# ============================================================
# 5) CV TRAINER
# ============================================================
@dataclass
class CVTrainer:
    df: pd.DataFrame
    features: list
    cat_features: list
    target: str

    def run(self):
        X = self.df[self.features]
        y = self.df[self.target].values
        cat_idx = [self.features.index(c) for c in self.cat_features]

        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

        oof = np.zeros(len(self.df))
        fold = 0
        scores = []
        models = []

        for train_idx, val_idx in kf.split(X, y):
            fold += 1
            print(f"\n--- Fold {fold} ---")

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            train_pool = Pool(X_train, y_train, cat_features=cat_idx)
            val_pool = Pool(X_val, y_val, cat_features=cat_idx)

            model = build_catboost()
            model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=200)

            val_pred = model.predict(X_val)
            rmse = mean_squared_error(y_val, val_pred, squared=False)
            scores.append(rmse)

            print(f"Fold {fold} RMSE: {rmse:.5f}")

            oof[val_idx] = val_pred
            models.append(model)

            os.makedirs("models", exist_ok=True)
            joblib.dump(model, f"models/catboost_fold{fold}.pkl")

        oof_rmse = mean_squared_error(y, oof, squared=False)

        print("\nCV Scores:", scores)
        print("Mean RMSE: {:.5f} ± {:.5f}".format(np.mean(scores), np.std(scores)))
        print("OOF RMSE:", round(oof_rmse, 5))

        return oof, models, scores


# ============================================================
# 6) RUN FULL PIPELINE
# ============================================================
# Create feature store
fs = FeatureStore(df=df, target=TARGET)

features = fs.get_features()
cat_features = fs.get_cat_features(features)
num_features = fs.get_num_features(features, cat_features)

print("Features:", len(features))
print("Categorical:", cat_features)

# Preprocess
pre = Preprocessor(df=df, features=features, cat_features=cat_features)
df_processed = pre.process()

# Train
trainer = CVTrainer(
    df=df_processed,
    features=features,
    cat_features=cat_features,
    target=TARGET
)
oof, models, cv_scores = trainer.run()

# Attach OOF to df
df_oof = df_processed.copy()
df_oof["oof"] = oof


specialist_model, threshold = train_high_yield_specialist(
    df=df_processed,
    features=features,
    cat_features=cat_features,
    target=TARGET,
    top_quantile=0.9
)
oof_blended = blend_high_yield_preds(
    df=df_processed,
    features=features,
    cat_features=cat_features,
    main_models=models,
    specialist_model=specialist_model,
    threshold=threshold
)


def build_submission_blended(
    df_test, 
    features, 
    cat_features, 
    main_models, 
    specialist_model=None, 
    threshold=None, 
    filename="submission_blended.csv"
):
    """
    Build Kaggle submission with optional high-yield blending.
    
    df_test: test dataframe
    features: list of feature columns
    cat_features: list of categorical features
    main_models: list of main K-Fold CatBoost models
    specialist_model: optional high-yield specialist model
    threshold: float, predicted threshold above which to use specialist
    filename: output CSV filename
    """
    
    cat_idx = [features.index(c) for c in cat_features]
    
    # Step 1: main model predictions (ensemble)
    main_preds = np.zeros(len(df_test))
    for model in main_models:
        pool = Pool(df_test[features], cat_features=cat_idx)
        main_preds += model.predict(pool)
    main_preds /= len(main_models)
    
    # Step 2: blend with specialist if provided
    if specialist_model is not None and threshold is not None:
        pool_spec = Pool(df_test[features], cat_features=cat_idx)
        specialist_preds = specialist_model.predict(pool_spec)
        
        # approximate high-yield samples using main_preds
        high_mask = main_preds >= threshold
        main_preds[high_mask] = (main_preds[high_mask] + specialist_preds[high_mask]) / 2
    
    # Step 3: build submission
    submission = pd.DataFrame({
        'id': df_test.index,
        'yield_tpha': main_preds
    })
    
    submission.to_csv(filename, index=False)
    print(f"Submission saved to: {filename}")
    
    return submission


submission = build_submission_blended(
    df_test=df_test,
    features=features,
    cat_features=cat_features,
    main_models=models,
    specialist_model=specialist_model,
    threshold=threshold,       
    filename="submission_high_yield[2].csv"
)


def analyze_high_yield_blend(df_test, main_models, specialist_model, features, cat_features, threshold):
    cat_idx = [features.index(c) for c in cat_features]
    
    # Step 1: main model predictions
    main_preds = np.zeros(len(df_test))
    for model in main_models:
        pool = Pool(df_test[features], cat_features=cat_idx)
        main_preds += model.predict(pool)
    main_preds /= len(main_models)
    
    # Step 2: specialist predictions
    pool_spec = Pool(df_test[features], cat_features=cat_idx)
    specialist_preds = specialist_model.predict(pool_spec)
    
    # Step 3: high-yield mask
    high_mask = main_preds >= threshold
    n_high = high_mask.sum()
    
    # Step 4: difference statistics
    diff = specialist_preds[high_mask] - main_preds[high_mask]
    avg_diff = diff.mean() if n_high > 0 else 0
    max_diff = diff.max() if n_high > 0 else 0
    min_diff = diff.min() if n_high > 0 else 0
    
    print("High-yield blending analysis:")
    print(f"Number of high-yield samples replaced: {n_high}")
    print(f"Fraction of test set: {n_high / len(df_test):.3%}")
    print(f"Average specialist adjustment: {avg_diff:.4f}")
    print(f"Max adjustment: {max_diff:.4f}, Min adjustment: {min_diff:.4f}")
    
    # Step 5: optional check of top 5 adjusted predictions
    if n_high > 0:
        top5_idx = diff.argsort()[::-1][:5]
        print("\nTop 5 adjusted predictions (specialist - main):")
        for i in top5_idx:
            idx = np.where(high_mask)[0][i]
            print(f"ID: {df_test.index[idx]}, Main: {main_preds[idx]:.4f}, Specialist: {specialist_preds[idx]:.4f}, Diff: {diff[i]:.4f}")
    
    return main_preds, specialist_preds, high_mask


main_preds, specialist_preds, high_mask = analyze_high_yield_blend(
    df_test=df_test,
    main_models=models,
    specialist_model=specialist_model,
    features=features,
    cat_features=cat_features,
    threshold=threshold
)

