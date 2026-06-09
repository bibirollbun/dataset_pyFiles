# data_toolkit
import pandas as pd
import numpy as np

#visualize
import seaborn as sns
import matplotlib.pyplot as plt

#sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV

# boosting
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# for display
from IPython.display import display

#save model
import joblib

# Creating a common Config which will be used in our Notebook
from dataclasses import dataclass

class config:
    random_seed : int = 42
    test_size: int = 0.2
    verbose: int = 0
    verbosity: int = 0
    score: str = 'mean_squared_error'
    cv: int = 6
    
# Warning suppression
import warnings
warnings.filterwarnings('ignore')

print("imported sucessfully")


import io
df = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
def meta_data(df):
    print("="*40)
    print(f"DataFrame Shape: {df.shape}")
    print("="*40)
    
    print("\nDataset Info:")
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    print(info_str)
    
    print("\nColumns and Data Types:")
    print(df.dtypes)
    
    print("\nColumns Names:")
    print(list(df.columns))
    
    print("\nUnique Values per Column:")
    print(df.nunique())
    
    print("\nMissing Values per Column:")
    print(df.isnull().sum())
    
    print("\nSummary Statistics:")
    display(df.describe(include='all'))


# Example usage
meta_data(df)



import math
sns.set_style("whitegrid")

num_cols = [col for col in df.columns if col != 'BeatsPerMinute' and col != 'id']

# Dynamically set subplot grid size
n_plots = len(num_cols)
n_cols = 5
n_rows = math.ceil(n_plots / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 4*n_rows))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    sns.histplot(df[col], kde=True, stat="density", bins=30, alpha=0.5, ax=axes[i], color='lightgreen')
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Density')

# Remove any unused subplots
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

sns.despine()
plt.tight_layout()
plt.show()


df_corr = df.corr()
plt.figure(figsize=(12, 6))
sns.heatmap(df_corr, annot=True, fmt=".2f", cmap='coolwarm', cbar_kws={"shrink": .8})
plt.title('Correlation Heatmap')
plt.show()



df_corr_target = df_corr[num_cols + ['BeatsPerMinute']]
result = df_corr_target.corr()['BeatsPerMinute'].sort_values(ascending=False).head(10)
print(result)


# Select key features based on potential importance
key_features = ['RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality', 
                'InstrumentalScore', 'Energy', 'BeatsPerMinute']

# Create pairplot to visualize relationships
plt.figure(figsize=(12, 10))
sns.pairplot(df[key_features], diag_kind='kde', plot_kws={'alpha': 0.6})
plt.tight_layout()
plt.show()


# Create bins for BeatsPerMinute to analyze feature distributions
df['BPM_Range'] = pd.cut(df['BeatsPerMinute'], bins=4, labels=['Very Slow', 'Slow', 'Medium', 'Fast'])

# Visualize distribution of key features across different BPM ranges
features_to_plot = ['RhythmScore', 'InstrumentalScore', 'AcousticQuality', 'AudioLoudness']

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for i, feature in enumerate(features_to_plot):
    sns.boxplot(x='BPM_Range', y=feature, data=df, ax=axes[i], palette='viridis')
    axes[i].set_title(f'{feature} by BPM Range')
    axes[i].set_xlabel('BPM Range')
    axes[i].set_ylabel(feature)

plt.tight_layout()
plt.show()

# Remove the temporary column
df.drop('BPM_Range', axis=1, inplace=True)


# Create engineered features that might help with prediction
df_features = df.copy()

# Create interaction features
df_features['Rhythm_x_Instrumental'] = df['RhythmScore'] * df['InstrumentalScore']
df_features['Quality_Index'] = df['RhythmScore'] + df['InstrumentalScore'] + df['AcousticQuality']
df_features['Energy_Loudness_Ratio'] = df['Energy'] / (df['AudioLoudness'].abs() + 1)  # Adding 1 to avoid division by zero

# Create polynomial features for key metrics
df_features['RhythmScore_Squared'] = df['RhythmScore'] ** 2
df_features['Energy_Squared'] = df['Energy'] ** 2

# Display head of the feature-engineered dataframe
df_features[['BeatsPerMinute', 'Rhythm_x_Instrumental', 'Quality_Index', 
             'Energy_Loudness_Ratio']].head()


# Prepare data for modeling
X = df_features.drop(['BeatsPerMinute', 'id'], axis=1)
y = df_features['BeatsPerMinute']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.test_size, random_state=config.random_seed)

# Train a Random Forest model with fewer estimators for faster training
rf = RandomForestRegressor(n_estimators=50, max_depth=15, random_state=config.random_seed, n_jobs=-1)
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)

# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Random Forest Performance:")
print(f"RMSE: {rmse:.4f}")
print(f"RÂ²: {r2:.4f}")


# Get feature importance from the model
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
})

# Sort by importance
feature_importance = feature_importance.sort_values('Importance', ascending=False).reset_index(drop=True)

# Visualize feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15), palette='viridis')
plt.title('Random Forest Feature Importance', fontsize=16)
plt.xlabel('Importance', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.tight_layout()
plt.show()

# Display the top 15 most important features
feature_importance.head(15)


# reloading the datasets proper categorization
train = df_features.copy()
test_final = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')


train.to_csv('featured.csv', index=False)


X = train.drop(['BeatsPerMinute', 'id'], axis=1)
y = train['BeatsPerMinute']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.test_size, random_state=config.random_seed)

print("done vai")


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.base import clone


cv = 3
random_state = config.random_seed

models = {
    "xgboost": xgb.XGBRegressor(
        n_estimators=5000,
        learning_rate=0.05,
        random_state=random_state,
        verbosity=0,
        tree_method="hist",
        n_jobs=-1
    ),
    "CatBoostRegressor": cb.CatBoostRegressor(
        n_estimators=5000,
        learning_rate=0.05,
        random_state=random_state,
        logging_level="Silent",
        eval_metric="RMSE"
    ),
    "lightgbm": lgb.LGBMRegressor(
        n_estimators=5000,
        learning_rate=0.05,
        random_state=random_state,
        n_jobs=-1,
        verbosity= -1,
        metric="rmse"
    )
}

kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
results = []

for name, base_model in models.items():
    print(f"\nTraining {name}...")
    fold_rmses, best_iters = [], []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model = clone(base_model)
        
        if name == "RandomForest":
            model.fit(X_tr, y_tr)
        elif name == "CatBoostRegressor":
            model.fit(
                X_tr, y_tr,
                eval_set=(X_val, y_val),
                early_stopping_rounds=50,
                verbose=False
            )
            best_iters.append(model.get_best_iteration())
        elif name == "xgboost":
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric="rmse",
                early_stopping_rounds=50,
                verbose=False
            )
            best_iters.append(model.get_booster().best_iteration)
        elif name == "lightgbm":
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric="rmse",
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
                
            )
            best_iters.append(model.best_iteration_)
        
        preds = model.predict(X_val)
        rmse = mean_squared_error(y_val, preds, squared=False)
        fold_rmses.append(rmse)
        print(f"  Fold {fold+1}: RMSE = {rmse:.4f}")
    
    rmse_mean = np.mean(fold_rmses)
    best_iter_mean = np.mean(best_iters) if best_iters else None
    results.append({"Model": name, "RMSE": round(rmse_mean, 4), "Best Iter": best_iter_mean})
    print(f">>> {name}: Mean CV RMSE = {rmse_mean:.4f}, Best Iter â‰ˆ {best_iter_mean}")

ranks = pd.DataFrame(results).sort_values(by="RMSE", ascending=True).reset_index(drop=True)
print("\nLeaderboard:")
print(ranks)



import optuna


# Use a larger sample (e.g., 200k rows) to stabilize tuning
sample_size = 200_000
X_sample = X_train.sample(sample_size, random_state=config.random_seed)
y_sample = y_train.loc[X_sample.index]

def objective(trial):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "n_jobs": -1,
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.07),
        "num_leaves": trial.suggest_int("num_leaves", 64, 128),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 30, 100),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.8, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.8, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 5),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-5, 1e-2, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-5, 1e-2, log=True),
    }

    X_tr, X_val, y_tr, y_val = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)

    model = lgb.LGBMRegressor(
        n_estimators=5000,
        **params
    )

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)]
    )

    preds = model.predict(X_val, num_iteration=model.best_iteration_)
    rmse = mean_squared_error(y_val, preds, squared=False)
    return rmse

# Setup Optuna study
study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=config.random_seed),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
)

# Optimize
study.optimize(objective, n_trials=50, n_jobs=1)

# Best parameters & RMSE
print("Best params:", study.best_params)
print("Best RMSE (sample):", study.best_value)



cv = 3
random_state = config.random_seed

# Use our tuned Optuna parameters
best_lgb_params = {
    "learning_rate": 0.03125,
    "num_leaves": 112,
    "min_data_in_leaf": 64,
    "feature_fraction": 0.8739,
    "bagging_fraction": 0.8905,
    "bagging_freq": 3,
    "lambda_l1": 0.0004745,
    "lambda_l2": 0.0015773,
}

kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
fold_rmses, best_iters = [], []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr = X_train.iloc[train_idx].reset_index(drop=True)
    X_val = X_train.iloc[val_idx].reset_index(drop=True)
    y_tr = y_train.iloc[train_idx].reset_index(drop=True)
    y_val = y_train.iloc[val_idx].reset_index(drop=True)

    model = lgb.LGBMRegressor(n_estimators=5000, **best_lgb_params)

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)]
    )

    preds = model.predict(X_val, num_iteration=model.best_iteration_)
    rmse = mean_squared_error(y_val, preds, squared=False)
    fold_rmses.append(rmse)
    best_iters.append(model.best_iteration_)

    print(f"Fold {fold+1}: RMSE = {rmse:.4f}, Best Iter = {model.best_iteration_}")

mean_rmse = np.mean(fold_rmses)
mean_best_iter = np.mean(best_iters)
print(f"\nLightGBM CV mean RMSE: {mean_rmse:.4f}, mean best iteration â‰ˆ {mean_best_iter:.1f}")



# Optuna params
best_lgb_params = {
    "learning_rate": 0.03125,
    "num_leaves": 112,
    "min_data_in_leaf": 64,
    "feature_fraction": 0.8739,
    "bagging_fraction": 0.8905,
    "bagging_freq": 3,
    "lambda_l1": 0.0004745,
    "lambda_l2": 0.0015773,
}

# Use slightly higher n_estimators with early stopping
n_estimators = 5000
early_stopping_rounds = 50

# Split full dataset for early stopping
from sklearn.model_selection import train_test_split
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=config.random_seed
)

model = lgb.LGBMRegressor(n_estimators=n_estimators, **best_lgb_params)

model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    eval_metric="rmse",
    callbacks=[lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(-1)]
)

# Predictions on validation
preds_val = model.predict(X_val, num_iteration=model.best_iteration_)

rmse_val = mean_squared_error(y_val, preds_val, squared=False)
print(f"Full dataset RMSE (validation): {rmse_val:.4f}")
print(f"Best iteration used: {model.best_iteration_}")


pred_test = model.predict(X_test)
rmse_test = mean_squared_error(y_test, pred_test, squared=False)
print(f"Test RMSE: {rmse_test:.4f}")



from sklearn.linear_model import LinearRegression
# ================================
# Define base models
# ================================
models = {
    "xgboost": xgb.XGBRegressor(
        n_estimators=5000,
        learning_rate=0.05,
        random_state=42,
        tree_method="hist",       # set to "gpu_hist" if GPU is available
        n_jobs=-1,
        verbosity=0
    ),
    "catboost": cb.CatBoostRegressor(
        n_estimators=5000,
        learning_rate=0.05,
        random_seed=42,
        logging_level="Silent",
        eval_metric="RMSE",
        task_type="GPU"           # change to "CPU" if no GPU
    ),
    "lightgbm": lgb.LGBMRegressor(
        n_estimators=5000,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1,
        metric="rmse",
        device="gpu"              # remove this if no GPU
    )
}

# ================================
# K-Fold setup
# ================================
kf = KFold(n_splits=3, shuffle=True, random_state=42)

# meta-data storage
meta_train = np.zeros((len(X_train), len(models)))
meta_test = np.zeros((len(X_test), len(models)))

# ================================
# Train base models with stacking
# ================================
for m_idx, (name, base_model) in enumerate(models.items()):
    print(f"\nTraining {name}...")
    test_fold_preds = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        mdl = clone(base_model)

        if name == "catboost":
            mdl.fit(X_tr, y_tr,
                    eval_set=(X_val, y_val),
                    early_stopping_rounds=50,
                    verbose=False)
        elif name == "lightgbm":
            mdl.fit(X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    eval_metric="rmse",
                    callbacks=[lgb.early_stopping(50),
                               lgb.log_evaluation(-1)])
        else:  # xgboost
            mdl.fit(X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    eval_metric="rmse",
                    early_stopping_rounds=50,
                    verbose=False)

        # Store OOF predictions
        meta_train[val_idx, m_idx] = mdl.predict(X_val)

        # Store test predictions per fold
        test_fold_preds.append(mdl.predict(X_test))

    # Average test predictions across folds
    meta_test[:, m_idx] = np.mean(test_fold_preds, axis=0)

# ================================
# Train Meta-Model
# ================================
meta_model = LinearRegression()
meta_model.fit(meta_train, y_train)

# Final stacked predictions
stacked_preds = meta_model.predict(meta_test)

# ================================
# Evaluate
# ================================
rmse = mean_squared_error(y_test, stacked_preds, squared=False)
print(f"\nFinal Stacked Model Test RMSE: {rmse:.4f}")



def feature_engineering(df):
    df_features = df.copy()
    
    # Interaction features
    df_features['Rhythm_x_Instrumental'] = (
        df_features['RhythmScore'] * df_features['InstrumentalScore']
    )
    df_features['Quality_Index'] = (
        df_features['RhythmScore'] + df_features['InstrumentalScore'] + df_features['AcousticQuality']
    )
    df_features['Energy_Loudness_Ratio'] = (
        df_features['Energy'] / (df_features['AudioLoudness'].abs() + 1)
    )
    
    # Polynomial features
    df_features['RhythmScore_Squared'] = df_features['RhythmScore'] ** 2
    df_features['Energy_Squared'] = df_features['Energy'] ** 2
    
    return df_features



final_test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
# Train features
X_train_fe = feature_engineering(X_train)

# Apply same transformation to test
X_final = feature_engineering(final_test)

# Ensure column order is same as training
X_final = X_final[X_train_fe.columns]




# base model predictions
meta_final = np.zeros((len(X_final), len(models)))

for m_idx, (name, base_model) in enumerate(models.items()):
    print(f"Retraining {name} on FULL training data...")
    
    mdl = clone(base_model)

    if name == "catboost":
        mdl.fit(X_train, y_train, verbose=False)
    elif name == "lightgbm":
        mdl.fit(X_train, y_train,
                eval_set=[(X_train, y_train)],
                eval_metric="rmse",
                callbacks=[lgb.log_evaluation(-1)])
    else:  # xgboost
        mdl.fit(X_train, y_train,
                eval_set=[(X_train, y_train)],
                eval_metric="rmse",
                verbose=False)

    meta_final[:, m_idx] = mdl.predict(X_final)

# 4. Meta-model final predictions
final_preds = meta_model.predict(meta_final)



# 5. Save submission
submission = pd.DataFrame({
    "id": final_test["id"],  
    "BeatsPerMinute": final_preds
})
submission.to_csv("submission.csv", index=False)

print(" Submission file saved as submission.csv")

