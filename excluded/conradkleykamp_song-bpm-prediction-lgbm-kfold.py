# Importing and loading necessary libraries and packages

# Fundamental libraries
import pandas as pd
import numpy as np

# Hiding warnings
import warnings
warnings.filterwarnings("ignore")

# Data viz
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style = 'white', palette = 'Set2')
pal = sns.color_palette('Set2')

# Scipy
from scipy.stats import skew, zscore

# Catboost
import catboost
from catboost.utils import eval_metric

# Sklearn
from sklearn.metrics import roc_curve, roc_auc_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold

# LightGBM
import lightgbm as lgb
from lightgbm import plot_importance

# Shap
import shap
shap.initjs()

# Optuna
import optuna


# Loading in the Kaggle datasets
df_train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')


# Viewing the first 5 rows in df_train
df_train.head()


# Viewing the info of df_train
df_train.info()


# Creating eda function (can be applied to both target variable and feature variables)

def eda(df, columns, method='iqr', z_thresh=3, plot=True):
    """
    This function performs exploratory data analysis on a given numeric feature/target:
    - Prints skewness
    - Detects outliers (IQR or Z-score)
    - Plots boxplot and histogram with outliers highlighted
    - Returns summary statistics with outlier info
    
    Parameters:
        df (pd.DataFrame): Input dataframe
        columns (str or list): Column(s) to analyze
        method (str): 'iqr' or 'zscore'
        z_thresh (float): Z-score threshold
        plot (bool): Whether to show boxplot + histogram
    """
    if isinstance(columns, str):
        columns = [columns]  # make it a list for uniform handling
    
    summaries = []
    
    for column in columns:
        data = df[column].dropna()
        skewness = data.skew()
        
        # Detect outliers
        if method == 'iqr':
            Q1, Q3 = data.quantile([0.25, 0.75])
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            outliers = data[(data < lower) | (data > upper)]
        elif method == 'zscore':
            z_scores = zscore(data)
            outliers = data[abs(z_scores) > z_thresh]
        else:
            raise ValueError("Method must be 'iqr' or 'zscore'")
        
        print(f"\n=== {column} ===")
        print(f"Skewness: {skewness:.4f}")
        print(f"Detected {len(outliers)} outliers using {method} method.")
        
        # Visualization (optional)
        if plot:
            
            plt.figure(figsize=(12,5))
            
            # Boxplot
            plt.subplot(1,2,1)
            sns.boxplot(x=data)
            plt.scatter(outliers, [0]*len(outliers), color='red', label='Outliers')
            plt.title(f'Boxplot of {column}')
            plt.legend()
            
            # Histogram
            plt.subplot(1,2,2)
            sns.histplot(data, bins=30, kde=True)
            for outlier in outliers:
                plt.axvline(outlier, color='red', linestyle='--', alpha=0.5)
            plt.title(f'Histogram of {column}')
            
            plt.tight_layout()
            plt.show()
        
        # Collect summary
        summaries.append({
            "column": column,
            "count": len(data),
            "mean": data.mean(),
            "median": data.median(),
            "std": data.std(),
            "min": data.min(),
            "max": data.max(),
            "skewness": skewness,
            "num_outliers": len(outliers),
            "pct_outliers": len(outliers) / len(data) * 100
        })
    
    return pd.DataFrame(summaries)


# Applying eda function to 'BeatsPerMinute'
eda(df_train, 'BeatsPerMinute', method='zscore', plot=True)


# Declaring numeric features
data_numeric = df_train.drop(['BeatsPerMinute', 'id'], axis=1)
features = data_numeric.columns


# Applying eda function to numeric features
eda(df_train, features, method='iqr', plot=False)


# Visualizing correlation matrix of numeric features and target variable in training data
plt.figure(figsize=(14,10))
corr=df_train.corr()
sns.heatmap(corr,annot=True,cmap='coolwarm', linewidths=0.5, fmt=',.2f', vmax=1, vmin=-1, center=0, mask=np.triu(corr))
plt.suptitle('Correlation Matrix', fontsize=16, fontweight='bold')
plt.show()


def preprocess_and_engineer(df, method='zscore', is_train=True):
    
    # Making a fresh copy of the data
    df = df.copy()
    
    # --- Handling missing values (median imputation) ---
    # Not needed for this particular dataset but handy to have in general
    for col in df.columns:
        if df[col].dtype in ['float64']:
            df[col] = df[col].fillna(df[col].median())
    
    # --- Outlier handling (IQR or Z-score) ---
    # TRAIN ONLY, DO NOT APPLY TO TEST SET
    if is_train:
        # IQR method
        if method == 'iqr':
            num_cols = df.select_dtypes(include=['float64']).columns
            for col in num_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        # Zscore method
        elif method == 'zscore':
            num_cols = df.select_dtypes(include=['float64']).columns
            z_scores = df[num_cols].apply(zscore)
            df = df[(z_scores.abs() <= 3).all(axis=1)]
        # No outlier removal method
        else:
            pass

    # --- Transformations of features with high skewness ---
    # 'LogTrackDuration' --> Skew: 1.0363
    df['TrackDurationMs'] = np.log1p(df['TrackDurationMs'])
    # 'VocalContent' --> Skew: 0.7891
    df['VocalContent'] = np.sqrt(df['VocalContent'])
    # 'AcousticQuality' --> 0.7860
    df['AcousticQuality'] = np.sqrt(df['AcousticQuality'])
    
    # --- New features ---
    # Time-based features
    df['TrackDurationMin'] = df['TrackDurationMs'] / 60000
    df['RhythmPerMs'] = df['RhythmScore'] / (df['TrackDurationMs'] + 1)
    # Interactions
    df['LoudnessEnergy'] = df['AudioLoudness'] * df['Energy']
    df['RhythmEnergy'] = df['RhythmScore'] * df['Energy']
    df['AcousticVocal'] = df['AcousticQuality'] * df['VocalContent']
    df['MoodLive'] = df['MoodScore'] * df['LivePerformanceLikelihood'] 
    # Ratios
    # Establishing min_val to preserve magnitude of ratios
    min_val = 1e-3
    df['InstrVocalRatio'] = df['InstrumentalScore'] / df['VocalContent'].clip(lower=min_val)
    df['EnergyRhythmRatio'] = df['Energy'] / df['RhythmScore'].clip(lower=min_val)
    df['VocalAcousticRatio'] = df['VocalContent'] / df['AcousticQuality'].clip(lower=min_val)
    
    # --- Quantile rank features ---
    for col in ['Energy', 'MoodScore', 'TrackDurationMs']:
        df[f'{col}_Rank'] = df[col].rank(pct=True)
    
    return df

# Applying function to df_train and df_test
train = preprocess_and_engineer(df_train, method='', is_train=True)
test = preprocess_and_engineer(df_test, method='', is_train=False)


# Verifying changes
train.info()


test.info()


# Creating fresh copies of train and test sets
train = train.copy()
test = test.copy()


# Splitting train data into X and y
X = train.drop(columns=['BeatsPerMinute', 'id'])
y = train['BeatsPerMinute']
X_test = test.drop(['id'], axis=1)

# Splitting the train data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =========================================
# Baseline LGBM + KFold
# =========================================

# --- If using the baseline LGBM (no tuned hyperparameters), set run_lgbm to 1, otherwise 0 to skip ---
run_lgbm = 1

if run_lgbm == 1:
    
    print("=== Running Baseline LGBM + KFold ===")
    # Creating KFold cross validation with 5 splits
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Initializing array to store predictions, lists to store each model and validation RMSEs
    y_preds = np.zeros(len(X_test))
    models = []
    val_rmses = []
    
    # Looping through each fold created by KFold
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        # Printing fold progress
        print(f"Training fold {fold + 1}/{n_splits} >>>")
        # Splitting the data into training and validation sets for each current fold
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
        # Defining baseline LGBM parameters
        model = lgb.LGBMRegressor(
            n_estimators=20000,
            learning_rate=0.001,
            num_leaves=100,
            max_depth=10,
            min_child_samples=10,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_alpha=2.0,
            reg_lambda=1.0,
            random_state=42,
            verbosity=-1,
            boosting_type='gbdt',
            metric='rmse'
        )
    
        # Fitting LGBM model on training data, evaluating with validation data
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            # Stopping training if validation doesn't improve for 100 rounds
            # Logging validation metrics every 100 iterations
            callbacks=[
                lgb.early_stopping(100),
                lgb.log_evaluation(period=100)
            ]
        )
        
        # Saving model 
        models.append(model)
    
        # Making predictions on test set for current fold (average prediction across all models)
        y_preds += model.predict(X_test) / n_splits
        # Making predictions on validation data
        val_pred = model.predict(X_val)
        val_rmse = mean_squared_error(y_val, val_pred, squared=False)
        val_rmses.append(val_rmse)
    # Printing average validation RMSE across folds
    print(f"Mean RMSE: {np.mean(val_rmses):.6f}")


# =========================================
# Optional Optuna-tuned LGBM + KFold
# =========================================

# --- If using the tuned LGBM (with Optuna), set run_optuna to 1, otherwise 0 to skip ---
# NOTE: This only has to run once, you can record the resulting best_params (below) and set use_optuna_params to 1
run_optuna = 0  

if run_optuna == 1:
    
    print("=== Running LGBM Tuned with Optuna ===")
    # Defining objective, suggesting parameter value ranges
    def objective(trial):
        params = {
            "n_estimators": 20000,
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
            "random_state": 42,
            "verbosity": -1,
            "boosting_type": "gbdt",
            "metric": "rmse"
        }

        val_rmses = []
        for train_idx, val_idx in kf.split(X, y):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
            )

            preds = model.predict(X_val)
            rmse = mean_squared_error(y_val, preds, squared=False)
            val_rmses.append(rmse)

        return np.mean(val_rmses)

    # Running Optuna study
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    print("Best params from Optuna:", study.best_params)
    print("Best RMSE from Optuna:", study.best_value)


# Saving best parameters from Optuna trials (above)
# Once saved, these can be plugged in below to the final LGBM model
best_params_1 = {'learning_rate': 0.0025770920018536995, 
                 'num_leaves': 220, 
                 'max_depth': 6, 
                 'min_child_samples': 85, 
                 'subsample': 0.6333262521157302, 
                 'colsample_bytree': 0.5464230684325477, 
                 'reg_alpha': 2.7317532956083355, 
                 'reg_lambda': 2.766633462787757}


# =========================================
# Optional Optuna-tuned LGBM + KFold (CONTINUED)
# =========================================

# This section uses the recorded best_params_1 and builds a LGBM model and stress tests it with KFold

# --- If using the tuned LGBM (with Optuna), set use_optuna_params to 1, otherwise 0 to skip ---
use_optuna_params = 0

if use_optuna_params == 1:

    print("=== Running LGBM Tuned with Optuna ===")
    
    best_params = best_params_1
    best_params.update({
        "n_estimators": 20000,
        "random_state": 42,
        "verbosity": -1,
        "boosting_type": "gbdt",
        "metric": "rmse"
    })

    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    y_preds = np.zeros(len(X_test))
    models = []
    val_rmses = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"Training fold {fold + 1}/{n_splits} >>>")
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        model = lgb.LGBMRegressor(**best_params)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(100),
                lgb.log_evaluation(period=100)
            ]
        )
        
        models.append(model)
    
        y_preds += model.predict(X_test) / n_splits
        val_pred = model.predict(X_val)
        val_rmse = mean_squared_error(y_val, val_pred, squared=False)
        val_rmses.append(val_rmse)
    print(f"Mean RMSE: {np.mean(val_rmses):.6f}")


# Plotting validation RMSE across folds
# Converting to numpy array just in case
val_rmses = np.array(val_rmses)

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(val_rmses) + 1), val_rmses, marker='o', linestyle='-')
plt.axhline(y=np.mean(val_rmses), color='r', linestyle='--', label=f"Mean RMSE = {np.mean(val_rmses):.4f}")

plt.title("Validation RMSE Across Folds")
plt.xlabel("Fold")
plt.ylabel("RMSE")
plt.xticks(range(1, len(val_rmses) + 1))
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()


# Plotting feature importance

# Getting feature importances from all folds
feature_importances = np.zeros(X.shape[1])

for model in models:
    feature_importances += model.feature_importances_

# Averaging over folds
feature_importances /= len(models)

# Putting into a DataFrame for sorting/plotting
fi_df = pd.DataFrame({
    "feature": X.columns,
    "importance": feature_importances
}).sort_values(by="importance", ascending=False)

# Plotting
plt.figure(figsize=(10, 6))
plt.barh(fi_df["feature"], fi_df["importance"])
plt.gca().invert_yaxis()  # Most important at the top
plt.title("LGBM Feature Importance (Averaged Across Folds)")
plt.xlabel("Importance")
plt.show()


# Creating submission file, populated with predictions
submission = pd.DataFrame({"id": test["id"], "BeatsPerMinute": y_preds})
submission.to_csv("submission.csv", index=False)


# Viewing first 5 rows of submission dataframe
submission.head()

