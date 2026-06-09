%load_ext cuml.accel


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


# Standard libraries
import os
import time
import warnings
import optuna

# Data manipulation
import numpy as np
import pandas as pd

from time import time

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image, display

# Statistical and mathematical functions
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist

# Machine learning - preprocessing
from sklearn.preprocessing import OneHotEncoder, RobustScaler, OrdinalEncoder

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline, make_pipeline

# Machine learning - models
from sklearn.linear_model import LinearRegression, RidgeCV, Ridge, ElasticNetCV, ElasticNet
from sklearn.ensemble import VotingRegressor, StackingRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.base import clone, BaseEstimator, TransformerMixin
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor, early_stopping
from catboost import CatBoostRegressor


from sklearn.neural_network import MLPRegressor

# Machine learning - model evaluation
from sklearn.model_selection import (
    KFold, 
    cross_validate, 
    train_test_split, 
    cross_val_score
)
from sklearn.metrics import make_scorer, mean_squared_log_error

# display all columns
pd.set_option('display.max_columns', None)

# Suppress warnings
warnings.filterwarnings(
    "ignore",
    message="Found unknown categories in columns.*during transform.",
    category=UserWarning
)
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but .* was fitted with feature names",
    category=UserWarning
)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

#setup of trails for tuning
TRAILS = 10


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv',index_col=0)
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv', index_col=0)
submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')


train.shape, test.shape, submission.shape


train.head()


def dataframe_summary(df):
    
    info_df = pd.DataFrame({
        'Feature': df.columns,
        'Non-Null Count': df.notnull().sum().values,
        'Dtype': df.dtypes.values
    })

    # Descriptive statistics
    describe_df = df.describe(include='all').transpose().reset_index().rename(columns={'index': 'Feature'})

    # Missing values per column
    missing_df = df.isna().sum().reset_index()
    missing_df.columns = ['Feature', 'Missing Values']

    # Unique value counts per column
    unique_vals = df.nunique().reset_index()
    unique_vals.columns = ['Feature', 'Unique Values']

    # Mode and frequency for categorical columns
    mode_df = df.mode().transpose().reset_index().rename(columns={'index': 'Feature', 0: 'Mode'})
    freq_df = df.apply(lambda x: x.value_counts().iloc[0] if x.dtype == 'object' else None).reset_index().rename(columns={'index': 'Feature', 0: 'Most Frequent Frequency'})

    # Merge all pieces 
    summary_df = info_df.merge(describe_df, on='Feature', how='left')
    summary_df = summary_df.merge(missing_df, on='Feature', how='left')
    summary_df = summary_df.merge(unique_vals, on='Feature', how='left')
    summary_df = summary_df.merge(mode_df, on='Feature', how='left')
    summary_df = summary_df.merge(freq_df, on='Feature', how='left')

    # Rearranging columns for readability
    preferred_order = ['Feature', 'Dtype', 'Non-Null Count', 'Missing Values', 'Unique Values', 'Mode', 'Most Frequent Frequency',
                       'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    summary_df = summary_df[[col for col in preferred_order if col in summary_df.columns]]

    # Count duplicated rows
    duplicate_count = df.duplicated().sum()

    return summary_df, duplicate_count


summary, dup_count = dataframe_summary(train)

print(f"\nTotal duplicated rows: {dup_count}")
summary.style.background_gradient(cmap='viridis', subset=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])


summary, dup_count = dataframe_summary(test)

print(f"\nTotal duplicated rows: {dup_count}")
summary.style.background_gradient(cmap='viridis', subset=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])


fig, axes = plt.subplots(1, 3, figsize=(20, 6))

sns.histplot(data=train, x='Calories', hue='Sex', kde=True, ax=axes[0], 
             palette='Set1', element='step', bins=30, alpha=0.1)
axes[0].set_title('Distribution of Calories by Sex', fontsize=14)
axes[0].set_xlabel('Calories', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].grid(axis='y', alpha=0.3)
axes[0].legend()

mean_calories = train['Calories'].mean()
median_calories = train['Calories'].median()

sns.histplot(data=train, x='Calories', kde=True, ax=axes[1], 
             color='purple', bins=30, alpha=0.4)
axes[1].set_title('Overall Distribution of Calories', fontsize=14)
axes[1].set_xlabel('Calories', fontsize=12)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].grid(axis='y', alpha=0.3)
axes[1].axvline(mean_calories, color='red', linestyle='--', 
               label=f'Global Mean: {mean_calories:.2f}')
axes[1].axvline(median_calories, color='green', linestyle='-', 
               label=f'Global Median: {median_calories:.2f}')
axes[1].legend()

sns.boxplot(data=train, y='Calories', x='Sex', hue='Sex', ax=axes[2], 
            palette='Set1', width=0.5, showfliers=False)
axes[2].set_title('Boxplot of Calories by Sex', fontsize=14)
axes[2].set_xlabel('Sex', fontsize=12)
axes[2].set_ylabel('Calories', fontsize=12)
axes[2].grid(axis='y', alpha=0.3)


plt.tight_layout()
plt.show()


def scatter_plot_all_features(df, target_col, sample_size=750000):
    # Subsample if necessary
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)

    # Identify numeric columns (excluding target)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    # Pre-filter male and female data
    male_df = df[df['Sex'] == 'male']
    female_df = df[df['Sex'] == 'female']

    sex_colors = {'male': 'blue', 'female': 'red'}

    # Set up the figure
    n_rows = len(numeric_cols)
    fig, axes = plt.subplots(nrows=n_rows, ncols=3, figsize=(20, 4.5 * n_rows))
    axes = axes.reshape(n_rows, 3)  # Ensures 2D indexing even for one row

    # Turn off interactive mode to speed up rendering
    plt.ioff()

    for i, col in enumerate(numeric_cols):
        # Combined
        sns.scatterplot(data=df, x=col, y=target_col, hue='Sex', palette=sex_colors,
                        ax=axes[i, 0], alpha=0.6, legend=False)
        axes[i, 0].set_title(f'Combined: {col} vs {target_col}')
        axes[i, 0].grid(alpha=0.3)

        # Male
        sns.scatterplot(data=male_df, x=col, y=target_col, color=sex_colors['male'],
                        ax=axes[i, 1], alpha=0.6)
        axes[i, 1].set_title(f'Male only: {col} vs {target_col}')
        axes[i, 1].grid(alpha=0.3)

        # Female
        sns.scatterplot(data=female_df, x=col, y=target_col, color=sex_colors['female'],
                        ax=axes[i, 2], alpha=0.6)
        axes[i, 2].set_title(f'Female only: {col} vs {target_col}')
        axes[i, 2].grid(alpha=0.3)

    # Layout and save
    plt.tight_layout()
    plt.ion()  # Reactivate interactive mode
    plt.show()
    
scatter_plot_all_features(train, 'Calories', sample_size=100000)


def plot_correlation_matrix(df):    
    
    corr = df.select_dtypes(include=[np.number]).corr(method="spearman")
    
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=False, fmt=".2f", cmap='coolwarm',
                square=True, cbar_kws={"shrink": .8}, linewidths=0.5, ax=axes[0])
    axes[0].set_title('Full Correlation Matrix\n(Spearman, Upper Triangle Masked)', fontsize=14)
    
    
    if 'Calories' in corr.columns:
        calories_corr = corr['Calories'].sort_values(ascending=False)
        colors = ['green' if val > 0 else 'red' for val in calories_corr.values]
        sns.barplot(x=calories_corr.values, y=calories_corr.index, palette=colors, ax=axes[1])
        axes[1].set_title('Features Correlation with Calories\n(Spearman)', fontsize=14)
        axes[1].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        axes[1].set_xlim(-1, 1)
        axes[1].grid(axis='x', alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "Calories column not found", ha='center', fontsize=14)
        axes[1].set_title('Features Correlation with Calories', fontsize=14)
    
    # 3. Strong correlations (|r| > 0.7), removing duplicates
    strong_corr = corr.stack().reset_index()
    strong_corr.columns = ['Feature_1', 'Feature_2', 'Correlation']
    strong_corr = strong_corr[strong_corr['Feature_1'] != strong_corr['Feature_2']]
    strong_corr['pair'] = strong_corr.apply(lambda x: tuple(sorted([x['Feature_1'], x['Feature_2']])), axis=1)
    strong_corr = strong_corr.drop_duplicates(subset='pair').drop(columns='pair')
    strong_corr = strong_corr[abs(strong_corr['Correlation']) > 0.7]
    strong_corr = strong_corr.sort_values('Correlation', ascending=False)

    if not strong_corr.empty:
        pivot_table = strong_corr.pivot(index='Feature_1', columns='Feature_2', values='Correlation')
        sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5, ax=axes[2])
        axes[2].set_title('Strong Feature Correlations\n(|r| > 0.7)', fontsize=14)
    else:
        axes[2].text(0.5, 0.5, "No strong correlations found", ha='center', fontsize=14)
        axes[2].set_title('Strong Feature Correlations', fontsize=14)
    
    
    plt.tight_layout()
    plt.show()

    
    # Interpretation
    print("\n๐ Key Insights on Multicollinearity and Feature Relevance:")
    print("1. Top features correlated with Calories:", ", ".join(calories_corr.head(5).index.tolist()))

    # Potential leakage detection
    possible_leak = calories_corr[calories_corr.abs() > 0.95].drop("Calories", errors='ignore')
    if not possible_leak.empty:
        print("\nโ�๏ธ Potential Data Leakage Detected!")
        for feature, corr_value in possible_leak.items():
            print(f"   โข {feature} (r = {corr_value:.2f})")

    print(f"\n2. Found {len(strong_corr)} pairs of features with strong correlations (|r| > 0.7).")
    if not strong_corr.empty:
        print("3. Top highly correlated feature pairs:")
        for _, row in strong_corr.head(5).iterrows():
            print(f"   โข {row['Feature_1']} & {row['Feature_2']}: r = {row['Correlation']:.2f}")

    print("\n4. Consider removing or combining highly correlated features to reduce multicollinearity.\n")

plot_correlation_matrix(train)


def create_features(df):
    df = df.copy()

    # 1. BMI Features
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    df['BMI_Category'] = pd.cut(
        df['BMI'],
        bins=[0, 18.5, 24.9, 29.9, 100],
        labels=['Underweight', 'Normal weight', 'Overweight', 'Obese']
    )

    # 2. Interaction & Ratio Features
    df['HR_per_kg'] = df['Heart_Rate'] / df['Weight']
    df['Age_Weight_Ratio'] = df['Age'] / df['Weight']
    df['Age_Duration_Interaction'] = df['Age'] * df['Duration']
    df['HRxDuration'] = df['Heart_Rate'] * df['Duration']
    df['Temp_x_Duration'] = df['Body_Temp'] * df['Duration']
    df['Weight_x_Duration'] = df['Weight'] * df['Duration']

    # 3. Age Grouping
    df['Age_Group'] = pd.cut(df['Age'], bins=[19, 30, 45, 60, 80], labels=['Young', 'Adult', 'Mid-age', 'Senior'])
    
    # 3. Metabolic/Cardio Features
    df['Max_HR'] = 220 - df['Age']
    df['HRR_Percent'] = (df['Heart_Rate'] - 60) / (df['Max_HR'] - 60)

    # 4. HR Zone Binning
    df['HR_Zone'] = pd.cut(
        df['Heart_Rate'],
        bins=[0, 100, 120, 140, 160, 180, 220],
        labels=['Very Light', 'Light', 'Moderate', 'Hard', 'Very Hard', 'Maximum']
    )
    # One-hot encoding for HR Zone
    hr_zone_dummies = pd.get_dummies(df['HR_Zone'], prefix='HR_Zone')
    df = pd.concat([df, hr_zone_dummies], axis=1)
    
    # 5. Log Features (nonlinear, controlled)
    df['Log_Duration'] = np.log1p(df['Duration'])

    # 6. Effort Score (Duration * Heart Rate)
    df['Effort_Score'] = df['Duration'] * df['Heart_Rate']

    # 7. ThermoCardio Index (Body_Temp * Heart Rate)
    df['ThermoCardio_Index'] = df['Body_Temp'] * df['Heart_Rate']
    
    # 8. Squared Features
    df['BMI_Squared'] = df['BMI'] ** 2
    df['Age_Squared'] = df['Age'] ** 2
    df['HR_Squared'] = df['Heart_Rate'] ** 2
    
    # Temperature Normalization or Deviation
    # Assuming 37.0 is the normal body temperature
    df['Temp_Deviation'] = df['Body_Temp'] - 37.0
    
    # A normalized exertion index
    df['Exertion_Index'] = (df['Heart_Rate'] * df['Duration'] * df['Body_Temp']) / df['Weight']
    
    # Additional physiological features
    df['HR_Reserve'] = df['Max_HR'] - df['Heart_Rate']  # Heart rate reserve
    df['VO2_Estimate'] = 15.3 * (df['Heart_Rate'] / df['Max_HR'])  # VO2 estimate
    
    # Metabolic equivalent features
    df['MET_Estimate'] = df['VO2_Estimate'] / 3.5  # Metabolic equivalent 
    
    # Exercise intensity ratio
    df['Intensity_Ratio'] = df['Heart_Rate'] / (220 - df['Age'])
    
    return df

train = create_features(train)
test = create_features(test)


X = train.drop(columns="Calories")
y = train["Calories"]

# Combined stratification keys to split the data 
stratify_col = X[['Sex', 'BMI_Category', 'Age_Group','HR_Zone']].astype(str).agg('-'.join, axis=1)

# Count frequencies
group_counts = stratify_col.value_counts()

# Mask for valid groups (with at least 2 samples)
valid_mask = stratify_col.isin(group_counts[group_counts >= 2].index)

# Filter data
X_valid = X[valid_mask]
y_valid = y[valid_mask]
stratify_col_valid = stratify_col[valid_mask]

# Now split
X_train, X_hold, y_train, y_hold = train_test_split(
    X_valid, y_valid,
    test_size=1/3,
    random_state=42,
    stratify=stratify_col_valid
)


# Define numerical and categorical columns
numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
ordinal_cols = ['Age_Group', 'HR_Zone']
binarty_cols = ['Sex']

# 1๏ธโฃ Numerical feature pipeline
numerical_pipeline = Pipeline([
    ("scaler", RobustScaler())
])

# 2๏ธโฃ ordinal feature pipeline OrdinalEncoder for 'Age_Group' and 'HR_Zone'
ordinal_pipeline = Pipeline([
    ("encoder", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

# 3๏ธโฃ Binary feature pipeline
binary_pipeline = Pipeline([
    ("encoder", OneHotEncoder(handle_unknown='ignore', drop='first'))
])


# 3๏ธโฃ ColumnTransformer to apply pipelines to the right columns
preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_cols),
    ("ord", ordinal_pipeline, ordinal_cols),
    ("bin", binary_pipeline, binarty_cols)
])

# Final preprocessor to apply transformations
preprocessor


#Objective_catboost function for Optuna
def objective_catboost(trial):
    params = {
        'iterations':           trial.suggest_int('iterations', 500, 2000),
        'depth':                trial.suggest_int('depth', 4, 10),
        'learning_rate':        trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'bagging_temperature':  trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'l2_leaf_reg':          trial.suggest_float('l2_leaf_reg', 1e-2, 10.0, log=True),
        'random_strength':      trial.suggest_float('random_strength', 1e-2, 10.0, log=True),
        'border_count':         trial.suggest_int('border_count', 32, 255),
        'loss_function':        'RMSE',
        'eval_metric':          'RMSE',
        'random_seed':          42,
        'verbose':              100,
        'use_best_model':       True,
        'task_type':            'GPU'
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []

    for fold_idx, (train_idx, valid_idx) in enumerate(kf.split(X_train), 1):
        X_tr_raw, X_va_raw = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_tr, y_va         = y_train.iloc[train_idx], y_train.iloc[valid_idx]

        prep = clone(preprocessor)
        X_tr = prep.fit_transform(X_tr_raw, y_tr)
        X_va = prep.transform(X_va_raw)

        model = CatBoostRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=(X_va, y_va),
            early_stopping_rounds=50,
            verbose=False
        )

        preds = model.predict(X_va).clip(0, 314)
        fold_scores.append(np.sqrt(mean_squared_log_error(y_va, preds)))

        trial.report(np.mean(fold_scores), step=fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(fold_scores)

# โโโ 3. Run the study_catboost โโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโ
study_catboost = optuna.create_study(
    direction='minimize',
    study_name="CatBoost_CV",
    storage="sqlite:///db.sqlite3",
    load_if_exists=True
)
study_catboost.optimize(objective_catboost, n_trials=TRAILS, show_progress_bar=True)


print("\nBest params:", study_catboost.best_params)
print("\nBest CV RMSLE:", study_catboost.best_value)

# โโโ 4. Train final model โโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโ
preprocessor_full = clone(preprocessor).fit(X_train, y_train)

final_catbbost_pipe = Pipeline([
    ("pre", preprocessor_full),
    ("model", CatBoostRegressor(
        **study_catboost.best_params,
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=42,
        verbose=100
    ))
])
final_model_catboost = TransformedTargetRegressor(
    regressor=final_catbbost_pipe,
    func=np.log1p,
    inverse_func=np.expm1
)

final_model_catboost.fit(X_train, y_train)
y_hold_pred = final_model_catboost.predict(X_hold).clip(0, 314)
hold_rmsle_catboost = np.sqrt(mean_squared_log_error(y_hold, y_hold_pred))
print("\nHold-out RMSLE:", hold_rmsle_catboost)

final_model_catboost


def objective_lgb(trial):
    params = {
        'n_estimators':      trial.suggest_int('n_estimators', 500, 2000),
        'max_depth':         trial.suggest_int('max_depth', 4, 12),
        'num_leaves':        trial.suggest_int('num_leaves', 16, 64),
        'learning_rate':     trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
        'reg_alpha':         trial.suggest_float('reg_alpha', 1e-2, 10.0, log=True),
        'reg_lambda':        trial.suggest_float('reg_lambda', 1e-2, 10.0, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
        'min_split_gain':    trial.suggest_float('min_split_gain', 0.0, 1.0),
        'subsample':         trial.suggest_float('subsample', 0.5, 0.8),
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.5, 0.8),
        'subsample_freq':    trial.suggest_int('subsample_freq', 1, 10),
        'objective':         'regression',
        'metric':            'rmse',
        'boosting_type':     'gbdt',
        'device':            'gpu',
        'random_state':      42,
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []

    for fold_idx, (train_idx, valid_idx) in enumerate(kf.split(X_train), 1):
        X_tr_raw, X_va_raw = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_tr, y_va         = y_train.iloc[train_idx], y_train.iloc[valid_idx]

        # Preprocess
        prep = clone(preprocessor)
        X_tr = prep.fit_transform(X_tr_raw, y_tr)
        X_va = prep.transform(X_va_raw)

        # Logโtransform
        y_tr_log = np.log1p(y_tr)
        y_va_log = np.log1p(y_va)

        # Fit with callbacks for early stopping
        model = LGBMRegressor(**params, verbose=-1)
        model.fit(
            X_tr, y_tr_log,
            eval_set=[(X_va, y_va_log)],
            callbacks=[
                early_stopping(stopping_rounds=50),
            ]
        )

        # Predict & invert
        preds = np.expm1(model.predict(X_va)).clip(0, 314)

        # Score
        fold_scores.append(np.sqrt(mean_squared_log_error(y_va, preds)))

        trial.report(np.mean(fold_scores), step=fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(fold_scores)


# โโโ Run the study โโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโ
study_lgb = optuna.create_study(
    direction='minimize',
    study_name="LGBM_CV",
    storage="sqlite:///db.sqlite3",
    load_if_exists=True
)
study_lgb.optimize(objective_lgb, n_trials=TRAILS, show_progress_bar=True)

print("\nBest LGBM params:", study_lgb.best_params)
print("\nBest CV RMSLE:", study_lgb.best_value)

# โโโ Final model โโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโ
preprocessor_full = clone(preprocessor).fit(X_train, y_train)

# Build pipeline + TransformedTargetRegressor for final
final_lgb_pipe = Pipeline([
    ("pre", preprocessor_full),
    ("model", LGBMRegressor(
        **study_lgb.best_params,
        objective='regression',
        metric='rmse',
        random_state=42,
        verbose=-1
    ))
])
final_model_lgb = TransformedTargetRegressor(
    regressor=final_lgb_pipe,
    func=np.log1p,
    inverse_func=np.expm1
)

# Train and evaluate
final_model_lgb.fit(X_train, y_train)
y_hold_pred_lgb = final_model_lgb.predict(X_hold).clip(0, 314)
rmsle_hold_lgb = np.sqrt(mean_squared_log_error(y_hold, y_hold_pred_lgb))
print("\nLGBM Hold-out RMSLE:", rmsle_hold_lgb)

final_model_lgb


def objective_xgb(trial):
    params = {
        'n_estimators':     trial.suggest_int('n_estimators', 500, 2000),
        'max_depth':        trial.suggest_int('max_depth', 3, 12),
        'learning_rate':    trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
        'gamma':            trial.suggest_float('gamma', 0.0, 5.0),
        'reg_alpha':        trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda':       trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'subsample':        trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'tree_method':      'gpu_hist',
        'objective':        'reg:squarederror',
        'eval_metric':      'rmse',
        'verbosity':        0,
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []

    for fold_idx, (train_idx, valid_idx) in enumerate(kf.split(X_train), 1):
        # Split
        X_tr_raw, X_va_raw = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_tr, y_va         = y_train.iloc[train_idx], y_train.iloc[valid_idx]

        # Preprocess
        prep = clone(preprocessor)
        X_tr = prep.fit_transform(X_tr_raw, y_tr)
        X_va = prep.transform(X_va_raw)

        # Log-transform target
        y_tr_log = np.log1p(y_tr)
        y_va_log = np.log1p(y_va)

        # Fit (no early stopping)
        model = XGBRegressor(**params, random_state=42)
        model.fit(
            X_tr,
            y_tr_log,
            eval_set=[(X_va, y_va_log)],
            verbose=False
        )

        # Predict & invert
        preds_log = model.predict(X_va)
        preds = np.expm1(preds_log).clip(0, 314)

        rmsle = np.sqrt(mean_squared_log_error(y_va, preds))
        fold_scores.append(rmsle)

        # Report & maybe prune
        trial.report(np.mean(fold_scores), step=fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(fold_scores)


# โโ Run the study โโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโ
study_xgb = optuna.create_study(
    direction='minimize',
    study_name="XGB_CV",
    storage="sqlite:///db.sqlite3",
    load_if_exists=True
)
study_xgb.optimize(objective_xgb, n_trials=TRAILS, show_progress_bar=True)


# โโ Final model โโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโ
preprocessor_full = clone(preprocessor).fit(X_train, y_train)
final_xgb_pipe = Pipeline([
    ("pre", preprocessor_full),
    ("model", XGBRegressor(
        **study_xgb.best_params,
        objective='reg:squarederror',
        tree_method='hist',
        verbosity=1, 
        random_state=42
    ))
])
final_model_xgb = TransformedTargetRegressor(
    regressor=final_xgb_pipe,
    func=np.log1p,
    inverse_func=np.expm1
)

final_model_xgb.fit(X_train, y_train)
y_hold_pred_xgb = final_model_xgb.predict(X_hold).clip(0, 314)
rmsle_hold_xgb = np.sqrt(mean_squared_log_error(y_hold, y_hold_pred_xgb))
print("\nXGB Hold-out RMSLE:", rmsle_hold_xgb)

final_model_xgb


def objective_extratrees(trial):
    params = {
        'n_estimators':     trial.suggest_int('n_estimators', 50, 200),
        'max_depth':        trial.suggest_int('max_depth', 3, 12),
        'min_samples_split':trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features':     trial.suggest_float('max_features', 0.1, 1.0),
        'bootstrap':        trial.suggest_categorical('bootstrap', [True, False]),
        'random_state':     42,
        'verbose':          0,
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []

    for fold_idx, (train_idx, valid_idx) in enumerate(kf.split(X_train), 1):
        # Split
        X_tr_raw, X_va_raw = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_tr, y_va         = y_train.iloc[train_idx], y_train.iloc[valid_idx]

        # Preprocess
        prep = clone(preprocessor)
        X_tr = prep.fit_transform(X_tr_raw, y_tr)
        X_va = prep.transform(X_va_raw)

        # Log-transform target
        y_tr_log = np.log1p(y_tr)

        model = ExtraTreesRegressor(**params)
        model.fit(X_tr,y_tr_log)
        
        # Predict & invert
        preds_log = model.predict(X_va)
        preds = np.expm1(preds_log).clip(0, 314)
        
        rmsle = np.sqrt(mean_squared_log_error(y_va, preds))
        fold_scores.append(rmsle)
        # Report & maybe prune
        trial.report(np.mean(fold_scores), step=fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return np.mean(fold_scores)

# โโ Run the study โโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโ
study_extratrees = optuna.create_study(
    direction='minimize',
    study_name="ExtraTrees_CV",
    storage="sqlite:///db.sqlite3",
    load_if_exists=True
)
study_extratrees.optimize(objective_extratrees, n_trials=TRAILS, show_progress_bar=True)
# โโ Final model โโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโ
preprocessor_full = clone(preprocessor).fit(X_train, y_train)
final_extratrees_pipe = Pipeline([
    ("pre", preprocessor_full),
    ("model", ExtraTreesRegressor(
        **study_extratrees.best_params,
        random_state=42
    ))
])
final_model_extratrees = TransformedTargetRegressor(
    regressor=final_extratrees_pipe,
    func=np.log1p,
    inverse_func=np.expm1
)
final_model_extratrees.fit(X_train, y_train)
y_hold_pred_extratrees = final_model_extratrees.predict(X_hold).clip(0, 314)
rmsle_hold_extratrees = np.sqrt(mean_squared_log_error(y_hold, y_hold_pred_extratrees))
print("\nExtraTrees Hold-out RMSLE:", rmsle_hold_extratrees)

final_model_extratrees


def objective_mlp(trial):
    params = {
        'hidden_layer_sizes': tuple([trial.suggest_int('hidden_units', 50, 200)] * trial.suggest_int('n_layers', 1, 3)),
        'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
        'solver': trial.suggest_categorical('solver', ['adam', 'lbfgs']),
        'alpha': trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
        'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True),
        'max_iter': 1000,
        'random_state': 42,
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []

    for fold_idx, (train_idx, valid_idx) in enumerate(kf.split(X_train), 1):
        X_tr_raw, X_va_raw = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_tr, y_va = y_train.iloc[train_idx], y_train.iloc[valid_idx]

        prep = clone(preprocessor)
        X_tr = prep.fit_transform(X_tr_raw, y_tr)
        X_va = prep.transform(X_va_raw)

        y_tr_log = np.log1p(y_tr)

        model = MLPRegressor(**params)
        model.fit(X_tr, y_tr_log)

        preds_log = model.predict(X_va)
        preds = np.expm1(preds_log).clip(0, 314)
        rmsle = np.sqrt(mean_squared_log_error(y_va, preds))
        fold_scores.append(rmsle)

        trial.report(np.mean(fold_scores), step=fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(fold_scores)

study_mlp = optuna.create_study(
    direction='minimize',
    study_name="MLP_CV",
    storage="sqlite:///db.sqlite3",
    load_if_exists=True
)
study_mlp.optimize(objective_mlp, n_trials=TRAILS, show_progress_bar=True)

preprocessor_full = clone(preprocessor).fit(X_train, y_train)

best = study_mlp.best_params.copy()

# hidden_layer_sizes from trial parameters
hidden_layer_sizes = tuple([best.pop('hidden_units')] * best.pop('n_layers'))

final_mlp_pipe = Pipeline([
    ("pre", preprocessor_full),
    ("model", MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        **best,
        max_iter=1000,
        random_state=42
    ))
])

final_model_mlp = TransformedTargetRegressor(
    regressor=final_mlp_pipe,
    func=np.log1p,
    inverse_func=np.expm1
)

final_model_mlp.fit(X_train, y_train)
y_hold_pred_mlp = final_model_mlp.predict(X_hold).clip(0, 314)
rmsle_hold_mlp = np.sqrt(mean_squared_log_error(y_hold, y_hold_pred_mlp))
print("\nMLP Hold-out RMSLE:", rmsle_hold_mlp)

final_model_mlp


def objective_hgbr(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'max_iter': trial.suggest_int('max_iter', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 100),
        'max_bins': trial.suggest_int('max_bins', 64, 255),
        'l2_regularization': trial.suggest_float('l2_regularization', 0.0, 1.0),
        'early_stopping': False,
        'random_state': 42
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []

    for fold_idx, (train_idx, valid_idx) in enumerate(kf.split(X_train), 1):
        X_tr_raw, X_va_raw = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_tr, y_va = y_train.iloc[train_idx], y_train.iloc[valid_idx]

        prep = clone(preprocessor)
        X_tr = prep.fit_transform(X_tr_raw, y_tr)
        X_va = prep.transform(X_va_raw)

        y_tr_log = np.log1p(y_tr)

        model = HistGradientBoostingRegressor(**params)
        model.fit(X_tr, y_tr_log)

        preds_log = model.predict(X_va)
        preds = np.expm1(preds_log).clip(0, 314)
        rmsle = np.sqrt(mean_squared_log_error(y_va, preds))
        fold_scores.append(rmsle)

        trial.report(np.mean(fold_scores), step=fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(fold_scores)
# โโ Run the study โโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโ
study_hgbr = optuna.create_study(
    direction='minimize',
    study_name="HGBR_CV",
    storage="sqlite:///db.sqlite3",
    load_if_exists=True
)
study_hgbr.optimize(objective_hgbr, n_trials=TRAILS, show_progress_bar=True)
# โโ Final model โโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโ
preprocessor_full = clone(preprocessor).fit(X_train, y_train)
final_hgbr_pipe = Pipeline([
    ("pre", preprocessor_full),
    ("model", HistGradientBoostingRegressor(
        **study_hgbr.best_params,
        random_state=42
    ))
])
final_model_hgbr = TransformedTargetRegressor(
    regressor=final_hgbr_pipe,
    func=np.log1p,
    inverse_func=np.expm1
)
final_model_hgbr.fit(X_train, y_train)
y_hold_pred_hgbr = final_model_hgbr.predict(X_hold).clip(0, 314)
rmsle_hold_hgbr = np.sqrt(mean_squared_log_error(y_hold, y_hold_pred_hgbr))
print("\nHGBR Hold-out RMSLE:", rmsle_hold_hgbr)

final_model_hgbr


def objective_elasticnet(trial):
    params = {
        'alpha': trial.suggest_float('alpha', 1e-4, 10.0, log=True),
        'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0),
        'max_iter': 10000,
        'random_state': 42
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []

    for fold_idx, (train_idx, valid_idx) in enumerate(kf.split(X_train), 1):
        X_tr_raw, X_va_raw = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_tr, y_va = y_train.iloc[train_idx], y_train.iloc[valid_idx]

        prep = clone(preprocessor)
        X_tr = prep.fit_transform(X_tr_raw, y_tr)
        X_va = prep.transform(X_va_raw)

        y_tr_log = np.log1p(y_tr)

        model = ElasticNet(**params, tol=1e-4)
        model.fit(X_tr, y_tr_log)

        preds_log = model.predict(X_va)
        preds = np.expm1(preds_log).clip(0, 314)
        rmsle = np.sqrt(mean_squared_log_error(y_va, preds))
        fold_scores.append(rmsle)

        trial.report(np.mean(fold_scores), step=fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(fold_scores)
# โโ Run the study โโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโ
study_elasticnet = optuna.create_study(
    direction='minimize',
    study_name="ElasticNet_CV",
    storage="sqlite:///db.sqlite3",
    load_if_exists=True
)
study_elasticnet.optimize(objective_elasticnet, n_trials=TRAILS, show_progress_bar=True)
# โโ Final model โโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโโ
preprocessor_full = clone(preprocessor).fit(X_train, y_train)
final_elasticnet_pipe = Pipeline([
    ("pre", preprocessor_full),
    ("model", ElasticNet(
        **study_elasticnet.best_params,
        tol=1e-4,
        random_state=42
    ))
])
final_model_elasticnet = TransformedTargetRegressor(
    regressor=final_elasticnet_pipe,
    func=np.log1p,
    inverse_func=np.expm1
)
final_model_elasticnet.fit(X_train, y_train)
y_hold_pred_elasticnet = final_model_elasticnet.predict(X_hold).clip(0, 314)
rmsle_hold_elasticnet = np.sqrt(mean_squared_log_error(y_hold, y_hold_pred_elasticnet))
print("\nElasticNet Hold-out RMSLE:", rmsle_hold_elasticnet)

final_model_elasticnet


results_df = pd.DataFrame({
    'Model': ['CatBoost', 'LightGBM', 'XGBoost', 'ExtraTrees', 'MLP', 'HistGradientBoosting', 'ElasticNet'],
    'Train RMSLE': [study_catboost.best_value, study_lgb.best_value, study_xgb.best_value, study_extratrees.best_value ,study_mlp.best_value, study_hgbr.best_value, study_elasticnet.best_value],
    'Hold-out RMSLE': [hold_rmsle_catboost, rmsle_hold_lgb, rmsle_hold_xgb, rmsle_hold_extratrees ,rmsle_hold_mlp, rmsle_hold_hgbr, rmsle_hold_elasticnet]
})

results_df


inv = np.array([
    1 / hold_rmsle_catboost,
    1 / rmsle_hold_lgb,
    1 / rmsle_hold_xgb,
    1 / rmsle_hold_extratrees,
    1 / rmsle_hold_mlp,
    1 / rmsle_hold_hgbr,
    1 / rmsle_hold_elasticnet,
    
])

# normalize weights
weights = (inv / inv.sum()).tolist()

print(f"Weights for the models: {weights}\n")
print(f"CatBoost weight: {weights[0]:.5f}")
print(f"XGBoost weight: {weights[1]:.5f}")
print(f"LightGBM weight: {weights[2]:.5f}")
print(f"ExtraTrees weight: {weights[3]:.5f}")
print(f"MLP weight: {weights[4]:.5f}")
print(f"HistGradientBoosting weight: {weights[5]:.5f}")
print(f"ElasticNet weight: {weights[6]:.5f}\n")


voter_full = VotingRegressor(
    estimators=[
        ('catboost', final_model_catboost),
        ('lgbm', final_model_lgb),
        ('xgb', final_model_xgb),
        ('extratrees', final_model_extratrees),
        ('mlp', final_model_mlp),
        ('hgbr', final_model_hgbr),
        ('elasticnet', final_model_elasticnet)
    ],
    weights=weights,
)
voter_full.fit(X_train, y_train)

y_train_pred_voter = np.clip(voter_full.predict(X_train), 0, 314)
rmsle_train_voter = np.sqrt(mean_squared_log_error(y_train, y_train_pred_voter))
print(f"\nVoting Regressor Train RMSLE: {rmsle_train_voter:.5f}")


y_hold_pred_voter = np.clip(voter_full.predict(X_hold), 0, 314)
rmsle_hold_voter = np.sqrt(mean_squared_log_error(y_hold, y_hold_pred_voter))
print(f"\nVoting Regressor Hold-out RMSLE: {rmsle_hold_voter:.5f}")

results_df = pd.concat([
    results_df,
    pd.DataFrame({
        'Model': ['Voting Regressor'],
        'Train RMSLE': [rmsle_train_voter],
        'Hold-out RMSLE': [rmsle_hold_voter]
    })
], ignore_index=True)

voter_full


stacking_model = StackingRegressor(
    estimators=[
        ("cat", final_model_catboost),
        ("xgb", final_model_xgb),
        ("lgb", final_model_lgb),
        ("extratrees", final_model_extratrees),
        ("mlp", final_model_mlp),
        ("hgbr", final_model_hgbr),
        ("elasticnet", final_model_elasticnet)
    ],
    final_estimator=Ridge(),
)

stacking_model.fit(X_train, y_train)

y_train_pred_stack = np.clip(stacking_model.predict(X_train), 0, 314)
rmsle_train_stack = np.sqrt(mean_squared_log_error(y_train, y_train_pred_stack))
print(f"\nStacking Regressor Train RMSLE: {rmsle_train_stack:.5f}\n")


# Predict
y_pred_hold_stack = np.clip(stacking_model.predict(X_hold), 0, 314)
rmsle_hold_stack = np.sqrt(mean_squared_log_error(y_hold, y_pred_hold_stack))
print(f"Stacking Regressor Hold-out RMSLE: {rmsle_hold_stack:.5f}")

results_df = pd.concat([
    results_df,
    pd.DataFrame({
        'Model': ['Stacking Regressor'],
        'Train RMSLE': [rmsle_train_stack],
        'Hold-out RMSLE': [rmsle_hold_stack]
    })
], ignore_index=True)

stacking_model


# weights of stacking model
stacking_weights = stacking_model.final_estimator_.coef_
stacking_weights = np.abs(stacking_weights) / np.sum(np.abs(stacking_weights))
print(f"\nStacking Regressor weights: {stacking_weights}")
print(f"CatBoost weight: {stacking_weights[0]:.5f}")
print(f"XGBoost weight: {stacking_weights[1]:.5f}")
print(f"LightGBM weight: {stacking_weights[2]:.5f}")
print(f"ExtraTrees weight: {stacking_weights[3]:.5f}")
print(f"MLP weight: {stacking_weights[4]:.5f}")
print(f"HistGradientBoosting weight: {stacking_weights[5]:.5f}")
print(f"ElasticNet weight: {stacking_weights[6]:.5f}")


results_df = results_df.sort_values(by='Hold-out RMSLE').reset_index(drop=True)
results_df


plt.figure(figsize=(12, 6))

ax = sns.barplot(x='Model', y='Hold-out RMSLE', data=results_df, palette='viridis')

for i, p in enumerate(ax.patches):
    ax.annotate(f'{p.get_height():.5f}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', va = 'bottom',
                xytext = (0, 5), textcoords = 'offset points')

plt.title('Model Performance Comparison (Hold-out RMSLE)', fontsize=14)
plt.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# predictions for all the mdoels
y_test_pred_catboost = np.clip(final_model_catboost.predict(test), 0, 314)
y_test_pred_xgb = np.clip(final_model_xgb.predict(test), 0, 314)
y_test_pred_lgb = np.clip(final_model_lgb.predict(test), 0, 314)
y_test_pred_voter = np.clip(voter_full.predict(test), 0, 314)
y_test_pred_stack = np.clip(stacking_model.predict(test), 0, 314)
y_test_pred_extratrees = np.clip(final_model_extratrees.predict(test), 0, 314)
y_test_pred_mlp = np.clip(final_model_mlp.predict(test), 0, 314)
y_test_pred_hgbr = np.clip(final_model_hgbr.predict(test), 0, 314)
y_test_pred_elasticnet = np.clip(final_model_elasticnet.predict(test), 0, 314)

predictions = pd.DataFrame({
    "CatBoost": y_test_pred_catboost,
    "XGBoost": y_test_pred_xgb,
    "LightGBM": y_test_pred_lgb,
    "ExtraTrees": y_test_pred_extratrees,
    "MLP": y_test_pred_mlp,
    "HistGradientBoosting": y_test_pred_hgbr,
    "ElasticNet": y_test_pred_elasticnet,
    "Voting Regressor": y_test_pred_voter,
    "Stacking Regressor": y_test_pred_stack
})
predictions.head()


plt.figure(figsize=(15, 6))
for col in predictions.columns:
    sns.kdeplot(predictions[col], label=col)
plt.title('Distribution of Predictions Across Models', fontsize=14)
plt.xlabel('Predicted Calories')
plt.ylabel('Density')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# --- Pairwise Scatterplots (Best vs All Others) ---
# best model from results_df (assumes it's available)
best_model = results_df.iloc[0]['Model']

# Sample for cleaner plots
sample_size = min(5000, len(predictions))
sample_idx = np.random.choice(predictions.index, sample_size, replace=False)
sample_preds = predictions.loc[sample_idx]

fig = plt.figure(figsize=(18, 18))
other_models = [m for m in predictions.columns if m != best_model]

for i, model in enumerate(other_models):
    ax = fig.add_subplot(3, 3, i + 1)
    ax.scatter(sample_preds[best_model], sample_preds[model], alpha=0.5, s=10)
    
    min_val = min(sample_preds[best_model].min(), sample_preds[model].min())
    max_val = max(sample_preds[best_model].max(), sample_preds[model].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--')

    ax.set_title(f'{best_model} vs {model}', fontsize=12)
    ax.set_xlabel(f'{best_model} Predictions')
    ax.set_ylabel(f'{model} Predictions')
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# --- 5. Mean Absolute Difference (MAD) Matrix ---
mad_matrix = pd.DataFrame(index=predictions.columns, columns=predictions.columns)
for col1 in predictions.columns:
    for col2 in predictions.columns:
        mad = (predictions[col1] - predictions[col2]).abs().mean()
        mad_matrix.loc[col1, col2] = mad

# Convert to float for heatmap
mad_matrix = mad_matrix.astype(float)

plt.figure(figsize=(10, 8))
sns.heatmap(mad_matrix, annot=True, fmt=".2f", cmap="viridis", square=True)
plt.title("Mean Absolute Difference Between Model Predictions")
plt.tight_layout()
plt.show()


def analyze_residuals(y_true, y_pred, title):
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Residual distribution
    sns.histplot(residuals, kde=True, ax=axes[0])
    axes[0].set_title('Residual Distribution')
    axes[0].axvline(0, color='red', linestyle='--')
    
    # Residuals vs Predicted
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.5, ax=axes[1])
    axes[1].set_title('Residuals vs Predicted Values')
    axes[1].axhline(0, color='red', linestyle='--')
    axes[1].set_xlabel('Predicted Values')
    axes[1].set_ylabel('Residuals')
    
    # QQ plot
    stats.probplot(residuals, plot=axes[2])
    axes[2].set_title('Q-Q Plot of Residuals')
    
    plt.suptitle(f'Residual Analysis: {title}', fontsize=16)
    plt.tight_layout()
    plt.show()
    
analyze_residuals(y_hold, y_pred_hold_stack, 'Picked Model')


# correlation distance
dist_matrix = pdist(predictions.T, metric='correlation')
linkage_matrix = linkage(dist_matrix, method='average')

# Dendrogram
plt.figure(figsize=(10, 6))
dendrogram(linkage_matrix, labels=predictions.columns, leaf_rotation=45)
plt.title("Clustering of Models Based on Prediction Similarity")
plt.tight_layout()
plt.show()


def get_feature_names(preprocessor):
    """Extract final feature names after ColumnTransformer preprocessing."""
    output_features = []
    for name, transformer, cols in preprocessor.transformers_:
        if transformer == 'drop':
            continue
        elif transformer == 'passthrough':
            output_features.extend(cols)
        elif hasattr(transformer, 'get_feature_names_out'):
            try:
                names = transformer.get_feature_names_out(cols)
            except TypeError:
                names = transformer.get_feature_names_out()
            output_features.extend(names)
        else:
            output_features.extend(cols)  # fallback
    return output_features

def plot_feature_importances(models_dict, preprocessor):
    importances = {}

    for name, model in models_dict.items():
        if hasattr(model, 'feature_importances_'):
            importances[name] = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances[name] = np.abs(model.coef_)
        else:
            print(f"โ�๏ธ Skipping {name} (no importances found)")

    feature_names = get_feature_names(preprocessor)

    n_models = len(importances)
    rows = int(np.ceil(n_models / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(18, 6 * rows))
    axes = axes.flatten()

    for i, (name, imp) in enumerate(importances.items()):
        if len(imp) == len(feature_names):
            features = feature_names
        else:
            features = [f'Feature {j}' for j in range(len(imp))]
            print(f"โ�๏ธ Feature name mismatch for {name}. Using generic names.")

        imp_df = pd.DataFrame({'Feature': features, 'Importance': imp})
        imp_df = imp_df.sort_values('Importance', ascending=False).head(15)

        sns.barplot(x='Importance', y='Feature', data=imp_df, ax=axes[i])
        axes[i].set_title(f'{name} Feature Importance', fontsize=14)
        axes[i].grid(axis='x', alpha=0.3)

    # Hide unused subplots if any
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

plot_feature_importances(
    {
        'CatBoost': final_model_catboost.regressor_.named_steps['model'],
        'XGBoost': final_model_xgb.regressor_.named_steps['model'],
        'LightGBM': final_model_lgb.regressor_.named_steps['model'],
        'ExtraTrees': final_model_extratrees.regressor_.named_steps['model'],
        'MLP': final_model_mlp.regressor_.named_steps['model'],  # MLP usually doesn't support importances
        'HistGradientBoosting': final_model_hgbr.regressor_.named_steps['model'],
        'ElasticNet': final_model_elasticnet.regressor_.named_steps['model']
        # Voting and Stacking regressors excluded, as they don't expose direct feature importances
    },
    preprocessor=final_model_catboost.regressor_.named_steps['pre']
)


# --- Predict on test set ---
X_test = test.copy()

# --- Prepare submission for best model ---
submission["Calories"] = y_test_pred_stack
submission.to_csv("submissions", index=False)
submission.head()

