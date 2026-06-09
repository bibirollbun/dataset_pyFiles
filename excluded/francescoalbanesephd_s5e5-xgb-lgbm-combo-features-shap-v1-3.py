# %%capture
!pip install itables
!pip install scikit-learn==1.4.0
# !pip install optuna-integration[xgboost]==4.3.0
# !pip install optuna-integration[lightgbm]==4.3.0


import pandas as pd
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from itables import init_notebook_mode, show
init_notebook_mode(all_interactive=False,connected=True)

# Sets the seed for reproducibility in numpy, random, torch CPU, and CUDA.
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED) # For multi-GPU setups.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False #May slightly slow down training, but ensures reproducibility

# Set plot style
sns.set_style('whitegrid')

# Silence FutureWarning
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# Import datasets (on Kaggle)
TRAIN_DF = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv',index_col='id')
TEST_DF = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv',index_col='id')

# Import datasets (on Colab)
# import os
# TRAIN_DF = pd.read_csv(os.path.join(playground_series_s5e5_path, 'train.csv'),index_col='id')
# TEST_DF = pd.read_csv(os.path.join(playground_series_s5e5_path, 'test.csv'),index_col='id')


# For each dataset...
for dataset in [TRAIN_DF, TEST_DF]:
    # check their shape
    print(dataset.shape)
    # and fix column names
    dataset.columns = [col.lower() for col in dataset.columns]


# Dummy encode 'sex' feature (male == 1)
for df in [TRAIN_DF,TEST_DF]:
    df['sex_male'] = (df['sex'] == 'male').astype(int)
    del df['sex']


# Display descriptive stats TRAIN_DF
print('\n',"="*50,f"TRAIN_DF description","="*50)
descriptive_stats_train = TRAIN_DF.describe().T.round(2)
descriptive_stats_train['Skew'] = TRAIN_DF.skew()
descriptive_stats_train['Kurt'] = TRAIN_DF.kurt()
display(descriptive_stats_train)

# Display descriptive stats of TEST_DF
print('\n',"="*50,f"TEST_DF description","="*50)
descriptive_stats_test = TEST_DF.describe().T.round(2)
descriptive_stats_test['Skew'] = TEST_DF.skew()
descriptive_stats_test['Kurt'] = TEST_DF.kurt()
display(descriptive_stats_test)


import missingno as msno
# plt.figure(figsize=(5,10))

# Visualize missing values as a matrix
for df, df_name in [
    (TRAIN_DF, "Train"),
    (TEST_DF, "Test")
]:
    msno.matrix(df,figsize=(12,6))
    plt.title(f"Missing Data in {df_name} Dataset: {df.isnull().sum().sum()}",
             fontsize=30)
    plt.show()


# TRAIN_DF Overview
print("="*30,"Show Training Dataset for initial data assessment","="*30)
show(TRAIN_DF)


# Identify Target
target = 'calories'

# Define a custom color palette
custom_colors = ['#219ebc', '#c1121f']

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 4))
annot_kws = {'xy': (0.6, 0.8), 'xycoords': 'axes fraction', 'fontsize': 10}

# Box plot
sns.boxplot(data=TRAIN_DF, x=target, ax=axes[0], color=custom_colors[0])
# sns.stripplot(data=TRAIN_DF, x=target, ax=axes[0], color=custom_colors[0], jitter=True, alpha=0.01)
axes[0].set_xlabel(target)
axes[0].set_title(f"Box Plot of {target}")

# Histogram
sns.histplot(data=TRAIN_DF, x=target, kde=True, bins=30, ax=axes[1], color=custom_colors[0])
axes[1].set_xlabel(target)
axes[1].set_ylabel("Frequency")
axes[1].set_title(f"Histogram of {target}")
axes[1].annotate(f"Skewness (TRAIN): {TRAIN_DF[target].skew():.2f}\nKurtosis (TRAIN): {TRAIN_DF[target].kurt():.2f}",
                 xy=annot_kws['xy'], xycoords=annot_kws['xycoords'], fontsize=annot_kws['fontsize'])



# Analysis of all NUMERICAL features
# Define a custom color palette
custom_palette = ['#219ebc', '#c1121f']

# Function to create and display plots for a single numerical variable
def create_variable_plots(train, test, variable):

    # Merge data for visualization (without modifying original DataFrames)
    train_temp = train.copy()
    test_temp = test.copy()
    train_temp["Dataset"] = "Train"
    test_temp["Dataset"] = "Test"
    combined_data = pd.concat([train_temp, test_temp])

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    annot_kws = {'xy': (0.03, 0.75), 'xycoords': 'axes fraction', 'fontsize': 10}

    # Box plot
    sns.boxplot(data=combined_data, x=variable, y="Dataset", palette=custom_palette, ax=axes[0])
    axes[0].set_xlabel(variable)
    axes[0].set_title(f"Box Plot of {variable}")

    # Histogram
    sns.histplot(data=train, x=variable, color=custom_palette[0], kde=True, bins=30, label="Train", ax=axes[1])
    sns.histplot(data=test, x=variable, color=custom_palette[1], kde=True, bins=30, label="Test", ax=axes[1])
    axes[1].set_xlabel(variable)
    axes[1].set_ylabel("Frequency")
    axes[1].set_title(f"Histogram of {variable} [Train, Test]")
    axes[1].legend()
    axes[1].annotate(f"Skewness (TRAIN): {train[variable].skew():.2f}\nKurtosis (TRAIN): {train[variable].kurt():.2f}",
                     xy=annot_kws['xy'], xycoords=annot_kws['xycoords'], fontsize=annot_kws['fontsize'])


    # Adjust spacing and show
    plt.tight_layout()
    plt.show()


# Perform univariate analysis for each numerical variable
for variable in TRAIN_DF.columns.difference([target]):
    create_variable_plots(TRAIN_DF, TEST_DF, variable)


# Create a heatmap to visualize the correlation matrix of the TRAIN_DF DataFrame
plt.figure(figsize=(12,8))
sns.heatmap(data=TRAIN_DF.corr().round(4),
            annot=True,
            cmap='coolwarm',
            linewidth = 0.3
           ); plt.show()

# Target Only
plt.figure(figsize=(12,3))
sns.heatmap(data=TRAIN_DF.corr()[[target]].T.round(4),
            annot=True,
            cmap='coolwarm',
            linewidth = 0.3
           ); plt.show()


# # Pairplot
# sns.pairplot(data=TRAIN_DF,
#              kind='reg',             # Use 'reg' for regression plots (including scatter)
#              diag_kind='kde',        # Show KDE on the diagonal
#              plot_kws={'scatter_kws': {'alpha': 0.05, 'color': custom_colors[1]},
#                        'lowess': True}) # Enable LOWESS and set colors

# plt.suptitle('Pairplot with LOWESS Regression Lines', y=1.02)
# plt.show()


def iqr_outlier_capping(train, valid=None, test=None, columns=None):
    """
    Applies IQR-based outlier capping to specified columns of one, two, or three DataFrames.

    Parameters:
        train (pd.DataFrame): The training DataFrame used to calculate IQR thresholds.
        valid (pd.DataFrame, optional): The validation DataFrame to cap using train thresholds.
        test (pd.DataFrame, optional): The test DataFrame to cap using train thresholds.
        columns (list, optional): List of column names to apply capping to. If None, applies to all numerical columns.

    Returns:
        tuple: A tuple containing:
            - train_capped (pd.DataFrame): Capped training DataFrame.
            - valid_capped (pd.DataFrame or None): Capped validation DataFrame (if provided).
            - test_capped (pd.DataFrame or None): Capped test DataFrame (if provided).

    Note: Make sure there are no nans
    """
    train_capped = train.copy() # Avoid modifying the original DataFrame
    valid_capped = valid.copy() if valid is not None else None
    test_capped = test.copy() if test is not None else None

    if columns is None:
        columns = train.select_dtypes(include='number').columns.tolist()  # All numerical columns

    # Calculate IQR-based thresholds from the training set
    for col in columns:
        Q1 = np.percentile(train[col], 25)
        Q3 = np.percentile(train[col], 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Show Values
        print(f'Columns {col}: \tLower Bound is: {lower_bound:.2f} \tUpper Bound is: {upper_bound:.2f}')

        # Cap outliers in the training set
        train_capped[col] = np.clip(train_capped[col], lower_bound, upper_bound)

        # If validation set is provided, cap using training set thresholds
        if valid is not None:
            valid_capped[col] = np.clip(valid[col], lower_bound, upper_bound)

        # If test set is provided, cap using training set thresholds
        if test is not None:
            test_capped[col] = np.clip(test[col], lower_bound, upper_bound)

    return train_capped, valid_capped, test_capped

# Cap target outliers in train and validation sets
TRAIN_capped, _, TEST_capped = iqr_outlier_capping(TRAIN_DF, None, TEST_DF, columns=TRAIN_DF.select_dtypes('number').columns.difference([target]))


# Log the target
log_y = np.log1p(TRAIN_capped['calories'])


# from tqdm import tqdm
# from itertools import combinations

# train_sample = TRAIN_capped.iloc[:10].copy()
# test_sample = TEST_capped.iloc[:10].copy()

# columns_to_encode = train_sample.columns.difference(['calories','sex_male'])

# pair_size = [2, 3, 4]

# for r in pair_size:
#     combinations_list = list(combinations(columns_to_encode,r))
#     batch_size = 20

#     for i in range(0, len(combinations_list), batch_size):
#         batch = combinations_list[i:i+batch_size]
#         for cols in tqdm(batch):
#             new_col_name = '_'.join(cols)

#             train_sample[new_col_name] = train_sample[list(cols)].astype(str).agg('_'.join, axis=1)
#             # train_sample[new_col_name] = train_sample[new_col_name].astype('category')

#             test_sample[new_col_name] = test_sample[list(cols)].astype(str).agg('_'.join, axis=1)
#             # test_sample[new_col_name] = test_sample[new_col_name].astype('category')


from tqdm import tqdm
from itertools import combinations

# Make small samples-copies for testing purposes
train_sample = TRAIN_capped.iloc[:10].copy()
test_sample = TEST_capped.iloc[:10].copy()

# Select columns to encode
columns_to_encode = TRAIN_capped.columns.difference(['calories','sex_male'])

# ==========================================================================
# INTERACTIONS

# Decide combo size
combo_size = [2, 3, 4]

# Make combos
for n in combo_size:
    combinations_list = list(combinations(columns_to_encode, n))
    for cols in tqdm(combinations_list):
      new_col_name = '_x_'.join(cols)  # Feature name

      # Calculate the interaction term (product)
      TRAIN_capped[new_col_name] = TRAIN_capped[list(cols)].prod(axis=1)
      TEST_capped[new_col_name] = TEST_capped[list(cols)].prod(axis=1)

# ==========================================================================
# RATIOS
combinations_list = list(combinations(columns_to_encode, 2)) # ratios are done in pairs
for cols in tqdm(combinations_list):
  col1, col2 = cols # Get the two columns for the ratio

  # Create two ratio features: col1 / col2 and col2 / col1
  new_col_name_ratio1 = f'{col1}_/_{col2}'
  new_col_name_ratio2 = f'{col2}_/_{col1}'

  # Add a small epsilon to the denominator to avoid division by zero
  Îµ = 1e-5

  # Calculate the ratio term
  TRAIN_capped[new_col_name_ratio1] = TRAIN_capped[col1] / (TRAIN_capped[col2] + Îµ)
  TEST_capped[new_col_name_ratio1] = TEST_capped[col1] / (TEST_capped[col2] + Îµ)

  TRAIN_capped[new_col_name_ratio2] = TRAIN_capped[col2] / (TRAIN_capped[col1] + Îµ)
  TEST_capped[new_col_name_ratio2] = TEST_capped[col1] / (TEST_capped[col1] + Îµ)


print("Modified TRAIN_capped with interaction terms:")
show(TRAIN_capped)
print("\nModified TEST_capped with interaction terms:")
show(TEST_capped)


# Define feature engineering function
def feature_engineer(df):
    Îµ = 1e-5
    df['BMI'] = df['weight'] / ((df['height'] / 100) ** 2)
    # [...]

    return df

# Apply function to datasets
feature_engineer(TRAIN_capped)
feature_engineer(TEST_capped)


from lightgbm import LGBMRegressor
import shap

# Define X and y
X = TRAIN_capped.copy()
X.pop(target)

# Train model
LGBM_model = LGBMRegressor(random_state=SEED, objective='regression', metric='rmsle', n_jobs=-1).fit(X, log_y)

# Use SHAP
explainer = shap.TreeExplainer(LGBM_model)
shap_values = explainer.shap_values(X)

# Global feature importance
shap.summary_plot(shap_values, X, max_display=25)  # Visualization
shap_df = pd.DataFrame(shap_values, columns=X.columns)
mean_abs_shap = pd.DataFrame(shap_df.abs().mean().sort_values(ascending=False),columns=['SHAP'])


# Example of local feature importance
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0], X.iloc[0])


from sklearn.feature_selection import mutual_info_regression

mutual_info = mutual_info_regression(X, log_y, random_state=SEED)

mutual_info = pd.Series(mutual_info)
mutual_info.index = X.columns
mutual_info = pd.DataFrame(mutual_info.sort_values(ascending=False), columns=['Mutual Information'])


# Embedded feature selection
importances = pd.DataFrame(LGBM_model.feature_importances_, index=X.columns, columns=['Importance'])


# Concatenate the results of shap, mutual information and feature_importances with style.bar to compare feature importances
pd.concat([
    mean_abs_shap,
    mutual_info,
    importances
    ], axis=1).style.bar(cmap='coolwarm')


# NOT USED YET

# from sklearn.model_selection import train_test_split

# def objective_xgb(trial):
#     params = {
#         "n_estimators": trial.suggest_int("n_estimators", 100, 3000, step=50),
#         "max_depth": trial.suggest_int("max_depth", 3, 12),
#         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
#         "subsample": trial.suggest_float("subsample", 0.6, 1.0),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
#         "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
#         "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
#         "n_jobs": -1,
#         "random_state": SEED
#     }

#     # MI feature selection
#     mi_percentile = trial.suggest_int("percentile", 70, 100, step=5)
#     selector = SelectPercentile(mutual_info_regression, percentile=mi_percentile)

#     X_selected = selector.fit_transform(X, y)

#     # train/valid split for early stopping
#     X_train, X_valid, y_train, y_valid = train_test_split(
#         X_selected, y, test_size=0.2, random_state=SEED
#     )

#     model = XGBRegressor(**params)

#     model.fit(
#         X_train, y_train,
#         eval_set=[(X_valid, y_valid)],
#         early_stopping_rounds=50,
#         verbose=False
#     )

#     preds = model.predict(X_valid)
#     score = mean_squared_error(y_valid, np.maximum(preds, 0), squared=False)

#     trial.report(score, step=0)
#     if trial.should_prune():
#         raise optuna.exceptions.TrialPruned()

#     return score



from lightgbm import LGBMRegressor, early_stopping
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error, make_scorer
import time

def kfold_lgbm_predict_with_early_stopping(model, X, y, X_test, k, random_state, eval_metric_name):
    """
    Performs K-Fold cross-validation with early stopping and returns OOF and test predictions.

    Args:
        model: The scikit-learn compatible model to train (LGBMRegressor or XGBRegressor).
        X: Training features (pandas DataFrame).
        y: Training target (pandas Series).
        X_test: Test features (pandas DataFrame).
        k: Number of folds.
        random_state: Random state for KFold.
        eval_metric_name: The name of the evaluation metric to use during model fitting (e.g., 'rmsle' for LightGBM, 'rmse' for XGBoost).

    Returns:
        A tuple containing:
            - mean_valid_score: The mean validation score across folds.
            - oof_predictions: Out-of-fold predictions for the training data.
            - test_predictions: Mean predictions for the test data across folds.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    fold_scores = []

    for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
        print(f"\n{'#'*10} Fold {fold+1}/{k} {'#'*10}")

        x_train = X.iloc[train_idx].copy()
        y_train = y.iloc[train_idx]
        x_valid = X.iloc[valid_idx].copy()
        y_valid = y.iloc[valid_idx]

        start = time.time()

        # Fit the model with early stopping
        model.fit(
            x_train, y_train,
            eval_set=[(x_valid, y_valid)],
            eval_metric=eval_metric_name,
            callbacks=[early_stopping(stopping_rounds=100, verbose=100)] # Pass the callback object here
        )

        # Predict OOF and test
        oof_preds[valid_idx] = model.predict(x_valid)
        test_preds += model.predict(X_test)

        # Calculate fold score (on the original scale using expm1 and rmsle)
        fold_score = mean_squared_log_error(np.expm1(y_valid), np.expm1(oof_preds[valid_idx]), squared=False)
        fold_scores.append(fold_score)
        print(f"Fold {fold+1} RMSLE: {fold_score:.4f}")

        end = time.time()
        print(f"Fold {fold+1} finished in {end - start:.2f} seconds")

    mean_valid_score = np.mean(fold_scores)
    test_predictions = test_preds / k

    return mean_valid_score, oof_preds, test_predictions


# Combine base features and top engineered features (SHAP)
baseline_features = ['age', 'height', 'weight', 'duration', 'heart_rate', 'body_temp', 'sex_male']
additional_features = [
    'duration_x_heart_rate', 'body_temp_x_duration_x_heart_rate', 'body_temp_x_heart_rate', 'age_x_body_temp_x_duration_x_heart_rate',
    'body_temp_/_heart_rate', 'duration_/_heart_rate']

final_features_set = baseline_features + additional_features
final_features_set


# Define X, y, and X_test
X = TRAIN_capped[final_features_set].copy()
y = log_y
X_test = TEST_capped[final_features_set].copy()

# Define LGBM_model
LGBM_model = LGBMRegressor(random_state=SEED, objective='regression_l2', n_estimators=5000, learning_rate=0.02, n_jobs=-1, colsample_bytree=0.75, subsample=0.9)

# Create the LightGBM early stopping callback instance
lgbm_early_stopping_instance = early_stopping(stopping_rounds=100, verbose=100)

# Run the K-Fold cross-validation with early stopping for LightGBM
mean_score_lgbm, oof_predictions_lgbm, test_predictions_lgbm = kfold_lgbm_predict_with_early_stopping(
    model=LGBM_model,
    X=X,
    y=y,
    X_test=X_test,
    k=5,
    random_state=SEED,
    eval_metric_name='rmsle'
)

print(f"\nMean Cross-Validation RMSLE (LightGBM): {mean_score_lgbm:.4f}")


from xgboost import XGBRegressor

# Train model
# Define XGBoost model parameters
XGB_model = XGBRegressor(
    random_state=SEED,
    objective='reg:squarederror',
    n_jobs=-1,
).fit(X, log_y)

# Use SHAP
explainer = shap.TreeExplainer(XGB_model)
shap_values = explainer.shap_values(X)

# Global feature importance
shap.summary_plot(shap_values, X, max_display=25)  # Visualization
shap_df = pd.DataFrame(shap_values, columns=X.columns)
mean_abs_shap = pd.DataFrame(shap_df.abs().mean().sort_values(ascending=False),columns=['SHAP'])


# Example of local feature importance
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0], X.iloc[0])


from xgboost import XGBRegressor

def kfold_xgb_predict_with_early_stopping(model, X, y, X_test, k, random_state, eval_metric_name):
    """
    Performs K-Fold cross-validation with early stopping and returns OOF and test predictions.

    Args:
        model: The scikit-learn compatible model to train (LGBMRegressor or XGBRegressor).
        X: Training features (pandas DataFrame).
        y: Training target (pandas Series).
        X_test: Test features (pandas DataFrame).
        k: Number of folds.
        random_state: Random state for KFold.
        eval_metric_name: The name of the evaluation metric to use during model fitting (e.g., 'rmsle' for LightGBM, 'rmse' for XGBoost).

    Returns:
        A tuple containing:
            - mean_valid_score: The mean validation score across folds.
            - oof_predictions: Out-of-fold predictions for the training data.
            - test_predictions: Mean predictions for the test data across folds.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    fold_scores = []

    for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
        print(f"\n{'#'*10} Fold {fold+1}/{k} {'#'*10}")

        x_train = X.iloc[train_idx].copy()
        y_train = y.iloc[train_idx]
        x_valid = X.iloc[valid_idx].copy()
        y_valid = y.iloc[valid_idx]

        start = time.time()

        # Fit the model with early stopping
        model.fit(
            x_train, y_train,
            eval_set=[(x_train, y_train), (x_valid, y_valid)],
            # eval_metric=eval_metric_name,
        )

        # Predict OOF and test
        oof_preds[valid_idx] = model.predict(x_valid)
        test_preds += model.predict(X_test)

        # Calculate fold score (on the original scale using expm1 and rmsle)
        fold_score = mean_squared_log_error(np.expm1(y_valid), np.expm1(oof_preds[valid_idx]), squared=False)
        fold_scores.append(fold_score)
        print(f"Fold {fold+1} RMSLE: {fold_score:.4f}")

        end = time.time()
        print(f"Fold {fold+1} finished in {end - start:.2f} seconds")

    mean_valid_score = np.mean(fold_scores)
    test_predictions = test_preds / k

    return mean_valid_score, oof_preds, test_predictions


# Combine base features and top engineered features (SHAP)
baseline_features = ['age', 'height', 'weight', 'duration', 'heart_rate', 'body_temp', 'sex_male']
additional_features = [
    'duration_x_heart_rate', 'body_temp_x_duration_x_heart_rate', 'body_temp_x_heart_rate', 'age_x_body_temp_x_duration_x_heart_rate',
    'age_x_heart_rate', 'age_x_duration_x_heart_rate',
    'body_temp_/_heart_rate', 'duration_/_heart_rate']

final_features_set = baseline_features + additional_features
final_features_set


from xgboost import XGBRegressor

# Define X, y, and X_test
X = TRAIN_capped[final_features_set].copy()
y = log_y
X_test = TEST_capped[final_features_set].copy()

# Define XGBoost model parameters
XGB_model = XGBRegressor(
    random_state=SEED,
    objective='reg:squarederror',
    n_estimators=5000,
    learning_rate=0.02,
    early_stopping_rounds=50,
    n_jobs=-1,
    colsample_bytree=0.75,
    subsample=0.9,
    verbose=True
)

# Run the K-Fold cross-validation with early stopping for XGBoost
mean_score_xgb, oof_predictions_xgb, test_predictions_xgb = kfold_xgb_predict_with_early_stopping(
    model=XGB_model,
    X=X,
    y=y,
    X_test=X_test,
    k=5,
    random_state=SEED,
    eval_metric_name='rmse',
)

print(f"\nMean Cross-Validation RMSLE (XGBoost): {mean_score_xgb:.4f}")


# Backtransform the predictions to the original scale
combined_predictions = (test_predictions_lgbm * 0.7) + (test_predictions_xgb * 0.3)
test_predictions_original_scale = np.expm1(combined_predictions)

# Create Submission File
submission_df = pd.DataFrame({
    'id': list(TEST_capped.index),
    'Calories': test_predictions_original_scale
})

# Save to CSV
submission_df.to_csv('submission.csv', index=False)

# Display the first 5 rows
display(submission_df)

# Plot preds distribution
sns.histplot(submission_df['Calories'],kde=True)
plt.show()

