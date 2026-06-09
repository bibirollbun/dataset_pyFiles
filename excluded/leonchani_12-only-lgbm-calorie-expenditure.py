# Table manipulation, calculating
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 100) # increase the maximum number of columns

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

# Learning
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import KFold
import lightgbm as lgb

# Saving model
import joblib

# Ignore all warnings
import warnings
warnings.simplefilter("ignore")


df_train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


df_train


df_test





df_train



sex_target_mean = df_train.groupby('Sex')['Calories'].mean()
sex_mapping = sex_target_mean.to_dict()
df_train['Sex_Encoded'] = df_train['Sex'].map(sex_mapping)
df_test['Sex_Encoded'] = df_test['Sex'].map(sex_mapping)


def create_binned_and_encoded_features(df_train, df_test, column_to_bin, target_column, bin_interval=5, suffix="_Binned_Encoded"):
    """
    Bin the specified numeric column and target encode the binned column.

    Args:
        df_train (pd.DataFrame): Training data frame.
        df_test (pd.DataFrame): Test dataframe.
        column_to_bin (str): Name of the numeric column to bin.ã€‚
        target_column (str): The name of the target column to use for target encoding.
        bin_interval (int or float): The interval between each bin. Default is 5.
        suffix (str): A suffix to add to newly generated column names. Defaults to "_Binned_Encoded".

    Returns:
        tuple:(df_train after processing, df_test after processing)
    """

    def generate_bins_and_labels(min_val, max_val, bin_interval):
        """
        Generates bins and labels used for binning time periods.

        Args:
            min_val (int or float): The minimum value of the column.
            max_val (int or float): The maximum value of the column.
            bin_interval (int or float): The interval between each bin.

        Returns:
            tuple: (list of bins, list of labels)
        """
        # Consider the max_val + bin_interval part of np.arange and adjust the last bin to include max_val
        bins = list(np.arange(min_val, max_val + bin_interval, bin_interval))

        # Generate labels until the last bin is reached
        labels = []
        for i in range(len(bins) - 1):
            # Labels are created in the format "lower limit - upper limit - 1"
            labels.append(f'{bins[i]}-{bins[i+1]-1}')
        return bins, labels

    # Get the minimum and maximum values â€‹â€‹of the binned column
    min_val = df_train[column_to_bin].min()
    max_val = df_train[column_to_bin].max()

    # Generate bins and labels
    bins, labels = generate_bins_and_labels(min_val, max_val, bin_interval)

    # New binned column names
    binned_encoded_column_name = f'{column_to_bin}{suffix}'

    # Binning columns
    df_train[binned_encoded_column_name] = pd.cut(df_train[column_to_bin], bins=bins, labels=labels, right=False)
    df_test[binned_encoded_column_name] = pd.cut(df_test[column_to_bin], bins=bins, labels=labels, right=False)

    # Target Encoding
    # Since NaN values may be included, you don't need to use dropna=False before groupby,
    # In case there is no bin with a NaN value during map processing, keys that do not exist in the dict will be set to NaN and used as is.
    target_mean_mapping = df_train.groupby(binned_encoded_column_name)[target_column].mean().to_dict()

    df_train[binned_encoded_column_name] = df_train[binned_encoded_column_name].map(target_mean_mapping)
    df_test[binned_encoded_column_name] = df_test[binned_encoded_column_name].map(target_mean_mapping)

    return df_train, df_test


df_train, df_test = create_binned_and_encoded_features(df_train, df_test, column_to_bin='Age', target_column='Calories', bin_interval=5)
df_train, df_test = create_binned_and_encoded_features(df_train, df_test, column_to_bin='Height', target_column='Calories', bin_interval=5)
df_train, df_test = create_binned_and_encoded_features(df_train, df_test, column_to_bin='Weight', target_column='Calories', bin_interval=5)
df_train, df_test = create_binned_and_encoded_features(df_train, df_test, column_to_bin='Duration', target_column='Calories', bin_interval=5)
df_train, df_test = create_binned_and_encoded_features(df_train, df_test, column_to_bin='Heart_Rate', target_column='Calories', bin_interval=5)
df_train, df_test = create_binned_and_encoded_features(df_train, df_test, column_to_bin='Body_Temp', target_column='Calories', bin_interval=5)


del df_train['Sex']
del df_train['Age']
del df_train['Height']
del df_train['Weight']
del df_train['Duration']
del df_train['Heart_Rate']
del df_train['Body_Temp']

del df_test['Sex']
del df_test['Age']
del df_test['Height']
del df_test['Weight']
del df_test['Duration']
del df_test['Heart_Rate']
del df_test['Body_Temp']


def standardize_dataframe(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Standardizes the columns of the specified DataFrame and returns the updated original DataFrame.
    
    Args:
        df (pd.DataFrame): The DataFrame to standardize.
        cols (list[str]): A list of column names to standardize.
    
    Returns:
        pd.DataFrame: The DataFrame with the specified columns standardized (modifies the original DataFrame).

    """
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(df[cols])
    scaled_values = scaler.transform(df[cols])
    df[cols] = scaled_values

    return df


# columns_to_standardize = ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads']
columns_to_drop = ['id', 'Calories']
columns_to_standardize = df_train.copy().drop(columns=columns_to_drop).columns

standardize_dataframe(df_train, columns_to_standardize)
standardize_dataframe(df_test, columns_to_standardize)





X = df_train.drop(columns=["id","Calories"])
# y = df_train["Calories"]
y = np.log1p(df_train["Calories"])


SEED = 42
NUM_SPLITS = 10 # try different folds


# Definition of the RMSLE evaluation function
def rmsle(y_true, y_pred):
    """Calculate the Root Mean Squared Logarithmic Error (RMSLE)"""
    y_pred_clipped = np.maximum(y_pred, 0) # Clip predicted values to be non-negative
    squared_log_error = (np.log1p(y_true) - np.log1p(y_pred_clipped)) ** 2
    return np.sqrt(np.mean(squared_log_error))

# Custom evaluation function for LightGBM (RMSLE)
def lgbm_rmsle(y_true, y_pred):
    """Function for evaluating RMSLE in LightGBM"""
    return 'RMSLE', rmsle(y_true, y_pred), False # eval_name, eval_result, is_higher_better


# Hyperparameters (initial values, room for tuning)

params = {   
    # Parameters that have a relatively small or indirect effect on the prediction
    'objective': 'regression',
    'metric': 'rmse',        # Key evaluation metrics during training (also using custom evaluation functions)
    'boosting_type': 'gbdt', # 'gbdt', 'dart', 'goss', 'rf'
    'verbosity': -1,         # -1 ~ 0 ~ 1
    'random_state': 42,
    'n_jobs': -1,

    # Parameters that are likely to have a large effect on the prediction
    'learning_rate': 0.05,   # 0.01 ~ 0.05 ~ 0.1.
    'n_estimators': 2000,    # 100 ~ 500 ~ 1,000?
    'max_depth': 7,          # 3 ~ 7 ~ 10.    
    'num_leaves': 32,        # 32 ~ 63 ~ 128. A value slightly smaller than 2^max_depth is recommended.
    'subsample': 0.8,        # 0.7 ~ 0.8 ~ 1.0.
    'colsample_bytree': 0.8, # 0.7 ~ 0.8 ~ 1.0.

    # Parameters that may have a moderate impact on the forecast
    'min_child_samples': 20, # 10 ~ 20 ~ 100?.
    'lambda_l1': 0.1,        # 0 ~ 0.1 ~
    'lambda_l2': 0.1,        # 0 ~ 0.1 ~
}

# K-Fold Cross-Validation
kf = KFold(n_splits=NUM_SPLITS, shuffle=True, random_state=SEED)
rmsle_scores = []
models = [] # A list to store trained models

for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
    print(f"Fold {fold+1}")
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric=lgbm_rmsle, # Use a custom RMSLE evaluation function
              callbacks=[lgb.early_stopping(100, verbose=False)])

    y_pred_val = model.predict(X_val)
    rmsle_val = rmsle(y_val, y_pred_val) # Calculate the RMSLE on validation data
    rmsle_scores.append(rmsle_val)
    models.append(model) # Save the trained model

# Show cross-validation results
print("\nCross-validation RMSLE scores:", rmsle_scores)
print(f'Optimized Cross-validated RMSLE score: {np.mean(rmsle_scores):.3f} +/- {np.std(rmsle_scores):.3f}')
print(f'Max RMSLE score: {np.max(rmsle_scores):.3f}')
print(f'Min RMSLE score: {np.min(rmsle_scores):.3f}')

print("\nCross-validation RMSLE scores:", np.expm1(rmsle_scores))
print(f'Optimized Cross-validated RMSLE score: {np.mean(np.expm1(rmsle_scores)):.3f} +/- {np.std(np.expm1(rmsle_scores)):.3f}')
print(f'Max RMSLE score: {np.max(np.expm1(rmsle_scores)):.3f}')
print(f'Min RMSLE score: {np.min(np.expm1(rmsle_scores)):.3f}')





# np.expm1(y_val) Version

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# feature_importance = model.feature_importances_
feature_importance = model.booster_.feature_importance(importance_type='gain')
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by="Importance", ascending=False)

# Feature Importance
sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis", ax=axes[0, 0])
axes[0, 0].set_title("Feature Importance (LightGBM)")
axes[0, 0].set_xlabel("Importance Score")
axes[0, 0].set_ylabel("Features")

# Actual vs. Predicted
sns.scatterplot(x=np.expm1(y_val), y=np.expm1(y_pred_val), alpha=0.6, edgecolors="k", ax=axes[0, 1])
# sns.scatterplot(x=y_val, y=y_pred_val, alpha=0.6, edgecolors="k", ax=axes[0, 1])
axes[0, 1].plot([min(np.expm1(y_val)), max(np.expm1(y_val))], [min(np.expm1(y_val)), max(np.expm1(y_val))], '--r', linewidth=2) 
axes[0, 1].set_title("Actual vs. Predicted Calorie Expenditure")
axes[0, 1].set_xlabel("Actual Values")
axes[0, 1].set_ylabel("Predicted Values")

# Residual Distribution
residuals = np.expm1(y_val) - np.expm1(y_pred_val)
# residuals = y_val - y_pred_val
sns.histplot(residuals, bins=30, kde=True, color='blue', ax=axes[1, 0])
axes[1, 0].axvline(0, color='red', linestyle='--')
axes[1, 0].set_title("Residual Distribution")
axes[1, 0].set_xlabel("Residuals")
axes[1, 0].set_ylabel("Frequency")

# Q-Q Plot of Residuals
stats.probplot(residuals, plot=axes[1, 1])
axes[1, 1].set_title("Q-Q Plot of Residuals")
axes[1, 1].set_xlabel("Theoretical Quantiles")
axes[1, 1].set_ylabel("Ordered Residuals")

plt.tight_layout()
plt.show()
plt.tight_layout()
plt.show()


# Original Version

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# feature_importance = model.feature_importances_
feature_importance = model.booster_.feature_importance(importance_type='gain')
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by="Importance", ascending=False)

# Feature Importance
sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis", ax=axes[0, 0])
axes[0, 0].set_title("Feature Importance (LightGBM)")
axes[0, 0].set_xlabel("Importance Score")
axes[0, 0].set_ylabel("Features")

# Actual vs. Predicted
sns.scatterplot(x=y_val, y=y_pred_val, alpha=0.6, edgecolors="k", ax=axes[0, 1])
axes[0, 1].plot([min(np.expm1(y_val)), max(np.expm1(y_val))], [min(np.expm1(y_val)), max(np.expm1(y_val))], '--r', linewidth=2) 
axes[0, 1].set_title("Actual vs. Predicted Calorie Expenditure")
axes[0, 1].set_xlabel("Actual Values")
axes[0, 1].set_ylabel("Predicted Values")

# Residual Distribution
residuals = y_val - y_pred_val
sns.histplot(residuals, bins=30, kde=True, color='blue', ax=axes[1, 0])
axes[1, 0].axvline(0, color='red', linestyle='--')
axes[1, 0].set_title("Residual Distribution")
axes[1, 0].set_xlabel("Residuals")
axes[1, 0].set_ylabel("Frequency")

# Q-Q Plot of Residuals
stats.probplot(residuals, plot=axes[1, 1])
axes[1, 1].set_title("Q-Q Plot of Residuals")
axes[1, 1].set_xlabel("Theoretical Quantiles")
axes[1, 1].set_ylabel("Ordered Residuals")

plt.tight_layout()
plt.show()
plt.tight_layout()
plt.show()





# import shap

# # LGBM SHAP values
# explainer_lgb = shap.TreeExplainer(model)
# shap_values_lgb = explainer_lgb.shap_values(X)


# shap.summary_plot(shap_values_lgb, X)


# # If shap_values_lgb is a list, convert it to a NumPy array
# if isinstance(shap_values_lgb, list):
#     shap_values_lgb = np.array(shap_values_lgb)

# # Handling the multiclass classification case
# if len(shap_values_lgb.shape) == 3:
#     shap_importance = np.abs(shap_values_lgb).mean(axis=1).mean(axis=0)
# # Handling binary classification cases
# else:
#     shap_importance = np.abs(shap_values_lgb).mean(axis=0)

# # Store in DataFrame
# df_importance = pd.DataFrame({
#     'feature': X.columns,
#     'shap_importance': shap_importance
# })

# # Sort by importance
# df_importance = df_importance.sort_values('shap_importance', ascending=False)

# # Show results
# display(df_importance)





test_id = df_test["id"]
test = df_test.drop(columns=['id'])
submit_score = []

for fold_, model in enumerate(models):
    # predict test data
    pred_ = model.predict(test)
    submit_score.append(pred_)

# predict test data
pred = np.mean(submit_score, axis=0)





# Visualizing Distribution of target variable

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# np.expm1 Distribution
sns.histplot(np.expm1(y), bins=30, kde=True, color='red', ax=axes[0, 0])
axes[0, 0].set_title("Train Object Variable Distribution (expm1)")
axes[0, 0].set_xlabel("Predicted calorie expenditure (y)")
axes[0, 0].set_ylabel("Frequency")

sns.histplot(np.expm1(y_pred_val), bins=30, kde=True, color='green', ax=axes[0, 1])
axes[0, 1].set_title("Validation Predictions Distribution (expm1)")
axes[0, 1].set_xlabel("Predicted calorie expenditure (y_pred_val)")
axes[0, 1].set_ylabel("Frequency")

sns.histplot(np.expm1(pred), bins=30, kde=True, color='blue', ax=axes[0, 2])
axes[0, 2].set_title("Test Predictions Distribution (expm1)")
axes[0, 2].set_xlabel("Predicted calorie expenditure (pred)")
axes[0, 2].set_ylabel("Frequency")

# Original Distribution
sns.histplot(y, bins=30, kde=True, color='red', ax=axes[1, 0])
axes[1, 0].set_title("Train Object Variable Distribution (Original)")
axes[1, 0].set_xlabel("Predicted calorie expenditure (y)")
axes[1, 0].set_ylabel("Frequency")

sns.histplot(y_pred_val, bins=30, kde=True, color='green', ax=axes[1, 1])
axes[1, 1].set_title("Validation Predictions Distribution (Original)")
axes[1, 1].set_xlabel("Predicted calorie expenditure (y_pred_val)")
axes[1, 1].set_ylabel("Frequency")

sns.histplot(pred, bins=30, kde=True, color='blue', ax=axes[1, 2])
axes[1, 2].set_title("Test Predictions Distribution (Original)")
axes[1, 2].set_xlabel("Predicted calorie expenditure (pred)")
axes[1, 2].set_ylabel("Frequency")

plt.tight_layout()
plt.show()





submission = pd.DataFrame({
    'id': test_id,
    'Calories': pred
})

submission['Calories'] = np.expm1(submission['Calories'])
submission['Calories'] = submission['Calories'].apply(lambda x: 0 if x < 0 else x)

# Save
submission.to_csv('submission.csv', index=False)

submission





display(len(models))
display(models)


# Saving Models
joblib.dump(models,'LightGBM.joblib')

# Loading Models
light_gbm =joblib.load('LightGBM.joblib')
light_gbm





!pip install watermark


%load_ext watermark
%watermark -n -u -v -iv -w -p pytensor,aeppl,xarray




