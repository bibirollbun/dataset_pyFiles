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
import xgboost as xgb 

# Saving model
import joblib

# Ignore all warnings
import warnings
warnings.simplefilter("ignore")


df_train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


df_train


df_test





mapping = {'male': 1.5, 'female': 1.0}
df_train['Sex'] = df_train['Sex'].replace(mapping)
df_test['Sex']  = df_test['Sex'].replace(mapping)


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

# Custom evaluation function for XGBoost (RMSLE)
def xgb_rmsle(y_pred, y_true):
    """Function for evaluating RMSLE in XGBoost"""
    y_true = y_true.get_label()
    return 'RMSLE', rmsle(y_true, y_pred)


# Hyperparameters for XGBoost (initial values, room for tuning)
params_xgb = {
    # # Parameters that have a relatively small or indirect effect on the prediction
    # 'objective': 'reg:squarederror',  # Regression objective
    # 'eval_metric': 'rmse',           # Initial evaluation metric (will use custom RMSLE)
    # 'booster': 'gbtree',
    # "tree_method": "gpu_hist",
    # 'device': "cuda",
    # 'nthread': -1,                   # Equivalent to 'n_jobs'
    # 'seed': 42,                      # Equivalent to 'random_state'

    # # Parameters that are likely to have a large effect on the prediction
    # 'learning_rate': 0.02,           # 0.01 ~ 0.05 ~ 0.1.
    # 'n_estimators': 1000,            # 100 ~ 500 ~ 1,000?
    # 'max_depth': 7,                  # 3 ~ 7 ~ 10. Equivalent to max_depth
    # 'subsample': 0.8,                # 0.7 ~ 0.8 ~ 1.0.
    # 'colsample_bytree': 0.8,         # 0.7 ~ 0.8 ~ 1.0.
    # # 'max_bin': 896, 
    
    # # Parameters that may have a moderate impact on the forecast
    # 'min_child_weight': 8,          # Equivalent to min_child_samples (but meaning is slightly different)
    # 'lambda': 0.1,                   # L2 regularization (similar to lambda_l2)
    # 'alpha': 0.1,                    # L1 regularization (similar to lambda_l1)
    # 'gamma': 0.1

    'eval_metric': 'rmse',
    'seed': 42,
    'max_depth': 10,
    'learning_rate': 0.00075,
    'reg_alpha': 2,
    'reg_lambda': 1,
    'max_delta_step': 2,
    'subsample': 0.9,
    'colsample_bytree': 0.55,
    'n_estimators': 1000,            # 100 ~ 500 ~ 1,000?
    # 'enable_categorical': True,
    'device': "cuda"
}

# K-Fold Cross-Validation
kf = KFold(n_splits=NUM_SPLITS, shuffle=True, random_state=SEED)
rmsle_scores = []
models = [] # A list to store trained XGBoost models

for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
    print(f"Fold {fold+1}")
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    dtrain_xgb = xgb.DMatrix(X_train, label=y_train)
    dval_xgb = xgb.DMatrix(X_val, label=y_val)

    model_xgb = xgb.train(
        params_xgb,
        dtrain_xgb,
        num_boost_round=params_xgb['n_estimators'],
        evals=[(dval_xgb, 'val')],
        custom_metric=xgb_rmsle, # Use the custom RMSLE evaluation function
        early_stopping_rounds=100,
        verbose_eval=False
    )

    y_pred_val = model_xgb.predict(dval_xgb)
    rmsle_val = rmsle(y_val, y_pred_val) # Calculate the RMSLE on validation data
    rmsle_scores.append(rmsle_val)
    models.append(model_xgb) # Save the trained XGBoost model

# Show cross-validation results
print("\nCross-validation RMSLE scores:", rmsle_scores)
print(f'Optimized Cross-validated RMSLE score: {np.mean(rmsle_scores):.3f} +/- {np.std(rmsle_scores):.3f}')
print(f'Max RMSLE score: {np.max(rmsle_scores):.3f}')
print(f'Min RMSLE score: {np.min(rmsle_scores):.3f}')

print("\nCross-validation RMSLE scores (expm1 applied):", np.expm1(rmsle_scores))
print(f'Optimized Cross-validated RMSLE score (expm1 applied): {np.mean(np.expm1(rmsle_scores)):.3f} +/- {np.std(np.expm1(rmsle_scores)):.3f}')
print(f'Max RMSLE score (expm1 applied): {np.max(np.expm1(rmsle_scores)):.3f}')
print(f'Min RMSLE score (expm1 applied): {np.min(np.expm1(rmsle_scores)):.3f}')





# if isinstance(models, list):
#     for i, model in enumerate(models):
#         plt.figure(figsize=(10, 6)) 
#         xgb.plot_importance(model, ax=plt.gca(), importance_type='gain')
#         plt.title(f'Feature Importance - Model {i+1}')
#         plt.show()
# else:
#     # If models is a single model
#     plt.figure(figsize=(10, 6))
#     xgb.plot_importance(models, ax=plt.gca(), importance_type='gain')
#     plt.title('Feature Importance')
#     plt.show()


def get_feature_importance_df(model):
    """
    Gets feature importance from an XGBoost model and returns it as a data frame.
    
    Args:
        model: A trained XGBoost model (XGBClassifier/XGBRegressor or Booster)
    
    Returns:
        pandas.DataFrame: A data frame containing feature names and importances

    """

    if isinstance(model, xgb.Booster):
        importance = model.get_score(importance_type='gain')
        feature_names = list(model.feature_names) if hasattr(model, 'feature_names') else list(importance.keys())
    else:
        importance = model.get_booster().get_score(importance_type='gain')
        feature_names = list(model.get_booster().feature_names)

    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': list(importance.values())})
    importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
    return importance_df

if isinstance(models, list):
    all_importance_dfs = []
    for i, model in enumerate(models):
        importance_df = get_feature_importance_df(model)
        importance_df['Model'] = f'Model {i+1}'
        all_importance_dfs.append(importance_df)
    # Combine importance of all models
    combined_importance_df = pd.concat(all_importance_dfs).reset_index(drop=True)
else:
    # If models is a single model
    importance_df = get_feature_importance_df(models)


# np.expm1(y_val) Version

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

feature_importance = combined_importance_df.groupby(['Feature'], as_index = False, dropna = False)['Importance'].mean()
importance_df = feature_importance.sort_values(by="Importance", ascending=False)

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

feature_importance = combined_importance_df.groupby(['Feature'], as_index = False, dropna = False)['Importance'].mean()
importance_df = feature_importance.sort_values(by="Importance", ascending=False)

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

# Convert test data to DMatrix
dtest = xgb.DMatrix(test)

for fold_, model in enumerate(models):
    # predict test data
    pred_ = model.predict(dtest)
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

# submission['Calories'] = np.expm1(submission['Calories'])
submission['Calories'] = submission['Calories'].apply(lambda x: 0 if x < 0 else x)

# Save
submission.to_csv('submission.csv', index=False)

submission





display(len(models))
display(models)


# Saving Models
joblib.dump(models,'XGBoost.joblib')

# Loading Models
light_gbm =joblib.load('XGBoost.joblib')
light_gbm





!pip install watermark


%load_ext watermark
%watermark -n -u -v -iv -w -p pytensor,aeppl,xarray




