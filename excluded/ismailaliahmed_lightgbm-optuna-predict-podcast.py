# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# Garbage Collector
import gc 

import pandas as pd
import numpy as np
import os

# Time Modules
import calendar
import time
import datetime
from datetime import datetime, timedelta

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Plots

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as sp
sns.set_style("whitegrid")
sns.set(rc={'figure.figsize':(18, 12)})
%matplotlib inline

# Statistics 
from scipy.stats import norm
from scipy.stats import zscore
from scipy import stats

import warnings
warnings.filterwarnings('ignore')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


##################################################################
# Installing GPU driver for LightGBM:-
!mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd
!sudo apt install nvidia-driver-460 nvidia-cuda-toolkit clinfo
!apt-get update --fix-missing
!pip install -q  lightgbm==4.1.0 \
  --config-settings=cmake.define.USE_GPU=ON \
  --config-settings=cmake.define.OpenCL_INCLUDE_DIR="/usr/local/cuda/include/" \
  --config-settings=cmake.define.OpenCL_LIBRARY="/usr/local/cuda/lib64/libOpenCL.so"


train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
sample = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')


all_columns = list(train.columns)


target = 'Listening_Time_minutes'


train.describe().T


float_columns = train.select_dtypes(include=['float64']).columns
object_columns = train.select_dtypes(include=['object']).columns


counting_unique = {}
for col in object_columns:
    print(f"'{col}' has {train[col].nunique()} unique categories.")


# Print top 10 unique value counts for each categorical column
for col in object_columns:
    print(f"\nTop value counts in '{col}':\n{train[col].value_counts().head(10)}")


train[float_columns].describe().T


missing_values = train.isnull().sum().sort_values(ascending=False)/len(train)
missing_values = pd.Series(missing_values.drop(missing_values[missing_values == 0].index).sort_values(ascending=False))

print("Missing Data:", missing_values, sep='\n')


float_columns = ['Episode_Length_minutes', 'Host_Popularity_percentage',
       'Guest_Popularity_percentage', 'Number_of_Ads',
       'Listening_Time_minutes']


palette = sns.color_palette(palette='husl', 
                  n_colors=len(float_columns))

color_dict = dict(zip(float_columns, palette))

# 1. Reshape the data into long-form
df_long = train[float_columns + ['Genre']].melt(id_vars='Genre', var_name='variable', value_name='value')

# 2. Set up the overall figure
n_cols = 3
n_rows = len(float_columns)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))

# Iteration process on float column
for i, col in enumerate(float_columns):
    # Histogram
    sns.histplot(
        data=train, 
        x=col, 
        hue='Genre', 
        multiple='stack', 
        alpha=0.7,
        linewidth=0.5, 
        ax=axes[i, 0]
    )
    axes[i, 0].set_title(f'{col} - Histogram')

    # ViolinPlot
    sns.violinplot(
        data=train, 
        x='Genre', 
        y=col, 
        palette='husl', 
        ax=axes[i, 1]
    )
    axes[i, 1].set_title(f'{col} - Violin Plot')

    # Boxplot
    sns.boxplot(
        data=train, 
        x='Genre', 
        y=col, 
        palette='husl', 
        ax=axes[i, 2]
    )
    axes[i, 2].set_title(f'{col} - Boxplot')

plt.tight_layout()
fig.suptitle("Distribution Plots by Float Column and Genre", fontsize=20, y=1.02, fontweight='bold')
plt.show()



mu, sigma = stats.norm.fit(train[target])
mu, sigma = stats.norm.fit(np.log1p(train[target]))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))
sns.distplot(train[target], ax=ax1, fit=norm).set_title('Target', fontsize=14, fontweight='bold')
sns.distplot(np.log1p(train[target]), ax=ax2, fit=norm).set_title('Target Logarithm(np.log1p)', fontsize=14, fontweight='bold')
plt.show()


fig = plt.figure()
res = stats.probplot(train[target], plot=plt)


print('Skewness: {}\t'.format(train[target].skew()))
print('Kurtosis: {}\t'.format(train[target].kurt()))

print('Mu: {}\t'.format(mu))
print('Sigma: {}\t'.format(sigma))


for i, col in enumerate(object_columns):
    grouped_data = train.groupby(col)[target].sum().sort_values()
    colors = sns.color_palette('husl', len(grouped_data))  # Match color based number of index Series

    plt.figure(figsize=(10, 6))
    grouped_data.plot(kind='barh', color=colors)
    plt.title(f'Sum of {target} by {col}', fontsize=14, fontweight='bold')
    plt.xlabel(f'Sum of {target}', fontweight='bold')
    plt.ylabel(col, fontweight='bold')
    plt.tight_layout()
    plt.show()


""" 
Categorical values plays a key roles, if your categorical data is ordinal or you want to force it into a numerical form:
"""


from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

train_copy = train.copy()

for col in object_columns:
    train_copy[col] = encoder.fit_transform(train_copy[col])


corr_numer = train_copy.corr(method='pearson')
mask = np.triu(corr_numer)

plt.figure(figsize=(30, 20))
fig = sns.heatmap(corr_numer, 
                  annot = True, 
                  fmt='.2g', 
                  mask = mask, 
                  vmax=0.99)

plt.show()


"""
This block of code will show how to achieve the same result above using pivot_table(), is a bit slower FYI.
"""

# pivot_table = train.pivot_table(
#     index=['Podcast_Name', 'Episode_Title', 'Genre'], 
#     columns=['Publication_Day','Publication_Time', 'Episode_Sentiment'], 
#     values=target, 
#     aggfunc='mean'
# )

# sns.heatmap(pivot_table,
#             cmap='coolwarm', 
#             annot = True, 
#             fmt='.2g', 
#             vmin=-0.99,
#             vmax=0.99, )
# plt.show()


y = np.array(train.Listening_Time_minutes)


X = train.drop(target, axis=1)


import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer


object_columns = X.select_dtypes(include=['object', 'category']).columns
float_columns = X.select_dtypes(include=['float64', 'int64']).columns

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), object_columns),

        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', MinMaxScaler())
        ]), float_columns)
    ]
)

# Full pipeline
pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor)
])

# Transform the data
X_processed = pipeline.fit_transform(X)
test_preprocessed = pipeline.transform(test)


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedStratifiedKFold, LeaveOneOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer

import optuna
import lightgbm as lgb


lgb_gbdt = lgb.LGBMRegressor(
    boosting_type='gbdt',
    random_state=42,
    objective='regression',
    metric='rmse',
    device='gpu',
    verbose=-1
)


lgb_goss = lgb.LGBMRegressor(
    boosting_type='goss',
    random_state=42,
    objective='regression',
    metric='rmse',
    device='gpu',
    verbose=-1
)



X_train, X_valid, y_train, y_valid = train_test_split(X_processed, y, test_size=0.25, random_state=42)
print('X_train: ', X_train.shape)
print('X_valid: ', X_valid.shape)
print('y_train: ', y_train.shape)
print('y_valid: ', y_valid.shape)


model = lgb.LGBMRegressor(**{
    'boosting_type':'goss',
    'random_state':42,
    'objective':'regression',
    'metric':'rmse',
    'device':'gpu',
    'verbose':-1,
    'learning_rate':0.030362233382902903,
    'n_estimators':998,
    'max_depth':9,
    'num_leaves':208,
    'min_child_samples':11,
    'subsample':0.8184667361186249,
    'colsample_bytree':0.8616459477375787,
    'reg_alpha':0.33080029457188864,
    'reg_lambda':0.20736962602335904,
})


n_splits = 10

#kf = RepeatedStratifiedKFold(n_splits=n_splits, random_state=42)
kf = LeaveOneOut()

# Cross-validation on the TRAIN set
cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='neg_root_mean_squared_error')
print("Cross-validated RMSE scores:", -cv_scores)
print("Mean CV RMSE:", np.mean(cv_scores))

# Train on TRAIN set
model.fit(X_train, y_train)

# Evaluate on VALIDATION set
val_preds = model.predict(X_valid)
val_rmse = np.sqrt(mean_squared_error(y_valid, val_preds))
print("Validation RMSE:", val_rmse)


# Initialize arrays to store OOF predictions and test predictions
oof_predictions_gbdt = np.zeros(len(X_train))
oof_predictions_goss = np.zeros(len(X_train))

test_predictions_gbdt = np.zeros((len(test_preprocessed), n_splits))
test_predictions_goss = np.zeros((len(test_preprocessed), n_splits))

# Store RMSE for each fold
fold_rmse_gbdt = []
fold_rmse_goss = []

# OOF Training for LightGBM (GBDT) and LightGBM (GOSS)
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"Training Fold {fold + 1}/{n_splits}...")
    
    # Split the data into training and validation sets
    X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
    y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
    
    # LightGBM (GBDT)
    lgb_gbdt.fit(
        X_train_fold, y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=10)
        ]
    )
    oof_predictions_gbdt[val_idx] = lgb_gbdt.predict(X_val_fold)
    test_predictions_gbdt[:, fold] = lgb_gbdt.predict(test_preprocessed)
    fold_rmse_gbdt.append(np.sqrt(mean_squared_error(y_val_fold, oof_predictions_gbdt[val_idx])))
    
    # LightGBM (GOSS)
    lgb_goss.fit(
        X_train_fold, y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),  # Increased to 50
            lgb.log_evaluation(period=10)
        ]
    )
    oof_predictions_goss[val_idx] = lgb_goss.predict(X_val_fold)
    test_predictions_goss[:, fold] = lgb_goss.predict(test_preprocessed)
    fold_rmse_goss.append(np.sqrt(mean_squared_error(y_val_fold, oof_predictions_goss[val_idx])))

# Compute average RMSE for each model
avg_rmse_gbdt = np.mean(fold_rmse_gbdt)
avg_rmse_goss = np.mean(fold_rmse_goss)

print("Average RMSE (GBDT):", avg_rmse_gbdt)
print("Average RMSE (GOSS):", avg_rmse_goss)


# Compute weights based on RMSLE
# Lower RMSLE => Higher Weight
total_weight = 1 / avg_rmse_gbdt + 1 / avg_rmse_goss
weight_gbdt = (1 / avg_rmse_gbdt) / total_weight
weight_goss = (1 / avg_rmse_goss) / total_weight

print("Weight for GBDT:", weight_gbdt)
print("Weight for GOSS:", weight_goss)


# Compute weighted average of predictions
final_test_predictions = (
    weight_gbdt * test_predictions_gbdt.mean(axis=1) +
    weight_goss * test_predictions_goss.mean(axis=1)
)


from sklearn.metrics import mean_squared_error
import lightgbm as lgb


# def objective(trial):
#     # Define parameter search space
#     param = {
#         "objective": "regression",
#         "metric": "rmse",
#         "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
#         "num_leaves": trial.suggest_int("num_leaves", 200, 512),
#         "learning_rate": trial.suggest_loguniform("learning_rate", 1e-4, 1e-1),
#         "feature_fraction": trial.suggest_uniform("feature_fraction", 0.2, 1.0),
#         "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.2, 1.0),
#         "bagging_freq": trial.suggest_int("bagging_freq", 5, 12),
#         "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
#         "max_depth": trial.suggest_int("max_depth", -1, 16),
#         "lambda_l1": trial.suggest_loguniform("lambda_l1", 1e-4, 10.0),
#         "lambda_l2": trial.suggest_loguniform("lambda_l2", 1e-4, 10.0),
#         "device_type": "gpu",
#         "seed": 42,
#         "verbose": -1
#     }

#     # Create a LightGBM dataset
#     dtrain = lgb.Dataset(X_train, label=y_train)
#     dval = lgb.Dataset(X_valid, label=y_valid, reference=dtrain)

#     # Train LightGBM model
#     model = lgb.train(
#         param,
#         dtrain,
#         valid_sets=[dval],
#     )

#     # Predict on validation set
#     y_val_pred = model.predict(X_valid)

#     rmse = np.sqrt(mean_squared_error(y_valid, y_val_pred))

#     return rmse

# # Run Optuna study
# study = optuna.create_study(direction="minimize")
# study.optimize(objective, n_trials=100)


# print('Best Trial:', study.best_trial)
"""
Best Trial: FrozenTrial(number=70, state=1, values=[0.3893298309779343], datetime_start=datetime.datetime(2025, 4, 29, 20, 37, 13, 1724), datetime_complete=datetime.datetime(2025, 4, 29, 20, 37, 22, 343700), params={'boosting_type': 'gbdt', 'num_leaves': 298, 'learning_rate': 0.09321505647261386, 'feature_fraction': 0.7562619850127539, 'bagging_fraction': 0.9385225190408361, 'bagging_freq': 11, 'min_data_in_leaf': 75, 'max_depth': 13, 'lambda_l1': 0.005704418121192958, 'lambda_l2': 1.167708799827015}, user_attrs={}, system_attrs={}, intermediate_values={}, distributions={'boosting_type': CategoricalDistribution(choices=('gbdt', 'dart')), 'num_leaves': IntDistribution(high=512, log=False, low=200, step=1), 'learning_rate': FloatDistribution(high=0.1, log=True, low=0.0001, step=None), 'feature_fraction': FloatDistribution(high=1.0, log=False, low=0.2, step=None), 'bagging_fraction': FloatDistribution(high=1.0, log=False, low=0.2, step=None), 'bagging_freq': IntDistribution(high=12, log=False, low=5, step=1), 'min_data_in_leaf': IntDistribution(high=100, log=False, low=10, step=1), 'max_depth': IntDistribution(high=16, log=False, low=-1, step=1), 'lambda_l1': FloatDistribution(high=10.0, log=True, low=0.0001, step=None), 'lambda_l2': FloatDistribution(high=10.0, log=True, low=0.0001, step=None)}, trial_id=70, value=None)
"""


# best_params = study.best_params
# print('Best Parameters:', study.best_params)
"""
Best Parameters: {'boosting_type': 'gbdt', 'num_leaves': 298, 'learning_rate': 0.09321505647261386, 'feature_fraction': 0.7562619850127539, 'bagging_fraction': 0.9385225190408361, 'bagging_freq': 11, 'min_data_in_leaf': 75, 'max_depth': 13, 'lambda_l1': 0.005704418121192958, 'lambda_l2': 1.167708799827015}
"""


# Calculate average predictions for test data
avg_test_predictions = (
    weight_gbdt * test_predictions_gbdt.mean(axis=1) +
    weight_goss * test_predictions_goss.mean(axis=1)
)


# Visualization of Prediction Distributions
viridis_cmap = cm.get_cmap("viridis", 3)

plt.figure(figsize=(12, 6))

# Plot true values (Train Data)
plt.hist(
    y_train, bins=30, color=viridis_cmap(0), alpha=0.6, edgecolor="black", label="True Values (Train)"
)

# Plot predicted values (OOF Predictions)
plt.hist(
    oof_predictions_gbdt, bins=30, color=viridis_cmap(0.5), alpha=0.6, edgecolor="black", label="Predicted Values (GBDT)"
)

plt.hist(
    oof_predictions_goss, bins=30, color=viridis_cmap(0.7), alpha=0.6, edgecolor="black", label="Predicted Values (GOSS)"
)

# Add titles and labels
plt.title("Prediction Distributions - Train and OOF Predictions", fontsize=16)
plt.xlabel("Listening_Time_minutes", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()


# Combine feature importance for both GBDT and GOSS models
feature_importance_gbdt = pd.DataFrame({
    'Feature': lgb_gbdt.feature_name_,
    'GBDT Importance': lgb_gbdt.feature_importances_,
})

feature_importance_goss = pd.DataFrame({
    'Feature': lgb_goss.feature_name_,
    'GOSS Importance': lgb_goss.feature_importances_,
})

# Merge the feature importance for comparison
combined_feature_importance = pd.merge(
    feature_importance_gbdt,
    feature_importance_goss,
    on='Feature',
    how='inner'
)

# Sort features by average importance
combined_feature_importance['Avg Importance'] = (
    combined_feature_importance['GBDT Importance'] +
    combined_feature_importance['GOSS Importance']
) / 2

combined_feature_importance = combined_feature_importance.sort_values(
    by='Avg Importance', ascending=False
)

# Plot the feature importance
plt.figure(figsize=(14, 8))
sns.barplot(
    x='Avg Importance', 
    y='Feature', 
    data=combined_feature_importance.head(10),  
    palette='viridis'
)
plt.title('Top 10 Features - Average Importance (GBDT & GOSS)', fontsize=16)
plt.xlabel('Average Importance Score', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Display top 10 features
print("Top 10 Features by Average Importance:\n")
display(combined_feature_importance.head(10))


params = {
    'boosting_type': 'gbdt',
    'num_leaves': 298,
    'learning_rate': 0.09321505647261386,
    'feature_fraction': 0.7562619850127539,
    'bagging_fraction': 0.9385225190408361,
    'bagging_freq': 11,
    'min_data_in_leaf': 75,
    'max_depth': 13,
    'lambda_l1': 0.005704418121192958,
    'lambda_l2': 1.167708799827015,
    "device_type": "gpu",
    "seed": 42,
    "verbose": -1,
    "metric": "rmse",
}

train_data = lgb.Dataset(X_processed, label=y)

prediction_model = lgb.train(
    params,
    train_data
)


# Make predictions on the test set
test_predictions_one = prediction_model.predict(test_preprocessed, num_iteration=prediction_model.best_iteration)

# Prepare submission file
submission = pd.DataFrame({'id': test['id'], 'Listening_Time_minutes': test_predictions_one})
submission.to_csv("submission.csv", index=False)

# Prepare oof prediction file 
oof_submission = pd.DataFrame({'id': test['id'], 'Listening_Time_minutes': final_test_predictions})
oof_submission.to_csv("oof_submission.csv", index=False)


model = lgb.LGBMRegressor(**{'n_jobs':-1,
                            'random_state':42, 
                            'boosting_type': 'gbdt', 
                            'num_leaves': 300, 
                            'learning_rate': 0.001, 
                        'feature_fraction': 0.75})
model.fit(X_processed, y)


# Make predictions on the test set
test_predictions = model.predict(test_preprocessed)

# Prepare submission file
submission = pd.DataFrame({'id': test['id'], 'Listening_Time_minutes': test_predictions})
submission.to_csv("submission_model.csv", index=False)


# 1. By split (number of times used)
importance_split = model.feature_importance(importance_type='split')

# 2. By gain (information gain)
importance_gain = model.feature_importance(importance_type='gain')

# 3. Get feature names
feature_names = model.feature_name()

# Example: view as DataFrame
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'split_importance': importance_split,
    'gain_importance': importance_gain
}).sort_values(by='gain_importance', ascending=False)


sns.histplot(data=feature_importance_df, binwidth=3, palette='husl')

