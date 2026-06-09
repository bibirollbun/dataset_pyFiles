#install scikit learn 1.5.2 as this version supports root_mean_squared_log_error
!pip uninstall scikit-learn -y
!pip install -q scikit-learn==1.5.2


import sklearn
sklearn.__version__


import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="seaborn")
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_log_error,mean_squared_error, mean_absolute_error, r2_score

# import optuna
import lightgbm as lgb

import torch
from sklearn.pipeline import Pipeline


train_df=pd.read_csv('/kaggle/input/big-oai-final-course-1/train.csv')
test_df=pd.read_csv('/kaggle/input/big-oai-final-course-1/test.csv')


# Check dataset shape and first rows
print(f"Dataset contains {train_df.shape[0]} rows and {train_df.shape[1]} columns.")
train_df.head()


train_df.info()


train_df.describe().round(2)


# Save 'id' column for submission
test_ids = test_df['id']

# Define the target column
target_column = 'Premium Amount'

# Select categorical and numerical columns (initial)
categorical_columns = train_df.select_dtypes(include=['object']).columns
numerical_columns = train_df.select_dtypes(exclude=['object']).columns

# Print out column information
print("Target Column:", target_column)
print("\nCategorical Columns:", categorical_columns.tolist())
print("\nNumerical Columns:", numerical_columns.tolist())


for column in categorical_columns:
    num_unique = train_df[column].nunique()
    print(f"'{column}' has {num_unique} unique categories.")


# plt.figure(figsize=(15,9))
# plt.title("Visualizing Missing Values")
# sns.heatmap(train_df.isnull(), cbar=False, cmap=sns.color_palette('magma'), yticklabels=False);
# plt.show()


# Function to calculate missing values, percentages, and data types
def missing_values_table(df):
    missing_count = df.isnull().sum()
    missing_percentage = 100 * missing_count / len(df)
    data_types = df.dtypes
    return pd.DataFrame({
        'Missing Values': missing_count,
        'Percentage (%)': missing_percentage,
        'Data Type': data_types
    })

# Create tables for train and test datasets
train_missing_table = missing_values_table(train_df)
test_missing_table = missing_values_table(test_df)

# Display the tables
print("Missing Values Table - Training Dataset:\n")
display(train_missing_table[train_missing_table['Missing Values'] > 0])  # Display only features with missing values
print("\n")

print("Missing Values Table - Test Dataset:\n")
display(test_missing_table[test_missing_table['Missing Values'] > 0])


# # Create a color palette for the columns
# palette = sns.color_palette('tab10', len(numerical_columns))
# color_dict = dict(zip(numerical_columns, palette))

# # Create a grid of subplots for histograms and boxplots only
# fig = plt.figure(figsize=(15, 10 * len(numerical_columns)))  # Adjusted width since only one column of plots
# gs = gridspec.GridSpec(2 * len(numerical_columns), 1, figure=fig)  # Single column grid

# for i, column in enumerate(numerical_columns):
#     if(column=="id"): continue
#     if train_df[column].nunique() > 50:
#         discrete = False
#     else:
#         discrete = True
    
#     # Plot histogram with a unique color
#     ax_hist = fig.add_subplot(gs[2 * i, 0])
#     sns.histplot(
#         data=train_df, x=column, fill=True, common_norm=False, alpha=0.6,
#         linewidth=0.8, color=color_dict[column], ax=ax_hist, discrete=discrete
#     )
    
#     # Plot boxplot with the same unique color
#     ax_box = fig.add_subplot(gs[2 * i + 1, 0])
#     sns.boxplot(data=train_df, x=column, ax=ax_box, color=color_dict[column])
#     ax_box.set_title(f'{column} (Boxplot)', fontsize=14)
#     sns.despine(ax=ax_box)

# plt.tight_layout()  # Adjust subplots to fit into the figure area
# plt.show()


# filtered_columns = [col for col in categorical_columns if col != 'Policy Start Date']
# fig, axes = plt.subplots(len(filtered_columns), 2, figsize=(15, 5 * len(filtered_columns)))

# for i, column in enumerate(filtered_columns):
#     # Barplot à gauche
#     sns.countplot(data=train_df, x=column, ax=axes[i, 0], palette='tab10')
#     axes[i, 0].set_title(f'Distribution of {column}', fontsize=14)
#     axes[i, 0].set_xlabel(column, fontsize=12)
#     axes[i, 0].set_ylabel('Count', fontsize=12)
#     sns.despine(ax=axes[i, 0])

#     # Boxplot à droite
#     sns.boxplot(data=train_df, x=column, y=target_column, ax=axes[i, 1], palette='tab10')
#     axes[i, 1].set_title(f'{column} vs {target_column}', fontsize=14)
#     axes[i, 1].set_xlabel(column, fontsize=12)
#     axes[i, 1].set_ylabel(target_column, fontsize=12)
#     sns.despine(ax=axes[i, 1])


# plt.tight_layout()  # Ajustement global des sous-graphiques
# plt.show()


from sklearn.preprocessing import  RobustScaler

def create_features(df):
    """Enhanced feature engineering function"""
    df['Policy Start Date'] = pd.to_datetime(df['Policy Start Date'])
    df['Year'] = df['Policy Start Date'].dt.year
    df['Day'] = df['Policy Start Date'].dt.day
    df['Month'] = df['Policy Start Date'].dt.month
    df['Month_name'] = df['Policy Start Date'].dt.month_name()
    df['Day_of_week'] = df['Policy Start Date'].dt.day_name()
    df['Week'] = df['Policy Start Date'].dt.isocalendar().week
    df['Year_sin'] = np.sin(2 * np.pi * df['Year'])
    df['Year_cos'] = np.cos(2 * np.pi * df['Year'])
    min_year = df['Year'].min()
    max_year = df['Year'].max()
    df['Year_sin'] = np.sin(2 * np.pi * (df['Year'] - min_year) / (max_year - min_year))
    df['Year_cos'] = np.cos(2 * np.pi * (df['Year'] - min_year) / (max_year - min_year))
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12) 
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['Day_sin'] = np.sin(2 * np.pi * df['Day'] / 31)  
    df['Day_cos'] = np.cos(2 * np.pi * df['Day'] / 31)
    
    df.drop('Policy Start Date', axis=1, inplace=True)

    return df
    


# Apply feature engineering
print("Creating features...")
train_df = create_features(train_df)
test_df = create_features(test_df)

# Reduce memory again after feature engineering
# train_df = reduce_mem_usage(train_df)
# test_df = reduce_mem_usage(test_df)

# Define feature types
numerical_features = [
    'Age', 'Annual Income', 'Number of Dependents', 'Health Score', 
    'Previous Claims', 'Vehicle Age', 'Credit Score', 'Insurance Duration', 
    'Year_sin', 'Year_cos', 'Month_sin', 'Month_cos', 'Day_sin', 'Day_cos'
]
categorical_features = [
    'Gender', 'Marital Status', 'Education Level', 'Occupation', 'Location',
    'Policy Type', 'Customer Feedback', 'Smoking Status', 'Exercise Frequency', 
    'Property Type', 'Month_name', 'Day_of_week'
]


# Advanced Preprocessing
print("\n" + "="*50)
print("ADVANCED PREPROCESSING")
print("="*50)

# Separate features and target
X = train_df.drop(['Premium Amount', 'id'], axis=1)
y = train_df['Premium Amount']
X_test = test_df.drop(['id'], axis=1)

# Log transform target
y_log = np.log1p(y)

# Create preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combine transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

X_processed = preprocessor.fit_transform(X)
test_processed = preprocessor.transform(test_df.drop(columns=['id', 'Year', 'Month', 'Day', 'Week']))


# Log-transform skewed features
train_df['Annual Income'] = np.log1p(train_df['Annual Income'])
test_df['Annual Income'] = np.log1p(test_df['Annual Income'])


# Separate features and target
X = train_df.drop(['Premium Amount', 'id'], axis=1)
y = train_df['Premium Amount']
X_test = test_df.drop(['id'], axis=1)

# Log transform target
y_log = np.log1p(y)


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from catboost import CatBoostRegressor

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_log_error


print("\n--- Training VotingRegressor(Ensemble) ---")
lr_model = LinearRegression()
ridge_model = Ridge(alpha=1.0, random_state=42)
lasso_model = Lasso(alpha=1.0, random_state=42, max_iter=2000)
elastic_net_model = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42, max_iter=2000) 

rf_model = RandomForestRegressor(n_estimators=30, random_state=42, n_jobs=1)
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=1)
lgb_model = lgb.LGBMRegressor(objective='regression', n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=1)
mlp_model = MLPRegressor(hidden_layer_sizes=(10, 5), activation='relu', solver='adam',
                         alpha=0.0001, learning_rate_init=0.001, max_iter=500, random_state=42,
                         early_stopping=True, n_iter_no_change=20, verbose=False)

cat_model = CatBoostRegressor(iterations=100, 
                              learning_rate=0.1,
                              depth=6,
                              random_seed=42,
                              verbose=0,
                              l2_leaf_reg=3,
                              loss_function='RMSE') 

gbr_model = GradientBoostingRegressor(n_estimators=100,
                                      learning_rate=0.1,
                                      max_depth=3,
                                      random_state=42)

hgb_model = HistGradientBoostingRegressor(max_iter=100,
                                          learning_rate=0.1,
                                          max_depth=None,
                                          random_state=42)

cv_voting_estimators = [
    #('linear_reg', lr_model),
    #('ridge_reg', ridge_model),

    # ('random_forest', rf_model),
    # ('lasso_reg', lasso_model),
    # ('elastic_net_reg', elastic_net_model),
    # ('xgboost', xgb_model),
    # ('lightgbm', lgb_model),
    # ('mlp_nn', mlp_model),
    # ('catboost', cat_model),
    # ('gradient_boosting', gbr_model),
    ('hist_gradient_boosting', hgb_model) # this one is the best
]

cv_voting_regressor = VotingRegressor(estimators=cv_voting_estimators, n_jobs=1)
rmsle_scores = []

def rmsle(y_true, y_pred):
    """
    Calculates the Root Mean Squared Logarithmic Error.
    y_true and y_pred must be in their original, non-log scale.
    """
    # Ensure no negative predictions for log calculation
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(mean_squared_log_error(y_true + 1, y_pred + 1))


# --- K-Fold Cross-Validation ---
print("\n--- Applying K-Fold Cross-Validation ---")
cv_voting_regressor = VotingRegressor(estimators=cv_voting_estimators, n_jobs=1)
rmsle_scores = []
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold_idx, (train_index, val_index) in enumerate(kf.split(X_processed)):
    print(f"--- Processing Fold {fold_idx + 1} ---")
    
    # Split data for this fold
    X_train_fold, X_val_fold = X_processed[train_index], X_processed[val_index]
    y_train_fold, y_val_fold = y_log.iloc[train_index], y_log.iloc[val_index]
    
    # # --- Apply LOF --- really slow
    # lof_fold = LocalOutlierFactor(n_neighbors=50, contamination='auto', n_jobs=1)
    # is_inlier_fold = lof_fold.fit_predict(X_train_fold)
    # X_train_fold = X_train_fold[is_inlier_fold == 1]
    # y_train_fold = y_train_fold[is_inlier_fold == 1]

    # --- Apply IsolationForest ---
    iso_forest_fold = IsolationForest(n_estimators=100, contamination=0.01, random_state=42, n_jobs=-1) # Tune contamination!
    is_inlier_fold = iso_forest_fold.fit_predict(X_train_fold)
    X_train_fold = X_train_fold[is_inlier_fold == 1]
    y_train_fold = y_train_fold[is_inlier_fold == 1]


    cv_voting_regressor.fit(X_train_fold, y_train_fold)
    y_pred_fold = cv_voting_regressor.predict(X_val_fold)
    
    y_pred_original = np.expm1(y_pred_fold)
    
    y_val_original = np.expm1(y_val_fold)
    
    y_pred_original = np.maximum(y_pred_original, 0)
    
    score = rmsle(y_val_original, y_pred_original)
    rmsle_scores.append(score)
    print(f"Fold {fold_idx + 1} RMSLE: {score:.4f}")

print(f"\nRMSLE scores for all folds: {rmsle_scores}")
print(f"Mean RMSLE across folds: {np.mean(rmsle_scores):.4f}")
print(f"Standard deviation of RMSLE across folds: {np.std(rmsle_scores):.4f}")


sample_submission = pd.read_csv("/kaggle/input/big-oai-final-course-1/sample_submission.csv")
sample_submission.head()


# Predict
y_pred = cv_voting_regressor.predict(test_processed)
y_pred = np.expm1(y_pred)
y_pred = np.maximum(y_pred, 0)


submission_df = pd.DataFrame()

if 'id' in test_df.columns:
    submission_df['id'] = test_df['id'].values
else:
    submission_df['id'] = sample_submission['id'].values

target_column_name = sample_submission.columns[1] # Assumes target is the second column
submission_df[target_column_name] = y_pred

# Save
submission_filename = 'submission.csv'
submission_df.to_csv(submission_filename, index=False)

print(f"Submission file '{submission_filename}' created.")
print(submission_df.head())

