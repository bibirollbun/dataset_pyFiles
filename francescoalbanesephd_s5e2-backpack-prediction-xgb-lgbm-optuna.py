#HIDDEN CELL

# %%capture
!pip install itables

import pandas as pd
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from itables import init_notebook_mode, show
init_notebook_mode(all_interactive=False,connected=True)


#HIDDEN CELL

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


# Import datasets
train_df = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv',index_col='id')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv',index_col='id')
train_extra = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv",index_col='id')


#HIDDEN CELL

# # Quick EDA
# import pandas_profiling
# train_df.profile_report()


#HIDDEN CELL

# Display datasets
show(train_df)
show(test_df)


train_df.info()


train_df.describe(include='all')


#HIDDEN CELL

# simplify col names
train_df.columns = train_df.columns.str.lower().str.replace(' ','_')
test_df.columns = test_df.columns.str.lower().str.replace(' ','_')
train_extra.columns = train_extra.columns.str.lower().str.replace(' ','_')


#HIDDEN CELL

# Get unique values for each categorical column in exploratory_df
for col in train_df.select_dtypes(include='object'):
    print(f"Unique values in {col}: {train_df[col].unique()}")


# Identify Target
target = 'price'


# HIDDEN CELL

plt.figure(figsize=(14, 6))

# Violinplots for Train and Train Extra side by side
sns.violinplot(data=[train_df[target].values, train_extra[target].values],
               inner='box', palette=['#ef233c', '#007bff'], orient='h', linewidth=2, cut=0)
sns.violinplot(data=[train_df[target].values, train_extra[target].values],
               inner='quartile', palette=['#ef233c', '#007bff'], orient='h', linewidth=2, cut=0)

plt.yticks([0, 1], ["Train", "Train Extra"]) #Set yticks to label each violinplot

plt.title(f'Violinplot of {target} (Train vs. Train Extra)', fontsize=11)
plt.xlabel(target, fontsize=10)
plt.ylabel('Dataset', fontsize=10) #Change label to reflect what the y axis actually represents.
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Skewness and kurtosis for both datasets
for train, train_name in [(train_df, "Train"), (train_extra, "Train Extra")]:
    print(f"Skewness ({train_name}): " + str(train[target].skew()))
    print(f"Kurtosis ({train_name}): " + str(train[target].kurt()))
    display(round(train[target].to_frame().describe()))


#HIDDEN CELL

print("Visualize Missing Data in Train, Train Extra and Test Datasets")

import missingno as msno
plt.figure(figsize=(12, 6))

# Visualize missing values as a matrix 
for df, df_name in [
    (train_df, "Train"), 
    (train_extra, "Train Extra"), 
    (test_df, "Test")]:
    
    msno.matrix(df)
    plt.title(f"Missing Data in {df_name} Dataset", fontsize = 30)
    plt.show()


# HIDDEN CELL

# Calculate feature missing value counts and proportions and make barplots
for df, df_name in [(train_df, "Train"), (train_extra, "Train Extra"), (test_df, "Test")]:
    total_values = df.shape[0]
    missing_values = df.isnull().sum()  # by feature

    # Create a DataFrame for display
    missing_df = pd.concat(
        [missing_values, round(missing_values / total_values, 2), df.dtypes],
        axis=1,
        keys=['N', '%', 'dtype']
    )
    
    # Horizontal barplot of missing value counts
    plt.figure(figsize=(12, 5))
    sns.barplot(y=missing_values.index, x=missing_values.values, orient='h', color="skyblue")
    plt.title(f"Missing Values in {df_name}",fontsize=20)
    plt.ylabel("Features")
    plt.xlabel("Number of Missing Values")
    plt.tight_layout()
    plt.grid(axis='x',alpha=0.5)
    plt.show()

    # # Horizontal barplot of missing value proportions
    # plt.figure(figsize=(10, 6))
    # sns.barplot(y=missing_values.index, x=round(missing_values / total_values, 2), orient='h')
    # plt.title(f"Missing Value Proportions in {df_name}")
    # plt.ylabel("Features")
    # plt.xlabel("Proportion of Missing Values")
    # plt.tight_layout()
    # plt.show()


import warnings

# Hide deprecation warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)

# Create a copy of the DataFrame to avoid modifying the original data
exploratory_df = train_df.copy()

# Select categorical and numerical columns
categorical_cols = exploratory_df.select_dtypes(include='object').columns
numerical_cols = exploratory_df.select_dtypes(exclude='object').columns

# Handle NaN values by filling them with a placeholder
exploratory_df[numerical_cols] = exploratory_df[numerical_cols].fillna(exploratory_df[numerical_cols].median())
# exploratory_df[categorical_cols] = exploratory_df[categorical_cols].fillna(exploratory_df[categorical_cols].mode())
exploratory_df[categorical_cols] = exploratory_df[categorical_cols].fillna('Missing')

print(len(numerical_cols))
print(len(categorical_cols))


from IPython.display import clear_output

# Plot categorical features with count plots
fig_cat, axes_cat = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))  # Adjust grid size based on data
axes_cat = axes_cat.flatten()

for i, column in enumerate(categorical_cols):
    clear_output(wait=False)
    print(f'Processing categorical feature: {column}')
    sns.countplot(x=exploratory_df[column], ax=axes_cat[i], order=exploratory_df[column].value_counts().index)
    axes_cat[i].set_title(column, fontsize=9)
    axes_cat[i].tick_params(axis='both', which='major', labelsize=6)

fig_cat.suptitle('Categorical Feature Distributions', fontsize=11)
plt.tight_layout()
plt.show()

# Categorical Cols insights
display(train_df[categorical_cols].describe())


# Loop through each categorical feature to display summary statistics and box plot
for column in categorical_cols:
    # Calculate summary statistics grouped by the categorical column
    stats = exploratory_df.groupby(column)[target].describe()
    
    # Display summary statistics
    print(f"\nSummary Statistics for price by {column}:")
    display(stats)
    
    # Plot violin plot
    plt.figure(figsize=(10, 4))
    sns.violinplot(data=exploratory_df, x=column, y=target, palette='Spectral',inner="quartile", linewidth=2, cut=0)
    sns.violinplot(data=exploratory_df, x=column, y=target, palette='Spectral',inner="box", linewidth=2, cut=0,)
    plt.title(f'price by {column}', fontsize=12)
    plt.xlabel(column, fontsize=11)
    plt.ylabel(target, fontsize=11)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


from sklearn.model_selection import train_test_split

# Merge the two train set
full_train_df = pd.concat([train_df, train_extra],axis=0)

# Split into train and validation sets
X = full_train_df.copy()
y = X.pop(target)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=SEED)

# Rename test_df (optional)
X_test = test_df.copy()


from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# Separate numerical and categorical columns, excluding 'occupation' from cat_cols
num_cols = X_train.select_dtypes(include=['number']).columns
cat_cols = X_train.select_dtypes(exclude=['number']).columns #.difference(['occupation']) in case you need a diff strategy for other columns

# Original column names
cols = list(num_cols) + list(cat_cols)

# Store original dtypes for later conversion
original_dtypes = X_train.dtypes[cols] # Store dtypes of selected columns

# Define the imputers
num_imputer = SimpleImputer(strategy='median')  # Use 'median', 'mean', or other strategies
cat_imputer_mode = SimpleImputer(strategy='most_frequent')  # For categorical columns
cat_imputer_missing = SimpleImputer(strategy='constant', fill_value='missing')  # Specific strategy for 'occupation'

# Create a column transformer to apply different imputers
column_transformer = make_column_transformer(
    (num_imputer, num_cols),             # Impute numerical columns
    (cat_imputer_mode, cat_cols),        # Impute other categorical columns
    # (cat_imputer_missing, ['feature']),  # Impute a column separately
    
)

# Fit and transform the training data
X_train_imputed = pd.DataFrame(
    column_transformer.fit_transform(X_train),  # Fit and transform the pipeline
    columns=cols,                     # Assign original column names
    index=X_train.index,               # Retain the original index
)

# Transform validation data
X_valid_imputed = pd.DataFrame(
    column_transformer.transform(X_valid),      # Transform only
    columns=cols,          
    index=X_valid.index               
)

# Transform test data
X_test_imputed = pd.DataFrame(
    column_transformer.transform(X_test),       # Transform only
    columns=cols,           
    index=X_test.index                
)

# Assign original dtypes to imputed DataFrames
X_train_imputed = X_train_imputed.astype(original_dtypes)
X_valid_imputed = X_valid_imputed.astype(original_dtypes)
X_test_imputed = X_test_imputed.astype(original_dtypes)


from sklearn.preprocessing import FunctionTransformer

# Create feature engineering function
def create_new_features(df):
    # Combine 'brand' and 'material'
    df['brand_material'] = df['brand'] + '_' + df['material']
    
    # Convert 'compartments' to an ordinal feature
    df['compartments_ordinal'] = pd.cut(df['compartments'], bins=[0, 3, 6, 10], labels=['few', 'moderate', 'abundant'])
    
    # Combine 'laptop_compartment' and 'waterproof'
    df['laptop_waterproof'] = df.apply(lambda row: 'both' if row['laptop_compartment'] == 'Yes' and row['waterproof'] == 'Yes' else 'One or None',axis=1)
    
    # Create an interaction term between 'Weight Capacity' and 'Compartments'
    df['weightXcompartment'] = df['weight_capacity_(kg)'] * df['compartments']

    return df

# ... and incorporate it into FunctionTransformer..
# feature_engineer = FunctionTransformer(create_new_features)
# feature_engineer.fit_transform(X_train_imputed.copy())

# or apply directly
X_train_transformed = create_new_features(X_train_imputed.copy())
X_valid_transformed = create_new_features(X_valid_imputed.copy())
X_test_transformed = create_new_features(X_test_imputed.copy())


# from sklearn.feature_selection import mutual_info_regression

# # Sample a subset of the data
# sampled_X = pd.get_dummies(X_train_transformed.copy()).sample(frac=0.25, random_state=SEED)
# sampled_y = y_train[sampled_X.index]

# mi_scores = mutual_info_regression(sampled_X, sampled_y, discrete_features='auto',random_state=SEED)
# mi_scores = pd.Series(mi_scores, name="MI Scores", index=sampled_X.columns)
# mi_scores = mi_scores.sort_values(ascending=False)
# mi_features = mi_scores[:10].keys()
# mi_features


# def plot_mi_scores(scores):
#     scores = scores.sort_values(ascending=True)
#     width = np.arange(len(scores))
#     ticks = list(scores.index)
#     plt.barh(width, scores)
#     plt.yticks(width, ticks)
#     plt.grid(alpha=0.2)
#     plt.title("Mutual Information Scores")

# plt.figure(dpi=100, figsize=(15, 10))
# plot_mi_scores(mi_scores)


def encode_ordinal(df):
    
    """Encodes ordinal categorical features into numerical values."""
    size = {"Small": 0, "Medium": 1, "Large": 2}
    df['size'] = df['size'].map(size)
    
    return df

X_train_transformed = encode_ordinal(X_train_transformed.copy())
X_valid_transformed = encode_ordinal(X_valid_transformed.copy())
X_test_transformed = encode_ordinal(X_test_transformed.copy())


from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor, StackingRegressor
# from sklearn.svm import SVR


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer

def model_test_pipeline(df, target, feature_selector, model, model_name):
    # Sample a subset of the data
    X = df
    y = target

    # Define scorer
    rmse_scorer = make_scorer(mean_squared_error, squared=False, greater_is_better = False)

    # Define the preprocessing for numerical and categorical features
    numerical_transformer = Pipeline(steps=[
        ('numeric_imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler()),
    ])
    
    # categorical_transformer = Pipeline(steps=[
    #     ('onehot', OneHotEncoder(handle_unknown='ignore')), # Not necessary for tree based method, but I'll include it in case i wanted to use other models | sparse_output for PCA
    # ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, X.select_dtypes(include='number').columns),
            # ("cat", categorical_transformer, X.select_dtypes(exclude='number').columns),
        ]
    )

    # Create a pipeline with preprocessing and the model
    pipeline = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            # ('feature_engineer', feature_engineer),
            ('feature_selection', feature_selector),
            ('model', model),
        ]
    )

    # Evaluate the pipeline using cross-validation
    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=5,
        scoring=rmse_scorer,
    )
 
    # Print the cross-validation results
    print(f"{model_name} RMSE: {-scores.mean():.4f}")


import optuna
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# # HIDDEN IN VIEWER

# # Optuna objective for XGBoost
# def objective_xgb(trial):
#     """
#     Objective function for optimizing XGBoost hyperparameters with Optuna.
#     Args:
#         trial: An Optuna trial object.
#     Returns:
#         RMSLE score on the validation set.
#     """
#     # Define model hyperparameter search space
#     params = {
#         "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
#         "max_depth": trial.suggest_int("max_depth", 3, 12),
#         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
#         "subsample": trial.suggest_float("subsample", 0.6, 1.0),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
#         "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),  # L1 regularization
#         "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),  # L2 regularization
#         "device": "cuda"  # Enable GPU support
#     }

#     # PCA Hyperparameter
#     # pca_n_components = trial.suggest_float("pca_n_components", 0.7, 0.99)

#     # Define the model
#     model = XGBRegressor(**params, random_state=SEED)

#     # Define the preprocessing for numerical and categorical features
#     numerical_transformer = Pipeline(steps=[
#         ('numeric_imputer', SimpleImputer(strategy='median')),
#         ('scaler', MinMaxScaler()),
#     ])
    
#     # categorical_transformer = Pipeline(steps=[
#     #     ('onehot', OneHotEncoder(handle_unknown='ignore')), # Not necessary for tree based method, but I'll include it in case i wanted to use other models | sparse_output for PCA
#     # ])

#     # Combine preprocessing steps
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ("num", numerical_transformer, X.select_dtypes(include='number').columns),
#             # ("cat", categorical_transformer, X.select_dtypes(exclude='number').columns),
#         ]
#     )

#     # Define feature selector
#     # PCA_selector = PCA(n_components=pca_n_components)
#     PCA_selector = PCA(n_components=.95)

#     # Create a pipeline with preprocessing and the model
#     pipeline = Pipeline(
#         steps=[
#             ('preprocessor', preprocessor),
#             ('feature_selection', PCA_selector),
#             ('model', model),
#         ]
#     )

#     # Fit the model and predict
#     pipeline.fit(X_train_transformed, y_train)
#     preds = pipeline.predict(X_valid_transformed)
    
#     # Calculate RMSE
#     return mean_squared_error(y_valid, np.maximum(preds, 0), squared=False)

# # Optimize with Optuna
# study_xgb = optuna.create_study(direction="minimize")
# study_xgb.optimize(objective_xgb, n_trials=100)

# # Best trial results
# print("Best XGBoost parameters:")
# print(study_xgb.best_trial.params)


xgb_params = dict({'n_estimators': 500, 'max_depth': 5, 'learning_rate': 0.02654256345205466, 'subsample': 0.9384786692487044, 'colsample_bytree': 0.9852280360210459, 'reg_alpha': 0.06988839793841407, 'reg_lambda': 0.01088893556703005})
xgb_params


# # HIDDEN IN VIEWER

# # Optuna objective for LightGBM
# def objective_lgbm(trial):
#     """
#     Objective function for optimizing LightGBM hyperparameters with Optuna.
#     Args:
#         trial: An Optuna trial object.
#     Returns:
#         RMSLE score on the validation set.
#     """
#     # Define model hyperparameter search space
#     params = {
#         "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
#         "num_leaves": trial.suggest_int("num_leaves", 20, 300),
#         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
#         "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
#         "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
#         "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 1.0, log=True),
#         "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 1.0, log=True),
#         "device_type": "gpu"  # Enable GPU support
#     }

#     # PCA Hyperparameter
#     # pca_n_components = trial.suggest_float("pca_n_components", 0.7, 0.99)

#     # Define the model
#     model = LGBMRegressor(**params, random_state=SEED, verbose=-1)

#     # Define the preprocessing for numerical and categorical features
#     numerical_transformer = Pipeline(steps=[
#         ('numeric_imputer', SimpleImputer(strategy='median')),
#         ('scaler', MinMaxScaler()),
#     ])
    
#     # categorical_transformer = Pipeline(steps=[
#     #     ('onehot', OneHotEncoder(handle_unknown='ignore')), # Not necessary for tree based method, but I'll include it in case i wanted to use other models | sparse_output for PCA
#     # ])

#     # Combine preprocessing steps
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ("num", numerical_transformer, X.select_dtypes(include='number').columns),
#             # ("cat", categorical_transformer, X.select_dtypes(exclude='number').columns),
#         ]
#     )

#     # Define feature selector
#     # PCA_selector = PCA(n_components=pca_n_components)
#     PCA_selector = PCA(n_components=.95)

#     # Create a pipeline with preprocessing and the model
#     pipeline = Pipeline(
#         steps=[
#             ('preprocessor', preprocessor),
#             ('feature_selection', PCA_selector),
#             ('model', model),
#         ]
#     )

#     # Fit the model and predict
#     pipeline.fit(X_train_transformed, y_train)
#     preds = pipeline.predict(X_valid_transformed)
    
#     # Calculate RMSE
#     return mean_squared_error(y_valid, np.maximum(preds, 0), squared=False)

# # Optimize with Optuna
# study_lgbm = optuna.create_study(direction="minimize")
# study_lgbm.optimize(objective_lgbm, n_trials=100)

# # Best trial results
# print("Best LightGBM parameters:")
# print(study_lgbm.best_trial.params)


lgbm_params = dict({'n_estimators': 250, 'num_leaves': 88, 'learning_rate': 0.01436331406205138, 'feature_fraction': 0.7938448041741858, 'bagging_fraction': 0.7391044648641961, 'lambda_l1': 4.8478740609655965e-06, 'lambda_l2': 0.015829294300280607})
lgbm_params


# Initialize the final models with the best hyperparameters
final_xgb = XGBRegressor(**xgb_params,random_state=SEED)
final_lgbm = LGBMRegressor(**lgbm_params,random_state=SEED,verbose=-1)


# Test with Model loop
models = [
    ('xgb', final_xgb),
    ('lgbm', final_lgbm),
]

# Define feature selector
PCA_selector = PCA(n_components=.95)

for model_name, model in models:
    model_test_pipeline(X_train_transformed, y_train, PCA_selector, model, model_name)


from sklearn.ensemble import VotingRegressor, StackingRegressor

# Define the VotingRegressor
voting_regressor = VotingRegressor([
    ('xgb', final_xgb),
    ('lgbm', final_lgbm),
    ('linear', LinearRegression()),
])

# Define the StackingRegressor_1
stacking_regressor_1 = StackingRegressor(
    estimators=[
    ('xgb', final_xgb),
    ('lgbm', final_lgbm),
    ],
    final_estimator=LinearRegression()
)

# Define the StackingRegressor_1
stacking_regressor_2 = StackingRegressor(
    estimators=[
    ('xgb', final_xgb),
    ('linear', LinearRegression()),
    ],
    final_estimator=final_lgbm
)

# Models list
combined_models = [
    ('vote', voting_regressor),
    ('stack_1', stacking_regressor_1),
    ('stack_2', stacking_regressor_2),
]

# Test Regressors
for model_name, model in combined_models:
    model_test_pipeline(X_train_transformed, y_train, PCA_selector, model, model_name)


# Create a pipeline with preprocessing and the model
numerical_transformer = Pipeline(steps=[
    ('numeric_imputer', SimpleImputer(strategy='median')),
    ('scaler', MinMaxScaler()),
])

# categorical_transformer = Pipeline(steps=[
#     ('onehot', OneHotEncoder(handle_unknown='ignore')), # Not necessary for tree based method, but I'll include it in case i wanted to use other models | sparse_output for PCA
# ])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, X.select_dtypes(include='number').columns),
        # ("cat", categorical_transformer, X.select_dtypes(exclude='number').columns),
    ]
)

# Create a pipeline with preprocessing and the model
pipeline = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        # ('feature_engineer', feature_engineer),
        ('feature_selection', PCA_selector),
        ('model', model),
    ]
)

# Fit the pipeline to the full filtered training data (using y_log for consistency)
pipeline.fit(X_train_transformed, y_train)

# Make predictions on the filtered test data
preds = pipeline.predict(X_test_transformed)

# Create the submission DataFrame using the correct index
submission_df = pd.DataFrame({'id': test_df.index, f'{target}': preds})

# Save the submission DataFrame to a CSV file
submission_df.to_csv('submission.csv', index=False)  # Avoid including the index in the CSV

