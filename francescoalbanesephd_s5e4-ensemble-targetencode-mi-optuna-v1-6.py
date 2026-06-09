# %%capture
!pip install itables
!pip install scikit-learn==1.3.1
!pip install optuna-integration[xgboost]==4.3.0
!pip install optuna-integration[lightgbm]==4.3.0


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





# Import datasets
TRAIN_DF = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv',index_col='id')
TEST_DF = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv',index_col='id')
# ORIGINAL_DATA = pd.read_csv("/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv")


# For each dataset...
for dataset in [TRAIN_DF, TEST_DF]:
    # check their shape
    print(dataset.shape)
    # and fix column names
    dataset.columns = [col.lower() for col in dataset.columns]


import missingno as msno
plt.figure(figsize=(5,10))

# Visualize missing values as a matrix 
for df, df_name in [
    (TRAIN_DF, "Train"), 
    (TEST_DF, "Test"), 
    # (ORIGINAL_DATA, "Original")
]:
    
    msno.matrix(df,figsize=(20,8))
    plt.title(f"Missing Data in {df_name} Dataset: {df.isnull().sum().sum()}", fontsize = 30)
    plt.show()


# Display missing vals in case the msno.matrix can't show some
pd.concat([
    pd.DataFrame(TRAIN_DF.isna().sum(),columns=['n missing: train']), 
    pd.DataFrame(TEST_DF.isna().sum(),columns=['n missing: test']),
    pd.DataFrame(TRAIN_DF.isna().sum()/TRAIN_DF.shape[0],columns=['% missing: train']).round(3)
],axis=1)


from sklearn.impute import SimpleImputer

# It's likely that Guest_Popularity_percentage is NaN because no guest is present. Therefore, I will impute it as 0
TRAIN_DF.fillna({'guest_popularity_percentage': 0}, inplace=True)
TEST_DF.fillna({'guest_popularity_percentage': 0}, inplace=True)

# Impute missing values (simple median imputation now, I'll consider better strategies later)
cols2impute = [col for col in TRAIN_DF.columns if TRAIN_DF[col].isna().any()]

# 1. Instantiate the imputer 
imputer = SimpleImputer(strategy='median')

# 2. Fit and transform
TRAIN_DF[cols2impute] = pd.DataFrame(imputer.fit_transform(TRAIN_DF[cols2impute]),index=TRAIN_DF.index)
TEST_DF[cols2impute] = pd.DataFrame(imputer.transform(TEST_DF[cols2impute]),index=TEST_DF.index)


# TRAIN_DF Overview
print("="*30,"Show Training Dataset for initial data assessment","="*30)
show(TRAIN_DF)


# Identify Numeric and Categorical Vars
num_vars = TRAIN_DF.select_dtypes(include='number').columns
cat_vars = TRAIN_DF.select_dtypes(exclude='number').columns


# TRAIN_DF Vars Overview
print("="*50,f"Description of Train Dataset","="*50)

# Display descriptive stats of numerical vars
print('\n',"="*50,f"Num Vars","="*50)
num_descriptive_stats = TRAIN_DF[num_vars].describe().T.round(2)
num_descriptive_stats['Skew'] = TRAIN_DF[num_vars].skew()
num_descriptive_stats['Kurt'] = TRAIN_DF[num_vars].kurt()
display(num_descriptive_stats)

# Display descriptive stats of cat vars
print('\n',"="*50,f"Cat Vars","="*50)
cat_descriptive_stats = TRAIN_DF[cat_vars].describe().T
display(cat_descriptive_stats)


# Visually check the cols
fig = plt.figure(figsize=(24, 8))

for i, col in enumerate(TRAIN_DF[num_vars].columns):
    fig.add_subplot(2, 3, i+1)
    sns.scatterplot(
        x=TRAIN_DF[col],
        y=TRAIN_DF['listening_time_minutes'],
    )


# Print unique values for 'number_of_ads'
print("Top 10 highest 'number_of_ads':")
print(TRAIN_DF['number_of_ads'].unique())
print("\n")

# Print the k highest values for 'episode_length_minutes'
print("Top 10 highest 'episode_length_minutes':")
print(TRAIN_DF['episode_length_minutes'].nlargest(5))


# Visually check the cols
fig = plt.figure(figsize=(24, 8))

for i, col in enumerate(TEST_DF.select_dtypes('number').columns):
    fig.add_subplot(2, 3, i+1)
    sns.boxplot(
        x=TEST_DF[col],
    )


# Print unique values for 'number_of_ads'
print("Top 10 highest 'number_of_ads':")
print(TEST_DF['number_of_ads'].unique())
print("\n")

# Print the k highest values for 'episode_length_minutes'
print("Top 10 highest 'episode_length_minutes':")
print(TEST_DF['episode_length_minutes'].nlargest(5))


ads_median = TRAIN_DF['number_of_ads'].median()
episode_length_median = TRAIN_DF['episode_length_minutes'].median()

for df in [TRAIN_DF, TEST_DF]:
    df['number_of_ads'].mask(df['number_of_ads'] > 3, ads_median, inplace=True)
    df['episode_length_minutes'].mask(df['episode_length_minutes'] > 121, episode_length_median, inplace=True)
    df['host_popularity_percentage'].clip(lower=0, upper=100, inplace=True)
    df['guest_popularity_percentage'].clip(lower=0, upper=100, inplace=True)
                                    


# Identify Target
target = 'listening_time_minutes'

# Define a custom color palette
custom_palette = ['#219ebc', '#c1121f']

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 4))
annot_kws = {'xy': (0.6, 0.8), 'xycoords': 'axes fraction', 'fontsize': 10}

# Box plot
sns.boxplot(data=TRAIN_DF, x=target, color='#219ebc', ax=axes[0])
axes[0].set_xlabel(target)
axes[0].set_title(f"Box Plot of {target}")

# Histogram
sns.histplot(data=TRAIN_DF, x=target, color='#219ebc', kde=True, bins=30, ax=axes[1])
axes[1].set_xlabel(target)
axes[1].set_ylabel("Frequency")
axes[1].set_title(f"Histogram of {target}")
axes[1].annotate(f"Skewness (TRAIN): {TRAIN_DF[target].skew():.2f}\nKurtosis (TRAIN): {TRAIN_DF[target].kurt():.2f}",
                 xy=annot_kws['xy'], xycoords=annot_kws['xycoords'], fontsize=annot_kws['fontsize'])



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
for variable in TRAIN_capped[num_vars].columns.difference([target]):
    create_variable_plots(TRAIN_capped, TEST_capped, variable)


TRAIN_capped.select_dtypes(exclude='number').columns


from IPython.display import clear_output

# Plot categorical features with count plots
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))  # Adjust grid size based on data
axes = axes.flatten()

for i, column in enumerate(cat_vars):
    clear_output(wait=False)
    print(f'Processing categorical feature: {column}')
    sns.countplot(y=TRAIN_capped[column], ax=axes[i], order=TRAIN_capped[column].value_counts().index,orient='h')
    axes[i].set_title(column, fontsize=9)
    axes[i].tick_params(axis='both', which='major', labelsize=6)
    # plt.xticks(rotation=45)

fig.suptitle('Categorical Feature Distributions', fontsize=11)
plt.tight_layout()
plt.show()

# Categorical Cols insights
display(TRAIN_capped[cat_vars].describe())


# Create a heatmap to visualize the correlation matrix of the TRAIN_DF DataFrame
plt.figure(figsize=(12,8))
sns.heatmap(data=TRAIN_capped.corr(numeric_only=True).round(4), annot=True, cmap="coolwarm", linewidth = 0.3,)


TRAIN_capped.columns


# Loop through low-cardinality categorical variables to display summary statistics and box plot
for column in ['podcast_name','genre','publication_day','publication_time','episode_sentiment']:
    # Calculate summary statistics grouped by the categorical column
    stats = TRAIN_capped.groupby(column)[target].describe()
    
    # Display summary statistics
    print(f"\nSummary Statistics for {target} by {column}:")
    display(stats.sort_values('50%',ascending=False))
    
    # Plot violin plot
    plt.figure(figsize=(12, 4))
    sns.violinplot(data=TRAIN_capped, x=column, y=target, palette='Spectral',inner="quartile", linewidth=2, cut=0)
    sns.violinplot(data=TRAIN_capped, x=column, y=target, palette='Spectral',inner="box", linewidth=2, cut=0,)
    plt.title(f'{target} by {column}', fontsize=12)
    plt.xlabel(column, fontsize=11)
    plt.ylabel(target, fontsize=11)
    plt.xticks(rotation=90,)
    plt.tight_layout()
    plt.show()


# Define categorical maps
day_map = {
    'Monday': 1,
    'Tuesday': 2,
    'Wednesday': 3,
    'Thursday': 4,
    'Friday': 5,
    'Saturday': 6,
    'Sunday': 7
}

time_of_day_map = {
    'Morning': 1,
    'Afternoon': 2,
    'Evening': 3, 
    'Night': 4
}

# Define feature engineering function
def feature_engineer(df):
    Îµ = 1e-4
    df['publication_day'] = df['publication_day'].map(day_map)
    df['publication_time'] = df['publication_time'].map(time_of_day_map)
    df['is_weekend'] = df['publication_day'].isin([6,7]).astype(int)
    # df['publication_day_sin'] = np.sin(2 * np.pi * df['publication_day'] / 7)
    # df['publication_day_cos'] = np.cos(2 * np.pi * df['publication_day'] / 7)
    # df['publication_time_sin'] = np.sin(2 * np.pi * df['publication_time'] / 4)
    # df['publication_time_cos'] = np.cos(2 * np.pi * df['publication_time'] / 4)
    df['host_guest_popularity_ratio'] = df['host_popularity_percentage'] / (df['guest_popularity_percentage'] + Îµ)
    # df['speakers_popularity_binned'] = ((df['host_popularity_percentage'] + df['guest_popularity_percentage']) / 2) // 10 * 10
    df['popularity_difference_abs'] = abs(df['host_popularity_percentage'] - df['guest_popularity_percentage'])
    # df['popularity_interaction'] = df['host_popularity_percentage'] * df['guest_popularity_percentage']
    df['guest_presence'] = (df['guest_popularity_percentage']!=0).astype(int)
    df['guest_over_host'] = (df['guest_popularity_percentage'] > df['host_popularity_percentage']).astype(int)
    df['ads_per_minute'] = df['number_of_ads'] / (df['episode_length_minutes'] + Îµ)
    df['over_one_hour'] = (df['episode_length_minutes'] > 60).astype(int)
    df['has_ads'] = (df['number_of_ads'] != 0).astype(int)
    # df['more_than_one_ad'] = (df['number_of_ads'] > 1).astype(int)
    df['episode_title'] = df['episode_title'].apply(lambda x: x.split()[1]).astype(int)


    return df

# Apply function to datasets
feature_engineer(TRAIN_capped)
feature_engineer(TEST_capped)


# Create a heatmap to visualize the correlation matrix of the TRAIN_DF DataFrame
plt.figure(figsize=(12,8))
sns.heatmap(data=TRAIN_capped.corr(numeric_only=True).round(4), 
            annot=True, 
            cmap="coolwarm", 
            linewidth = 0.3,
            annot_kws={"rotation": 90,
                       "fontsize": 10})

plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.title('Correlation Heatmap (Full)')
plt.show()

# Create a heatmap to visualize the correlation matrix of the engineered train_capped DataFrame
plt.figure(figsize=(22,2))
sns.heatmap(data=TRAIN_capped.corr(numeric_only=True)[[target]].T.round(4), 
            annot=True, 
            cmap="coolwarm", 
            linewidth = 0.3,
            annot_kws={"rotation": 90})

plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.title('Correlation Heatmap (Target only)')
plt.show()


from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler, TargetEncoder
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import KFold, cross_validate, train_test_split, RandomizedSearchCV
from sklearn.feature_selection import SelectPercentile, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

import optuna
from optuna.integration import XGBoostPruningCallback
from optuna.integration import LightGBMPruningCallback


def cross_validate_model(model, X, y, k):
    
    # Define scorer
    rmse_scorer = make_scorer(mean_squared_error, squared=False, greater_is_better = False)
    # Define Kfold
    cv = KFold(n_splits=k, shuffle=True, random_state=SEED)

    # Get Scores
    scores = cross_validate(model, X, y, return_train_score=True, cv=cv,
                       scoring=rmse_scorer, verbose=10, 
                            # n_jobs=-1 # maybe counterproductive if n_jobs -1 is already set in the models
                           )

    # Return the cross-validation results
    # print(f"RMSE score: {-scores['test_score'].mean():.4f}")
    return -scores['test_score'].mean()


# Select device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Define Features
cyclical_cols = [col for col in TRAIN_capped.columns if col.endswith('sin') or col.endswith('cos')]
cols_to_exclude = list([target])+list(cyclical_cols) # Rationale: Skip MinMax for cyclical features, remove target from pipeline 
numerical_variables = TRAIN_capped.select_dtypes(include='number').columns.difference(cols_to_exclude)
categorical_variables = TRAIN_capped.select_dtypes(exclude='number').columns

# Define X and y
X = TRAIN_capped.copy()
y = X.pop(target)

# Preprocessing step 1: numerical vars
numerical_transformer = Pipeline([
    ('scaler', MinMaxScaler()),
])

# Preprocessing step 2a: categorical vars OHE
categorical_transformer_ohe = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Preprocessing step 2b: categorical vars OE (suitable strategy for tree based models: much faster with often similar accuracy)
categorical_transformer_oe = Pipeline([
    ('ordinal_encoder', OrdinalEncoder())
])

# Preprocessing step 2c: categorical vars TE 
categorical_transformer_te = Pipeline([
    ('target_encoder', TargetEncoder(target_type='continuous', smooth='auto', cv=5, shuffle=True, random_state=SEED))
])

# Define the preprocessors
preprocessor = make_column_transformer(
    # (numerical_transformer, numerical_variables), # Only Tree models used here, so it's not necessary
    (categorical_transformer_te, categorical_variables),
    remainder='passthrough'
)


# Split Data for faster hyperparameter search
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.33, random_state=SEED)


# Test the pipelines with small samples before running the code with full data
X_reduced = X.sample(5_000)
y_reduced = y.iloc[X_reduced.index]


# # Optuna objective for XGBoost
# USE_CV = True

# def objective_xgb(trial):
#     """
#     Objective function for optimizing XGBoost hyperparameters with Optuna.
#     Args:
#         trial: An Optuna trial object.
#     Returns:
#         RMSE score on cross validation.
#     """
#     # Define model hyperparameter search space
#     params = {
#         "n_estimators": trial.suggest_int("n_estimators", 100, 3000, step=50),
#         "max_depth": trial.suggest_int("max_depth", 3, 12),
#         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
#         "subsample": trial.suggest_float("subsample", 0.6, 1.0),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
#         "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
#         "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
#         # "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
#         # "gamma": trial.suggest_float("gamma", 0, 5),
#         "n_jobs":-1,
#         "device":device
#     }

#     # MI Hyperparameter
#     mi_percentile = trial.suggest_int("percentile", 70, 100, step=5)

#     # Define the model
#     model = XGBRegressor(**params, random_state=SEED)

#     # Define feature selector
#     mi_selector = SelectPercentile(mutual_info_regression,percentile=mi_percentile)

#     # Create a pipeline with preprocessing and the model
#     pipeline = Pipeline(
#         steps=[
#             ('preprocessor', preprocessor),
#             ('feature_selection', mi_selector),
#             ('model', model),
#         ]
#     )

#     if USE_CV:
#         # Cross validate
#         score = cross_validate_model(pipeline, X, y, 3)

#         # Report Intermediate Result
#         trial.report(score, step=0)

#         # Check if trial should be pruned
#         if trial.should_prune():
#             raise optuna.exceptions.TrialPruned()
        
#         return score
        
#     else:
#         # Fit the model and predict
#         pipeline.fit(X_train, y_train)
#         preds = pipeline.predict(X_valid)
        
#         # Calculate RMSE
#         score = mean_squared_error(y_valid, np.maximum(preds, 0), squared=False) 

#         # Report Intermediate Result
#         trial.report(score, step=0)

#         # Check if trial should be pruned
#         if trial.should_prune():
#             raise optuna.exceptions.TrialPruned()
        
#         return score 

# # Optimize with Optuna
# pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
# study_xgb = optuna.create_study(direction="minimize", pruner=pruner)
# study_xgb.optimize(objective_xgb, n_trials=20)

# # Best trial results
# print("Best XGBoost parameters:")
# print(study_xgb.best_trial.params)


# # Optuna objective for LightGBM
# USE_CV = True

# def objective_lgbm(trial):
#     """
#     Objective function for optimizing LightGBM hyperparameters with Optuna.
#     Args:
#         trial: An Optuna trial object.
#     Returns:
#         RMSE score on cross validation.
#     """
#     # Define model hyperparameter search space
#     params = {
#         'n_estimators': trial.suggest_int('n_estimators', 100, 3000, step=50),
#         'max_depth': trial.suggest_int('max_depth', 3, 15),
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
#         'num_leaves': trial.suggest_int('num_leaves', 32, 512),
#         'min_child_samples': trial.suggest_int('min_child_samples', 5, 255),
#         'subsample': trial.suggest_float('subsample', 0.7, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
#         'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
#         'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
#         'max_bin': trial.suggest_int('max_bin', 96, 255),
#         'verbosity': -1,
#         'device': 'gpu',
#         'njobs': -1
#     }

#     # MI Hyperparameter
#     mi_percentile = trial.suggest_int("percentile", 70, 100, step=5)

#     # Define the model
#     model = LGBMRegressor(**params, random_state=SEED,
#                           objective='regression', metric='rmse', 
#                           # early_stopping_round=250,
#                          )

#     # Define feature selector
#     mi_selector = SelectPercentile(mutual_info_regression,percentile=mi_percentile)

#     # Create a pipeline with preprocessing and the model
#     pipeline = Pipeline(
#         steps=[
#             ('preprocessor', preprocessor),
#             ('feature_selection', mi_selector),
#             ('model', model),
#         ]
#     )

#     if USE_CV:
#         # Cross validate
#         score = cross_validate_model(pipeline, X, y, 3)

#         # Report intermediate result
#         trial.report(score, step=0)

#         # Check for pruning
#         if trial.should_prune():
#             raise optuna.exceptions.TrialPruned()
        
#         return score
        
#     else:
#         # Fit the model and predict
#         pipeline.fit(X_train, y_train)
#         preds = pipeline.predict(X_valid)

#         # Report intermediate result
#         trial.report(score, step=0)

#         # Check for pruning
#         if trial.should_prune():
#             raise optuna.exceptions.TrialPruned()
        
#         # Calculate RMSE
#         score = mean_squared_error(y_valid, np.maximum(preds, 0), squared=False)
#         return score

# # Optimize with Optuna
# pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
# study_lgbm = optuna.create_study(direction="minimize", pruner=pruner)
# study_lgbm.optimize(objective_lgbm, n_trials=20)

# # Best trial results
# print("Best LightGBM parameters:")
# print(study_lgbm.best_trial.params)


# Define XGB model hyperparameters
xgb_params = {
    'n_estimators': 2900, 
    'max_depth': 11, 
    'learning_rate': 0.018401897581513407, 
    'subsample': 0.8461201872634104, 
    'colsample_bytree': 0.9172130210962273, 
    'reg_alpha': 1.893197079814006e-05, 
    'reg_lambda': 0.7276018034234313, 
 }

xgb_percentile = 75

# Define LGBM model hyperparameters (partial, I had to stop optimization early)

lgbm_params = {
    'n_estimators': 850, 
     'max_depth': 15, 
     'learning_rate': 0.036521889830212026, 
     'num_leaves': 246, 
     'min_child_samples': 28, 
     'subsample': 0.9103904433563347, 
     'colsample_bytree': 0.7573705433934719, 
     'reg_alpha': 3.64033791618735, 
     'reg_lambda': 9.200519530479228, 
     'max_bin': 218, 
}
lgbm_percentile = 95


models = {
    "LGBMRegressor": LGBMRegressor(**lgbm_params, random_state=SEED, objective='regression', metric='rmse'),
    "XGBoost": XGBRegressor(**xgb_params, random_state=SEED, device=device),
    "CatBoost": CatBoostRegressor(random_state=SEED, verbose=0, 
                                 task_type= "GPU",devices='0'
                                 ),
}


# # For loop iterating models

# results = {}
# for name, model in models.items():
#     print("="*20,f"KFold with model: {name}","="*20)

#     if name == 'LGBMRegressor':
#         percentile = lgbm_percentile
#     else:
#         percentile = xgb_percentile

#     # Preprocessor w OrdinalEncoder
#     pipeline = Pipeline(
#         steps=[
#             ('preprocessor', preprocessor),
#             ('selector', SelectPercentile(mutual_info_regression,percentile=percentile)),
#             ('model', model)
#         ]
#     )

#     # CV
#     cross_validate_model(pipeline, X, y, 5)
#     # cross_validate_model(pipeline, X_reduced, y_reduced, 5)


# Baseline Stacked Model
estimators = []
for name, model in models.items():          
    
    if name == 'LGBMRegressor':
        percentile = lgbm_percentile
    else:
        percentile = xgb_percentile # applies the same to CatBoost
        
    pipeline = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('selector', SelectPercentile(mutual_info_regression,percentile=percentile)),
            ('model', model)
        ]
    )
    
    estimators.append((name, pipeline))

stacking_model = StackingRegressor(estimators=estimators, final_estimator=Ridge(random_state=SEED))

# CV
cross_validate_model(stacking_model, X, y, 5)
# cross_validate_model(stacking_model, X_reduced, y_reduced, 5)


stacking_model


# Train PentaModel StackEnsemble on the Full Train Data
stacking_model.fit(X,y)

# Display Model Framework
stacking_model


# Create Submission File
submission_df = pd.DataFrame({
    'id': list(TEST_capped.index),
    'Listening_Time_minutes': stacking_model.predict(TEST_capped)
})

# Save to CSV
submission_df.to_csv('submission.csv', index=False)

# Display the first 5 rows
submission_df.head(5)

