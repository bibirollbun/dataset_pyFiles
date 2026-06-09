import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, early_stopping
from sklearn.model_selection import cross_val_predict, TimeSeriesSplit
from sklearn.metrics import roc_auc_score,auc,roc_curve, mean_squared_error, mean_squared_log_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor
from category_encoders import HashingEncoder
from sklearn.feature_selection import SelectFromModel
import optuna
from boruta import BorutaPy
from scipy import stats
from sklearn.model_selection import KFold

import gc
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')


train_data = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


# Number of rows [examples], Number of columns [Features] 
print("Total rows in train data: {0}, Total columns in train data: {1}".
      format(train_data.shape[0], train_data.shape[1]))
      
print("Total rows in test data: {0}, Total columns in test data: {1}".
      format(test_data.shape[0], test_data.shape[1]))


train_data.info()


test_data.info()


# Combine features from training and test data
X = pd.concat([train_data.drop(['id', 'Listening_Time_minutes'], axis=1), test_data.drop('id', axis=1)], axis=0, ignore_index=True)

# Create target variable (0 for train data, 1 for test data)
y = [0] * len(train_data) + [1] * len(test_data)

# Define categorical columns
categorical_columns = ['Podcast_Name', 'Episode_Title', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']

# Make sure categorical columns are properly typed
for col in categorical_columns:
    if col in X.columns:
        X[col] = X[col].astype('category')

# Train an XGBoost model with categorical feature support
model = XGBClassifier(random_state=0, enable_categorical=True)

# Perform cross-validation and get predicted probabilities
cv_preds = cross_val_predict(model, X, y, cv=5, n_jobs=-1, method='predict_proba')

# Calculate ROC-AUC score
score = roc_auc_score(y_true=y, y_score=cv_preds[:, 1])
print(f"ROC-AUC score: {score:.3f}")


# drop id
train_data.drop('id', axis=1, inplace=True)
test_data.drop('id', axis=1, inplace=True)


plt.figure(figsize=(5, 5))
# Plotting
sns.histplot(data=train_data, x='Listening_Time_minutes',bins=30, kde=True, alpha=0.5)

# Adding a title
plt.title('Target Distribution', fontsize=18, weight='bold')
plt.show()


print("Skewness: %f" % train_data['Listening_Time_minutes'].skew())
print("Kurtosis: %f" % train_data['Listening_Time_minutes'].kurt())


# describe the Numarical columns in training data
train_data.describe()


# describe the Numarical columns testing data   
test_data.describe()


numeric_vars = train_data.select_dtypes("number")

def diagnostic(df, var):
    fig = plt.figure(figsize = (10, 4))
    plt.subplot(1,3,1)
    df[var].hist(bins = 40)
    plt.title("Distribution of {}".format(var))
    
    plt.subplot(1,3,2)
    stats.probplot(df[var], dist = "norm", plot = plt)
    plt.ylabel("Quantiles")
    
    plt.subplot(1,3,3)
    sns.boxplot(y = df[var])
    plt.title("Boxplot")
    plt.show()
    
for var in numeric_vars:
    diagnostic(train_data, var)


# describe the Categorical columns

train_data.describe(include=['O'])


test_data.describe(include=['O'])



df_cat = train_data.select_dtypes(include=['object'])

# it is good to look at the list of distinct values and check for ordinal variables
uniques = []
for f in df_cat.columns:
    item = {'feature':f}
    item['No. of Unique'] = df_cat[f].nunique()
    item['values'] = df_cat[f].unique().tolist()
    value_counts = df_cat[f].value_counts(normalize=True) * 100
    item['percentages'] = [round(value_counts.get(val, 0),0) for val in item['values']]
    uniques.append(item)

df_uniques = pd.DataFrame(uniques)
df_uniques = df_uniques.set_index('feature')
df_uniques


sns.boxplot(x = "Episode_Sentiment",y = "Listening_Time_minutes",
            color = "blue",
            data = train_data).set(title = "Listening_Time_minutes by Publication_Time");


sns.boxplot(x = "Publication_Day",y = "Listening_Time_minutes",
            color = "blue",
            data = train_data).set(title = "Listening_Time_minutes by Publication_Time");


import pandas as pd
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')
columns_to_impute = ['Episode_Length_minutes', 'Guest_Popularity_percentage', 'Number_of_Ads']
train_data[columns_to_impute] = imputer.fit_transform(train_data[columns_to_impute])
test_data[columns_to_impute] = imputer.fit_transform(test_data[columns_to_impute])



# Number of Ads: Round then Cap
train_data['Number_of_Ads'] = np.round(train_data['Number_of_Ads'])
# Calculate percentile AFTER rounding
p99_ads = train_data['Number_of_Ads'].quantile(0.99) 
print(f"Capping Number_of_Ads at {p99_ads}")
train_data['Number_of_Ads'] = train_data['Number_of_Ads'].clip(upper=p99_ads)
test_data['Number_of_Ads'] = test_data['Number_of_Ads'].clip(upper=p99_ads)


# Episode Length: Cap at percentiles
p99_len = train_data['Episode_Length_minutes'].quantile(0.99)
train_data['Episode_Length_minutes'] = train_data['Episode_Length_minutes'].clip( upper=p99_len)
test_data['Episode_Length_minutes'] = test_data['Episode_Length_minutes'].clip( upper=p99_len)


# --- 1. Podcast_Name: Hashing Encoding ---
n_components_hash = 8
hasher = HashingEncoder(n_components=n_components_hash, cols=['Podcast_Name'], return_df=True)

# Fit on training data and transform both train and test
# HashingEncoder automatically handles new categories in the test set
train_Podcast_Name_hashed = hasher.fit_transform(train_data['Podcast_Name'])
test_Podcast_Name_hashed = hasher.transform(test_data['Podcast_Name'])

# Rename columns for clarity (optional)
hash_cols = [f'PodcastName_Hash_{i}' for i in range(n_components_hash)]
train_Podcast_Name_hashed.rename(columns={f'col_{i}': hash_cols[i] for i in range(n_components_hash)}, inplace=True)
test_Podcast_Name_hashed.rename(columns={f'col_{i}': hash_cols[i] for i in range(n_components_hash)}, inplace=True)

# Add the NEW hash columns from the results back to the ORIGINAL DataFrames
# We select *only* the hash columns from the temporary dataframes
train_data = pd.concat([train_data, train_Podcast_Name_hashed], axis=1)
test_data = pd.concat([test_data, test_Podcast_Name_hashed], axis=1)



# --- 2. Episode_Title: Extract Number ---
def extract_episode_number(df, col_name='Episode_Title'):
    # Extract first sequence of digits found in the string
    numbers = df[col_name].str.extract(r'(\d+)', expand=False)
    # Convert to numeric, coercing errors (non-matches) to NaN, then fill NaN with 0
    numbers_numeric = pd.to_numeric(numbers, errors='coerce').fillna(0).astype(int)
    return numbers_numeric

train_data['Episode_Number'] = extract_episode_number(train_data, 'Episode_Title')
test_data['Episode_Number'] = extract_episode_number(test_data, 'Episode_Title')


# --- 3. Genre: One-Hot Encoding ---
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore') # ignore unknown categories in test set

# Fit on training data
ohe.fit(train_data[['Genre']])

# Get feature names for the new columns
ohe_feature_names = ohe.get_feature_names_out(['Genre'])

# Transform train data
train_genre_encoded = ohe.transform(train_data[['Genre']])
train_genre_data = pd.DataFrame(train_genre_encoded, columns=ohe_feature_names, index=train_data.index)

# Transform test data
test_genre_encoded = ohe.transform(test_data[['Genre']])
test_genre_data = pd.DataFrame(test_genre_encoded, columns=ohe_feature_names, index=test_data.index)

# Concatenate the new columns and drop the original 'Genre' column
train_data = pd.concat([train_data, train_genre_data], axis=1)
test_data= pd.concat([test_data, test_genre_data], axis=1)


# --- 4. Publication_Day: Cyclical Encoding ---
day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
days_in_week = 7

def encode_cyclical(df, col, max_val, map_dict=None):
    if map_dict:
         df[col + '_num'] = df[col].map(map_dict)
    else:
         df[col + '_num'] = df[col] 

    df[col + '_sin'] = np.sin(2 * np.pi * df[col + '_num'] / max_val)
    df[col + '_cos'] = np.cos(2 * np.pi * df[col + '_num'] / max_val)
    return df.drop([col, col + '_num'], axis=1) # Drop original and intermediate numerical column

train_data = encode_cyclical(train_data, 'Publication_Day', days_in_week, day_map)
test_data = encode_cyclical(test_data, 'Publication_Day', days_in_week, day_map)


# --- 5. Publication_Time: Ordinal Encoding ---
# Define the explicit order
time_order = ['Morning', 'Afternoon', 'Evening', 'Night']
# Use handle_unknown and unknown_value to manage categories present in test but not train
# or unexpected values. np.nan or -1 are common choices for unknown_value.
time_encoder = OrdinalEncoder(categories=[time_order], handle_unknown='use_encoded_value', unknown_value=np.nan) # Or unknown_value=-1

# Fit on training data (to learn the mapping structure, although we provided it)
time_encoder.fit(train_data[['Publication_Time']])

# Transform both sets
train_data['Publication_Time_Ordinal'] = time_encoder.transform(train_data[['Publication_Time']])
test_data['Publication_Time_Ordinal'] = time_encoder.transform(test_data[['Publication_Time']])


# --- 6. Episode_Sentiment: Ordinal Encoding ---
# Define the explicit order (using -1, 0, 1 might be slightly more interpretable for sentiment)
# Let's stick to 0, 1, 2 for consistency with typical OrdinalEncoder output
sentiment_order = ['Negative', 'Neutral', 'Positive']
sentiment_encoder = OrdinalEncoder(categories=[sentiment_order], handle_unknown='use_encoded_value', unknown_value=np.nan) # Or -1

# Fit on training data
sentiment_encoder.fit(train_data[['Episode_Sentiment']])

# Transform both sets
train_data['Episode_Sentiment_Ordinal'] = sentiment_encoder.transform(train_data[['Episode_Sentiment']])
test_data['Episode_Sentiment_Ordinal'] = sentiment_encoder.transform(test_data[['Episode_Sentiment']])

# Drop the original column
train_data.drop(['Episode_Sentiment','Publication_Time','Episode_Title','Genre','Podcast_Name'], axis=1, inplace=True)
test_data.drop(['Episode_Sentiment','Publication_Time','Episode_Title','Genre','Podcast_Name'], axis=1, inplace=True)


plt.figure(figsize=(20, 15))  # Adjust the size for clarity
sns.heatmap(train_data.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap", fontsize=14, fontweight='bold')
plt.show()


y = train_data['Listening_Time_minutes']
X = train_data.drop(['Listening_Time_minutes'], axis=1)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)



def objective_catboost_gpu(trial):
    """
    Optuna objective function for CatBoostRegressor with GPU acceleration.
    """

    # --- Hyperparameters for CatBoost ---

    # iterations: Number of boosting iterations (trees)
    # Note: With early stopping, this acts as a maximum limit.
    iterations = trial.suggest_int('iterations', 400, 2500) # GPU can handle more trees

    # learning_rate: Step size shrinkage
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.2, log=True) # Log scale often good for LR

    # depth: Depth of the trees (CatBoost alias for max_depth)
    depth = trial.suggest_int('depth', 4, 10)

    # l2_leaf_reg: L2 regularization coefficient (Lambda)
    l2_leaf_reg = trial.suggest_float('l2_leaf_reg', 0.1, 10.0, log=True)

    # border_count: Number of splits for numerical features 
    border_count = trial.suggest_categorical('border_count', [32, 64, 128, 255])

    # random_strength: Amount of randomness added to splits (helps prevent overfitting)
    random_strength = trial.suggest_float('random_strength', 1e-8, 5.0, log=True)

    # bagging_temperature: Controls intensity of Bayesian bootstrapping (0: no bagging, 1: standard bagging)
    bagging_temperature = trial.suggest_float('bagging_temperature', 0.0, 1.0)

    # --- Model Initialization ---
    params = {
        'iterations': iterations,
        'learning_rate': learning_rate,
        'depth': depth,
        'l2_leaf_reg': l2_leaf_reg,
        'loss_function': 'RMSE',      # Use RMSE loss for regression
        'eval_metric': 'RMSE',        # Evaluate performance using RMSE
        'task_type': "GPU",           # Enable GPU usage
        'devices': '0',               # Specify GPU device (usually '0')
        'random_seed': 123,
        'verbose': 0,                 # Suppress verbose output during Optuna trials
        'border_count': border_count,
        'random_strength': random_strength,
        'bagging_temperature': bagging_temperature,
        'early_stopping_rounds': 50   # Stop if validation RMSE doesn't improve for 50 rounds
    }


    # Instantiate the CatBoostRegressor model
    model = CatBoostRegressor(**params)

    # --- Train & Evaluate ---
    try:
        # Train the model with early stopping
        # Pass validation set for early stopping and internal evaluation
        model.fit(X_train, y_train,
                  eval_set=(X_val, y_val),
                  verbose=0) # Suppress fit output within the trial

        # Get the best score (RMSE) achieved on the validation set during training
        # Optuna aims to minimize this score.
        score = model.get_best_score()['validation']['RMSE']

        # # Alternative: Predict manually and calculate score (less efficient if using early stopping)
        # y_pred = model.predict(X_val)
        # score = mean_squared_error(y_val, y_pred, squared=False) # Calculate RMSE

    except Exception as e:
        import traceback
        print(f"Error during trial for CatBoost GPU with params {trial.params}:")
        print(traceback.format_exc())
        # Return a large value to tell Optuna this trial failed or was bad
        return float('inf')

    return score


# Create an Optuna study
study = optuna.create_study(direction='minimize', sampler=optuna.samplers.RandomSampler(seed=123))
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Function to log the best trial
def log_best_trial(study, trial):
    if study.best_trial == trial:
        print(f"New best trial: {trial.number} with value: {trial.value} and params: {trial.params}")

# Run the optimization
study.optimize(objective_catboost_gpu, n_trials=25, callbacks=[log_best_trial])

# Get the best parameters
best_params = study.best_params
best_score = study.best_value
print(f"Best Hyperparameters: {best_params}")
print(f"Best RMSE: {best_score:.6f}")


final_catboost_model = CatBoostRegressor(**best_params,
                                          task_type='GPU',
                                          devices='0',
                                          random_seed=123,
                                          loss_function='RMSE',
                                          eval_metric='RMSE',
                                          )
final_catboost_model.fit(X_train, y_train,eval_set=(X_val, y_val),verbose=0)
y_pred = final_catboost_model.predict(X_val)
score = mean_squared_error(y_val, y_pred, squared=False)
print(f"Best RMSE: {score:.6f}")


# Feature importance
feature_importances = final_catboost_model.get_feature_importance()

feature_names = X.columns

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.barh(feature_names, feature_importances)
plt.xlabel('Feature Importance')
plt.title('CatBoost Feature Importances')
plt.show()


# Residual Analysis
residuals = y_val - y_pred

# Residuals vs Predicted Values
plt.figure(figsize=(12, 6))
sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
plt.title("Residuals vs Predicted Values", fontsize=18, fontweight='bold')
plt.xlabel("Predicted Values", fontsize=12)
plt.ylabel("Residuals", fontsize=12)
plt.tight_layout()
plt.show()

# Residual Distribution
plt.figure(figsize=(10, 6))
sns.histplot(residuals, bins=30, kde=True)
plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
plt.title("Distribution of Residuals", fontsize=16, fontweight='bold')
plt.xlabel("Residuals", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.tight_layout()
plt.show()


results = final_catboost_model.predict(test_data)


submission = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")
submission['Listening_Time_minutes'] = results
submission.to_csv('submission.csv', index=False)

