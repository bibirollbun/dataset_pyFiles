# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# --- Visualization Libraries ---
import matplotlib.pyplot as plt 
import seaborn as sns
%matplotlib inline
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_style('whitegrid')

# --- Scikit-learn for Preprocessing and Metrics ---
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, QuantileTransformer, RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from scipy.stats import uniform, randint

# --- TensorFlow and Keras for Deep Learning ---
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# --- A Classical Model for Baseline/Comparison ---
import xgboost as xgb

# --- Utilities ---
import warnings
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# --- Load the datasets ---

BASE_PATH = '/kaggle/input/playground-series-s5e4/'

train_df = pd.read_csv(BASE_PATH + 'train.csv', index_col='id')

test_df = pd.read_csv(BASE_PATH + 'test.csv', index_col='id')

submission_df = pd.read_csv(BASE_PATH + 'sample_submission.csv')

print(f"Training data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")
print(f"Sample submission shape: {submission_df.shape}")


# --- Initial Data Inspection ---

print("\n--- Training Data Info ---")
train_df.info()
print("\n--- Test Data Info ---")
test_df.info()


train_df.head()


train_df.describe()


print("\n--- Missing Values in Training Data ---")
print(train_df.isnull().sum()[train_df.isnull().sum() > 0])
print("\n--- Missing Values in Test Data ---")
print(test_df.isnull().sum()[test_df.isnull().sum() > 0])


plt.figure(figsize=(12, 5))
sns.histplot(train_df['Listening_Time_minutes'], kde=True, bins=50)
plt.title('Distribution of Listening Time (Target)')
plt.xlabel('Listening Time (minutes)')
plt.ylabel('Frequency')
plt.show()


plt.figure(figsize=(10, 6))
sns.scatterplot(x=train_df['Episode_Length_minutes'].fillna(train_df['Episode_Length_minutes'].median()),
                y=train_df['Listening_Time_minutes'])
plt.title('Listening Time vs. Episode Length')
plt.xlabel('Episode Length (minutes)')
plt.ylabel('Listening Time (minutes)')
plt.show()


numerical_features = ['Episode_Length_minutes', 'Host_Popularity_percentage',
                      'Guest_Popularity_percentage', 'Number_of_Ads']
fig, axes = plt.subplots(len(numerical_features), 1, figsize=(10, len(numerical_features) * 4))
for i, col in enumerate(numerical_features):
    sns.histplot(train_df[col], kde=True, ax=axes[i], bins=50)
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')
plt.tight_layout()
plt.show()


print("\n--- Investigating Potential Anomalies ---")

print("\nHost Popularity > 100:")
print(train_df[train_df['Host_Popularity_percentage'] > 100]['Host_Popularity_percentage'].describe())

print("\nGuest Popularity > 100:")
print(train_df[train_df['Guest_Popularity_percentage'] > 100]['Guest_Popularity_percentage'].describe())

print("\nNumber of Ads - Unusual Values:")
non_integer_ads = train_df[train_df['Number_of_Ads'].notna() & (train_df['Number_of_Ads'] % 1 != 0)]
print(f"Number of rows with non-integer 'Number_of_Ads': {len(non_integer_ads)}")
print("Examples of non-integer 'Number_of_Ads':")
print(non_integer_ads['Number_of_Ads'].head())

print("\nEpisode Length = 0:")
print(f"Number of rows with Episode_Length_minutes == 0: {len(train_df[train_df['Episode_Length_minutes'] == 0])}")



plot_categorical_features = ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']

print("\n--- Visualizing Distributions of Low/Medium Cardinality Categorical Features (with Counts) ---")

for col in plot_categorical_features:
    if col in train_df.columns:
        plt.figure(figsize=(12, 6))
        
        ax = sns.countplot(data=train_df,
                           x=col,
                           order=train_df[col].value_counts().index,
                           palette='viridis')

        ax.bar_label(ax.containers[0], fmt='%.0f')

        plt.title(f'Frequency Distribution of {col} (with Counts)')
        plt.xlabel(col)
        plt.ylabel('Frequency (Count)')
        plt.xticks(rotation=45, ha='right')

        plt.ylim(0, ax.get_ylim()[1] * 1.05)

        plt.tight_layout()
        plt.show()


# --- Training Data ---
missing_train_counts = train_df.isnull().sum()
missing_train_counts = missing_train_counts[missing_train_counts > 0]
total_train_rows = len(train_df)
missing_train_perc = (missing_train_counts / total_train_rows) * 100

missing_train_summary = pd.DataFrame({
    'Missing Count': missing_train_counts,
    'Percentage (%)': missing_train_perc
})
missing_train_summary.sort_values(by='Percentage (%)', ascending=False, inplace=True)

print("\nMissing Values Summary - Training Data:")
if missing_train_summary.empty:
    print("No missing values found in the training data.")
else:
    display(missing_train_summary.round(2)) 

    plt.figure(figsize=(10, 4))
    sns.barplot(x=missing_train_summary.index, y=missing_train_summary['Percentage (%)'], palette='viridis')
    plt.title('Percentage of Missing Values by Feature (Training Data)')
    plt.xlabel('Features')
    plt.ylabel('Percentage Missing (%)')
    plt.xticks(rotation=45, ha='right')

    ax = plt.gca()
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f%%')

    plt.tight_layout()
    plt.show()

# --- Test Data ---
missing_test_counts = test_df.isnull().sum()
missing_test_counts = missing_test_counts[missing_test_counts > 0]
total_test_rows = len(test_df)
missing_test_perc = (missing_test_counts / total_test_rows) * 100

missing_test_summary = pd.DataFrame({
    'Missing Count': missing_test_counts,
    'Percentage (%)': missing_test_perc
})
missing_test_summary.sort_values(by='Percentage (%)', ascending=False, inplace=True)

print("\nMissing Values Summary - Test Data:")
if missing_test_summary.empty:
    print("No missing values found in the test data.")
else:
    display(missing_test_summary.round(2)) # 

    plt.figure(figsize=(10, 4))
    sns.barplot(x=missing_test_summary.index, y=missing_test_summary['Percentage (%)'], palette='viridis')
    plt.title('Percentage of Missing Values by Feature (Test Data)')
    plt.xlabel('Features')
    plt.ylabel('Percentage Missing (%)')
    plt.xticks(rotation=45, ha='right')

    ax = plt.gca() 
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f%%')

    plt.tight_layout()
    plt.show()


# Create copies to avoid modifying original dataframes
train_clean = train_df.copy()
test_clean = test_df.copy()
print("Created copies: train_clean, test_clean")

# a) Cap Popularity Percentages at 100
print("\nCapping popularity percentages at 100...")
train_clean['Host_Popularity_percentage'] = train_clean['Host_Popularity_percentage'].clip(upper=100)
test_clean['Host_Popularity_percentage'] = test_clean['Host_Popularity_percentage'].clip(upper=100)
train_clean['Guest_Popularity_percentage'] = train_clean['Guest_Popularity_percentage'].clip(upper=100)
test_clean['Guest_Popularity_percentage'] = test_clean['Guest_Popularity_percentage'].clip(upper=100)
print("Popularity capped.")

# b) Handle Number_of_Ads (Impute missing, Round, Cap Outliers)
ads_median_train = train_clean['Number_of_Ads'].median()
print(f"\nImputing missing 'Number_of_Ads' in train set with median: {ads_median_train}")
train_clean['Number_of_Ads'].fillna(ads_median_train, inplace=True)
if test_clean['Number_of_Ads'].isnull().any():
     print("Warning: Unexpected missing values found in test set 'Number_of_Ads'. Imputing with train median.")
     test_clean['Number_of_Ads'].fillna(ads_median_train, inplace=True)

print("Rounding 'Number_of_Ads'...")
train_clean['Number_of_Ads'] = train_clean['Number_of_Ads'].round().astype(int)
test_clean['Number_of_Ads'] = test_clean['Number_of_Ads'].round().astype(int)

q_cap = 0.999
ads_cap_value = train_clean['Number_of_Ads'].quantile(q_cap)
ads_cap_value = max(ads_cap_value, ads_median_train)
print(f"Capping 'Number_of_Ads' at the {q_cap*100:.1f}th percentile (or median if higher): {int(ads_cap_value)}")
train_clean['Number_of_Ads'] = train_clean['Number_of_Ads'].clip(upper=int(ads_cap_value))
test_clean['Number_of_Ads'] = test_clean['Number_of_Ads'].clip(upper=int(ads_cap_value))
print("'Number_of_Ads' processed.")

# c) Handle Episode_Length_minutes = 0 (Treat as missing for imputation)
print("\nReplacing Episode_Length_minutes = 0 with NaN for imputation...")
train_clean['Episode_Length_minutes'] = train_clean['Episode_Length_minutes'].replace(0, np.nan)
test_clean['Episode_Length_minutes'] = test_clean['Episode_Length_minutes'].replace(0, np.nan)
print("Zero episode lengths marked as NaN.")

print("\nAnomaly Handling Complete.")


# # --- ADD INDICATOR COLUMNS (NEW STEP) ---
# print("Adding missing value indicator columns...")

# features_with_nan = ['Episode_Length_minutes', 'Guest_Popularity_percentage']

# for feature in features_with_nan:
#     if feature in train_clean.columns:
#         indicator_col_train = f'{feature}_Missing'
#         train_clean[indicator_col_train] = train_clean[feature].isnull().astype(int)
#         print(f"Created '{indicator_col_train}' in train_clean.")

#     if feature in test_clean.columns:
#         indicator_col_test = f'{feature}_Missing'
#         test_clean[indicator_col_test] = test_clean[feature].isnull().astype(int)
#         print(f"Created '{indicator_col_test}' in test_clean.")
# print("Indicator columns added.")


print("Starting missing value imputation...")

# a) Impute Episode_Length_minutes (using median)
length_median_train = train_clean['Episode_Length_minutes'].median()
print(f"Imputing missing 'Episode_Length_minutes' with median: {length_median_train:.2f}")
train_clean['Episode_Length_minutes'].fillna(length_median_train, inplace=True)
test_clean['Episode_Length_minutes'].fillna(length_median_train, inplace=True)
print("'Episode_Length_minutes' imputed.")

# b) Impute Guest_Popularity_percentage (using median)
guest_pop_median_train = train_clean['Guest_Popularity_percentage'].median()
print(f"Imputing missing 'Guest_Popularity_percentage' with median: {guest_pop_median_train:.2f}")
train_clean['Guest_Popularity_percentage'].fillna(guest_pop_median_train, inplace=True)
test_clean['Guest_Popularity_percentage'].fillna(guest_pop_median_train, inplace=True)
print("'Guest_Popularity_percentage' imputed.")

print("\nImputation Complete.")


print("Verifying cleaning steps...")

print("\nMissing values in cleaned Train data:")
missing_train_final = train_clean.isnull().sum()
if missing_train_final.sum() == 0: print("No missing values remain in train_clean.")

print("\nMissing values in cleaned Test data:")
missing_test_final = test_clean.isnull().sum()
if missing_test_final.sum() == 0: print("No missing values remain in test_clean.")


print("\nChecking dtypes after initial cleaning:")
train_clean.info()


print("\nChecking descriptive statistics after cleaning (numerical features):")
numerical_cols_clean = train_clean.select_dtypes(include=np.number).columns
display(train_clean[numerical_cols_clean].describe().round(2))

print("\nPreprocessing - Initial Cleaning Verification Complete.")


print("Starting categorical feature encoding...")

categorical_to_encode = ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']

high_cardinality_to_drop = ['Podcast_Name', 'Episode_Title']

# --- Apply One-Hot Encoding using pandas get_dummies ---

print(f"Applying One-Hot Encoding to: {categorical_to_encode}")
train_encoded = pd.get_dummies(train_clean,
                               columns=categorical_to_encode,
                               drop_first=False,
                               dtype=int)        

test_encoded = pd.get_dummies(test_clean,
                              columns=categorical_to_encode,
                              drop_first=False,
                              dtype=int)

# --- Align columns after get_dummies ---

train_labels = train_encoded['Listening_Time_minutes']
train_ids = train_encoded.index

train_cols = train_encoded.drop(columns=['Listening_Time_minutes']).columns
test_ids = test_encoded.index

test_encoded = test_encoded.reindex(columns=train_cols, fill_value=0)

train_encoded['Listening_Time_minutes'] = train_labels

print(f"\nShape of training data after encoding: {train_encoded.shape}")
print(f"Shape of test data after encoding: {test_encoded.shape}")

# --- Drop High Cardinality and Original ID Column ---

print(f"\nDropping high cardinality columns: {high_cardinality_to_drop}")
cols_to_drop_train = [col for col in high_cardinality_to_drop if col in train_encoded.columns]
train_final = train_encoded.drop(columns=cols_to_drop_train)

cols_to_drop_test = [col for col in high_cardinality_to_drop if col in test_encoded.columns]
test_final = test_encoded.drop(columns=cols_to_drop_test)


print(f"\nShape of final training data (before scaling): {train_final.shape}")
print(f"Shape of final test data (before scaling): {test_final.shape}")

print("\nChecking dtypes after encoding:")
train_final.info()

print("\nCategorical Encoding and Column Dropping Complete.")
train_final.head()


TARGET = 'Listening_Time_minutes'

if TARGET in train_final.columns:
    X = train_final.drop(columns=[TARGET])
    y = train_final[TARGET]
else:
    raise ValueError(f"Target column '{TARGET}' not found in train_final DataFrame.")

X_competition_test = test_final.copy()

print(f"Shape of training features (X): {X.shape}")
print(f"Shape of training target (y): {y.shape}")
print(f"Shape of competition test features (X_competition_test): {X_competition_test.shape}")


# --- Create Training and Validation Sets ---

VAL_SIZE = 0.20
RANDOM_STATE = 42

X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                  test_size=VAL_SIZE,
                                                  random_state=RANDOM_STATE)

print(f"\nShape of training subset (X_train): {X_train.shape}")
print(f"Shape of validation subset (X_val): {X_val.shape}")
print(f"Shape of training target subset (y_train): {y_train.shape}")
print(f"Shape of validation target subset (y_val): {y_val.shape}")


# --- Scale Numerical Features ---

print("\nScaling features")
# scaler = QuantileTransformer(output_distribution='uniform',
#                              n_quantiles=min(len(X_train), 1000),
#                              random_state=RANDOM_STATE,
#                              subsample=10**9)

# --- QuantileTransformer (Normal Output - GaussRank trick) ---
# print("\nScaling features using QuantileTransformer (Normal output - GaussRank)...")
# scaler = QuantileTransformer(output_distribution='normal',
#                              n_quantiles=min(len(X_train), 1000),
#                              random_state=RANDOM_STATE,
#                              subsample=10**9)

# --- RobustScaler (Uses Median/IQR - Robust to outliers) ---
# print("\nScaling features using RobustScaler...")
# scaler = RobustScaler()

# --- StandardScaler (Original - Keep for reference) ---
print("\nScaling features using StandardScaler...")
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_competition_test_scaled = scaler.transform(X_competition_test)

print(f"Type after scaling: {type(X_train_scaled)}")
print(f"Shape of scaled training features: {X_train_scaled.shape}")
print(f"Shape of scaled validation features: {X_val_scaled.shape}")
print(f"Shape of scaled competition test features: {X_competition_test_scaled.shape}")

print("\nData Splitting and Scaling Complete.")


input_dim = X_train_scaled.shape[1]
print(f"Number of input features for MLP: {input_dim}")

mlp_model = Sequential(name='Simple_MLP')
mlp_model.add(Input(shape=(input_dim,), name='Input_Layer'))
mlp_model.add(Dense(128, activation='relu', name='Hidden_Layer_1'))
mlp_model.add(BatchNormalization())
mlp_model.add(Dropout(0.2))
mlp_model.add(Dense(64, activation='relu', name='Hidden_Layer_2'))
mlp_model.add(BatchNormalization())
mlp_model.add(Dropout(0.2))
mlp_model.add(Dense(1, name='Output_Layer'))

print("\nMLP Model Summary:")
mlp_model.summary()

# --- Compile the Model ---

optimizer = Adam(learning_rate=0.001)

mlp_model.compile(optimizer=optimizer,
                  loss='mean_squared_error',
                  metrics=[RootMeanSquaredError(name='rmse'), 'mae'])

# --- Define Callbacks ---
early_stopping = EarlyStopping(monitor='val_loss',
                              patience=10,
                              restore_best_weights=True,
                              verbose=1)

# --- Train the Model ---

print("\nStarting model training...")

EPOCHS = 100
BATCH_SIZE = 128

history = mlp_model.fit(X_train_scaled, y_train,
                       epochs=EPOCHS,
                       batch_size=BATCH_SIZE,
                       validation_data=(X_val_scaled, y_val),
                       callbacks=[early_stopping],
                       verbose=1)

print("\nModel Training Complete.")

# --- Plot training history ---

def plot_history(history, metric_name='rmse', loss_name='loss'):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.xlabel('Epoch')
    plt.ylabel(f'{loss_name.replace("_"," ").title()}')
    plt.plot(hist['epoch'], hist[loss_name], label='Train Loss')
    plt.plot(hist['epoch'], hist[f'val_{loss_name}'], label='Val Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.xlabel('Epoch')
    plt.ylabel(f'{metric_name.upper()}')
    plt.plot(hist['epoch'], hist[metric_name], label=f'Train {metric_name.upper()}')
    plt.plot(hist['epoch'], hist[f'val_{metric_name}'], label=f'Val {metric_name.upper()}')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

print("\nPlotting training history (Loss and RMSE)...")
plot_history(history, metric_name='rmse', loss_name='loss')
plot_history(history, metric_name='mae', loss_name='loss')


import time

# --- Train and Evaluate Baseline XGBoost Model ---

print("Initializing XGBoost Regressor...")

xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror', 
    n_estimators=5000,            
    learning_rate=0.05,           
    max_depth=7,                  
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    tree_method='gpu_hist'
)

print("Starting XGBoost model training...")
start_time = time.time()

xgb_model.fit(X_train_scaled, y_train,
              eval_set=[(X_val_scaled, y_val)],
              eval_metric='rmse',
              early_stopping_rounds=50,
              verbose=100)


end_time = time.time()
print(f"\nXGBoost Training Time: {end_time - start_time:.2f} seconds")


# --- Evaluate the XGBoost model on the validation set ---
print("\nEvaluating XGBoost model on validation set...")
y_pred_xgb_val = xgb_model.predict(X_val_scaled)

rmse_xgb_val = mean_squared_error(y_val, y_pred_xgb_val, squared=False)

print(f"\nXGBoost Validation RMSE: {rmse_xgb_val:.4f}")

# Compare with MLP's best validation RMSE
try:
    best_val_rmse_mlp = min(history.history['val_rmse'])
    print(f"MLP Best Validation RMSE: {best_val_rmse_mlp:.4f}")
    improvement = best_val_rmse_mlp - rmse_xgb_val
    print(f"Difference (MLP - XGB): {improvement:.4f}")
    if improvement > 0:
        print("XGBoost performed better on the validation set.")
    else:
        print("MLP performed better or similarly on the validation set.")
except NameError:
    print("MLP history not found, cannot compare directly in this cell.")
    print("(Check previous cell's output for MLP validation RMSE)")


# --- Hyperparameter Tuning with RandomizedSearchCV ---

print("Setting up RandomizedSearchCV for XGBoost...")

param_dist = {
    'n_estimators': randint(100, 1000),
    'learning_rate': uniform(0.01, 0.1),
    'max_depth': randint(3, 7),
    'subsample': uniform(0.7, 0.3),
    'colsample_bytree': uniform(0.7, 0.3),
}

xgb_base = xgb.XGBRegressor(
    objective='reg:squarederror',
    random_state=RANDOM_STATE,
    n_jobs=-1,
    tree_method='gpu_hist',
    gamma=0.1,
    min_child_weight=3
)

# Set up RandomizedSearchCV
N_ITER = 50
CV_FOLDS = 3

print(f"Running Randomized Search with {N_ITER} iterations and {CV_FOLDS}-fold CV...")
start_time = time.time()

random_search = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=param_dist,
    n_iter=N_ITER,
    scoring='neg_root_mean_squared_error',
    cv=CV_FOLDS,
    verbose=2,
    random_state=RANDOM_STATE,
    n_jobs=1
)


print("Starting .fit() - This may take a while...")
random_search.fit(X_train_scaled, y_train)
print("Finished .fit()")

end_time = time.time()
print(f"\nRandomized Search finished. Time taken: {end_time - start_time:.2f} seconds")

# --- Show Results ---
print("\nBest parameters found by Randomized Search:")
print(random_search.best_params_)

best_cv_rmse = -random_search.best_score_
print(f"\nBest cross-validation RMSE: {best_cv_rmse:.4f}")

# --- Evaluate the Best Found Model on the Hold-Out Validation Set ---
print("\nEvaluating the best estimator found on the hold-out validation set...")
best_xgb_model = random_search.best_estimator_

y_pred_best_xgb_val = best_xgb_model.predict(X_val_scaled)
rmse_best_xgb_val = mean_squared_error(y_val, y_pred_best_xgb_val, squared=False)

print(f"\nRMSE of Best XGBoost model on Validation Set: {rmse_best_xgb_val:.4f}")

# Compare with the initial baseline XGBoost
print(f"Baseline XGBoost Validation RMSE: {rmse_xgb_val:.4f}")
improvement_tuned = rmse_xgb_val - rmse_best_xgb_val
print(f"Improvement from tuning: {improvement_tuned:.4f}")


# --- Ensemble Evaluation (MLP + Baseline XGBoost) ---
print("\n--- Evaluating Ensemble Performance on Validation Set ---")

print("MLP predicting on validation set...")
if 'mlp_model' in locals() and hasattr(mlp_model, 'predict'):
    mlp_pred_val = mlp_model.predict(X_val_scaled).flatten()
else:
    print("Error: mlp_model not found or not trained. Cannot generate MLP predictions.")
    mlp_pred_val = None

print("Baseline XGBoost predicting on validation set...")
if 'xgb_model' in locals() and hasattr(xgb_model, 'predict'):
    xgb_pred_val = xgb_model.predict(X_val_scaled)
else:
    print("Error: Baseline xgb_model not found or not trained. Cannot generate XGBoost predictions.")
    xgb_pred_val = None

if mlp_pred_val is not None and xgb_pred_val is not None:
    ensemble_pred_val = (mlp_pred_val + xgb_pred_val) / 2

    ensemble_rmse_val = mean_squared_error(y_val, ensemble_pred_val, squared=False)

    print("\n--- Validation Set Performance Comparison ---")
    best_val_rmse_mlp = np.inf
    best_val_rmse_xgb = np.inf

    try:
        if 'history' in locals() and 'val_rmse' in history.history:
             best_val_rmse_mlp = min(history.history['val_rmse'])
             print(f"MLP Best Validation RMSE: {best_val_rmse_mlp:.4f}")
        else:
             print("MLP history not available.")
    except Exception as e:
        print(f"Could not retrieve MLP score: {e}")

    try:
        if hasattr(xgb_model, 'best_score'):
            best_val_rmse_xgb = xgb_model.best_score
            print(f"Baseline XGBoost Best Validation RMSE: {best_val_rmse_xgb:.4f}")
        elif 'rmse_xgb_val' in locals():
             best_val_rmse_xgb = rmse_xgb_val
             print(f"Baseline XGBoost Validation RMSE (calculated): {best_val_rmse_xgb:.4f}")
        else:
             print("Baseline XGBoost score not available.")
    except Exception as e:
        print(f"Could not retrieve XGBoost score: {e}")

    print(f"Ensemble (Average) Validation RMSE: {ensemble_rmse_val:.4f}")

    best_score = min(best_val_rmse_mlp, best_val_rmse_xgb, ensemble_rmse_val)
    print(f"\nOverall Best Validation RMSE: {best_score:.4f}")

    if best_score == ensemble_rmse_val:
        print(">> Ensemble (Average) performed best on validation set.")
        best_model_name = 'ensemble'
    elif best_score == best_val_rmse_xgb:
        print(">> Baseline XGBoost performed best on validation set.")
        best_model_name = 'xgb'
    else:
        print(">> MLP performed best on validation set.")
        best_model_name = 'mlp'
else:
    print("\nCould not perform ensemble evaluation due to missing model predictions.")
    best_model_name = None


print("Generating predictions on the competition test set using the BASELINE XGBoost model...")

# Predict on the scaled competition test data
try:
    predictions_test = xgb_model.predict(X_competition_test_scaled)
    print("Predictions generated using 'xgb_model'.")
except NameError:
    print("Error: 'xgb_model' not found.")
    predictions_test = None

if predictions_test is not None:
    # --- Check prediction range ---
    print("\nBasic statistics of predictions:")
    print(f"Min prediction: {predictions_test.min():.4f}")
    print(f"Max prediction: {predictions_test.max():.4f}")
    print(f"Mean prediction: {predictions_test.mean():.4f}")

    if (predictions_test < 0).any():
        print("\nWarning: Negative predictions found! Clipping to 0.")
        predictions_test = np.clip(predictions_test, a_min=0, a_max=None)
    else:
        print("\nNo negative predictions found.")

    print("\nPredictions generated successfully.")
else:
    print("\nPrediction generation skipped due to missing model.")


print("Creating submission file...")

# Check if predictions exist
if predictions_test is not None:

    # Create a pandas DataFrame for the submission
    submission_final = pd.DataFrame({
        'id': test_ids,
        TARGET: predictions_test
    })

    submission_filename = 'submission.csv'

    submission_final.to_csv(submission_filename, index=False)

    print(f"\nSubmission file '{submission_filename}' created successfully!")
    display(submission_final.head())

else:
    print("\nSubmission file creation skipped because predictions were not generated.")

