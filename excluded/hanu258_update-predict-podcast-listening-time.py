# Core Libraries
import numpy as np
import pandas as pd
import warnings

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Modeling
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import xgboost as xgb
import category_encoders as ce

# Suppress Warnings
warnings.filterwarnings("ignore")


train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


train


test


# Add a 'dataset' column to track source
train['dataset'] = 'train'
test['dataset'] = 'test'

# Combine train and test datasets for unified preprocessing
df = pd.concat([train, test], axis=0).reset_index(drop=True)


df


df.shape


df.describe()


df.info()


df.columns


#Identifying Missing Values
df.isnull().sum()


# Identifying Duplicates
df.duplicated().sum()


#Data Cleaning
#Imputation
# Fill numeric column with median or mean
df['Episode_Length_minutes'] = df['Episode_Length_minutes'].fillna(train['Episode_Length_minutes'].mean())
df['Guest_Popularity_percentage'] = df['Guest_Popularity_percentage'].fillna(train['Guest_Popularity_percentage'].mean())
df['Number_of_Ads'] = df['Number_of_Ads'].fillna(train['Number_of_Ads'].mean())

df['Episode_Title'] = df['Episode_Title'].str.replace("Episode ", "", regex=False).astype(int)


# Handle outliers
df['Episode_Length_minutes'] = df['Episode_Length_minutes'].clip(upper=325)

df['Number_of_Ads'] = df['Number_of_Ads'].clip(upper=10)

# df['Episode_Length_log'] = np.log1p(df['Episode_Length_minutes']+17)  # log1p handles zeroes


import matplotlib.pyplot as plt
import seaborn as sns

# Plot histogram for numerical columns
numerical_cols = ["Episode_Length_minutes", "Host_Popularity_percentage", 
                  "Guest_Popularity_percentage", "Number_of_Ads", "Listening_Time_minutes"]

plt.figure(figsize=(12, 8))
for i, col in enumerate(numerical_cols):
    plt.subplot(3, 2, i+1)
    sns.histplot(df[col], bins=50, kde=True)  # kde=True adds Kernel Density Estimation
    plt.title(f"Distribution of {col}")
plt.tight_layout()
plt.show()


# Select categorical columns
categorical_cols = df.select_dtypes(include=["object", "category"]).columns

# Set figure size
plt.figure(figsize=(12, len(categorical_cols) * 4))

# Loop through each categorical column
for i, col in enumerate(categorical_cols, 1):
    plt.subplot(len(categorical_cols), 1, i)
    
    # Countplot sorted by frequency
    ax = sns.countplot(x=df[col], order=df[col].value_counts().index, palette="viridis")
    
    plt.title(f"Count of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    
    # Rotate labels for better readability
    plt.xticks(rotation=45)

# Adjust layout
plt.tight_layout()
plt.show()


df.head()


cat_columns = df.select_dtypes(include=['object', 'category']).columns
cat_columns


df.head()


# Separate train and test datasets
train_df = df[df['dataset'] == 'train'].drop(columns=['dataset', 'Podcast_Name'], errors='ignore')
test_df  = df[df['dataset'] == 'test'].drop(columns =['dataset', 'Podcast_Name'], errors='ignore')


# Drop unnecessary columns from both datasets
train_df = train_df.drop(columns=['id'], errors='ignore')
test_df = test_df.drop(columns=['Listening_Time_minutes'], errors='ignore')


# Separate features and target
X = train_df.drop(['Listening_Time_minutes'], axis=1)
y = train_df['Listening_Time_minutes']


X


# Set random state and number of CV splits for reproducibility
RANDOM_STATE = 42
N_SPLITS = 5

# Define categorical columns for target encoding
categorical_columns = ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']

# Define XGBoost model parameters
xgb_params = {
    'n_estimators': 400,
    'max_depth': 14,
    'learning_rate': 0.0345,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'tree_method': 'hist',
    'n_jobs': -1
}


# Initialize K-Fold Cross-Validation
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
rmse_scores = []

# Cross-validation loop with target encoding
for fold, (train_index, val_index) in enumerate(kf.split(X)):
    print(f"Fold {fold + 1}")

    # Split data
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # Apply target encoding on categorical columns
    for col in categorical_columns:
        encoder = ce.TargetEncoder(cols=[col], smoothing=5)
        X_train[f'{col}_encoded'] = encoder.fit_transform(X_train[col], y_train)
        X_val[f'{col}_encoded'] = encoder.transform(X_val[col])


    # Drop original categorical columns
    X_train = X_train.drop(categorical_columns, axis=1)
    X_val = X_val.drop(categorical_columns, axis=1)


    # Train XGBoost model with early stopping
    model = XGBRegressor(**xgb_params)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              early_stopping_rounds=10,
              verbose=False)

    # Make predictions and calculate RMSE
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    rmse_scores.append(rmse)

    print(f"RMSE: {rmse:.4f}")

# Print average RMSE across all folds
print(f"\nAverage RMSE across {N_SPLITS} folds: {np.mean(rmse_scores):.4f} Â± {np.std(rmse_scores):.4f}")


# Plotting top 10 features by gain
fig, ax = plt.subplots(figsize=(10, 6))  # Bigger figure for clarity

xgb.plot_importance(
    model,
    ax=ax,
    max_num_features=10,
    importance_type='gain',
    title='Top 10 Feature Importances',
    color='skyblue',  # Soft color
    height=0.5        # Thinner bars for cleaner look
)

# Styling
ax.set_title('ğŸ”� Top 10 Feature Importances (by Gain)', fontsize=16, fontweight='bold')
ax.set_xlabel('Gain', fontsize=12)
ax.set_ylabel('Features', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, axis='x', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()


categorical_columns = ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']

# Define model parameters (same as before)
xgb_params = {
    'n_estimators': 400,
    'max_depth': 14,
    'learning_rate': 0.0345,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'tree_method': 'hist',
    'n_jobs': -1
}

# 1. Retrain the Encoder
encoders = {}  # Store encoders for each column
for col in categorical_columns:
    encoder = ce.TargetEncoder(cols=[col], smoothing=5)
    X[f'{col}_encoded'] = encoder.fit_transform(X[col], y)
    encoders[col] = encoder
X_encoded = X.drop(categorical_columns, axis=1)

# 2. Retrain the XGBoost Model
model = XGBRegressor(**xgb_params)
model.fit(X_encoded, y)

# Prepare New Row Data

test_features = test_df.drop(columns=['id'], errors='ignore')  # Drop unnecessary columns

# 3. Transform the New Row Data
for col in categorical_columns:
    test_features[f'{col}_encoded'] = encoders[col].transform(test_features[col])
test_features = test_features.drop(categorical_columns, axis=1)

# 4. Make Predictions
test_df['Listening_Time_minutes'] = model.predict(test_features)


# Preprocess test data
# test_features = test_df.drop(columns=['id'], errors='ignore')  # Drop unnecessary columns



# test_df['Listening_Time_minutes'] = model.predict(test_features)


# Create submission file
submission = test_df[['id', 'Listening_Time_minutes']]  # Include 'id' and the predicted target column
submission.to_csv('submission.csv', index=False)

print("Submission file created: submission.csv")


submission




