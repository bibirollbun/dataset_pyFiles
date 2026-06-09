# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Kaggle Input Path
TRAIN_PATH = "/kaggle/input/playground-series-s5e4/train.csv"
TEST_PATH = "/kaggle/input/playground-series-s5e4/test.csv"
SAMPLE_SUB_PATH = "/kaggle/input/playground-series-s5e4/sample_submission.csv"

# Load the Datasets
lt_train = pd.read_csv(TRAIN_PATH)
lt_test = pd.read_csv(TEST_PATH)
lt_submission = pd.read_csv(SAMPLE_SUB_PATH)


# 2.1: Shape & Columns
print("Train shape:", lt_train.shape)
print("Test shape:", lt_test.shape)

print("\nTrain columns:")
print(lt_train.columns.tolist())

print("\nTest columns:")
print(lt_test.columns.tolist())


lt_train.head()


lt_train.info()


missing_train = lt_train.isnull().sum()
missing_test = lt_test.isnull().sum()

print("\nğŸ§¼ Missing values in train set:")
print(missing_train[missing_train > 0])

print("\nğŸ§¼ Missing values in test set:")
print(missing_test[missing_test > 0])


# Plotting target variable distribution
plt.figure(figsize=(10, 5))
sns.histplot(lt_train['Listening_Time_minutes'], bins=50, kde=True, color='mediumseagreen')
plt.title('Distribution of Listening Time (in minutes)', fontsize=14)
plt.xlabel('Listening Time (minutes)')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()

# Summary statistics
lt_train['Listening_Time_minutes'].describe()



plt.figure(figsize=(8, 5))
sns.scatterplot(
    x='Episode_Length_minutes',
    y='Listening_Time_minutes',
    data=lt_train,
    alpha=0.3,
    edgecolor=None
)
plt.title('Listening Time vs Episode Length')
plt.xlabel('Episode Length (minutes)')
plt.ylabel('Listening Time (minutes)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Correlation check
lt_train[['Episode_Length_minutes', 'Listening_Time_minutes']].corr()



# Group by Genre and calculate mean listening time
genre_avg = lt_train.groupby('Genre')['Listening_Time_minutes'].mean().sort_values(ascending=False)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x=genre_avg.values, y=genre_avg.index, palette='coolwarm')
plt.title('Average Listening Time by Genre')
plt.xlabel('Average Listening Time (minutes)')
plt.ylabel('Genre')
plt.grid(True, axis='x')
plt.tight_layout()
plt.show()



plt.figure(figsize=(8, 5))
sns.scatterplot(
    x='Host_Popularity_percentage',
    y='Listening_Time_minutes',
    data=lt_train,
    alpha=0.3,
    edgecolor=None
)
plt.title('Listening Time vs Host Popularity')
plt.xlabel('Host Popularity (%)')
plt.ylabel('Listening Time (minutes)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Correlation
lt_train[['Host_Popularity_percentage', 'Listening_Time_minutes']].corr()



plt.figure(figsize=(8, 5))
sns.scatterplot(
    x='Guest_Popularity_percentage',
    y='Listening_Time_minutes',
    data=lt_train,
    alpha=0.3,
    edgecolor=None
)
plt.title('Listening Time vs Guest Popularity')
plt.xlabel('Guest Popularity (%)')
plt.ylabel('Listening Time (minutes)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Correlation
lt_train[['Guest_Popularity_percentage', 'Listening_Time_minutes']].corr()



import missingno as msno

# Visualise missing values
msno.matrix(lt_train.sample(10000), figsize=(10, 5))
plt.title("Missing Values in Sample of Train Set")
plt.show()



lt_train = lt_train.drop(columns=['Podcast_Name', 'Episode_Title'])
lt_test = lt_test.drop(columns=['Podcast_Name', 'Episode_Title'])


# Fill numeric columns with median (for stability)
lt_train['Episode_Length_minutes'].fillna(lt_train['Episode_Length_minutes'].median(), inplace=True)
lt_train['Guest_Popularity_percentage'].fillna(lt_train['Guest_Popularity_percentage'].median(), inplace=True)
lt_train['Number_of_Ads'].fillna(lt_train['Number_of_Ads'].median(), inplace=True)

lt_test['Episode_Length_minutes'].fillna(lt_test['Episode_Length_minutes'].median(), inplace=True)
lt_test['Guest_Popularity_percentage'].fillna(lt_test['Guest_Popularity_percentage'].median(), inplace=True)



# Combine train + test for consistent encoding
lt_train['is_train'] = 1
lt_test['is_train'] = 0
combined = pd.concat([lt_train, lt_test], axis=0)

# One-hot encode
combined = pd.get_dummies(combined, columns=['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment'], drop_first=True)

# Split back
lt_train = combined[combined['is_train'] == 1].drop(columns=['is_train'])
lt_test = combined[combined['is_train'] == 0].drop(columns=['is_train', 'Listening_Time_minutes'])



# Exclude ID column and target for heatmap
heatmap_data = lt_train.drop(columns=['id', 'Listening_Time_minutes'])

# Append target temporarily to check correlation
heatmap_data['Listening_Time_minutes'] = lt_train['Listening_Time_minutes']

# Compute correlation matrix
corr_matrix = heatmap_data.corr()

# Plot heatmap
plt.figure(figsize=(16, 12))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, linewidths=0.5)
plt.title("Full Feature Correlation with Listening Time", fontsize=14)
plt.tight_layout()
plt.show()



# Correlation with target only
target_corr = corr_matrix['Listening_Time_minutes'].sort_values(ascending=False)

# Plot top 20 positive + negative correlations
plt.figure(figsize=(10, 8))
target_corr.drop('Listening_Time_minutes').sort_values().tail(20).plot(kind='barh', color='teal')
plt.title('Top Correlated Features with Listening Time')
plt.xlabel('Correlation Coefficient')
plt.grid(True)
plt.tight_layout()
plt.show()



# Split features and target
X = lt_train.drop(columns=['id', 'Listening_Time_minutes'])
y = lt_train['Listening_Time_minutes']

# Optional: scale numeric features (good for neural networks)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)



import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Basic feedforward network
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)  # Output layer for regression
])

# Compile with RMSE-friendly loss
model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)



history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=512,
    verbose=1
)



import matplotlib.pyplot as plt

# Plot RMSE over epochs
plt.plot(history.history['root_mean_squared_error'], label='Train RMSE')
plt.plot(history.history['val_root_mean_squared_error'], label='Val RMSE')
plt.title('Training vs Validation RMSE')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



# Prepare feature matrix and target again
X_lgb = lt_train.drop(columns=['id', 'Listening_Time_minutes'])
y_lgb = lt_train['Listening_Time_minutes']

# Train-validation split
from sklearn.model_selection import train_test_split

X_train_lgb, X_val_lgb, y_train_lgb, y_val_lgb = train_test_split(
    X_lgb, y_lgb, test_size=0.2, random_state=42
)



from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

# 1. Initialise the model
lgb_model = LGBMRegressor(
    objective='regression',
    metric='rmse',
    learning_rate=0.05,
    n_estimators=300,        # You can change this based on how well it does
    num_leaves=31,
    random_state=42,
    verbose=100
)

# 2. Fit the model
lgb_model.fit(
    X_train_lgb, y_train_lgb,
    eval_set=[(X_val_lgb, y_val_lgb)],
    eval_metric='rmse'
)

# 3. Evaluate RMSE
y_pred_val = lgb_model.predict(X_val_lgb)
val_rmse = mean_squared_error(y_val_lgb, y_pred_val, squared=False)
print(f"ğŸ“Š Validation RMSE: {val_rmse:.5f}")



from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Try slightly deeper trees and more trees
xgb_model_tuned = XGBRegressor(
    objective='reg:squarederror',
    learning_rate=0.05,
    n_estimators=500,   # increased from 300
    max_depth=7,        # increased from 6 to 7
    random_state=42,
    verbosity=0
)

# Fit model
xgb_model_tuned.fit(
    X_train_lgb, y_train_lgb,
    eval_set=[(X_val_lgb, y_val_lgb)],
    early_stopping_rounds=30,
    verbose=False
)

# Predict and evaluate
y_pred_xgb_tuned = xgb_model_tuned.predict(X_val_lgb)
rmse_xgb_tuned = mean_squared_error(y_val_lgb, y_pred_xgb_tuned, squared=False)
print(f"ğŸ“ˆ Tuned XGBoost Validation RMSE: {rmse_xgb_tuned:.5f}")



# 1. Retrain on full training data
final_xgb_model = XGBRegressor(
    objective='reg:squarederror',
    learning_rate=0.05,
    n_estimators=500,
    max_depth=7,
    random_state=42,
    verbosity=0
)

final_xgb_model.fit(X_lgb, y_lgb)

# 2. Predict on test set
final_preds = final_xgb_model.predict(lt_test.drop(columns=['id']))

# 3. Prepare submission
submission_final = lt_submission.copy()
submission_final['Listening_Time_minutes'] = final_preds

# 4. Save submission
submission_final.to_csv("submission_xgb_final.csv", index=False)

print("âœ… Submission file created: submission_xgb_final.csv")


