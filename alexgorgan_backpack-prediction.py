import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import warnings

# Warnings ignorieren
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


print("Csv hinzufugen")
try:
    train_df = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
    test_df = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
except FileNotFoundError as e:
    print(f"Error{e}")
    exit()

try:
    extra_train_df = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")
    print(f"Extra train data:{extra_train_df.shape}")
    train_df = pd.concat([train_df, extra_train_df], ignore_index=True)
except FileNotFoundError:
    print("train_extra.csv nicht gefunden.")
    train_df = train_df_original

print(f"Training final shape: {train_df.shape}")
print(f"Test final shape: {test_df.shape}")

# --- Plot Price Distribution ---
plt.figure(figsize=(10, 6))
plt.hist(train_df['Price'], bins=50, edgecolor='black', alpha=0.7)
plt.title('Price Distribution in Training set')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show()


# Combining train and test for consistent preprocessing
combined_df = pd.concat([train_df.drop('Price', axis=1), test_df], ignore_index=True)

# Identify categorical and numerical features
categorical_features = combined_df.select_dtypes(include=['object']).columns
numerical_features = combined_df.select_dtypes(include=np.number).columns.drop('id') # Exclude id

print("\nMissing values before imputation:")
print(combined_df.isnull().sum())
for col in categorical_features:
    # Fill with mode or a placeholder like 'Missing'
    mode_val = combined_df[col].mode()[0] if not combined_df[col].mode().empty else 'Missing'
    combined_df[col].fillna(mode_val, inplace=True)
for col in numerical_features:
    # Fill with median (less sensitive to outliers than mean)
    median_val = combined_df[col].median()
    combined_df[col].fillna(median_val, inplace=True)

print("\nMissing values after imputation:")
print(combined_df.isnull().sum())


for col in ['Laptop Compartment', 'Waterproof']:
    if col in combined_df.columns:
        combined_df[col] = combined_df[col].map({'Yes': 1, 'No': 0}).fillna(0) # Fill potential NaNs introduced by mapping with 0
        # Ensure the column is numeric after mapping
        combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce').fillna(0).astype(int)
combined_df = pd.get_dummies(combined_df, columns=[col for col in categorical_features if col not in ['Laptop Compartment', 'Waterproof']], dummy_na=False)

# Separate train and test data again
train_processed = combined_df.iloc[:len(train_df)]
test_processed = combined_df.iloc[len(train_df):]

# Align columns - crucial if get_dummies created different columns due to unique values only in train or test
train_labels = train_df['Price']
train_ids = train_processed['id']
test_ids = test_processed['id']

# Drop ID columns before training
train_processed = train_processed.drop('id', axis=1)
test_processed = test_processed.drop('id', axis=1)

# Ensure both dataframes have the same columns after one-hot encoding
common_cols = list(set(train_processed.columns) & set(test_processed.columns))
train_processed = train_processed[common_cols]
test_processed = test_processed[common_cols]

print("\nTrain processed shape:", train_processed.shape)
print("Test processed shape:", test_processed.shape)



model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10, min_samples_split=10)

print("\nTraining model...")
model.fit(train_processed, train_labels)

# --- Plot Feature Importances ---
importances = model.feature_importances_
feature_names = train_processed.columns
# Create a DataFrame for easier sorting and plotting
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
# Sort features by importance
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=True) # Ascending for horizontal bar plot

# Plotting top 20 features
top_n = 20
plt.figure(figsize=(10, 8)) # Adjusted figure size for better readability
plt.barh(feature_importance_df['feature'].tail(top_n), feature_importance_df['importance'].tail(top_n), color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Characteristik')
plt.title(f'The Importance (Top {top_n})')
plt.gca().margins(y=0.01) # Add some margin to y-axis
plt.tight_layout() # Adjust layout to prevent labels from overlapping
plt.show()

# --- Prediction --- 
print("Making predictions...")
predictions = model.predict(test_processed)

# --- Submission File --- 
submission_df = pd.DataFrame({'id': test_ids, 'Price': predictions})
submission_df.to_csv("submission.csv", index=False)

