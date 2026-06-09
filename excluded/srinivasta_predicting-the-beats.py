import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

import warnings
# Suppress the specific FutureWarning related to 'use_inf_as_na'
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")




train_df = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")

display("Train DataFrame shape:", train_df.shape)
display("Test DataFrame shape:", test_df.shape)


display(train_df.head())
display(test_df.head())

train_df.info()
test_df.info()

display(train_df.describe())
display(test_df.describe())


display(train_df.nunique())
display(test_df.nunique())


plt.figure(figsize=(10, 6))
sns.histplot(train_df['BeatsPerMinute'], kde=True)
plt.title('Distribution of BeatsPerMinute in Train Data')
plt.xlabel('BeatsPerMinute')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 8))
correlation_matrix = train_df.drop('id', axis=1).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features in Train Data')
plt.show()


numerical_features = correlation_matrix.index.tolist()
numerical_features.remove('BeatsPerMinute')

plt.figure(figsize=(15, 10))
for i, feature in enumerate(numerical_features):
    plt.subplot(3, 3, i + 1)
    sns.scatterplot(x=train_df[feature], y=train_df['BeatsPerMinute'])
    plt.title(f'BeatsPerMinute vs {feature}')
    plt.xlabel(feature)
    plt.ylabel('BeatsPerMinute')
plt.tight_layout()
plt.show()



# Identify numerical features (all features except 'id')
numerical_features = train_df.columns.tolist()
numerical_features.remove('id')
if 'BeatsPerMinute' in numerical_features:
    numerical_features.remove('BeatsPerMinute')

# Check for missing values (as per previous exploration, none expected)
display("Missing values in train_df:\n", train_df.isnull().sum())
display("Missing values in test_df:\n", test_df.isnull().sum())

# Initialize StandardScaler
scaler = StandardScaler()

# Fit scaler on training data numerical features and transform
X_train = train_df[numerical_features]
y_train = train_df['BeatsPerMinute']
X_test = test_df[numerical_features]

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert scaled arrays back to DataFrames for easier handling in subsequent steps
X_train_scaled = pd.DataFrame(X_train_scaled, columns=numerical_features)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=numerical_features)

# Align columns - Although in this case they should already be aligned as we used the same numerical_features list,
# this is a good practice for robustness, especially if categorical features were involved.
train_cols = X_train_scaled.columns
test_cols = X_test_scaled.columns

missing_in_test = set(train_cols) - set(test_cols)
for c in missing_in_test:
    X_test_scaled[c] = 0

missing_in_train = set(test_cols) - set(train_cols)
for c in missing_in_train:
    X_train_scaled[c] = 0

X_test_scaled = X_test_scaled[train_cols] # Ensure the order is the same

display("\nShape of X_train_scaled:", X_train_scaled.shape)
display("Shape of y_train:", y_train.shape)
display("Shape of X_test_scaled:", X_test_scaled.shape)





# Instantiate the LightGBM regressor model
lgbm_model = lgb.LGBMRegressor(random_state=42)

# Fit the model to the scaled training data
lgbm_model.fit(X_train_scaled, y_train)




# Make predictions on the training data using the LightGBM model
y_train_pred_lgbm = lgbm_model.predict(X_train_scaled)

# Calculate the Root Mean Squared Error (RMSE) for the LightGBM model
rmse_lgbm = np.sqrt(mean_squared_error(y_train, y_train_pred_lgbm))

display(f"Root Mean Squared Error on the training data (LightGBM): {rmse_lgbm}")


# Use the trained LightGBM model to make predictions on the scaled test data
predictions_lgbm = lgbm_model.predict(X_test_scaled)


# Create the submission DataFrame with 'id' from the test data and the LightGBM predictions
submission_df_lgbm = pd.DataFrame({'id': test_df['id'], 'BeatsPerMinute': predictions_lgbm})

# Save the submission DataFrame to a CSV file
submission_df_lgbm.to_csv('submission.csv', index=False)

# Display the first few rows of the submission file to verify the format
display(submission_df_lgbm.head())

