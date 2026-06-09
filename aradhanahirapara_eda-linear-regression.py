import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


train_df = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
train_df


test_df= pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
test_df


print("Missing values in training data:\n", train_df.isnull().sum())
print("Missing values in test data:\n", test_df.isnull().sum())


# Separate target and features for training data
if 'id' in train_df.columns:
    X = train_df.drop(['id', 'Price'], axis=1)
else:
    X = train_df.drop('Price', axis=1)
y = train_df['Price']

# For test data, drop 'id' if present
if 'id' in test_df.columns:
    X_test_final = test_df.drop('id', axis=1)
else:
    X_test_final = test_df.copy()


# One-hot encode categorical features in both training and test sets
X = pd.get_dummies(X)
X_test_final = pd.get_dummies(X_test_final)


# Align the columns of the training and test sets (fill missing columns with 0)
X, X_test_final = X.align(X_test_final, join='left', axis=1, fill_value=0)


# Impute missing values with the median value of each column
X = X.fillna(X.median())
X_test_final = X_test_final.fillna(X_test_final.median())


# Optional: Visualize feature distributions (example using the first feature)
plt.figure(figsize=(8, 4))
sns.histplot(X.iloc[:, 0], bins=30, kde=True)
plt.title("Distribution of Feature 1")
plt.show()


# Split training data into train and validation sets for model evaluation.
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Function to compute RMSE
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))



# ----- Model 1: Linear Regression -----
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred_val = lr_model.predict(X_val)
lr_rmse = rmse(y_val, lr_pred_val)
print("Linear Regression RMSE on validation set:", lr_rmse)


rf_model = RandomForestRegressor(
    n_estimators=50,           # Reduce number of trees to limit CPU usage
    max_depth=10,              # Limit tree depth to prevent excessive computation
    min_samples_split=5,       # Prevent very deep trees
    min_samples_leaf=2,        # Ensure each leaf has at least 2 samples
    random_state=42,           # Reproducibility
    n_jobs=2,                  # Use only 2 CPU cores to prevent excessive usage
    warm_start=True            # Allow incremental training if needed
)

# Train the model
rf_model.fit(X_train, y_train)
# Make predictions on the validation set
rf_pred_val = rf_model.predict(X_val)
# Calculate RMSE
rf_rmse = rmse(y_val, rf_pred_val)
print("Optimized Random Forest RMSE on validation set:", rf_rmse)


# Compare the two models
if lr_rmse < rf_rmse:
    best_model = lr_model
    print("Selected Model: Linear Regression")
else:
    best_model = rf_model
    print("Selected Model: Random Forest Regressor")


# Retrain the best model on the entire training dataset
best_model.fit(X, y)


# Predict on the test dataset
test_predictions = best_model.predict(X_test_final)


# Ensure the 'id' column is included if available; otherwise, create an index-based id.
if 'id' in test_df.columns:
    submission = pd.DataFrame({
        'id': test_df['id'],
        'Price': test_predictions
    })
else:
    submission = pd.DataFrame({
        'id': np.arange(len(test_predictions)),
        'Price': test_predictions
    })


submission.to_csv('sample_submission.csv', index=False)
print("Submission file created: sample_submission.csv")

