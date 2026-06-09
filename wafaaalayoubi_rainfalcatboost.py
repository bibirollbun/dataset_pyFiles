import pandas as pd
import catboost
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as pl
from sklearn.model_selection import GridSearchCV
import optuna
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import numpy as np
from sklearn.metrics import accuracy_score


# Load datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv', index_col='id')

# Quick look at the data
train.head()


# Check dataset info
train.info()



# Check for duplicates in the training data
duplicates = train.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")
duplicates = test.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")


# Check for missing values in the dataset
missing_values = train.isnull().sum()
print(f"Missing values per column:\n{missing_values}")

# Handle missing values (impute or drop)
train = train.dropna()  # Dropping rows with missing values
# Or you can use imputation methods like filling with the median or mean
# train = train.fillna(train.median())



# Check for missing values in the dataset
missing_values = test.isnull().sum()
print(f"Missing values per column:\n{missing_values}")

# Handle missing values (impute or drop)
test = test.dropna()  # Dropping rows with missing values
# Or you can use imputation methods like filling with the median or mean
# train = train.fillna(train.median())


# Drop the 'day' column from the dataset
train = train.drop(columns=['day'])

# If you're working with the test data, don't forget to drop it there as well
test = test.drop(columns=['day'])


# Calculate the correlation matrix
correlation_matrix = train.corr()

# Plot the heatmap of the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()


# Define features and target
X = train.drop(columns=["rainfall"])
y = train["rainfall"]

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



# Step 3: Train CatBoost model
catboost_model = CatBoostClassifier(iterations=1000, depth=10, learning_rate=0.05, cat_features=[], verbose=200)
catboost_model.fit(X_train, y_train)


# Step 4: Train XGBoost model
xgb_model = XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=10, random_state=42)
xgb_model.fit(X_train, y_train)


# Step 5: Get predictions from both models on training and validation sets
catboost_train_pred = catboost_model.predict(X_train)
xgb_train_pred = xgb_model.predict(X_train)

catboost_val_pred = catboost_model.predict(X_val)
xgb_val_pred = xgb_model.predict(X_val)


# Step 6: Stack predictions from both models as features for the meta-model
X_stack_train = np.column_stack((catboost_train_pred, xgb_train_pred))
X_stack_val = np.column_stack((catboost_val_pred, xgb_val_pred))


# Step 7: Train the meta-model (Logistic Regression)
meta_model = LogisticRegression()
meta_model.fit(X_stack_train, y_train)


# Step 8: Evaluate the stacked model on the validation set
stacked_val_pred = meta_model.predict(X_stack_val)
accuracy = accuracy_score(y_val, stacked_val_pred)
print(f"Stacked Model Accuracy: {accuracy}")



# Get predictions from both models on the test set
catboost_test_pred = catboost_model.predict(test)
xgb_test_pred = xgb_model.predict(test)

# Stack the predictions for the meta-model
X_stack_test = np.column_stack((catboost_test_pred, xgb_test_pred))

# Get the final predictions using the meta-model
stacked_test_pred = meta_model.predict(X_stack_test)

# Step 10: Prepare the submission DataFrame
submission = pd.DataFrame({
    'id': test.index,  # Keep the test set ID for submission
    'rainfall': stacked_test_pred
})

# Save submission file
submission.to_csv("submission.csv", index=False)




