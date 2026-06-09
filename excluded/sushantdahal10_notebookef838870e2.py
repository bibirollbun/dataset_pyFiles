import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# Load the train and test data
train = pd.read_csv('/kaggle/input/ucs-654-kaggle-hack-lab-exam-2/train.csv')
test = pd.read_csv('/kaggle/input/ucs-654-kaggle-hack-lab-exam-2/test.csv')

# Define target variable and features
target = 'target'
X = train.drop(columns=[target])
y = train[target]

# Split the training data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
test_scaled = scaler.transform(test.drop(columns=['id']))

# Initialize the linear regression model
lr = LinearRegression()

# Hyperparameter tuning for linear regression
param_grid = {'fit_intercept': [True, False], 'positive': [True, False]}

# Grid search to find the best hyperparameters
grid_search = GridSearchCV(estimator=lr, param_grid=param_grid, scoring='r2', cv=5, verbose=1)
grid_search.fit(X_train_scaled, y_train)

# Get the best model and its parameters
best_lr = grid_search.best_estimator_

# Evaluate the model on the validation set
y_val_pred = best_lr.predict(X_val_scaled)
r2 = r2_score(y_val, y_val_pred)
print(f"R² Score on Validation Set: {r2}")

# If R² is perfect, finalize the model
if r2 == 1:
    print("Perfect R² achieved on validation set. Finalizing the model...")
else:
    print(f"R² is not 1 yet, current score: {r2}. Fine-tuning may be required.")

# Make predictions on the test data
test_predictions = best_lr.predict(test_scaled)

# Create a submission file
submission = pd.DataFrame({
    'id': test['id'],
    'target': test_predictions
})
submission.to_csv('submission.csv', index=False)

print("Submission File Created:")
print(submission.head())




