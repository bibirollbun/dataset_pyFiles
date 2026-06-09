# Step 1: Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error

# Step 2: Load the data
train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")

# Step 3: Separate features and target
X = train.drop(['id', 'Price'], axis=1)
y = train['Price']
X_test = test.drop(['id'], axis=1)

# Step 4: Handle missing values
# Numeric columns: fill NaNs with median
X = X.fillna(X.median(numeric_only=True))
X_test = X_test.fillna(X_test.median(numeric_only=True))

# Categorical columns: fill NaNs with mode
for col in X.columns:
    if X[col].dtype == 'object':
        mode = X[col].mode()[0]
        X[col] = X[col].fillna(mode)
        X_test[col] = X_test[col].fillna(mode)

# Step 5: Encode categorical features
for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        X_test[col] = le.transform(X_test[col])

# Step 6: Optional validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Define base ensemble models
rf = RandomForestRegressor(n_estimators=100, random_state=42)
gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
et = ExtraTreesRegressor(n_estimators=100, random_state=42)

# Step 8: Create and train Voting Regressor
voting = VotingRegressor([('rf', rf), ('gb', gb), ('et', et)])
voting.fit(X_train, y_train)

# Step 9: Evaluate model on validation set
val_preds = voting.predict(X_val)
rmse = mean_squared_error(y_val, val_preds, squared=False)
print("Validation RMSE:", rmse)

# Step 10: Predict on test set
test_preds = voting.predict(X_test)

# Step 11: Prepare submission file
submission = pd.DataFrame({
    'id': test['id'],
    'Price': test_preds
})
submission.to_csv("submission.csv", index=False)
print("submission.csv created!")


