import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_log_error


# Load dataset
df = pd.read_csv('/kaggle/input/gdsc-workshop-april-2025/train.csv')
# Use only numerical features (excluding target and Id)
numerical_features = df.select_dtypes(include=['int64', 'float64']).drop(columns=['id', 'Rings'])
y = df['Rings'].values.reshape(-1, 1)
# reshape used to convert the 1d to 2d

# Fill missing values
X = numerical_features.fillna(numerical_features.mean()).values

# Normalize features  - 1 to +1 normilzied the vlaue of big data 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split  # we used the not take the whole data only take the chunk of data
#  training mijority chunk- 50% testing data- small for traning model  then we optimized the data validation data final
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)



df.head()


y



y=df['Rings']



X_scaled





X


df.isna().sum()



import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error

# Ensure target is a column vector
y_train = y_train.reshape(-1, 1)
y_val = y_val.reshape(-1, 1)

# Add bias term (intercept) to X manually
X_train_manual = np.c_[np.ones((X_train.shape[0], 1)), X_train]
X_val_manual = np.c_[np.ones((X_val.shape[0], 1)), X_val]

# Initialize weights
theta = np.zeros((X_train_manual.shape[1], 1))
learning_rate = 0.01
epochs = 1000
m = X_train_manual.shape[0]

# Gradient Descent Loop
for epoch in range(epochs):
    preds = X_train_manual.dot(theta)
    error = preds - y_train
    gradient = (2 / m) * X_train_manual.T.dot(error)
    theta -= learning_rate * gradient

# Make predictions on validation set
manual_preds = X_val_manual.dot(theta)

# Optional: Clip negative predictions for RMSLE compatibility
manual_preds_clipped = np.maximum(0, manual_preds)

# Evaluation Metrics
manual_rmse = np.sqrt(mean_squared_error(y_val, manual_preds))
manual_mae = mean_absolute_error(y_val, manual_preds)
manual_r2 = r2_score(y_val, manual_preds)
manual_rmsle = np.sqrt(mean_squared_log_error(y_val, manual_preds_clipped))

# Display Results
print("ğŸ“� Manual Linear Regression:")
print(f"   â�¤ RMSE : {manual_rmse:.2f}")
print(f"   â�¤ MAE  : {manual_mae:.2f}")
print(f"   â�¤ RÂ²   : {manual_r2:.4f}")
print(f"   â�¤ RMSLE: {manual_rmsle:.4f}")




# Train Scikit-learn Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on validation set
predictions = model.predict(X_val)

# Clip predictions to avoid negative values (important for RMSLE)
predictions_clipped = np.maximum(0, predictions)

# Evaluation Metrics
rmse = np.sqrt(mean_squared_error(y_val, predictions))
mae = mean_absolute_error(y_val, predictions)
r2 = r2_score(y_val, predictions)
rmsle = np.sqrt(mean_squared_log_error(y_val, predictions_clipped))

# Display Results
print("\nâš™ï¸� Scikit-learn Linear Regression Performance:")
print(f"   â�¤ RMSE   : {rmse:.2f}")
print(f"   â�¤ MAE    : {mae:.2f}")
print(f"   â�¤ RÂ²     : {r2:.4f}")
print(f"   â�¤ RMSLE  : {rmsle:.4f}")

