import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer





data = pd.read_csv("/kaggle/input/datasetsss/train_data_modified.csv")

data['Percentage of Federal Land'] = data['Percentage of Federal Land'].apply(lambda x: str(x).replace('%', '')).astype(float)


data = data.dropna()  # or use imputation methods like fillna()

# Convert categorical columns to numerical (if necessary)
data['state'] = data['state'].astype('category').cat.codes
data['year_month'] = data['year_month'].astype('category').cat.codes

# Step 2: Split into Features and Target
# Assuming 'total_fire_size' is the target variable
X = data.drop(columns=['total_fire_size', 'wildfire_size_lag_1', 'wildfire_size_lag_2'])  # Features
y = data['total_fire_size']  # Target

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Feature Scaling (Optional)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Train the Decision Tree
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train_scaled, y_train)

# Step 6: Make Predictions and Evaluate the Model
y_pred = model.predict(X_test_scaled)
print(y_pred)
# Evaluate with mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')



data['state'] = data['state'].astype('category').cat.codes
data['year_month'] = data['year_month'].astype('category').cat.codes

# Step 2: Split into Features and Target
# Assuming 'total_fire_size' is the target variable
X = data.drop(columns=['total_fire_size', 'wildfire_size_lag_1', 'wildfire_size_lag_2'])  # Features
y = data['total_fire_size']  # Target

# Step 3: Feature Scaling (Optional)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Train the Decision Tree on the entire dataset
model = DecisionTreeRegressor(random_state=42, max_depth=20, min_samples_split=3, min_samples_leaf=2)
model.fit(X_scaled, y)




predict_data = pd.read_csv("/kaggle/input/datasetsss/test_data_modified.csv").drop(columns=['total_fire_size', 'wildfire_size_lag_1', 'wildfire_size_lag_2', 'ID'])

predict_data['Percentage of Federal Land'] = predict_data['Percentage of Federal Land'].apply(lambda x: str(x).replace('%', '')).astype(float)
predict_data['state'] = predict_data['state'].astype('category').cat.codes
predict_data['year_month'] = predict_data['year_month'].astype('category').cat.codes
X_predict = scaler.transform(predict_data)

imputer = SimpleImputer(strategy="mean")  # or "median"
X_predict_imputed = imputer.fit_transform(X_predict)

#predictions = model.predict(X_predict)
predictions = model.predict(X_predict_imputed)
print(len(predictions))
print(predictions)



submit_data = pd.read_csv("/kaggle/input/forest-fire-prediction-epoch-hackathon/zero_submission.csv")

submit_data['total_fire_size'] = predictions
print(submit_data.head())

submit_data.to_csv("/kaggle/working/submission.csv")




