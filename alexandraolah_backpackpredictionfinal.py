import numpy as np 
import pandas as pd 

train_file_path = "/kaggle/input/playground-series-s5e2/train.csv"
test_file_path = "/kaggle/input/playground-series-s5e2/test.csv"

train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

train_data.describe()


from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Make a copy of the training data
data = train_data.copy()

# Columns where we want to fill missing values
cols_to_fill = ["Brand", "Material", "Size", "Compartments", "Laptop Compartment", "Waterproof"]

# Temporarily fill missing values with -999 so LabelEncoder can work
data.fillna(-999, inplace=True)

# Encode categorical (object) columns to numbers
encoders = {}
for col in data.select_dtypes(include='object').columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    encoders[col] = le

# Fill missing values using Random Forest
for col in cols_to_fill:
    if train_data[col].isnull().sum() == 0:
        continue  # Skip if no missing values in this column

    print(f"Filling missing values in column: {col}")
    
    X = data.drop(columns=[col])
    y = train_data[col]

    X_train = X[y.notnull()]
    y_train = y[y.notnull()]
    X_pred = X[y.isnull()]

    # Choose model based on data type
    if y.dtype == 'object':
        model = RandomForestClassifier(n_estimators=100, random_state=0)
        y_le = LabelEncoder()
        y_train_encoded = y_le.fit_transform(y_train.astype(str))
        model.fit(X_train, y_train_encoded)
        y_pred = model.predict(X_pred)
        train_data.loc[y.isnull(), col] = y_le.inverse_transform(y_pred)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_pred)
        train_data.loc[y.isnull(), col] = y_pred

print("Missing values in train_data filled!.")



y = train_data["Price"]

features = ["Brand","Material","Size","Compartments","Laptop Compartment","Waterproof"]

test_data_cleaned = test_data.dropna(axis=0)

X = train_data[features]
X_test = test_data[features]

X_encoded = pd.get_dummies(X)
X_test_encoded = pd.get_dummies(X_test)

from sklearn.tree import DecisionTreeRegressor

# Train the model
backpack_model = DecisionTreeRegressor(random_state=1)
backpack_model.fit(X_encoded, y)



from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Split data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X_encoded, y, random_state=1)

# Train model on training set
model = DecisionTreeRegressor(random_state=1)
model.fit(X_train, y_train)

# Predict on validation set
val_preds = model.predict(X_valid)

# Evaluate the model
mae = mean_absolute_error(y_valid, val_preds)
print(f"MAE on validation set: {mae:.2f}")


from xgboost import XGBRegressor

xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=1)
xgb_model.fit(X_train, y_train)
preds = xgb_model.predict(X_valid)
mae = mean_absolute_error(y_valid, preds)
print(f"XGBoost MAE: {mae:.2f}")


test_preds = xgb_model.predict(X_test_encoded.loc[test_data_cleaned.index])

submission = pd.DataFrame({'id': test_data_cleaned['id'],
                           'Price': test_preds})

submission.to_csv('submission.csv', index=False)
print("Submission file created: submission.csv")

