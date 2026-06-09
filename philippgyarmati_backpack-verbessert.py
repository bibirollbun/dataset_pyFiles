import numpy as np 
import pandas as pd 

train_data = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")


from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

train_data_copy = train_data.copy()
train_data_copy.fillna(-999, inplace=True)

encoders = {}
cols_to_fill = ["Brand","Material","Size","Compartments","Laptop Compartment","Waterproof"]

# Encode string columns
for col in train_data_copy.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    train_data_copy[col] = le.fit_transform(train_data_copy[col].astype(str))
    encoders[col] = le

for target_col in cols_to_fill:
    if train_data[target_col].isnull().sum() > 0:
        
        feature_cols = [col for col in train_data.columns if col != target_col]

        X = train_data_copy[feature_cols].select_dtypes(include=['number'])

        X_train = X[train_data[target_col].notnull()]
        y_train = train_data[target_col][train_data[target_col].notnull()]
        X_pred = X[train_data[target_col].isnull()]

        if len(X_train) == 0 or len(X_pred) == 0:
            print(f"Skipping {target_col}, not enough data to train.")
            continue
        
        # Choose model
        if train_data[target_col].dtype == 'object':
            model = RandomForestClassifier(n_estimators=100, random_state=0)
            y_le = LabelEncoder()
            y_train = y_le.fit_transform(y_train)
            encoders[target_col] = y_le  # Save encoder for decoding predicted values
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=0)


        model.fit(X_train, y_train)

        predicted = model.predict(X_pred)

        # If it was encoded, decode prediction
        if target_col in encoders:
            predicted = encoders[target_col].inverse_transform(predicted)


        # Fill predicted values
        train_data.loc[train_data[target_col].isnull(), target_col] = predicted

print("Done filling missing values!")



y = train_data["Price"]

features = ["Brand","Material","Size","Compartments","Laptop Compartment","Waterproof"]  #Feature Selection
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])


from xgboost import XGBRegressor

model = XGBRegressor(n_estimators=300, max_depth=8, learning_rate=0.05, random_state=1)
model.fit(X,y)
predictions = model.predict(X_test)

output = pd.DataFrame({'id': test_data.id, 'Price': predictions})
output.to_csv('submission.csv', index=False)

print("Your submission was successfully saved!")

