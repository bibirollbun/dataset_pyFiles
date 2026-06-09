import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


def read_data(train: str, test: str):
    train_data = pd.read_csv(train)
    test_data = pd.read_csv(test)
    return train_data, test_data


train_data, test_data = read_data('/kaggle/input/widsdatathon2022/train.csv', '/kaggle/input/widsdatathon2022/test.csv')


train_data = train_data.drop(columns=['id'])
test_ids = test_data['id']
test_data = test_data.drop(columns=['id'])


X = train_data.drop(columns=['site_eui'])
y = train_data['site_eui']


num_cols = X.select_dtypes(include=['float64', 'int64']).columns
num_imputer = SimpleImputer(strategy='median')
X[num_cols] = num_imputer.fit_transform(X[num_cols])
test_data[num_cols] = num_imputer.transform(test_data[num_cols])


cat_cols = X.select_dtypes(include=['object']).columns
cat_imputer = SimpleImputer(strategy='most_frequent')
X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])
test_data[cat_cols] = cat_imputer.transform(test_data[cat_cols])


label_encoders = {col: LabelEncoder() for col in cat_cols}
for col, le in label_encoders.items():
    X[col] = le.fit_transform(X[col])
    test_data[col] = le.transform(test_data[col])


scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])
test_data[num_cols] = scaler.transform(test_data[num_cols])


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


gbr_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
gbr_model.fit(X_train, y_train)


y_val_pred = gbr_model.predict(X_val)
rmse = mean_squared_error(y_val, y_val_pred) ** 0.5
print(f"Validation RMSE: {rmse}")


test_predictions = gbr_model.predict(test_data)


output = pd.DataFrame({'id': test_ids, 'site_eui': test_predictions})


output.head()


output.to_csv("submission.csv", index=False)


print("Predictions saved to 'submission.csv'")

