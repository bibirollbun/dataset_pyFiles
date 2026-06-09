import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


train_file_path = "/kaggle/input/playground-series-s5e2/train.csv"
test_file_path = "/kaggle/input/playground-series-s5e2/test.csv"
sample_submission_file_path = "/kaggle/input/playground-series-s5e2/sample_submission.csv"


train_df = pd.read_csv(train_file_path)
test_df = pd.read_csv(test_file_path)


train_df.head(5)


train_df.info()


train_df.describe()


train_df.describe(include=object)


train_df.isnull().sum()


test_df.isnull().sum()


def oneHotEncdoing(df):
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    print(f"categorical_columns \n{categorical_columns}")

    encoder = OneHotEncoder(sparse_output=False)
    one_hot_encoded = encoder.fit_transform(df[categorical_columns])
    one_hot_df = pd.DataFrame(
        one_hot_encoded,
        columns=encoder.get_feature_names_out(categorical_columns)
    )
    df_encoded = pd.concat([df, one_hot_df], axis=1)
    df_encoded = df_encoded.drop(categorical_columns, axis=1)
    
    return df_encoded


def imputation(df):
    my_imputer = SimpleImputer()
    imputed_df = pd.DataFrame(my_imputer.fit_transform(df))
    imputed_df.columns = df.columns
    return imputed_df


new_train_df = oneHotEncdoing(train_df)
new_train_df = imputation(new_train_df)
new_train_df.head(5)


new_train_df.isnull().sum()


new_train_df.info()


y = new_train_df.Price
X = new_train_df.drop(['Price'], axis=1)

# Divide data into training and validation subsets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)


model = RandomForestRegressor(n_estimators=50, random_state=0)
model.fit(X_train, y_train)


pred = model.predict(X_valid)
mae = mean_absolute_error(y_valid, pred)
rmse = mean_squared_error(y_valid, pred, squared=False)
print(f"MAE: {mae}")
print(f"RMSE: {rmse}")


new_test_df = oneHotEncdoing(test_df)
new_test_df = imputation(new_test_df)


Final_model = RandomForestRegressor(n_estimators=50, random_state=0)
Final_model.fit(X, y)
final_prediction = Final_model.predict(new_test_df)


submission = pd.DataFrame({'id': new_test_df['id'], 'Price': final_prediction})
submission.to_csv('submission.csv', index=False)

