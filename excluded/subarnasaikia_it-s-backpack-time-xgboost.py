import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor

print("Import Completed!!")


train_file_path = "/kaggle/input/playground-series-s5e2/train.csv"
extra_train_file_path = "/kaggle/input/playground-series-s5e2/training_extra.csv"
test_file_path = "/kaggle/input/playground-series-s5e2/test.csv"
sample_submission_file_path = "/kaggle/input/playground-series-s5e2/sample_submission.csv"


train_df = pd.read_csv(train_file_path)
extra_train_df = pd.read_csv(extra_train_file_path)
test_df = pd.read_csv(test_file_path)


extra_train_df.head(5)


train_df.head(5)


train_df.info()


train_df.shape


extra_train_df.shape


test_df.shape


extra_train_df.describe()


train_df.describe()


train_df.describe(include=object)


extra_train_df.describe(include=object)


def missing_details(df):
    print("\n------------------------------------------------")
    print("Missing Values :")
    print("------------------------------------------------")
    print( df.isnull().sum()[df.isnull().sum() > 0] )
    
    
    missing_percentage = (df.isnull().sum() / len(df)) * 100 
    print("\n------------------------------------------------")
    print("Percentage of Missing values: (%) ")
    print("------------------------------------------------")
    print(missing_percentage[missing_percentage > 0])
    

    
    total_missing_percentage = (df.isnull().sum().sum() / (df.size)) * 100
    print("\n------------------------------------------------")
    print(f"Total missing values percentage: {total_missing_percentage:.2f}%")
    print("------------------------------------------------")


print("\n****************************************************")
print("Missing Details of train_df: ")
print("\n****************************************************")
missing_details(train_df)


print("\n****************************************************")
print("Missing Details of extra_train_df: ")
print("\n****************************************************")
missing_details(extra_train_df)

print("\n****************************************************")
print("Missing Details of test_df: ")
print("\n****************************************************")
missing_details(test_df)


def oneHotEncdoing(df):
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    print("\n------------------------------------------------")
    print(f"categorical_columns \n{categorical_columns}")
    print("------------------------------------------------")

    df[categorical_columns] = df[categorical_columns].fillna('missing')
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    one_hot_encoded = encoder.fit_transform(df[categorical_columns])
    one_hot_df = pd.DataFrame(
        one_hot_encoded,
        columns=encoder.get_feature_names_out(categorical_columns)
    )
    df_encoded = pd.concat([df.reset_index(drop=True), one_hot_df], axis=1)
    # for column in categorical_columns:
    #     new_column = column + "_nan"
    #     if new_column in df_encoded.columns.tolist():
    #         df_encoded = df_encoded.drop(new_column, axis=1)
    df_encoded = df_encoded.drop(categorical_columns, axis=1)
    assert df_encoded.shape[0] == df.shape[0], "Row count mismatch after encoding"
    
    return df_encoded


def imputation(X_train, X_valid, test):
    my_imputer = SimpleImputer()
    
    imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
    imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))
    imputed_test = pd.DataFrame(my_imputer.transform(test))
    
    imputed_X_train.columns = X_train.columns
    imputed_X_valid.columns = X_valid.columns
    imputed_test.columns = test.columns
    
    return imputed_X_train, imputed_X_valid, imputed_test


new_train = pd.concat([train_df, extra_train_df], axis=0, ignore_index=True)
print(new_train.shape)
new_train.head(5)


y = new_train.Price
X = new_train.drop(['Price'], axis=1)

print(X.shape)
print(y.shape)

# Divide data into training and validation subsets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)


print(X_train.shape)
print(y_train.shape)


# new_X_train, new_X_valid, new_test_df = imputation(X_train, X_valid, test_df)


print(X_train.shape)
new_X_train = oneHotEncdoing(X_train)
new_X_valid = oneHotEncdoing(X_valid)
new_test_df = oneHotEncdoing(test_df)
print(new_X_train.shape)

final_X_train, final_X_valid, final_test_df = imputation(new_X_train, new_X_valid, new_test_df)


final_X_train.head(5)


print(final_X_train.shape)
print(y_train.shape)


print("\n****************************************************")
print("Missing Details of final_X_train: ")
print("\n****************************************************")
missing_details(final_X_train)


print("\n****************************************************")
print("Missing Details of final_X_valid: ")
print("\n****************************************************")
missing_details(final_X_valid)

print("\n****************************************************")
print("Missing Details of final_test_df: ")
print("\n****************************************************")
missing_details(final_test_df)


model = XGBRegressor(
    n_estimators=5000, 
    learning_rate=0.01,
    early_stopping_rounds=100
)

model.fit(final_X_train, y_train, 
         eval_set=[(final_X_valid, y_valid)], 
         verbose=True
    )


pred = model.predict(final_X_valid)
mae = mean_absolute_error(y_valid, pred)
rmse = mean_squared_error(y_valid, pred, squared=False)
print(f"MAE: {mae}")
print(f"RMSE: {rmse}")


final_prediction = model.predict(final_test_df)


submission = pd.DataFrame({'id': new_test_df['id'], 'Price': final_prediction})
submission.to_csv('submission.csv', index=False)

