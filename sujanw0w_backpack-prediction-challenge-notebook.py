import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt # visualization
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import root_mean_squared_error

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


df_train_1 = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")
df_train_2 = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")

# Concatenate two training data into one
df = pd.concat((df_train_1, df_train_2), axis=0)

# Remove id column
df = df.drop(columns=["id"], axis=1)

print(df.dtypes)


print(df.isnull().sum())


weight_imputation_value = df['Weight Capacity (kg)'].median()
df['Weight Capacity (kg)'] = df['Weight Capacity (kg)'].fillna(weight_imputation_value)

object_cols = df.select_dtypes('object').columns.tolist()
df[object_cols] = df[object_cols].fillna('None')


print(df.isnull().sum())


data = np.array(df)

categorical_columns = [i for i in range(0, len(data[0])) if isinstance(data[0][i], str)]

categorical_data = data[:, categorical_columns]
non_categorical_data = data[:, [i for i in range(0, len(data[0])) if i not in categorical_columns]]

# print(categorical_data)
# print(non_categorical_data)

one_hot_encoder = OneHotEncoder()
one_hot_encoder.fit(X=categorical_data)


encoded_data = one_hot_encoder.transform(X=categorical_data).toarray()
print(encoded_data)
print(encoded_data.shape)


# Concatenate numerical columns with encoded columns

preprocessed_data = np.concatenate((encoded_data, non_categorical_data), axis=1)

print(preprocessed_data)
print(preprocessed_data.shape)


X = preprocessed_data[:, 0 : -1]
y = preprocessed_data[:, -1]

print(f"X shape = {X.shape}")
print(f"y shape = {y.shape}")


X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2, random_state=42)


standard_scalar = StandardScaler()

X_train[:, -2:] = standard_scalar.fit_transform(X=X_train[:,-2: ])

print(X_train[0:2])


# SGD Regression
regression = SGDRegressor(
    max_iter=1000,
    tol=1e-3,
    random_state=42,
    eta0=0.01,
    learning_rate="constant",
    early_stopping=True,
    n_iter_no_change=4
)
regression.fit(X=X_train, y=y_train)

print(regression.coef_)
print("total iterations: ", regression.n_iter_)
print("Number of changes in weights: ", regression.t_)


# Testing data

test_data = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")

id_column = test_data['id']

test_data.drop(columns=["id"], inplace=True)
print(test_data.shape)
# test_data_output = pd.read_csv("./sample_submission.csv")

# test_data = pd.concat([test_data_input, test_data_output], join="inner", axis=1).drop(columns=["id"], axis=1)


print(test_data.isnull().sum())


test_data['Weight Capacity (kg)'] = test_data['Weight Capacity (kg)'].fillna(weight_imputation_value)

test_data[object_cols] = test_data[object_cols].fillna('None')

print(test_data.isnull().sum())


test_data_array = np.array(test_data)

test_categorical_columns = [i for i in range(0, len(test_data_array[0])) if isinstance(test_data_array[0][i], str)]

test_categorical_data = test_data_array[:, categorical_columns]
test_non_categorical_data = test_data_array[:, [i for i in range(0, len(test_data_array[0])) if i not in categorical_columns]]

test_encoded_data = one_hot_encoder.transform(test_categorical_data).toarray()

final_test_data = np.concatenate((test_encoded_data, test_non_categorical_data), axis=1)


X_test = final_test_data
print(X_test.shape)


# Scaling features

X_test[:, -2:] = standard_scalar.transform(X_test[:, -2:])
print(X_test[0:2])


y_pred = regression.predict(X=X_validation)

error = root_mean_squared_error(y_true=y_validation, y_pred=y_pred)
print(error)


# Predict the label

y_target = regression.predict(X=X_test)
print(y_target)


id_column = np.array(id_column)

output_array = np.stack((id_column, y_target), axis=1)
print(output_array)

# Save into a CSV file
np.savetxt("SGD_test_output.csv", output_array, delimiter=',', fmt=['%d', '%f'], header='id, Price', comments='')

