# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df_train = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")

df_train


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import pandas as pd

def preprocess_data(input_data):

    input_data = input_data.copy()

    # Separate the 'date' column
    date_col = 'date'
    
    date_data = input_data[[date_col]]
    other_data = input_data.drop(columns=[date_col])

    # Identify numerical and categorical columns
    numerical_cols = other_data.select_dtypes(include=['number']).columns
    categorical_cols = other_data.select_dtypes(include=['object', 'category']).columns

    # Impute missing values
    imputer = SimpleImputer(strategy='constant', fill_value=0)
    imputed_date_data = pd.DataFrame(
        imputer.fit_transform(date_data),
        columns=[date_col],
        index=input_data.index
    )
    imputed_numerical_data = pd.DataFrame(
        imputer.fit_transform(other_data[numerical_cols]),
        columns=numerical_cols,
        index=input_data.index
    )
    imputed_categorical_data = pd.DataFrame(
        imputer.fit_transform(other_data[categorical_cols]),
        columns=categorical_cols,
        index=input_data.index
    )

    # Ordinal encode the 'date' column
    ordinal_encoder = OrdinalEncoder()
    encoded_date_data = pd.DataFrame(
        ordinal_encoder.fit_transform(imputed_date_data),
        columns=[date_col],
        index=input_data.index
    )
    new_date_data = encoded_date_data['date']
    encoded_date_data = pd.DataFrame(new_date_data/(np.max(new_date_data)),
                                columns=[date_col],
                                index=input_data.index)

    # One-hot encode all categorical columns
    one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    encoded_categorical_data = pd.DataFrame(
        one_hot_encoder.fit_transform(imputed_categorical_data),
        columns=one_hot_encoder.get_feature_names_out(categorical_cols),
        index=input_data.index
    )

    
    # Combine numerical, one-hot encoded categorical, and ordinally encoded date columns
    processed_data = pd.concat([imputed_numerical_data, encoded_categorical_data, encoded_date_data], axis=1)

    return processed_data


df_processed=preprocess_data(df_train)


from sklearn.model_selection import train_test_split

X = df_processed.drop(columns=['num_sold','id'])
y = df_processed['num_sold']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.7, random_state=0)


from sklearn.metrics import mean_absolute_error

def eval_model(model):
    pred = model.predict(X_valid)
    pred = np.round(pred)
    mae = mean_absolute_error(pred,y_valid)
    print("MAE for this model:",mae)


from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=450, random_state=0)
model.fit(X_train, y_train)
eval_model(model)


X_test = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")
X_test_p = preprocess_data(X_test).drop(['id'],axis=1)

pred = model.predict(X_test_p)
final_pred = np.round(pred).astype(int)

output = pd.DataFrame({
    'id':X_test['id'],
    'num_sold':final_pred
})

output.to_csv('predictions.csv',index=False)

print('Predictions saved to csv!')

