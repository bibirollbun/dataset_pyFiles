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


train= pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv')
test= pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv')
data_dict= pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/data_dictionary.csv')



from sklearn.impute import KNNImputer,SimpleImputer 
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, RobustScaler


#train.info()
#train.head(7)


# Step 1: Handle missing values
train.replace({'N/A': np.nan, 'NaN': np.nan}, inplace=True)  # Convert string "N/A" and "NaN" to real NaN
train.dropna(axis=0, how='all', inplace=True)  # Remove rows where all values are NaN
train.dropna(axis=1, how='all', inplace=True)  # Remove columns where all values are NaN

# Step 2: Convert categorical variables to numerical (if necessary)
categorical_cols = train.select_dtypes(include=['object']).columns  # Select categorical columns
train[categorical_cols] = train[categorical_cols].apply(lambda x: pd.factorize(x)[0])  # Encode categories

# Step 3: Convert numeric columns with potential issues
numeric_cols = train.select_dtypes(include=['number']).columns
train[numeric_cols] = train[numeric_cols].apply(pd.to_numeric, errors='coerce')  # Convert to float, force NaNs where errors occur

# Step 4: Fill remaining missing values (if any)
train.fillna(train.median(), inplace=True)  # Replace NaNs with median of each column

# Step 5: Normalize numeric columns (if needed)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
train[numeric_cols] = scaler.fit_transform(train[numeric_cols])






# Step 1: Handle missing values
test.replace({'N/A': np.nan, 'NaN': np.nan}, inplace=True)  # Convert string "N/A" and "NaN" to real NaN
test.dropna(axis=0, how='all', inplace=True)  # Remove rows where all values are NaN
test.dropna(axis=1, how='all', inplace=True)  # Remove columns where all values are NaN

# Step 2: Convert categorical variables to numerical (if necessary)
categorical_cols = test.select_dtypes(include=['object']).columns  # Select categorical columns
test[categorical_cols] = test[categorical_cols].apply(lambda x: pd.factorize(x)[0])  # Encode categories

# Step 3: Convert numeric columns with potential issues
numeric_cols = test.select_dtypes(include=['number']).columns
test[numeric_cols] = test[numeric_cols].apply(pd.to_numeric, errors='coerce')  # Convert to float, force NaNs where errors occur

# Step 4: Fill remaining missing values (if any)
test.fillna(test.median(), inplace=True)  # Replace NaNs with median of each column

# Step 5: Normalize numeric columns (if needed)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
test[numeric_cols] = scaler.fit_transform(test[numeric_cols])






X = train.drop(columns= ['efs', 'efs_time'])
y = train['efs']


X_train, X_val, y_train, y_val = train_test_split(X,y, train_size= 0.75)


train.head()



# Step 6: Save cleaned dataset
#df.to_csv('cleaned_dataset.csv', index=False)



pipe = Pipeline([
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
    ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore')),
    ('scaler', StandardScaler()),
    ('classifier', XGBClassifier(use_label_encoder=True, eval_metric='logloss'))
])


pipe.fit(X_train, y_train)


pipe.score(X_val, y_val)





test.head()


X_test =test
preds = pipe.predict(X_test)
final_df = pd.DataFrame({
    'ID': X_test.index,
    'prediction': preds,
})

final_df.ID = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv").ID.values

final_df.to_csv('submission.csv', index = False)


final_df.ID

