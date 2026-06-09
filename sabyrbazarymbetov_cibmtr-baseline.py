import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

import warnings
warnings.filterwarnings('ignore')



train_df = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv', index_col='ID')
test_df = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv', index_col='ID')


train_df.head()


test_df.head()


data_dict = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/data_dictionary.csv')
data_dict


print(len(train_df.columns), '\n')
train_df.columns


print(len(test_df.columns), '\n')
test_df.columns


X = train_df.drop(columns=['efs', 'efs_time'])
y = train_df['efs']

X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8)


pipe = Pipeline([
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
    ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore')),
    ('scaler', RobustScaler()), 
    ('Dtree', DecisionTreeClassifier())
])


X_train.shape


y_train.shape


pipe.fit(X_train, y_train)


pipe.score(X_val, y_val)


X_test = test_df
preds = pipe.predict(X_test)

ss = pd.DataFrame({
    'ID': X_test.index, 
    'prediction': preds
})

ss.to_csv('submission.csv', index=False)

