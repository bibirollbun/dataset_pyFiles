import numpy as np 
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
sub = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")


print(f'The shape of the train dataset is {train.shape}')
print(f'The shape of the test dataset is {test.shape}')


train.head(10)


import seaborn as sns


sns.heatmap(train.corr(numeric_only=True),annot=True)


train["Genre"].nunique()


train.nunique()


train.info()


train.describe()





train.drop(columns=['id'], inplace=True)
test.drop(columns=['id'], inplace=True)


train.shape


train.head()


X = train.iloc[:,:-1]
y = train.iloc[:,-1]


X.head()


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer


numerical_pipeline = Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler',StandardScaler())])
categorical_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),('encoder',OrdinalEncoder())])


numerical_features = X.select_dtypes(include=['int64','float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()


numerical_features


preprocessor = ColumnTransformer([('num', numerical_pipeline, numerical_features),('cat', categorical_pipeline, categorical_features)])


X_train_preprocessed = preprocessor.fit_transform(X_train)
X_val_preprocessed = preprocessor.transform(X_val)


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

# model = LinearRegression()
# model = DecisionTreeRegressor(max_depth=5, min_samples_split=5, min_samples_leaf=5)
model = AdaBoostRegressor(random_state=42, n_estimators=100)


model.fit(X_train_preprocessed, y_train)

y_pred = model.predict(X_val_preprocessed)


from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print(f'MAE: {mae}, MSE: {mse}, r2: {r2}')


test_preprocessed = preprocessor.transform(test)

predictions = model.predict(test_preprocessed)


sub['Listening_Time_minutes'] = predictions

sub.to_csv("submission.csv", index=False)

