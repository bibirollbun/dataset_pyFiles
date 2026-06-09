#Necsessary Libraries
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


#Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


#Regression Models
from sklearn.linear_model import LinearRegression


df = pd.read_csv('/kaggle/input/playground-series-s4e9/train.csv')
df.drop('id', axis=1, inplace=True)
df.head()


categorical_features = []
numerical_features = []
features = []
target = 'price'

for name in df.columns:
    if df[name].dtype == 'object':
        categorical_features.append(name)
    else:
        numerical_features.append(name)
numerical_features.remove(target)
features = numerical_features + categorical_features
print(features)


df.isnull().sum()


numerical_transformer = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

for name in categorical_features:
    encoder = LabelEncoder()
    df[name] = encoder.fit_transform(df[name])

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_features),
])


X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)


model = LinearRegression()
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model)
])
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print('---------------- Linear Regression ----------------')
print('MSE:', mse)
print('MAE:', mae)
print('R2:', r2)


testX = pd.read_csv('/kaggle/input/playground-series-s4e9/test.csv')

for name in categorical_features:
    encoder = LabelEncoder()
    testX[name] = encoder.fit_transform(testX[name])


X_testX = testX[features]
model = LinearRegression()
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model)
])
pipeline.fit(X, y)

y_pred = pipeline.predict(X_testX)
submission = pd.DataFrame({
    'id': testX['id'],
    'price': y_pred
})
submission.to_csv('submission.csv', index=False)

