#Necesaary Libraries
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


#Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


#Regression Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


df = pd.read_csv('/kaggle/input/autoam-car-price-prediction/train.csv')
df['running'] = df['running'].str.extract(r'(\d+)').astype(float)
df.head()


numerical_features = []
categorical_features = []
features = []
target = 'price'

for name in df.columns:
    features.append(name)
    if df[name].dtype == 'object':
        categorical_features.append(name)
    else:
        numerical_features.append(name)
numerical_features.remove(target)
features.remove(target)
print(features)


encoder = LabelEncoder()
for name in categorical_features:
    df[name] = encoder.fit_transform(df[name])


preprocessor = ColumnTransformer(
    transformers = [
        ('features', StandardScaler(), features)
    ]
)


X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)


models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'XGBoost': XGBRegressor(),
    'LightGBM': LGBMRegressor(),
    'Cat Boost': CatBoostRegressor(logging_level='Silent')
}


for name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print(f'--------{name}--------')
    print(f'Mean Squared Error: {mse}')
    print(f'Mean Absolute Error: {mae}')
    print(f'R2 Score: {r2}\n')


testX = pd.read_csv('/kaggle/input/autoam-car-price-prediction/test.csv')
testX['running'] = testX['running'].str.extract(r'(\d+)').astype(float)
testX.head()


encoder = LabelEncoder()
for name in categorical_features:
    testX[name] = encoder.fit_transform(testX[name])


X_test = testX[features]


for name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    pipeline.fit(X, y)

    y_pred = pipeline.predict(X_test)
    submission = pd.DataFrame({
        'Id': testX['Id'],
        'price': y_pred
    })
    submission.to_csv('1.'+name[:5]+'_submission.csv')


df = pd.read_csv('/kaggle/input/autoam-car-price-prediction/train.csv')
df['running'] = df['running'].str.extract(r'(\d+)').astype(float)


numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers = [
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)


X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)


for name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print(f'--------{name}--------')
    print(f'Mean Squared Error: {mse}')
    print(f'Mean Absolute Error: {mae}')
    print(f'R2 Score: {r2}\n')


testX = pd.read_csv('/kaggle/input/autoam-car-price-prediction/test.csv')
testX['running'] = testX['running'].str.extract(r'(\d+)').astype(float)
testX.head()


X_test = testX[features]
X_test.head()


for name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    pipeline.fit(X, y)

    y_pred = pipeline.predict(X_test)
    submission = pd.DataFrame({
        'Id': testX['Id'],
        'price': y_pred
    })
    submission.to_csv('2.'+name[:5]+'_submission.csv')

