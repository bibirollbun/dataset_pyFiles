#Necessary Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


#Preprocessing Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


#Regression Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
# train_ex_df = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')


num_cols = []
obj_cols = []

for name in train.columns:
    if train[name].dtype == 'object':
        obj_cols.append(name)
    else:
        num_cols.append(name)
num_cols.remove('Price')
num_cols.remove('id')
print(num_cols)
features = num_cols + obj_cols
print(features)


plt.figure(figsize=(12, 4))
i = 1
for name in num_cols:
    if name != 'Price' and name != 'id':
        plt.subplot(1, 2, i)
        i += 1
        plt.scatter(train[name], train['Price'])
        plt.xlabel(name)
        plt.ylabel('Price')
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 6))
for i, col in enumerate(num_cols):
    plt.subplot(2, 3, i*3+1)
    sns.histplot(train[col], bins=10, color='blue')
    plt.title(f'Train [{col}] Distribution')
    plt.xlabel(col)

    plt.subplot(2, 3, i*3+2)
    sns.histplot(test_df[col], bins=10, color='green')
    plt.title(f'Test [{col}] Distribution')
    plt.xlabel(col)
plt.tight_layout()
plt.show()


train.isnull().sum()


num_tra = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers = [
        ('num', num_tra, features)
    ]
)


# train = pd.concat([train_df, train_ex_df], axis=0, ignore_index=True)


encoder = LabelEncoder()
for name in obj_cols:
    train[name] = encoder.fit_transform(train[name])
    test_df[name] = encoder.fit_transform(test_df[name])


train.head()


X = train[features]
y = train['Price']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'RandomForest': RandomForestRegressor(
        n_jobs=-1
    ),
    'GradientBoosting': GradientBoostingRegressor(),
    'XGBoost': XGBRegressor(
        n_jobs = -1
    ),
    'LightGBM': LGBMRegressor(
        n_jobs = -1
    ),
    'CatBoost': CatBoostRegressor(
        logging_level='Silent'
    )
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

    print(f'\n{name} Model Results:')
    print('R2-SCORE:', r2)
    print('Mean Absolute Error:', mae)
    print('Mean Squared Error:', mse)


X_test = test_df[features]


for name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    pipeline.fit(X, y)

    y_pred = pipeline.predict(X_test)
    submission = pd.DataFrame({
        'id': test_df['id'],
        'Price': y_pred
    })
    submission.to_csv(name[:5]+'_submission.csv', index=False)

