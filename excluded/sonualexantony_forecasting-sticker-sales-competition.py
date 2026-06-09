#Required libraries
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

#Preprocessing
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#Linear Models
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor


df = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
df = df.dropna(subset=['num_sold']) #Dropping rows without target value
df.head()


df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df.drop('id', axis=1, inplace=True)


NUMERICAL_DATA = []
CATEGORICAL_DATA = []

for name in df.columns:
    if df[name].dtype == 'object':
        CATEGORICAL_DATA.append(name)
    else:
        NUMERICAL_DATA.append(name)

NUMERICAL_DATA.remove('num_sold')
FEATURES = NUMERICAL_DATA + CATEGORICAL_DATA
print('Features:',FEATURES)
print('Numerical Features:', NUMERICAL_DATA)
print('Categorical Features:', CATEGORICAL_DATA)


numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers = [
        ('num', numeric_transformer, NUMERICAL_DATA),
        ('cat', categorical_transformer, CATEGORICAL_DATA)
    ]
)


X = df.drop('num_sold', axis=1)
y = df['num_sold']

X_scaled = preprocessor.fit_transform(X)


# Used for tuning Hyperparameters
# tscv = TimeSeriesSplit(n_splits=2)
# param_grid = {
#     'loss': ['squared_error', 'absolute_error', 'huber', 'quantile'],
#     'n_estimators': [10, 50, 100],
#     'max_depth': [None, 5, 10],
#     'learning_rate': [0.0, 0.1, 0.5]
# }

# grid_search = GridSearchCV(
#     estimator = GradientBoostingRegressor(),
#     param_grid = param_grid,
#     cv = tscv,
#     n_jobs=-1
# )

# grid_search.fit(X_scaled, y)
# print(grid_search.best_params_)


# Used for tuning Hyperparameters
# tscv = TimeSeriesSplit(n_splits=2)
# param_grid = {
#     'n_estimators': [10, 50, 100],
#     'max_depth': [None, 5, 10],
#     'learning_rate': [0.0, 0.1, 0.5]
# }

# grid_search = GridSearchCV(
#     estimator = XGBRegressor(),
#     param_grid = param_grid,
#     cv = tscv,
#     n_jobs=-1
# )

# grid_search.fit(X_scaled, y)
# print(grid_search.best_params_)


#Training and evaluating the model
tscv = TimeSeriesSplit(n_splits=5)

models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=0.0, max_iter=100),
    'Lasso': Lasso(alpha=1.0),
    'Random Forest': RandomForestRegressor(
        n_estimators=10,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    ),
    'Gradient Boosting': GradientBoostingRegressor(
        loss='absolute_error',
        n_estimators=50,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    ),
    'XGBoost': XGBRegressor(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
}

results = {}

#Training and evaluating every single model
for name, model in models.items():
    mse = []
    mae = []
    r2 = []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        #Preprocessing features
        X_train_scaled = preprocessor.fit_transform(X_train)
        X_test_scaled = preprocessor.transform(X_test)

        #Train the model
        model.fit(X_train_scaled, y_train)

        #Make predections
        y_pred = model.predict(X_test_scaled)

        #Calculate metrics
        mse.append(mean_squared_error(y_test, y_pred))
        mae.append(mean_absolute_error(y_test, y_pred))
        r2.append(r2_score(y_test, y_pred))

    results[name] = {
        'MSE': np.mean(mse),
        'MAE': np.mean(mae),
        'R2': np.mean(r2)
    }


# Print results
for model_name, metrics in results.items():
    print(f"\nResults for {model_name}:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")


#Training the entire data on Lasso
test_df = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
X_test = test_df.drop('id', axis=1)
X_test.set_index('date', inplace=True)

model = GradientBoostingRegressor(
    loss='absolute_error',
    n_estimators=50,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

X_scaled = preprocessor.fit_transform(X)
X_scaled_test = preprocessor.transform(X_test)

model.fit(X_scaled, y)
y_pred = model.predict(X_scaled_test)


submission = pd.DataFrame({
    'id': test_df['id'],
    'sales': y_pred
})

submission.to_csv('submission.csv', index=False)




