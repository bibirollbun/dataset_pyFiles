import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


train_data = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')


train_data['Brand_Material'] = train_data['Brand'].fillna('Unknown') + '_' + train_data['Material'].fillna('Unknown')

bins = [0, 5, 10, np.inf]
labels = ['Low', 'Medium', 'High']
train_data['WeightCapacity_Bin'] = pd.cut(train_data['Weight Capacity (kg)'], bins=bins, labels=labels)

test_data['Brand_Material'] = test_data['Brand'].fillna('Unknown') + '_' + test_data['Material'].fillna('Unknown')
test_data['WeightCapacity_Bin'] = pd.cut(test_data['Weight Capacity (kg)'], bins=bins, labels=labels)

X = train_data.drop(['Price', 'id'], axis=1)
y = train_data['Price']

numeric_features = ['Compartments', 'Weight Capacity (kg)']
categorical_features = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color',
                        'Brand_Material', 'WeightCapacity_Bin']



numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

ridge_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', Ridge())
])



X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'regressor__alpha': [0.01, 0.1, 1, 10, 100],
    'regressor__solver': ['auto', 'svd', 'cholesky', 'lsqr']
}

grid_search = GridSearchCV(ridge_pipeline, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)



y_pred = grid_search.predict(X_valid)

mse = mean_squared_error(y_valid, y_pred)
mae = mean_absolute_error(y_valid, y_pred)
r2 = r2_score(y_valid, y_pred)

print(f" Mean Squared Error: {mse:.2f}")
print(f" Mean Absolute Error: {mae:.2f}")
print(f" R-squared: {r2:.4f}")


X_test = test_data.drop(['id'], axis=1)
test_preds = grid_search.predict(X_test)

submission = test_data[['id']].copy()
submission['Price'] = test_preds
submission.to_csv('/kaggle/working/submission.csv', index=False)

print("✅ Submission file saved to /kaggle/working/submission.csv")
print(submission.head())




