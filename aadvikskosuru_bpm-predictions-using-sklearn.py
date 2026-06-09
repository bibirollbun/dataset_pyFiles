from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd


train_df = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")

test_ids = test_df['id']


train_df = train_df.drop('id', axis=1)
test_df = test_df.drop('id', axis=1)


X_train = train_df.drop('BeatsPerMinute', axis=1)
y_train = train_df['BeatsPerMinute']


X_test = test_df.copy()


pipeline = Pipeline([
    ('poly', PolynomialFeatures(include_bias=False)),
    ('scaler', StandardScaler()), 
    ('ridge', Ridge(random_state=42))
])

param_grid = { 
    'poly__degree': [2, 3],
    'ridge__alpha': np.logspace(-3, 5, 9)
}


grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5,
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)


best_alpha = grid_search.best_params_['ridge__alpha']
best_mse = -grid_search.best_score_
best_rmse = np.sqrt(best_mse)

print(f"Best Alpha found: {best_alpha}")
print(f"Best cross-validated RMSE: {best_rmse:.4f}")

best_model = grid_search.best_estimator_


predictions = best_model.predict(X_test)


submission = pd.DataFrame({'id': test_ids, 'BeatsPerMinute': predictions})
submission.to_csv('submission.csv', index=False)

