#Some Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Preprocessing
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#Linear Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor


df = pd.read_csv('/kaggle/input/demand-forecasting-kernels-only/train.csv')
df.head()


df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)


X = df.drop('sales', axis=1)
y = df['sales']
df.head()


#Training and evaluating the model
tscv = TimeSeriesSplit(n_splits=5)

models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=1.0),
    'Random Forest': RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    ),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    ),
    'XGBoost': XGBRegressor(
        n_estimators=100,
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

        #Scale Features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

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
test_df = pd.read_csv('/kaggle/input/demand-forecasting-kernels-only/test.csv')
X_test = test_df.drop('id', axis=1)
X_test.set_index('date', inplace=True)

model = Lasso(alpha=1.0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_test = scaler.transform(X_test)

model.fit(X_scaled, y)
y_pred = model.predict(X_test)


submission = pd.DataFrame({
    'id': test_df['id'],
    'sales': y_pred
})

submission.to_csv('submission.csv', index=False)

