import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


solution = pd.read_csv('/kaggle/input/aviachipta-narxini-bashorat-qilish/sample_solution.csv', index_col=0)
test = pd.read_csv("/kaggle/input/aviachipta-narxini-bashorat-qilish/test_data.csv", index_col=0)
dataset = pd.read_csv("/kaggle/input/aviachipta-narxini-bashorat-qilish/train_data.csv", index_col=0)
dataset.head()


# Outlier(Noan'anaviy) qiymatlarni olib tashlash
def filter_out(data):
    data = data[data['days_left'] > 9]
    data = data[data['duration'] > 1]
    return data

data = filter_out(dataset)

# Datasetni bo'lish
X = data.drop('price', axis=1)
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def preprocess_data(train_data=None, test_data=None):
    # Sonli va matnli ustunlarni ajratib olish
    categorical_columns = test_data.select_dtypes(include=['object']).columns.to_list()
    numerical_columns = test_data.select_dtypes(include=['int64', 'float64']).columns.to_list()

    # Normalizatsiya qilish
    scaler = MinMaxScaler()
    
    if train_data is not None:
        train_data[numerical_columns] = scaler.fit_transform(train_data[numerical_columns])
        test_data[numerical_columns] = scaler.transform(test_data[numerical_columns])
    else:
        test_data[numerical_columns] = scaler.fit_transform(test_data[numerical_columns])

    # Qo'lda kodlash uchun qiymatlar
    encoded = {
        'departure_time': {'Early_Morning': 0, 'Evening': 1, 'Morning': 2, 'Afternoon': 3, 'Night': 4, 'Late_Night': 5},
        'stops': {'zero': 0, 'one': 1, 'two_or_more': 2},
        'arrival_time': {'Night': 0, 'Evening': 1, 'Morning': 2, 'Afternoon': 3, 'Early_Morning': 4, 'Late_Night': 5}
    }

    # Matnli ustunlarni avtomatik kodlash
    for col in categorical_columns:
        if col not in encoded.keys():
            combined_data = pd.concat([train_data[col], test_data[col]], axis=0) if train_data is not None else test_data[col]
            le = LabelEncoder()
            le.fit(combined_data)
            if train_data is not None:
                train_data[col] = le.transform(train_data[col])
            test_data[col] = le.transform(test_data[col])

        # Qo'lda kodlash
        for key, value in encoded.items():
            if key in col:
                if train_data is not None:
                    train_data[key] = train_data[key].replace(value)
                test_data[key] = test_data[key].replace(value)

    return (train_data, test_data, scaler) if train_data is not None else (test_data, scaler)

X_train, X_test, scaler = preprocess_data(X_train, X_test)


# Optimal parametrlarni kiritib train qilamiz
model = RandomForestRegressor(random_state=42, max_depth=30, min_samples_leaf=1, min_samples_split=2, n_estimators=500)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(y_pred)


# Parametrlarni belgilash
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
# GridSearchCV bilan parametrlarni optimallashtirish
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=3,  # 3-fold cross-validation
    scoring='neg_mean_squared_error',
    n_jobs=-1,  # Parallel ishlash
    verbose=2
)
grid_search.fit(X_train, y_train)
gs_pred = grid_search.predict(X_test)
print("Eng yaxshi parametrlar:", grid_search.best_params_)
print("Natija:", gs_pred)


# Baholash
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("rmse:", rmse)
print("mae:", mae)
print(f"R-squared: {r2:.2f}")


# Taqqoslash
comparison = pd.DataFrame({
    'Haqiqiy narx': y_test,
    'Bashorat narx': y_pred
})
print(comparison.head())

plt.plot(y_test.values, label='Haqiqiy narx', color='blue')
plt.plot(y_pred, label='Bashorat narx', color='red')
plt.legend()
plt.show()


test, scaler = preprocess_data(test_data=test)
result = model.predict(test)
solution['price'] = result
solution.to_csv('solution.csv')
df = pd.read_csv("/kaggle/working/solution.csv", index_col=0)
df

