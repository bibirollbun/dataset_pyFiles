import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import joblib
import io


df_sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
df = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
df_extra = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')


df


def no_unknown_yes_encoder(df, name_column: str):
    new_column = []
    for value in df[name_column]:
        if value == 'No':
            new_column.append(-1) # no
        elif value == 'Yes':
            new_column.append(1) # yes
        else:
            new_column.append(0) # unknown
            
    df[name_column] = new_column


for dataframe in [df, df_extra, df_test]:
    for column in ['Laptop Compartment', 'Waterproof']:
        no_unknown_yes_encoder(dataframe, column)


for dataframe in [df, df_extra, df_test]:
    for column in ['Brand', 'Material', 'Style', 'Color']:
        df[column] = df[column].fillna('Unknown_' + column)


def category_encoder(df, names_of_columns: list):
    # Ініціалізація OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False)
    
    # Кодування всіх категоріальних колонок
    encoded = encoder.fit_transform(df[names_of_columns])
    
    # Отримуємо назви нових колонок
    new_columns = encoder.get_feature_names_out(names_of_columns)
    
    # Перетворюємо результат у DataFrame
    encoded_df = pd.DataFrame(encoded, columns=new_columns)
    
    # Додаємо закодовані колонки до початкового DataFrame
    df = pd.concat([df.drop(columns=names_of_columns), encoded_df], axis=1)

    return df


categorical_columns = ['Brand', 'Material', 'Style', 'Color']
df = category_encoder(df, categorical_columns)
df_extra = category_encoder(df_extra, categorical_columns)
df_test = category_encoder(df_test, categorical_columns)


def size_encoder(df, name_column: str):
    new_column = []
    for value in df[name_column]:
        if value == 'Small':
            new_column.append(1)
        elif value == 'Medium':
            new_column.append(2)
        elif value == 'Large':
            new_column.append(3)
        else:
            new_column.append(-1) # unknown
            
    df[name_column] = new_column


for dataframe in [df, df_extra, df_test]:
    size_encoder(dataframe, 'Size')


df.plot.box(
    column="Weight Capacity (kg)",
)
plt.show()


df['Weight Capacity (kg)'] = df['Weight Capacity (kg)'].fillna(df['Weight Capacity (kg)'].mean())
df_extra['Weight Capacity (kg)'] = df_extra['Weight Capacity (kg)'].fillna(df_extra['Weight Capacity (kg)'].mean())
df_test['Weight Capacity (kg)'] = df_test['Weight Capacity (kg)'].fillna(df_test['Weight Capacity (kg)'].mean())


df_test.isnull().sum()


df


df_extra


def model_regression_report(model, model_name, X_test, y_test):
    # Прогнозування
    y_pred = model.predict(X_test)
    
    # Оцінка за допомогою метрик регресії
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    # r2 = r2_score(y_test, y_pred)
    
    print(f"Regression report for the model {model_name}:")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    # print(f"R² Score: {r2}")


def grid_search_fun(alg, param_grid, X, y):
    grid_search = GridSearchCV(
        alg,
        param_grid,
        scoring="neg_root_mean_squared_error",
        cv=5,
    )

    grid_search.fit(X, y)

    print("Best parameters:", grid_search.best_params_)
    print("Best accuracy:", grid_search.best_score_)


def X_y_split(df):
    X = df.drop(columns=['id', 'Price'])
    y = df['Price'].values
    return X, y


X, y = X_y_split(df_extra)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=13
)


param_grid_XGBR = {
    'n_estimators': [100, 500, 1000],  # Кількість дерев
    'learning_rate': [0.1, 0.2, 0.01],  # Швидкість навчання
    'max_depth': [6, 9, 3],  # Глибина дерев
    # 'min_child_weight': [1, 3, 5, 7],  # Мінімальна вага вузла
    # 'subsample': [0.6, 0.7, 0.8, 1.0],  # Частка вибірки для навчання
    # 'colsample_bytree': [0.6, 0.7, 0.8, 1.0],  # Частка ознак
    # 'gamma': [0, 0.1, 0.2, 0.5],  # Регуляризація
    # 'reg_alpha': [0, 0.01, 0.1, 1],  # L1-регуляризація
    # 'reg_lambda': [0.5, 1, 2, 5],  # L2-регуляризація
    # 'booster': ['gbtree', 'dart'],  # Тип бустингу
    # 'tree_method': ['auto', 'hist', 'gpu_hist']  # Метод побудови дерева
}


GrS_XGBR = 0
if GrS_XGBR:
    grid_search_fun(
        XGBRegressor(
            # n_estimators = 61,
            # max_depth = 4,
            # learning_rate = 0.5,
            # min_child_weight = 5,
            # alpha = 2.5,
            # objective='reg:squarederror', 
            eval_metric='rmse'
        ),
        param_grid_XGBR,
        X_train,
        y_train,
    )


model_XGBR = XGBRegressor( 
    eval_metric='rmse'
)

model_XGBR.fit(X_train, y_train)

model_regression_report(
    model_XGBR, "XGBRegressor", X_test, y_test
)


model_LGBM = LGBMRegressor()

model_LGBM.fit(X_train, y_train)

model_regression_report(
    model_LGBM, "LGBMRegressor", X_test, y_test
)


y_pred_test = model_LGBM.predict(df_test[df_test.columns[1:]])


df_sample_submission['id'] = df_test['id']
df_sample_submission['Price'] = y_pred_test


df_sample_submission


df_sample_submission.to_csv('submission.csv', index=False)

