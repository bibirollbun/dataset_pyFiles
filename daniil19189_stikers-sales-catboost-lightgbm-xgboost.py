import pandas as pd
import numpy as np

# Модели
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Для валидации и подбора параметров
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error



# ======== ШАГ 1: ЗАГРУЗКА И АГРЕГАЦИЯ  ===================================

# Допустим, train.csv имеет столбцы: [id, date, country, store, product, num_sold]
# А test.csv: [id, date, country, store, product] (без num_sold).
train_df = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
test_df  = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")

train_df['date'] = pd.to_datetime(train_df['date'])
test_df['date']  = pd.to_datetime(test_df['date'])

# Уберём/заполним пропуски в num_sold, если есть.
train_df = train_df.dropna(subset=['num_sold'])

# 1.1. Собираем общий временной ряд (агрегированный)
aggregated_train = (
    train_df
    .groupby('date', as_index=False)['num_sold']
    .sum()
    .rename(columns={'num_sold': 'total_num_sold'})
)



# ======== ШАГ 2: ВЫЧИСЛЕНИЕ RATIO (COUNTRY/STORE/PRODUCT)  ===============

#
# 2.1. Пример: средняя доля страны от общего объёма продаж за весь train.
#     Если в реальности доля меняется во времени, нужно строить time-based ratio (как в вашем примере).
#

# Пример: country_ratio_df: [country, ratio_country]
country_ratio_df = (
    train_df
    .groupby('country', as_index=False)['num_sold'].sum()
    .rename(columns={'num_sold': 'sum_country'})
)

sum_all = country_ratio_df['sum_country'].sum()
country_ratio_df['ratio_country'] = country_ratio_df['sum_country'] / sum_all

# Аналогично для store
store_ratio_df = (
    train_df
    .groupby('store', as_index=False)['num_sold'].sum()
    .rename(columns={'num_sold': 'sum_store'})
)
sum_store_all = store_ratio_df['sum_store'].sum()
store_ratio_df['ratio_store'] = store_ratio_df['sum_store'] / sum_store_all

# Аналогично для product
product_ratio_df = (
    train_df
    .groupby('product', as_index=False)['num_sold'].sum()
    .rename(columns={'num_sold': 'sum_product'})
)
sum_product_all = product_ratio_df['sum_product'].sum()
product_ratio_df['ratio_product'] = product_ratio_df['sum_product'] / sum_product_all


# ======== ШАГ 3: ОБУЧАЕМ АНСАМБЛЬ НА АГРЕГИРОВАННОМ РЯДЕ ================
# Пример с CatBoost, XGBoost, LightGBM + RandomizedSearchCV

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error

# 3.1. Фичи для временного ряда (упрощённо)
aggregated_train['day_of_week'] = aggregated_train['date'].dt.dayofweek
aggregated_train['month']       = aggregated_train['date'].dt.month
aggregated_train['year']        = aggregated_train['date'].dt.year - aggregated_train['date'].dt.year.min()

# Можно добавить sin/cos и т.д.  (как в вашем коде)
# aggregated_train['month_sin'] = np.sin(...)
# aggregated_train['month_cos'] = np.cos(...)

# Уберём дату из фич
X_agg = aggregated_train.drop(['date','total_num_sold'], axis=1)
y_agg = aggregated_train['total_num_sold']

# Разделяем на train/val
X_train, X_val, y_train, y_val = train_test_split(X_agg, y_agg, test_size=0.2, shuffle=False)

# Параметры для CatBoost (минимально)
cat_param = {
    'iterations': [50,100],
    'depth': [4,6]
}
cat_model = CatBoostRegressor(verbose=False, random_seed=42)
cat_search = RandomizedSearchCV(
    cat_model, cat_param, n_iter=2, scoring='neg_root_mean_squared_error', cv=3, random_state=42
)

# Параметры для XGB
xgb_param = {
    'n_estimators': [100,200],
    'max_depth': [3,6],
}
xgb_model = XGBRegressor(random_state=42)
xgb_search = RandomizedSearchCV(
    xgb_model, xgb_param, n_iter=2, scoring='neg_root_mean_squared_error', cv=3, random_state=42
)

# Параметры для LGBM
lgbm_param = {
    'n_estimators': [100,200],
    'max_depth': [3,6,-1],
}
lgbm_model = LGBMRegressor(random_state=42)
lgbm_search = RandomizedSearchCV(
    lgbm_model, lgbm_param, n_iter=2, scoring='neg_root_mean_squared_error', cv=3, random_state=42
)

# Обучаем
cat_search.fit(X_train, y_train)
xgb_search.fit(X_train, y_train)
lgbm_search.fit(X_train, y_train)

best_cat  = cat_search.best_estimator_
best_xgb  = xgb_search.best_estimator_
best_lgbm = lgbm_search.best_estimator_

val_cat  = best_cat.predict(X_val)
val_xgb  = best_xgb.predict(X_val)
val_lgbm = best_lgbm.predict(X_val)

rmse_cat  = np.sqrt(mean_squared_error(y_val, val_cat))
rmse_xgb  = np.sqrt(mean_squared_error(y_val, val_xgb))
rmse_lgbm = np.sqrt(mean_squared_error(y_val, val_lgbm))

print("Val RMSE CatBoost:", rmse_cat)
print("Val RMSE XGBoost: ", rmse_xgb)
print("Val RMSE LGBM:    ", rmse_lgbm)

# Ансамбль (усреднение)
val_ensemble = (val_cat + val_xgb + val_lgbm) / 3
rmse_ensemble = np.sqrt(mean_squared_error(y_val, val_ensemble))
print("Val RMSE Ensemble:", rmse_ensemble)


# ======== ШАГ 4: ПОЛУЧАЕМ ПРОГНОЗ НА TEST, РАЗБИВАЕМ ПО RATIO ===========

# 4.1. Подготовим тестовый набор (агрегированный) - т.е. получим список дат из test_df
test_dates_df = test_df[['date']].drop_duplicates().sort_values('date').reset_index(drop=True)
test_dates_df['day_of_week'] = test_dates_df['date'].dt.dayofweek
test_dates_df['month']       = test_dates_df['date'].dt.month
test_dates_df['year']        = test_dates_df['date'].dt.year - aggregated_train['date'].dt.year.min()

X_test_agg = test_dates_df.drop(['date'], axis=1)

# Предсказываем общий прогноз (на все страны, все магазины, все продукты)
pred_cat  = best_cat.predict(X_test_agg)
pred_xgb  = best_xgb.predict(X_test_agg)
pred_lgbm = best_lgbm.predict(X_test_agg)

pred_ensemble = (pred_cat + pred_xgb + pred_lgbm) / 3

# 4.2. "Размножаем" итоговый ряд на все (country, store, product)
#     Предположим, в test_df каждая строка - уникальная (date, country, store, product).
#     Хотим для каждой строки получить итоговый num_sold = total_pred * ratio_country * ratio_store * ratio_product

test_merged = test_df.copy()
test_merged = test_merged.sort_values('date').reset_index(drop=True)

# Добавим столбец total_pred напрямую
test_dates_df['total_pred'] = pred_ensemble

# Соединяем по дате, чтобы в каждой строке test_merged появилась соответствующая total_pred
test_merged = pd.merge(
    test_merged,
    test_dates_df[['date','total_pred']], 
    on='date',
    how='left'
)

# Присоединим ratio_country, ratio_store, ratio_product (здесь они постоянные)
test_merged = pd.merge(test_merged, country_ratio_df[['country','ratio_country']], on='country', how='left')
test_merged = pd.merge(test_merged, store_ratio_df[['store','ratio_store']], on='store', how='left')
test_merged = pd.merge(test_merged, product_ratio_df[['product','ratio_product']], on='product', how='left')

# 4.3. Считаем финальный прогноз: num_sold = total_pred * ratio_country * ratio_store * ratio_product
test_merged['num_sold'] = (
    test_merged['total_pred']
    * test_merged['ratio_country']
    * test_merged['ratio_store']
    * test_merged['ratio_product']
).round()

# 4.4. Формируем submission
submission = test_merged[['id','num_sold']].copy()
submission.to_csv("submission_with_ratios.csv", index=False)

print(submission.head(10))


