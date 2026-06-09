import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)


df_train = pd.read_csv('/kaggle/input/prediction-interval-competition-ii-house-price/dataset.csv')
df_test = pd.read_csv('/kaggle/input/prediction-interval-competition-ii-house-price/test.csv')


df_train.head(5)


df_train.shape


df_train.isna().sum()


df_train.duplicated().sum()


df_train.describe()


df_train.info()


num_cols = df_train.select_dtypes(include=['float64', 'int64']).columns
obj_cols = df_train.select_dtypes(include='object').columns

print(f'numeric columns: {num_cols} \n')
print(f'object columns: {obj_cols} \n')


# Liczba kolumn numerycznych
n_cols = 2  # Liczba wykresów w jednym wierszu
n_rows = (len(num_cols) + n_cols - 1) // n_cols  # Liczba wierszy

# Tworzenie siatki wykresów
fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 80))
axes = axes.flatten()  # Spłaszczenie tablicy osi

# Rysowanie wykresów
for i, col in enumerate(num_cols):
    axes[i].hist(df_train[col])
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')
    axes[i].set_title(f'Frequency of {col}')
    axes[i].grid(linestyle='--', alpha=0.6)

# Ukrywanie pustych osi, jeśli liczba wykresów jest mniejsza niż liczba osi
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()  # Dopasowanie układu
plt.show()


num = df_train.select_dtypes(include=np.number)
correlation_matrix = num.corr()

plt.figure(figsize=(32, 32))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.1f')
plt.title('Correlation Matrix')
plt.title
plt.show()




from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

df_train['sale_date'] = pd.to_datetime(df_train['sale_date'])
df_train['sale_year'] = df_train['sale_date'].dt.year
df_train = df_train.sort_values(by='sale_date')


df_train['sale_nbr'] = df_train['sale_nbr'].fillna(df_train['sale_nbr'].mode()[0])
df_train['subdivision'] = df_train['subdivision'].fillna(df_train['subdivision'].mode()[0])
df_train['submarket'] = df_train['submarket'].fillna(df_train['submarket'].mode()[0])

scaler = MinMaxScaler()
df_train[num_cols] = scaler.fit_transform(df_train[num_cols])

encoder = LabelEncoder()

df_train['sale_warning'] = encoder.fit_transform(df_train['sale_warning'])
df_train['join_status'] = encoder.fit_transform(df_train['join_status'])
df_train['city'] = encoder.fit_transform(df_train['city'])
df_train['zoning'] = encoder.fit_transform(df_train['zoning'])
df_train['sale_warning'] = encoder.fit_transform(df_train['sale_warning'])
df_train['subdivision'] = encoder.fit_transform(df_train['subdivision'])
df_train['submarket'] = encoder.fit_transform(df_train['submarket'])         

X = df_train.drop(columns=['id', 'sale_date', 'sale_price', 'sale_nbr'])
y = df_train['sale_price']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

rf_model = RandomForestRegressor()
xgb_model = XGBRegressor()
lgbm_model = XGBRegressor()
gb_model = XGBRegressor()


rf_model.fit(x_train, y_train)
xgb_model.fit(x_train, y_train)
lgbm_model.fit(x_train, y_train)
gb_model.fit(x_train, y_train)

rf_pred = rf_model.predict(x_test)
xgb_pred = xgb_model.predict(x_test)
lgbm_pred = lgbm_model.predict(x_test)
gb_pred = gb_model.predict(x_test)


rf_r2 = r2_score(y_test, rf_pred)
xgb_r2 = r2_score(y_test, xgb_pred)
lgbm_r2 = r2_score(y_test, lgbm_pred)
gb_r2 = r2_score(y_test, gb_pred)

rf_mse= mean_squared_error(y_test, rf_pred)
xgb_mse = mean_squared_error(y_test, xgb_pred)
lgbm_mse = mean_squared_error(y_test, lgbm_pred)
gb_mse = mean_squared_error(y_test, gb_pred)

rf_mae = mean_absolute_error(y_test, rf_pred)
xgb_mae = mean_absolute_error(y_test, xgb_pred)
lgbm_mae = mean_absolute_error(y_test, lgbm_pred)
gb_mae = mean_absolute_error(y_test, gb_pred)

print('RandomForestRegressor : \n')
print(f'r2 score: {rf_r2} \n')
print(f'mean absolute error: {rf_mse} \n')
print(f'mean squered error: {rf_mae} \n')

print('XGBRegressor: \n')
print(f'r2 score: {xgb_r2} \n')
print(f'mean absolute error: {xgb_mse} \n')
print(f'mean squered error: {xgb_mae} \n')

print('LGBMRegressor: \n')
print(f'r2 score: {lgbm_r2} \n')
print(f'mean absolute error: {lgbm_mse} \n')
print(f'mean squered error: {lgbm_mae} \n')

print('GradientBoostingRegressor: \n')
print(f'r2 score: {gb_r2} \n')
print(f'mean absolute error: {gb_mse} \n')
print(f'mean squered error: {gb_mae} \n')


model_scores = {
    'RandomForestRegressor' : rf_r2,
    'XGBRegressor' : xgb_r2,
    'LGBMRegressor' : lgbm_r2,
    'GradientBoostingRegressor' : gb_r2
}

plt.bar(model_scores.keys(), model_scores.values(), color='skyblue', edgecolor='black')
plt.grid(linestyle='--', alpha=0.6)
plt.xlabel('Model')
plt.ylabel('r2 score')
plt.xticks(rotation=45)
plt.title('Model Scores')
plt.show()

