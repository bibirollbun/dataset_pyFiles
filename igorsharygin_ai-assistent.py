


import pandas as pd  
import numpy as np  

# Загрузка данных  
train_df = pd.read_csv('D:\\MLD\\kaggle\\Student Bag Price Prediction\\data\\train.csv')  
test_df = pd.read_csv('D:\\MLD\\kaggle\\Student Bag Price Prediction\\data\\test.csv')  

# Базовая информация о тренировочном датасете  
print("Информация о тренировочном датасете:")  
print(train_df.info())  
print("\nПервые 5 строк данных:")  
print(train_df.head())  
print("\nОписательная статистика:")  
print(train_df.describe())  
print("\nПропущенные значения:")  
print(train_df.isnull().sum())


# Анализ категориальных переменных  
categorical_columns = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']  

print("Уникальные значения в категориальных переменных:")  
for col in categorical_columns:  
    print(f"\n{col}:")  
    print(train_df[col].value_counts().head())  
    print(f"Всего уникальных значений: {train_df[col].nunique()}")  

# Проверка на выбросы в числовых переменных  
numeric_columns = ['Compartments', 'Weight Capacity (kg)', 'Price']  

import matplotlib.pyplot as plt  
import seaborn as sns  

plt.figure(figsize=(15, 5))  
for i, col in enumerate(numeric_columns, 1):  
    plt.subplot(1, 3, i)  
    sns.boxplot(y=train_df[col])  
    plt.title(f'Boxplot for {col}')  
plt.tight_layout()  
plt.show()  

# Корреляция между числовыми переменными  
correlation = train_df[numeric_columns].corr()  
print("\nКорреляционная матрица:")  
print(correlation)


# Создаем копию датафрейма  
train_processed = train_df.copy()  
test_processed = test_df.copy()  

# Обработка пропущенных значений  
# Для категориальных переменных создаем категорию 'Unknown'  
categorical_columns = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']  
for col in categorical_columns:  
    train_processed[col].fillna('Unknown', inplace=True)  
    test_processed[col].fillna('Unknown', inplace=True)  

# Для числовых переменных используем медиану  
numeric_columns = ['Weight Capacity (kg)']  
for col in numeric_columns:  
    median_value = train_processed[col].median()  
    train_processed[col].fillna(median_value, inplace=True)  
    test_processed[col].fillna(median_value, inplace=True)  

# Преобразование категориальных переменных  
from sklearn.preprocessing import LabelEncoder  

# Создаем словарь для хранения энкодеров  
encoders = {}  

# Применяем LabelEncoder к каждой категориальной переменной  
for col in categorical_columns:  
    encoders[col] = LabelEncoder()  
    train_processed[col] = encoders[col].fit_transform(train_processed[col])  
    test_processed[col] = encoders[col].transform(test_processed[col])  

# Проверяем результат  
print("Проверка на пропущенные значения после обработки:")  
print(train_processed.isnull().sum())  

print("\nПример преобразованных данных (первые 5 строк):")  
print(train_processed.head())


import pandas as pd  
import numpy as np  
import seaborn as sns  
import matplotlib.pyplot as plt  
from sklearn.preprocessing import StandardScaler  

# Создаем список признаков для анализа  
features = ['Compartments', 'Weight Capacity (kg)', 'Brand', 'Material',   
           'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']  

# Масштабируем числовые признаки для лучшего анализа  
scaler = StandardScaler()  
numeric_features = ['Compartments', 'Weight Capacity (kg)']  
train_processed[numeric_features] = scaler.fit_transform(train_processed[numeric_features])  

# Анализ корреляций  
plt.figure(figsize=(12, 8))  
correlation_matrix = train_processed[features + ['Price']].corr()  
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)  
plt.title('Корреляционная матрица')  
plt.tight_layout()  
plt.show()  

# Анализ важности признаков через средние цены  
categorical_features = ['Brand', 'Material', 'Size', 'Style', 'Color']  
fig, axes = plt.subplots(2, 3, figsize=(15, 10))  
axes = axes.ravel()  

for idx, feature in enumerate(categorical_features):  
    if idx < len(axes):  
        avg_price = train_processed.groupby(feature)['Price'].mean().sort_values(ascending=False)  
        avg_price.plot(kind='bar', ax=axes[idx])  
        axes[idx].set_title(f'Средняя цена по {feature}')  
        axes[idx].tick_params(axis='x', rotation=45)  

plt.tight_layout()  
plt.show()  

# Анализ распределения цен  
plt.figure(figsize=(10, 6))  
sns.histplot(data=train_processed, x='Price', bins=50)  
plt.title('Распределение цен')  
plt.show()  

# Анализ числовых признаков  
fig, axes = plt.subplots(1, 2, figsize=(15, 5))  

sns.scatterplot(data=train_processed, x='Compartments', y='Price', ax=axes[0])  
axes[0].set_title('Зависимость цены от количества отделений')  

sns.scatterplot(data=train_processed, x='Weight Capacity (kg)', y='Price', ax=axes[1])  
axes[1].set_title('Зависимость цены от грузоподъемности')  

plt.tight_layout()  
plt.show()  

# Статистический анализ  
print("\nСтатистика по ценам для разных категорий:")  
for feature in categorical_features:  
    print(f"\nСредняя цена по {feature}:")  
    print(train_processed.groupby(feature)['Price'].mean().sort_values(ascending=False))  
    print("\nСтандартное отклонение:")  
    print(train_processed.groupby(feature)['Price'].std().sort_values(ascending=False))


# Создаем копию датафрейма для новых признаков  
train_enhanced = train_processed.copy()  
test_enhanced = test_processed.copy()  

# 1. Создаем взаимодействия между признаками  
train_enhanced['weight_per_compartment'] = train_enhanced['Weight Capacity (kg)'] / (train_enhanced['Compartments'] + 1)  
test_enhanced['weight_per_compartment'] = test_enhanced['Weight Capacity (kg)'] / (test_enhanced['Compartments'] + 1)  

# 2. Создаем бинарные признаки для премиум характеристик  
train_enhanced['is_premium'] = ((train_enhanced['Waterproof'] == 1) &   
                              (train_enhanced['Laptop Compartment'] == 1)).astype(int)  
test_enhanced['is_premium'] = ((test_enhanced['Waterproof'] == 1) &   
                             (test_enhanced['Laptop Compartment'] == 1)).astype(int)  

# 3. Создаем категории по вместимости  
train_enhanced['capacity_category'] = pd.qcut(train_enhanced['Weight Capacity (kg)'],   
                                            q=5, labels=['Very Small', 'Small', 'Medium', 'Large', 'Very Large'])  
test_enhanced['capacity_category'] = pd.qcut(test_enhanced['Weight Capacity (kg)'],   
                                           q=5, labels=['Very Small', 'Small', 'Medium', 'Large', 'Very Large'])  

# 4. Нормализация числовых признаков  
numeric_features = ['Compartments', 'Weight Capacity (kg)', 'weight_per_compartment']  
scaler = StandardScaler()  
train_enhanced[numeric_features] = scaler.fit_transform(train_enhanced[numeric_features])  
test_enhanced[numeric_features] = scaler.transform(test_enhanced[numeric_features])  

# Проверяем корреляции новых признаков с ценой  
correlations = train_enhanced[numeric_features + ['Price']].corr()['Price'].sort_values(ascending=False)  
print("Корреляции с ценой:")  
print(correlations)  

# Базовая статистика по новым признакам  
print("\nСтатистика по новым признакам:")  
print(train_enhanced[['weight_per_compartment', 'is_premium']].describe())  

# Средние цены по категориям вместимости  
print("\nСредние цены по категориям вместимости:")  
print(train_enhanced.groupby('capacity_category')['Price'].mean().sort_values(ascending=False))


import numpy as np  
from sklearn.preprocessing import PolynomialFeatures  
from sklearn.feature_selection import SelectKBest, f_regression  
from sklearn.ensemble import RandomForestRegressor  

# Подготовка данных для анализа  
numeric_features = ['Compartments', 'Weight Capacity (kg)', 'weight_per_compartment']  
categorical_features = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']  

# 1. Создаем полиномиальные признаки для числовых переменных  
poly = PolynomialFeatures(degree=2, include_bias=False)  
numeric_poly = poly.fit_transform(train_enhanced[numeric_features])  
numeric_poly_df = pd.DataFrame(numeric_poly, columns=poly.get_feature_names_out(numeric_features))  

# 2. Создаем комбинированные категориальные признаки  
train_enhanced['brand_material'] = train_enhanced['Brand'].astype(str) + '_' + train_enhanced['Material'].astype(str)  
train_enhanced['size_style'] = train_enhanced['Size'].astype(str) + '_' + train_enhanced['Style'].astype(str)  

# 3. Оценка важности признаков с помощью Random Forest  
X = pd.concat([  
    train_enhanced[numeric_features],  
    train_enhanced[categorical_features]  
], axis=1)  

rf = RandomForestRegressor(n_estimators=100, random_state=42)  
rf.fit(X, train_enhanced['Price'])  

# Важность признаков  
feature_importance = pd.DataFrame({  
    'feature': X.columns,  
    'importance': rf.feature_importances_  
}).sort_values('importance', ascending=False)  

print("Важность признаков (Random Forest):")  
print(feature_importance)  

# 4. Анализ нелинейных зависимостей  
plt.figure(figsize=(15, 5))  

# График зависимости цены от веса  
plt.subplot(131)  
plt.scatter(train_enhanced['Weight Capacity (kg)'], train_enhanced['Price'], alpha=0.1)  
plt.title('Цена vs Вес')  
plt.xlabel('Вес')  
plt.ylabel('Цена')  

# График зависимости цены от количества отделений  
plt.subplot(132)  
plt.scatter(train_enhanced['Compartments'], train_enhanced['Price'], alpha=0.1)  
plt.title('Цена vs Отделения')  
plt.xlabel('Отделения')  
plt.ylabel('Цена')  

# График зависимости цены от веса на отделение  
plt.subplot(133)  
plt.scatter(train_enhanced['weight_per_compartment'], train_enhanced['Price'], alpha=0.1)  
plt.title('Цена vs Вес/Отделение')  
plt.xlabel('Вес/Отделение')  
plt.ylabel('Цена')  

plt.tight_layout()  
plt.show()  

# 5. Статистика по комбинированным признакам  
print("\nТоп-10 комбинаций бренд-материал по средней цене:")  
print(train_enhanced.groupby('brand_material')['Price']  
      .agg(['mean', 'count', 'std'])  
      .sort_values('mean', ascending=False)  
      .head(10))  

print("\nТоп-10 комбинаций размер-стиль по средней цене:")  
print(train_enhanced.groupby('size_style')['Price']  
      .agg(['mean', 'count', 'std'])  
      .sort_values('mean', ascending=False)  
      .head(10))


import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split  
from lightgbm import LGBMRegressor, early_stopping  
from sklearn.metrics import mean_squared_error, r2_score  
import matplotlib.pyplot as plt  

# Загрузка данных  
train_df = pd.read_csv('D:\\MLD\\kaggle\\Student Bag Price Prediction\\data\\train.csv')  
test_df = pd.read_csv('D:\\MLD\\kaggle\\Student Bag Price Prediction\\data\\test.csv')  

# Создаем копии для обработки  
train_processed = train_df.copy()  
test_processed = test_df.copy()  

# Обработка категориальных переменных  
categorical_columns = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']  
for col in categorical_columns:  
    train_processed[col] = pd.Categorical(train_processed[col]).codes  
    test_processed[col] = pd.Categorical(test_processed[col]).codes  

# Создание дополнительных признаков  
def create_features(df):  
    df['weight_per_compartment'] = df['Weight Capacity (kg)'] / (df['Compartments'] + 1)  
    df['is_premium'] = ((df['Waterproof'] == 1) & (df['Laptop Compartment'] == 1)).astype(int)  
    return df  

train_processed = create_features(train_processed)  
test_processed = create_features(test_processed)  

# Подготовка финального набора признаков  
final_features = [  
    'Weight Capacity (kg)',  
    'weight_per_compartment',  
    'Compartments',  
    'Brand',  
    'Material',  
    'Size',  
    'Style',  
    'Color',  
    'Laptop Compartment',  
    'Waterproof',  
    'is_premium'  
]  

X = train_processed[final_features]  
y = train_processed['Price']  

# Разделение на обучающую и валидационную выборки  
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)  

# Настройка и обучение модели  
model = LGBMRegressor(  
    n_estimators=1000,  
    learning_rate=0.05,  
    num_leaves=31,  
    min_child_samples=20,  
    random_state=42  
)  

# Обучение модели с eval_set  
model.fit(  
    X_train, y_train,  
    eval_set=[(X_val, y_val)],  
    eval_metric='rmse',  
    callbacks=[early_stopping(stopping_rounds=50)]  
)  

# Оценка качества модели  
y_pred = model.predict(X_val)  
mse = mean_squared_error(y_val, y_pred)  
rmse = np.sqrt(mse)  
r2 = r2_score(y_val, y_pred)  

print(f"\nРезультаты модели:")  
print(f"RMSE: {rmse:.2f}")  
print(f"R2 score: {r2:.4f}")  

# Важность признаков в итоговой модели  
feature_importance = pd.DataFrame({  
    'feature': final_features,  
    'importance': model.feature_importances_  
}).sort_values('importance', ascending=False)  

print("\nВажность признаков в итоговой модели:")  
print(feature_importance)  

# Анализ ошибок  
errors = y_val - y_pred  
plt.figure(figsize=(15, 5))  

# Распределение ошибок  
plt.subplot(131)  
plt.hist(errors, bins=50)  
plt.title('Распределение ошибок')  
plt.xlabel('Ошибка')  
plt.ylabel('Частота')  

# Предсказанные vs реальные значения  
plt.subplot(132)  
plt.scatter(y_val, y_pred, alpha=0.1)  
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)  
plt.title('Предсказанные vs Реальные значения')  
plt.xlabel('Реальные значения')  
plt.ylabel('Предсказанные значения')  

# Ошибки vs предсказанные значения  
plt.subplot(133)  
plt.scatter(y_pred, errors, alpha=0.1)  
plt.axhline(y=0, color='r', linestyle='--')  
plt.title('Ошибки vs Предсказанные значения')  
plt.xlabel('Предсказанные значения')  
plt.ylabel('Ошибка')  

plt.tight_layout()  
plt.show()  

# Подготовка предсказаний для тестового набора  
X_test = test_processed[final_features]  
test_predictions = model.predict(X_test)  

# Создание файла с предсказаниями  
submission = pd.DataFrame({  
    'id': test_df['id'],  
    'Price': test_predictions  
})  

# Сохранение предсказаний  
submission.to_csv('submission.csv', index=False)  

print("\nФайл с предсказаниями сохранен как 'submission.csv'")  

# Дополнительная информация о модели  
print("\nДополнительная статистика:")  
print(f"Среднее абсолютное отклонение: {np.mean(np.abs(errors)):.2f}")  
print(f"Медианное абсолютное отклонение: {np.median(np.abs(errors)):.2f}")  
print(f"Максимальная ошибка: {np.max(np.abs(errors)):.2f}")  
print(f"95-й перцентиль ошибки: {np.percentile(np.abs(errors), 95):.2f}")  

# Анализ предсказаний  
print("\nСтатистика предсказаний:")  
print(f"Минимальная предсказанная цена: {min(test_predictions):.2f}")  
print(f"Максимальная предсказанная цена: {max(test_predictions):.2f}")  
print(f"Средняя предсказанная цена: {np.mean(test_predictions):.2f}")  
print(f"Медианная предсказанная цена: {np.median(test_predictions):.2f}")


import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split  
from lightgbm import LGBMRegressor  
from sklearn.preprocessing import StandardScaler  
from sklearn.metrics import mean_squared_error, r2_score  
import matplotlib.pyplot as plt  

# Загрузка данных  
train_df = pd.read_csv('D:\\MLD\\kaggle\\Student Bag Price Prediction\\data\\train.csv')  
test_df = pd.read_csv('D:\\MLD\\kaggle\\Student Bag Price Prediction\\data\\test.csv')  

# Создаем копии для обработки  
train_processed = train_df.copy()  
test_processed = test_df.copy()  

# Обработка категориальных переменных  
categorical_columns = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']  
for col in categorical_columns:  
    train_processed[col] = pd.Categorical(train_processed[col]).codes  
    test_processed[col] = pd.Categorical(test_processed[col]).codes  

# Расширенное создание признаков  
def create_advanced_features(df):  
    # Базовые признаки  
    df['weight_per_compartment'] = df['Weight Capacity (kg)'] / (df['Compartments'] + 1)  
    df['is_premium'] = ((df['Waterproof'] == 1) & (df['Laptop Compartment'] == 1)).astype(int)  
    
    # Новые признаки  
    df['total_features'] = df['Laptop Compartment'] + df['Waterproof']  
    df['weight_category'] = pd.qcut(df['Weight Capacity (kg)'], q=5, labels=False)  
    
    # Взаимодействия признаков  
    df['size_weight'] = df['Size'] * df['Weight Capacity (kg)']  
    df['brand_material'] = df['Brand'] * 10 + df['Material']  
    
    # Нелинейные преобразования  
    df['weight_squared'] = df['Weight Capacity (kg)'] ** 2  
    df['compartments_squared'] = df['Compartments'] ** 2  
    
    return df  

# Применяем расширенные признаки  
train_processed = create_advanced_features(train_processed)  
test_processed = create_advanced_features(test_processed)  

# Обновленный набор признаков  
final_features = [  
    'Weight Capacity (kg)', 'weight_per_compartment', 'Compartments',  
    'Brand', 'Material', 'Size', 'Style', 'Color',  
    'Laptop Compartment', 'Waterproof', 'is_premium',  
    'total_features', 'weight_category', 'size_weight',  
    'brand_material', 'weight_squared', 'compartments_squared'  
]  

# Нормализация числовых признаков  
numeric_features = ['Weight Capacity (kg)', 'weight_per_compartment', 'Compartments',  
                   'size_weight', 'weight_squared', 'compartments_squared']  
scaler = StandardScaler()  
train_processed[numeric_features] = scaler.fit_transform(train_processed[numeric_features])  
test_processed[numeric_features] = scaler.transform(test_processed[numeric_features])  

X = train_processed[final_features]  
y = train_processed['Price']  

# Разделение на обучающую и валидационную выборки  
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)  

# Настройка улучшенной модели  
model = LGBMRegressor(  
    n_estimators=2000,  
    learning_rate=0.01,  
    num_leaves=63,  
    min_child_samples=10,  
    subsample=0.8,  
    colsample_bytree=0.8,  
    random_state=42  
)  

# Обучение модели  
model.fit(X_train, y_train)  

# Оценка качества модели  
y_pred = model.predict(X_val)  
mse = mean_squared_error(y_val, y_pred)  
rmse = np.sqrt(mse)  
r2 = r2_score(y_val, y_pred)  

print(f"\nРезультаты улучшенной модели:")  
print(f"RMSE: {rmse:.2f}")  
print(f"R2 score: {r2:.4f}")  

# Важность признаков  
feature_importance = pd.DataFrame({  
    'feature': final_features,  
    'importance': model.feature_importances_  
}).sort_values('importance', ascending=False)  

print("\nВажность признаков в улучшенной модели:")  
print(feature_importance)  

# Подготовка предсказаний для тестового набора  
X_test = test_processed[final_features]  
test_predictions = model.predict(X_test)  

# Создание файла с предсказаниями  
submission = pd.DataFrame({  
    'id': test_df['id'],  
    'Price': test_predictions  
})  

# Сохранение предсказаний  
submission.to_csv('submission_improved.csv', index=False)  

print("\nСтатистика предсказаний:")  
print(f"Минимальная предсказанная цена: {min(test_predictions):.2f}")  
print(f"Максимальная предсказанная цена: {max(test_predictions):.2f}")  
print(f"Средняя предсказанная цена: {np.mean(test_predictions):.2f}")  
print(f"Медианная предсказанная цена: {np.median(test_predictions):.2f}")


import pandas as pd  
import numpy as np  
from sklearn.model_selection import KFold  
from sklearn.preprocessing import RobustScaler  
from lightgbm import LGBMRegressor  
from xgboost import XGBRegressor  
from sklearn.ensemble import RandomForestRegressor  
from sklearn.metrics import mean_squared_error, r2_score  

# Загрузка данных  
train_df = pd.read_csv('D:\\MLD\\kaggle\\Student Bag Price Prediction\\data\\train.csv')  
test_df = pd.read_csv('D:\\MLD\\kaggle\\Student Bag Price Prediction\\data\\test.csv')  

def create_features(df):  
    df = df.copy()  
    
    # Базовые признаки  
    df['weight_per_compartment'] = df['Weight Capacity (kg)'] / (df['Compartments'] + 1)  
    df['total_features'] = df['Laptop Compartment'] + df['Waterproof']  
    
    # Взаимодействия признаков  
    df['size_weight'] = df['Size'] * df['Weight Capacity (kg)']  
    df['brand_material'] = df['Brand'] * 10 + df['Material']  
    
    # Нелинейные преобразования  
    df['weight_squared'] = df['Weight Capacity (kg)'] ** 2  
    df['compartments_squared'] = df['Compartments'] ** 2  
    
    return df  

def preprocess_data(df):  
    df_processed = df.copy()  
    
    # Обработка категориальных переменных  
    categorical_columns = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']  
    
    # Label encoding для категориальных переменных  
    for col in categorical_columns:  
        df_processed[col] = pd.Categorical(df_processed[col]).codes  
    
    # Создание дополнительных признаков  
    df_processed = create_features(df_processed)  
    
    return df_processed  

# Предобработка данных  
train_processed = preprocess_data(train_df)  
test_processed = preprocess_data(test_df)  

# Определение признаков  
numeric_features = [  
    'Weight Capacity (kg)',   
    'Compartments',  
    'weight_per_compartment',  
    'total_features',  
    'size_weight',  
    'brand_material',  
    'weight_squared',  
    'compartments_squared'  
]  

categorical_features = [  
    'Brand',  
    'Material',  
    'Size',  
    'Style',  
    'Color',  
    'Laptop Compartment',  
    'Waterproof'  
]  

final_features = numeric_features + categorical_features  

# Нормализация числовых признаков  
scaler = RobustScaler()  
train_processed[numeric_features] = scaler.fit_transform(train_processed[numeric_features])  
test_processed[numeric_features] = scaler.transform(test_processed[numeric_features])  

X = train_processed[final_features]  
y = train_processed['Price']  

# Настройка моделей  
lgb_params = {  
    'n_estimators': 2000,  
    'learning_rate': 0.01,  
    'num_leaves': 31,  
    'subsample': 0.8,  
    'colsample_bytree': 0.8,  
    'random_state': 42  
}  

xgb_params = {  
    'n_estimators': 2000,  
    'learning_rate': 0.01,  
    'max_depth': 7,  
    'subsample': 0.8,  
    'colsample_bytree': 0.8,  
    'random_state': 42  
}  

rf_params = {  
    'n_estimators': 200,  
    'max_depth': 15,  
    'min_samples_split': 5,  
    'random_state': 42  
}  

# Кросс-валидация  
n_folds = 5  
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)  
predictions = np.zeros(len(test_processed))  
scores = []  

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):  
    print(f"\nFold {fold + 1}")  
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]  
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]  
    
    # Обучение моделей  
    lgb_model = LGBMRegressor(**lgb_params)  
    xgb_model = XGBRegressor(**xgb_params)  
    rf_model = RandomForestRegressor(**rf_params)  
    
    lgb_model.fit(X_train, y_train)  
    xgb_model.fit(X_train, y_train)  
    rf_model.fit(X_train, y_train)  
    
    # Предсказания  
    lgb_pred = lgb_model.predict(X_val)  
    xgb_pred = xgb_model.predict(X_val)  
    rf_pred = rf_model.predict(X_val)  
    
    # Взвешенное среднее  
    fold_pred = 0.4 * lgb_pred + 0.4 * xgb_pred + 0.2 * rf_pred  
    
    # Оценка качества  
    fold_score = r2_score(y_val, fold_pred)  
    fold_rmse = np.sqrt(mean_squared_error(y_val, fold_pred))  
    scores.append((fold_score, fold_rmse))  
    
    # Предсказания для тестового набора  
    lgb_test = lgb_model.predict(test_processed[final_features])  
    xgb_test = xgb_model.predict(test_processed[final_features])  
    rf_test = rf_model.predict(test_processed[final_features])  
    
    predictions += (0.4 * lgb_test + 0.4 * xgb_test + 0.2 * rf_test) / n_folds  

# Вывод результатов  
print("\nРезультаты кросс-валидации:")  
for fold, (r2, rmse) in enumerate(scores, 1):  
    print(f"Fold {fold}: R2 = {r2:.4f}, RMSE = {rmse:.2f}")  

print(f"\nСреднее R2: {np.mean([s[0] for s in scores]):.4f}")  
print(f"Среднее RMSE: {np.mean([s[1] for s in scores]):.2f}")  

# Анализ распределения предсказаний  
print("\nСтатистика предсказаний:")  
print(f"Минимальная предсказанная цена: {min(predictions):.2f}")  
print(f"Максимальная предсказанная цена: {max(predictions):.2f}")  
print(f"Средняя предсказанная цена: {np.mean(predictions):.2f}")  
print(f"Медианная предсказанная цена: {np.median(predictions):.2f}")  

# Анализ важности признаков  
feature_importance = pd.DataFrame({  
    'feature': final_features,  
    'importance': lgb_model.feature_importances_  
}).sort_values('importance', ascending=False)  

print("\nВажность признаков:")  
print(feature_importance)  

# Создание файла с предсказаниями  
submission = pd.DataFrame({  
    'id': test_df['id'],  
    'Price': predictions  
})  

submission.to_csv('submission_ensemble.csv', index=False)  
print("\nФайл с предсказаниями сохранен как 'submission_ensemble.csv'")  

# Дополнительный анализ данных  
print("\nСтатистика обучающего набора:")  
print(f"Среднее значение цены: {train_df['Price'].mean():.2f}")  
print(f"Медиана цены: {train_df['Price'].median():.2f}")  
print(f"Стандартное отклонение цены: {train_df['Price'].std():.2f}")


import pandas as pd  
import numpy as np  
import lightgbm as lgb  
from sklearn.model_selection import train_test_split  
from sklearn.impute import SimpleImputer  

# Загрузка данных  
train_df = pd.read_csv('D:\\MLD\\kaggle\\Student Bag Price Prediction\\data\\train.csv')  
test_df = pd.read_csv('D:\\MLD\\kaggle\\Student Bag Price Prediction\\data\\test.csv')  

def preprocess_data(df, imputers=None, is_training=True):  
    df = df.copy()  
    
    # Категориальные и числовые признаки  
    cat_features = ['Brand', 'Material', 'Size', 'Style', 'Color', 'Laptop Compartment', 'Waterproof']  
    num_features = ['Weight Capacity (kg)', 'Compartments']  
    
    if is_training:  
        imputers = {  
            'categorical': SimpleImputer(strategy='most_frequent'),  
            'numerical': SimpleImputer(strategy='median')  
        }  
        
        # Заполнение пропущенных значений  
        df[cat_features] = imputers['categorical'].fit_transform(df[cat_features])  
        df[num_features] = imputers['numerical'].fit_transform(df[num_features])  
        
        # Преобразование категориальных признаков в числовые  
        for col in cat_features:  
            df[col] = pd.Categorical(df[col]).codes  
    else:  
        # Заполнение пропущенных значений  
        df[cat_features] = imputers['categorical'].transform(df[cat_features])  
        df[num_features] = imputers['numerical'].transform(df[num_features])  
        
        # Преобразование категориальных признаков в числовые  
        for col in cat_features:  
            df[col] = pd.Categorical(df[col]).codes  
    
    return df, imputers  

# Предобработка данных  
train_processed, imputers = preprocess_data(train_df, is_training=True)  
test_processed, _ = preprocess_data(test_df, imputers, is_training=False)  

# Создание новых признаков  
def add_features(df):  
    df = df.copy()  
    # Взаимодействие числовых признаков  
    df['Weight_per_Compartment'] = df['Weight Capacity (kg)'] / df['Compartments']  
    
    # Квадратичные признаки  
    df['Weight_Squared'] = df['Weight Capacity (kg)'] ** 2  
    df['Compartments_Squared'] = df['Compartments'] ** 2  
    
    return df  

train_processed = add_features(train_processed)  
test_processed = add_features(test_processed)  

# Подготовка данных для модели  
feature_columns = ['Weight Capacity (kg)', 'Compartments',  
                  'Brand', 'Material', 'Size', 'Style', 'Color',   
                  'Laptop Compartment', 'Waterproof',  
                  'Weight_per_Compartment', 'Weight_Squared', 'Compartments_Squared']  

X = train_processed[feature_columns]  
y = train_processed['Price']  

# Разделение на обучающую и валидационную выборки  
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)  

# Создание модели LightGBM  
model = lgb.LGBMRegressor(  
    objective='regression',  
    metric='rmse',  
    num_leaves=31,  
    learning_rate=0.05,  
    feature_fraction=0.9,  
    bagging_fraction=0.8,  
    bagging_freq=5,  
    n_estimators=1000,  
    early_stopping_rounds=50,  
    verbose=100  
)  

# Обучение модели  
print("Обучение модели...")  
model.fit(  
    X_train, y_train,  
    eval_set=[(X_val, y_val)],  
    eval_metric='rmse',  
    callbacks=[  
        lgb.early_stopping(stopping_rounds=50),  
        lgb.log_evaluation(period=100)  
    ]  
)  

# Оценка модели  
val_predictions = model.predict(X_val)  
val_mse = np.mean((y_val - val_predictions) ** 2)  
val_rmse = np.sqrt(val_mse)  
val_r2 = 1 - np.sum((y_val - val_predictions) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2)  

print("\nРезультаты на валидационной выборке:")  
print(f"RMSE: {val_rmse:.2f}")  
print(f"R2: {val_r2:.4f}")  

# Анализ важности признаков  
feature_importance = pd.DataFrame({  
    'feature': feature_columns,  
    'importance': model.feature_importances_  
})  
print("\nВажность признаков:")  
print(feature_importance.sort_values('importance', ascending=False))  

# Предсказания для тестового набора  
test_predictions = model.predict(test_processed[feature_columns])  

# Проверка и корректировка предсказаний  
test_predictions = np.clip(test_predictions, 15, 150)  # Ограничиваем предсказания диапазоном из обучающей выборки  

# Создание файла с предсказаниями  
submission = pd.DataFrame({  
    'id': test_df['id'],  
    'Price': test_predictions  
})  

submission.to_csv('submission_lightgbm.csv', index=False)  

print("\nСтатистика предсказаний:")  
print(f"Минимальная предсказанная цена: {np.min(test_predictions):.2f}")  
print(f"Максимальная предсказанная цена: {np.max(test_predictions):.2f}")  
print(f"Средняя предсказанная цена: {np.mean(test_predictions):.2f}")  
print(f"Медианная предсказанная цена: {np.median(test_predictions):.2f}")  

# Анализ ошибок  
residuals = y_val - val_predictions  
print("\nАнализ ошибок:")  
print(f"Средняя ошибка: {np.mean(residuals):.2f}")  
print(f"Стандартное отклонение ошибки: {np.std(residuals):.2f}")  
print(f"Медиана ошибки: {np.median(residuals):.2f}")  

# Анализ ошибок по ценовым диапазонам  
bins = pd.qcut(y_val, q=5)  
error_by_price = pd.DataFrame({  
    'true_price': y_val,  
    'predicted_price': val_predictions,  
    'abs_error': np.abs(y_val - val_predictions),  
    'price_bin': bins  
})  

print("\nСредняя абсолютная ошибка по ценовым диапазонам:")  
print(error_by_price.groupby('price_bin')['abs_error'].mean())  

# Анализ выбросов в предсказаниях  
large_errors = error_by_price[error_by_price['abs_error'] > error_by_price['abs_error'].quantile(0.95)]  
print("\nАнализ 5% наихудших предсказаний:")  
print(large_errors.describe())  

# Сохранение важных признаков для дальнейшего анализа  
feature_importance.to_csv('feature_importance.csv', index=False)


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
/kaggle/input/playground-series-s5e2/sample_submission.csv
/kaggle/input/playground-series-s5e2/train.csv
/kaggle/input/playground-series-s5e2/test.csv
/kaggle/input/playground-series-s5e2/training_extra.csv
train_df = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
import tensorflow as tf  
import numpy as np  
import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import LabelEncoder, StandardScaler  
from sklearn.impute import SimpleImputer  
from tensorflow.keras import layers, models, mixed_precision  

# Проверка доступности GPU  
print("TensorFlow видит следующие устройства:")  
print(tf.config.list_physical_devices())  
print("\nGPU доступны:")  
print(tf.config.list_physical_devices('GPU'))  
print("\nТекущее устройство по умолчанию:", tf.test.gpu_device_name())  

# Настройка GPU  
gpus = tf.config.experimental.list_physical_devices('GPU')  
if gpus:  
    try:  
        for gpu in gpus:  
            tf.config.experimental.set_memory_growth(gpu, True)  
        # Включаем mixed precision  
        mixed_precision.set_global_policy('mixed_float16')  
    except RuntimeError as e:  
        print(e)  

def preprocess_data(df, encoders=None, scalers=None, imputers=None, is_training=True):  
    df = df.copy()  
    
    # Категориальные и числовые признаки  
    cat_features = ['Brand', 'Material', 'Size', 'Style', 'Color', 'Laptop Compartment', 'Waterproof']  
    num_features = ['Weight Capacity (kg)', 'Compartments']  
    
    if is_training:  
        encoders = {}  
        scalers = {}  
        imputers = {  
            'categorical': SimpleImputer(strategy='most_frequent'),  
            'numerical': SimpleImputer(strategy='median')  
        }  
        
        # Заполнение пропущенных значений  
        df[cat_features] = imputers['categorical'].fit_transform(df[cat_features])  
        df[num_features] = imputers['numerical'].fit_transform(df[num_features])  
        
        # Label Encoding для категориальных признаков  
        for col in cat_features:  
            encoders[col] = LabelEncoder()  
            df[col] = encoders[col].fit_transform(df[col])  
            
        # Нормализация числовых признаков  
        scaler = StandardScaler()  
        df[num_features] = scaler.fit_transform(df[num_features])  
        scalers['numeric'] = scaler  
        
        # Нормализация целевой переменной  
        if 'Price' in df.columns:  
            price_scaler = StandardScaler()  
            df['Price'] = price_scaler.fit_transform(df[['Price']])  
            scalers['Price'] = price_scaler  
    else:  
        # Заполнение пропущенных значений  
        df[cat_features] = imputers['categorical'].transform(df[cat_features])  
        df[num_features] = imputers['numerical'].transform(df[num_features])  
        
        # Применение существующих преобразований  
        for col in cat_features:  
            df[col] = encoders[col].transform(df[col])  
        df[num_features] = scalers['numeric'].transform(df[num_features])  
    
    return df, encoders, scalers, imputers  

# Предобработка данных  
train_processed, encoders, scalers, imputers = preprocess_data(train_df, is_training=True)  
test_processed, _, _, _ = preprocess_data(test_df, encoders, scalers, imputers, is_training=False)  

# Подготовка данных для модели  
feature_columns = ['Weight Capacity (kg)', 'Compartments',  
                  'Brand', 'Material', 'Size', 'Style', 'Color',   
                  'Laptop Compartment', 'Waterproof']  

X = train_processed[feature_columns].values  
y = train_processed['Price'].values  

# Разделение на обучающую и валидационную выборки  
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)  

# Создание tf.data.Dataset  
BATCH_SIZE = 512  
AUTOTUNE = tf.data.AUTOTUNE  

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))\
    .batch(BATCH_SIZE)\
    .prefetch(AUTOTUNE)  

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))\
    .batch(BATCH_SIZE)\
    .prefetch(AUTOTUNE)  

# Создание модели  
def create_model(input_dim):  
    model = models.Sequential([  
        layers.Input(shape=(input_dim,)),  
        layers.Dense(64, activation='relu', kernel_initializer='he_normal',  
                    kernel_regularizer=tf.keras.regularizers.l2(0.01)),  
        layers.BatchNormalization(),  
        layers.Dropout(0.2),  
        
        layers.Dense(32, activation='relu', kernel_initializer='he_normal',  
                    kernel_regularizer=tf.keras.regularizers.l2(0.01)),  
        layers.BatchNormalization(),  
        
        layers.Dense(1, kernel_initializer='he_normal')  
    ])  
    
    return model  

# Создание и компиляция модели с использованием GPU  
with tf.device('/GPU:0'):  
    model = create_model(X_train.shape[1])  
    model.compile(  
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  
        loss='huber',  
        metrics=['mae']  
    )  

    # Обучение модели  
    history = model.fit(  
        train_dataset,  
        validation_data=val_dataset,  
        epochs=50,  
        callbacks=[  
            tf.keras.callbacks.EarlyStopping(  
                monitor='val_loss',  
                patience=10,  
                restore_best_weights=True  
            ),  
            tf.keras.callbacks.ReduceLROnPlateau(  
                monitor='val_loss',  
                factor=0.5,  
                patience=5,  
                min_lr=0.0001  
            )  
        ],  
        verbose=1  
    )  

# Оценка модели  
val_predictions = model.predict(X_val, batch_size=BATCH_SIZE)  
val_predictions_original = scalers['Price'].inverse_transform(val_predictions)  
y_val_original = scalers['Price'].inverse_transform(y_val.reshape(-1, 1))  

val_mse = np.mean((y_val_original - val_predictions_original) ** 2)  
val_rmse = np.sqrt(val_mse)  
val_r2 = 1 - np.sum((y_val_original - val_predictions_original) ** 2) / np.sum((y_val_original - np.mean(y_val_original)) ** 2)  

print("\nРезультаты на валидационной выборке:")  
print(f"RMSE: {val_rmse:.2f}")  
print(f"R2: {val_r2:.4f}")  

# Предсказания для тестового набора  
test_dataset = tf.data.Dataset.from_tensor_slices(test_processed[feature_columns].values)\
    .batch(BATCH_SIZE)\
    .prefetch(AUTOTUNE)  

test_predictions = model.predict(test_dataset)  
test_predictions_original = scalers['Price'].inverse_transform(test_predictions)  

# Создание файла с предсказаниями  
submission = pd.DataFrame({  
    'id': test_df['id'],  
    'Price': test_predictions_original.flatten()  
})  

submission.to_csv('submission_nn.csv', index=False)  

print("\nСтатистика предсказаний:")  
print(f"Минимальная предсказанная цена: {np.min(test_predictions_original):.2f}")  
print(f"Максимальная предсказанная цена: {np.max(test_predictions_original):.2f}")  
print(f"Средняя предсказанная цена: {np.mean(test_predictions_original):.2f}")  
print(f"Медианная предсказанная цена: {np.median(test_predictions_original):.2f}")  

# Визуализация процесса обучения  
import matplotlib.pyplot as plt  

plt.figure(figsize=(12, 4))  

plt.subplot(1, 2, 1)  
plt.plot(history.history['loss'], label='Training Loss')  
plt.plot(history.history['val_loss'], label='Validation Loss')  
plt.title('Model Loss')  
plt.xlabel('Epoch')  
plt.ylabel('Loss')  
plt.legend()  

plt.subplot(1, 2, 2)  
plt.plot(history.history['mae'], label='Training MAE')  
plt.plot(history.history['val_mae'], label='Validation MAE')  
plt.title('Model MAE')  
plt.xlabel('Epoch')  
plt.ylabel('MAE')  
plt.legend()  

plt.tight_layout()  
plt.show()
TensorFlow видит следующие устройства:
[PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU'), PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]

GPU доступны:
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]

Текущее устройство по умолчанию: /device:GPU:0
Epoch 1/50
469/469 ━━━━━━━━━━━━━━━━━━━━ 10s 10ms/step - loss: 1.9140 - mae: 0.9860 - val_loss: 0.8224 - val_mae: 0.8704 - learning_rate: 0.0010
Epoch 2/50
469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 0.7149 - mae: 0.8730 - val_loss: 0.5228 - val_mae: 0.8645 - learning_rate: 0.0010
Epoch 3/50
469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 0.5074 - mae: 0.8698 - val_loss: 0.4719 - val_mae: 0.8645 - learning_rate: 0.0010
Epoch 4/50
469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 0.4722 - mae: 0.8688 - val_loss: 0.4639 - val_mae: 0.8640 - learning_rate: 0.0010
Epoch 5/50
469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 0.4665 - mae: 0.8683 - val_loss: 0.4640 - val_mae: 0.8654 - learning_rate: 0.0010
Epoch 6/50
469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 0.4655 - mae: 0.8681 - val_loss: 0.4640 - val_mae: 0.8654 - learning_rate: 0.0010
Epoch 7/50
469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 0.4653 - mae: 0.8681 - val_loss: 0.4630 - val_mae: 0.8646 - learning_rate: 0.0010
Epoch 8/50
469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 0.4649 - mae: 0.8678 - val_loss: 0.4637 - val_mae: 0.8654 - learning_rate: 0.0010
Epoch 9/50
469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 0.4648 - mae: 0.8679 - val_loss: 0.4618 - val_mae: 0.8635 - learning_rate: 0.0010
Epoch 10/50
469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 0.4647 - mae: 0.8677 - val_loss: 0.4622 - val_mae: 0.8640 - learning_rate: 0.0010
Epoch 11/50
469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 0.4645 - mae: 0.8676 - val_loss: 0.4618 - val_mae: 0.8638 - learning_rate: 0.0010
Epoch 12/50
469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 0.4644 - mae: 0.8676 - val_loss: 0.4617 - val_mae: 0.8637 - learning_rate: 0.0010
Epoch 13/50
469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 0.4643 - mae: 0.8676 - val_loss: 0.4618 - val_mae: 0.8638 - learning_rate: 0.0010
Epoch 14/50
469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 0.4643 - mae: 0.8676 - val_loss: 0.4613 - val_mae: 0.8634 - learning_rate: 0.0010
Epoch 15/50
469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 0.4642 - mae: 0.8676 - val_loss: 0.4608 - val_mae: 0.8629 - learning_rate: 0.0010
Epoch 16/50
469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 0.4641 - mae: 0.8676 - val_loss: 0.4610 - val_mae: 0.8632 - learning_rate: 0.0010
Epoch 17/50
469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 0.4639 - mae: 0.8674 - val_loss: 0.4606 - val_mae: 0.8628 - learning_rate: 0.0010
Epoch 18/50
469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 0.4640 - mae: 0.8675 - val_loss: 0.4607 - val_mae: 0.8630 - learning_rate: 0.0010
Epoch 19/50
469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 0.4638 - mae: 0.8674 - val_loss: 0.4603 - val_mae: 0.8626 - learning_rate: 0.0010
Epoch 20/50
469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 0.4637 - mae: 0.8673 - val_loss: 0.4601 - val_mae: 0.8625 - learning_rate: 0.0010
Epoch 21/50
469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 0.4636 - mae: 0.8673 - val_loss: 0.4600 - val_mae: 0.8624 - learning_rate: 0.0010
Epoch 22/50
469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 0.4636 - mae: 0.8673 - val_loss: 0.4602 - val_mae: 0.8626 - learning_rate: 0.0010
Epoch 23/50
469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 0.4636 - mae: 0.8673 - val_loss: 0.4600 - val_mae: 0.8625 - learning_rate: 0.0010
Epoch 24/50
469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 0.4635 - mae: 0.8672 - val_loss: 0.4600 - val_mae: 0.8625 - learning_rate: 0.0010
Epoch 25/50
469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 0.4635 - mae: 0.8673 - val_loss: 0.4600 - val_mae: 0.8625 - learning_rate: 0.0010
Epoch 26/50
469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 0.4634 - mae: 0.8672 - val_loss: 0.4599 - val_mae: 0.8625 - learning_rate: 0.0010
Epoch 27/50
469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 0.4633 - mae: 0.8672 - val_loss: 0.4598 - val_mae: 0.8623 - learning_rate: 0.0010
Epoch 28/50
469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 0.4633 - mae: 0.8672 - val_loss: 0.4598 - val_mae: 0.8624 - learning_rate: 0.0010
Epoch 29/50
469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 0.4632 - mae: 0.8671 - val_loss: 0.4597 - val_mae: 0.8623 - learning_rate: 0.0010
Epoch 30/50
469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 0.4632 - mae: 0.8671 - val_loss: 0.4597 - val_mae: 0.8623 - learning_rate: 0.0010
Epoch 31/50
469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 0.4632 - mae: 0.8671 - val_loss: 0.4598 - val_mae: 0.8623 - learning_rate: 0.0010
Epoch 32/50
469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 0.4632 - mae: 0.8671 - val_loss: 0.4600 - val_mae: 0.8625 - learning_rate: 0.0010
Epoch 33/50
469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 0.4630 - mae: 0.8670 - val_loss: 0.4597 - val_mae: 0.8623 - learning_rate: 5.0000e-04
Epoch 34/50
469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 0.4631 - mae: 0.8670 - val_loss: 0.4597 - val_mae: 0.8622 - learning_rate: 5.0000e-04
Epoch 35/50
469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 0.4630 - mae: 0.8670 - val_loss: 0.4597 - val_mae: 0.8623 - learning_rate: 5.0000e-04
Epoch 36/50
469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 0.4630 - mae: 0.8670 - val_loss: 0.4597 - val_mae: 0.8623 - learning_rate: 5.0000e-04
Epoch 37/50
469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 0.4630 - mae: 0.8670 - val_loss: 0.4599 - val_mae: 0.8624 - learning_rate: 5.0000e-04
Epoch 38/50
469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 0.4631 - mae: 0.8670 - val_loss: 0.4597 - val_mae: 0.8623 - learning_rate: 5.0000e-04
Epoch 39/50
469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 0.4630 - mae: 0.8670 - val_loss: 0.4596 - val_mae: 0.8622 - learning_rate: 2.5000e-04
Epoch 40/50
469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 0.4630 - mae: 0.8669 - val_loss: 0.4597 - val_mae: 0.8622 - learning_rate: 2.5000e-04
Epoch 41/50
469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 0.4630 - mae: 0.8669 - val_loss: 0.4596 - val_mae: 0.8622 - learning_rate: 2.5000e-04
Epoch 42/50
469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 0.4630 - mae: 0.8670 - val_loss: 0.4596 - val_mae: 0.8622 - learning_rate: 2.5000e-04
Epoch 43/50
469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 0.4630 - mae: 0.8669 - val_loss: 0.4596 - val_mae: 0.8622 - learning_rate: 2.5000e-04
Epoch 44/50
469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 0.4629 - mae: 0.8669 - val_loss: 0.4596 - val_mae: 0.8622 - learning_rate: 1.2500e-04
Epoch 45/50
469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 0.4629 - mae: 0.8669 - val_loss: 0.4596 - val_mae: 0.8622 - learning_rate: 1.2500e-04
Epoch 46/50
469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 0.4629 - mae: 0.8669 - val_loss: 0.4596 - val_mae: 0.8622 - learning_rate: 1.2500e-04
Epoch 47/50
469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 0.4629 - mae: 0.8669 - val_loss: 0.4596 - val_mae: 0.8622 - learning_rate: 1.2500e-04
Epoch 48/50
469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 0.4629 - mae: 0.8669 - val_loss: 0.4596 - val_mae: 0.8622 - learning_rate: 1.2500e-04
Epoch 49/50
469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 0.4629 - mae: 0.8669 - val_loss: 0.4596 - val_mae: 0.8622 - learning_rate: 1.0000e-04
Epoch 50/50
469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - loss: 0.4629 - mae: 0.8669 - val_loss: 0.4596 - val_mae: 0.8622 - learning_rate: 1.0000e-04
118/118 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step

Результаты на валидационной выборке:
RMSE: 38.94
R2: 0.0004
391/391 ━━━━━━━━━━━━━━━━━━━━ 2s 3ms/step

Статистика предсказаний:
Минимальная предсказанная цена: 76.25
Максимальная предсказанная цена: 81.88
Средняя предсказанная цена: 80.94
Медианная предсказанная цена: 81.88





