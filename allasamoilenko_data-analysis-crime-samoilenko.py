import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from warnings import filterwarnings
filterwarnings('ignore')


def load_and_analyze_data(filepath):
    df = pd.read_csv(filepath)
    df.info()
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print("\n=== Числові ознаки ===")
    print(numeric_cols)
    print("\n=== Категоріальні ознаки ===")
    print(categorical_cols)
    
    print("\n=== Пропущені значення ===")
    print(df.isnull().sum())
    
    return df, numeric_cols, categorical_cols


df, numeric_cols, categorical_cols = load_and_analyze_data('/kaggle/input/crime-cast-forecasting-crime-categories/train.csv')


def plot_numeric_features(df, numeric_cols):
    n_cols = int(sqrt(len(df)))
    for col in numeric_cols:
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        df[col].hist(bins=n_cols)
        plt.title(f'Гістограма {col}')
        
        skewness = df[col].skew()
        peaks = len(pd.cut(df[col], bins=n_cols).value_counts().nlargest(3))
        print(f"\nАналіз {col}:")
        print(f"- Викиди: {'так' if len(df[col][(df[col] - df[col].mean()).abs() > 3*df[col].std()]) > 0 else 'ні'}")
        print(f"- Кількість піків: {peaks}")
        print(f"- Зміщення: {'вліво' if skewness < 0 else 'вправо'} (коефіцієнт асиметрії: {skewness:.2f})")
        
        plt.subplot(1, 2, 2)
        plt.boxplot(df[col].dropna(), vert=False)
        plt.title(f'Ящик з вусами {col}')
        
        q1, q2, q3 = df[col].quantile([0.25, 0.5, 0.75])
        iqr = q3 - q1
        outliers = ((df[col] < (q1 - 1.5*iqr))) | ((df[col] > (q3 + 1.5*iqr))).sum()
        print(f"- 25% даних: до {q1:.2f}")
        print(f"- 50% даних: до {q2:.2f}")
        print(f"- 75% даних: до {q3:.2f}")
        print(f"- Викиди: {outliers} спостережень")
        
        plt.tight_layout()
        plt.show()

plot_numeric_features(df, numeric_cols)


# 3. Індивідуальна обробка викидів
def custom_outlier_handling(df):
    df_clean = df.copy()
    
    # Обробка Latitude - фільтрація значень < 30
    df_clean = df_clean[df_clean['Latitude'] >= 30]
    
    # Обробка Victim_Age - фільтрація значень поза діапазоном [10, 100]
    df_clean = df_clean[(df_clean['Victim_Age'] >= 10) & (df_clean['Victim_Age'] <= 100)]
    
    # Отримуємо список числових стовпців
    numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Визначаємо оптимальну кількість рядків і стовпців для subplot
    n_cols = 4  # Фіксована кількість стовпців
    n_rows = len(numeric_cols) // n_cols + (1 if len(numeric_cols) % n_cols != 0 else 0)
    
    # Візуалізація після обробки викидів
    plt.figure(figsize=(15, 5*n_rows))  # Динамічний розмір залежно від кількості рядків
    
    for i, column in enumerate(numeric_cols, 1):
        plt.subplot(n_rows, n_cols, i)
        plt.boxplot(df_clean[column].dropna(), vert=False, patch_artist=True)
        plt.title(f'Boxplot: {column}', pad=10)
        plt.xlabel(column)
    
    plt.tight_layout()
    plt.show()
    
    return df_clean

# Застосування індивідуальної обробки викидів
df_clean = custom_outlier_handling(df)

# Перевірка результатів
print("\n=== Статистика після обробки викидів ===")
numb_columns = df_clean.select_dtypes(include=['int64', 'float64']).columns.tolist()
print(df_clean[numb_columns].describe())

# Порівняння розмірів датафреймів
print(f"\nРозмір оригінального датафрейму: {df.shape}")
print(f"Розмір очищеного датафрейму: {df_clean.shape}")
print(f"Видалено записів: {df.shape[0] - df_clean.shape[0]} ({((df.shape[0] - df_clean.shape[0])/df.shape[0])*100:.2f}%)")


def plot_categorical_features(df, categorical_cols, max_categories=15):
    for col in categorical_cols:
        # Отримуємо топ max_categories категорій
        value_counts = df[col].value_counts()
        
        # Якщо категорій забагато - обмежуємо кількість
        if len(value_counts) > max_categories:
            value_counts = value_counts[:max_categories]
            title_suffix = f" (топ {max_categories})"
        else:
            title_suffix = ""
        
        plt.figure(figsize=(12, 5))
        
        # Спрощений barplot
        bars = plt.bar(value_counts.index.astype(str), value_counts.values)
        
        # Додаємо значення на стовпці
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=9)
        
        plt.title(f'Розподіл {col}{title_suffix}', pad=20)
        plt.xlabel('')
        plt.ylabel('Кількість')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

# Виклик функції
plot_categorical_features(df, categorical_cols)


def plot_group_analysis(df, numeric_col, categorical_col):
    plt.figure(figsize=(12, 6))
    df.boxplot(column=numeric_col, by=categorical_col, vert=False)
    plt.title(f'Вплив {categorical_col} на {numeric_col}')
    plt.suptitle('')
    plt.show()

plot_group_analysis(df, 'Victim_Age', 'Crime_Category')


from itertools import combinations
def plot_scatter_matrix(df, numeric_cols):
    for col1, col2 in combinations(numeric_cols, 2):
        plt.figure(figsize=(8, 6))
        plt.scatter(df[col1], df[col2], alpha=0.5)
        plt.title(f'Залежність між {col1} та {col2}')
        plt.xlabel(col1)
        plt.ylabel(col2)
        
        # Розрахунок кореляції
        corr = df[[col1, col2]].corr().iloc[0,1]
        print(f"Кореляція між {col1} та {col2}: {corr:.2f}")
        
        plt.show()

plot_scatter_matrix(df, numeric_cols[:4]) 


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['Crime_Category'])
y_clean = label_encoder.transform(df_clean['Crime_Category'])


high_card_cols = ['Location', 'Cross_Street', 'Modus_Operandi']
low_card_cols = ['Area_Name', 'Part 1-2', 'Victim_Sex', 'Victim_Descent',
                'Premise_Description', 'Weapon_Description',
                'Status', 'Status_Description']


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def frequency_encoder(X):
    if isinstance(X, pd.DataFrame):
        df = X.copy()
    else:
        df = pd.DataFrame(X, columns=high_card_cols)
    
    for col in df.columns:
        freq = df[col].value_counts(normalize=True)
        df[col] = df[col].map(freq)
    return df


high_card_transformer = Pipeline(steps=[ 
    ('imputer', SimpleImputer(strategy='constant', fill_value='UNKNOWN')),
    ('freq_encoder', FunctionTransformer(frequency_encoder))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='UNKNOWN')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numerical_transformer = Pipeline(steps=[
    ('imputer', IterativeImputer(random_state=42))
])

preprocessor = ColumnTransformer(transformers=[
    ('high_card', high_card_transformer, high_card_cols),
    ('low_card', categorical_transformer, low_card_cols),
    ('num', numerical_transformer, numeric_cols)
])


import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score, make_scorer

def objective(trial):
    params = {
        'classifier__n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'classifier__max_depth': trial.suggest_int('max_depth', 3, 10),
        'classifier__learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'classifier__subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'classifier__colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
    }
    
    model = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', XGBClassifier(
            random_state=42,
            eval_metric='mlogloss',
            use_label_encoder=False,
            **params
        ))
    ])
    
    cv_score = cross_val_score(
        model, df_clean.drop('Crime_Category', axis=1), y_clean, 
        cv=5, scoring=make_scorer(f1_score, average='weighted'), n_jobs=-1
    )
    
    return cv_score.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)
best_params = study.best_params


final_model = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', XGBClassifier(
        random_state=42,
        eval_metric='mlogloss',
        use_label_encoder=False,
        **best_params
    ))
])


final_model.fit(df_clean.drop('Crime_Category', axis=1), y_clean)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df.drop('Crime_Category', axis=1), y, test_size=0.2, random_state=42
)

X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(
    df_clean.drop('Crime_Category', axis=1), y_clean, test_size=0.2, random_state=42
)


model_raw = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', XGBClassifier(
        random_state=42,
        eval_metric='mlogloss',
        use_label_encoder=False,
        **best_params
    ))
])
model_raw.fit(X_train, y_train)


from sklearn.metrics import classification_report

print("=== Результати на неочищених даних ===")
y_pred_raw = model_raw.predict(X_test)
print(classification_report(y_test, y_pred_raw, target_names=label_encoder.classes_))

print("\n=== Результати на очищених даних ===")
y_pred_clean = final_model.predict(X_test_clean)
print(classification_report(y_test_clean, y_pred_clean, target_names=label_encoder.classes_))


import joblib
joblib.dump(final_model, 'best_crime_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')


optuna.visualization.plot_optimization_history(study)
optuna.visualization.plot_param_importances(study)

