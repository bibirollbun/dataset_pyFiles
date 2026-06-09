import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Загрузка данных
train = pd.read_csv('/kaggle/input/forest-cover-type-prediction/train.csv')
test = pd.read_csv('/kaggle/input/forest-cover-type-prediction/test.csv')

# Признаки и метки
X = train.drop(columns=['Id', 'Cover_Type'])
y = train['Cover_Type']
X_test = test.drop(columns=['Id'])

# Разделение признаков
numeric_features = X.columns[:10]   
binary_features = X.columns[10:]   

# Предобработка
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(interaction_only=True))
        ]), numeric_features),
        ('bin', 'passthrough', binary_features)
    ]
)

# Пайплайн
pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('clf', LogisticRegression(
        solver='saga',
        multi_class='multinomial',
        max_iter=5000,
        random_state=42,
        n_jobs=-1
    ))
])

# Подбор гиперпараметров
param_grid = {
    'preprocess__num__poly__degree': [1, 2],
    'clf__C': [0.1, 1, 10, 100]
}

# Кросс-валидация и обучение
grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
grid.fit(X, y)

# Вывод лучшего результата
print("Лучшая точность (CV):", grid.best_score_)
print("Лучшие параметры:", grid.best_params_)

# Предсказание и сохранение
y_pred = grid.predict(X_test)
submission = pd.DataFrame({'Id': test['Id'], 'Cover_Type': y_pred})
submission.to_csv('submission.csv', index=False)

