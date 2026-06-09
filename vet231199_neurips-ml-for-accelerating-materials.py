import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV

# Загрузка данных
train_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
test_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')

# Просто признаки из SMILES
def simple_smiles_features(smiles):
    return {
        'length': len(smiles),
        'num_atoms': sum(c.isdigit() for c in smiles),
        'has_ring': int('1' in smiles),
        'has_double_bond': int('=' in smiles),
        'has_aromatic': int('c' in smiles or 'C' in smiles),
    }

# Создаем признаки для тестового набора
X_test = []
for s in test_df['SMILES']:
    feats = simple_smiles_features(s)
    X_test.append(list(feats.values()))
X_test = np.array(X_test)

targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
models = {}

for target in targets:
    # Обрабатываем NaN
    mask = train_df[target].notnull()
    train_df_clean = train_df[mask]

    # Создаем признаки для обучающего набора
    X_train = []
    for s in train_df_clean['SMILES']:
        feats = simple_smiles_features(s)
        X_train.append(list(feats.values()))
    X_train = np.array(X_train)

    y = train_df_clean[target]

    # Разделение для оценки
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y, test_size=0.2, random_state=42
    )

    # Гиперпараметры с помощью RandomizedSearchCV
    param_dist = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'min_samples_split': [2, 5, 10]
    }

    regressor = GradientBoostingRegressor(random_state=42)
    rand_search = RandomizedSearchCV(regressor, param_distributions=param_dist, n_iter=10, scoring='neg_mean_absolute_error', cv=3, random_state=42)
    rand_search.fit(X_tr, y_tr)

    best_model = rand_search.best_estimator_
    models[target] = best_model

    # Можно оценить качество на валидационной выборке
    val_pred = best_model.predict(X_val)
    mae = np.mean(np.abs(y_val - val_pred))
    print(f"{target} MAE: {mae}")

# Предсказываем для тестового набора
predictions = {}
for target in targets:
    predictions[target] = models[target].predict(X_test)

# Создаем DataFrame для submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'Tg': predictions['Tg'],
    'FFV': predictions['FFV'],
    'Tc': predictions['Tc'],
    'Density': predictions['Density'],
    'Rg': predictions['Rg']
})

# Сохраняем
submission.to_csv('submission.csv', index=False)

