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


train_df = pd.read_csv('/kaggle/input/forest-cover-type-prediction/train.csv')
test_df = pd.read_csv('/kaggle/input/forest-cover-type-prediction/test.csv')
print(f"Размер train: {train_df.shape}")
print(f"Размер test: {test_df.shape}")

print("\nTrain info:")
print(train_df.info())

print("\nTest info:")
print(train_df.info())

print("\nРаспределение Cover_Type:")
print(train_df['Cover_Type'].value_counts().sort_index())


print("\Train:")
train_df.head()


import matplotlib.pyplot as plt
import seaborn as sns
# непрервные признаки
continuous_features = [
    'Elevation', 'Aspect', 'Slope', 
    'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
    'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 
    'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points'
]


print("статистика непрерывных признаков:")
print(train_df[continuous_features].describe())

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.ravel()

for i, feature in enumerate(continuous_features):
    axes[i].hist(train_df[feature], bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[i].set_title(feature)
    axes[i].set_xlabel('Значение')
    axes[i].set_ylabel('Частота')

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
corr_matrix = train_df[continuous_features].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Корр матрица')
plt.show()

wilderness_cols = [col for col in train_df.columns if 'Wilderness_Area' in col]
soil_cols = [col for col in train_df.columns if 'Soil_Type' in col]


X = train_df.drop(['Id', 'Cover_Type'], axis=1)
y = train_df['Cover_Type']
X_test = test_df.drop(['Id'], axis=1)

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"X_train: {X_train.shape}")
print(f"X_val: {X_val.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_val: {y_val.shape}")


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_val_scaled = X_val.copy()
X_test_scaled = X_test.copy()


X_train_scaled[continuous_features] = scaler.fit_transform(X_train[continuous_features])
X_val_scaled[continuous_features] = scaler.transform(X_val[continuous_features])
X_test_scaled[continuous_features] = scaler.transform(X_test[continuous_features])


print("Статистика после масштабирования (первые 3 непрерывных признака):")
print("Средние значения:")
print(X_train_scaled[continuous_features[:3]].mean())
print("\nСтандартные отклонения:")
print(X_train_scaled[continuous_features[:3]].std())


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print("\n---RandomForestClassifier---")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_scaled, y_train)

y_val_pred = rf_model.predict(X_val_scaled)

val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"\nТочность на валидационной выборке: {val_accuracy:.4f}")


from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

y_train_adj = y_train - 1
y_val_adj = y_val - 1

models = {
    'RandomForest': RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    ),
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=100, random_state=42
    ),
    'ExtraTrees': ExtraTreesClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    ),
    'XGBoost': XGBClassifier(
        n_estimators=100, random_state=42, eval_metric='mlogloss',
        verbosity=0
    ),
    'LightGBM': LGBMClassifier(
        n_estimators=100, random_state=42, verbose=-1
    ),
    'CatBoost': CatBoostClassifier(
        iterations=100, random_state=42, verbose=0
    )
}


results = {}
for name, model in models.items():
    print(f"\n--- Обучение {name} ---")
    
    if name in ['XGBoost', 'LightGBM', 'CatBoost']:

        model.fit(X_train_scaled, y_train_adj)
        y_val_pred_adj = model.predict(X_val_scaled)

        y_val_pred = y_val_pred_adj + 1
    else:
        model.fit(X_train_scaled, y_train)
        y_val_pred = model.predict(X_val_scaled)
    
    accuracy = accuracy_score(y_val, y_val_pred)
    results[name] = accuracy
    print(f"Точность {name}: {accuracy:.4f}")

for name, accuracy in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{name}: {accuracy:.4f}")


from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder

best_models = {
    'RandomForest': RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    ),
    'ExtraTrees': ExtraTreesClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    ),
    'XGBoost': XGBClassifier(
        n_estimators=100, random_state=42, eval_metric='mlogloss',
        verbosity=0
    ),
    'LightGBM': LGBMClassifier(
        n_estimators=100, random_state=42, verbose=-1
    )
}


print("---VotingClassifier ---")

y_train_encoded = y_train - 1
y_val_encoded = y_val - 1

trained_models = []
for name, model in best_models.items():
    print(f"Обучение {name} для ансамбля...")
    if name in ['XGBoost', 'LightGBM']:
        model.fit(X_train_scaled, y_train_encoded)
    trained_models.append((name, model))


voting_clf = VotingClassifier(
    estimators=trained_models,
    voting='soft' 
)


voting_clf.fit(X_train_scaled, y_train_encoded)


y_val_pred_encoded = voting_clf.predict(X_val_scaled)
y_val_pred = y_val_pred_encoded + 1  # преобразуем обратно к 1-7


voting_accuracy = accuracy_score(y_val, y_val_pred)
print(f"\nТочность VotingClassifier: {voting_accuracy:.4f}")

print(f"Лучшая одиночная модель (ExtraTrees): {results['ExtraTrees']:.4f}")
print(f"Улучшение: {voting_accuracy - results['ExtraTrees']:.4f}")



from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

print("--- нрастройка ExtraTrees ---")

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    ExtraTreesClassifier(random_state=42, n_jobs=-1),
    param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_scaled, y_train_encoded)

print(f"\nЛучшие параметры: {grid_search.best_params_}")
print(f"Лучшая точность (кросс-валидация): {grid_search.best_score_:.4f}")

best_et_model = grid_search.best_estimator_
y_val_pred_et = best_et_model.predict(X_val_scaled)
y_val_pred_et_original = y_val_pred_et + 1
best_et_accuracy = accuracy_score(y_val, y_val_pred_et_original)

print(f"Точность настроенного ExtraTrees на валидации: {best_et_accuracy:.4f}")
print(f"Улучшение по сравнению с базовым ExtraTrees: {best_et_accuracy - results['ExtraTrees']:.4f}")

best_models['ExtraTrees_tuned'] = grid_search.best_estimator_



final_models = [
    ('ExtraTrees', ExtraTreesClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    )),
    ('RandomForest', RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    )),
    ('XGBoost', XGBClassifier(
        n_estimators=100, random_state=42, eval_metric='mlogloss',
        verbosity=0
    ))
]

final_voting_clf = VotingClassifier(
    estimators=final_models,
    voting='soft'
)

print("Обучение финального VotingClassifier...")
final_voting_clf.fit(X_train_scaled, y_train_encoded)

y_val_final_pred_encoded = final_voting_clf.predict(X_val_scaled)
y_val_final_pred = y_val_final_pred_encoded + 1

final_accuracy = accuracy_score(y_val, y_val_final_pred)
print(f"\nТочность финального ансамбля на валидации: {final_accuracy:.4f}")

print("\n--- Предсказание на тестовых данных ---")
print(f"Размер тестовых данных: {X_test_scaled.shape}")


test_predictions_encoded = final_voting_clf.predict(X_test_scaled)
test_predictions = test_predictions_encoded + 1 


print("\nРаспределение предсказанных классов:")
unique, counts = np.unique(test_predictions, return_counts=True)
for cls, count in zip(unique, counts):
    print(f"Класс {cls}: {count} примеров ({count/len(test_predictions)*100:.2f}%)")


submission = pd.DataFrame({
    'Id': test_df['Id'],
    'Cover_Type': test_predictions
})


submission_path = '/kaggle/working/submission.csv'
submission.to_csv(submission_path, index=False)


submission.head(10)

