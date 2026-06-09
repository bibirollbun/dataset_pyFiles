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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Завантаження даних
try:
    train_data = pd.read_csv('/kaggle/input/data-science-london-scikit-learn/train.csv', header=None)
    train_labels = pd.read_csv('/kaggle/input/data-science-london-scikit-learn/trainLabels.csv', header=None)
    test_data = pd.read_csv('/kaggle/input/data-science-london-scikit-learn/test.csv', header=None)
except FileNotFoundError:
    print("Помилка: Перевірте, чи файли доступні за вказаними шляхами.")
    raise

# Перейменування стовпців для зручності
train_data.columns = [f'feature_{i}' for i in range(train_data.shape[1])]
train_labels.columns = ['target']
test_data.columns = [f'feature_{i}' for i in range(test_data.shape[1])]


# Перегляд перших 5 рядків даних
print("Перші 5 рядків тренувальних даних:")
print(train_data.head())

# Перевірка пропущених значень
print("\nКількість пропущених значень у тренувальних даних:")
print(train_data.isnull().sum())

# Розподіл цільової змінної
print("\nРозподіл цільової змінної:")
print(train_labels['target'].value_counts())

# Гістограми розподілу ознак
train_data.hist(figsize=(15, 15))
plt.show()

# Кореляційна матриця
sns.heatmap(train_data.corr(), cmap='coolwarm')
plt.show()


# Попередня обробка даних
X = train_data
y = train_labels['target']

# Масштабування ознак
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test_data)

# Відбір ознак
selector = SelectKBest(f_classif, k=20)  # Вибираємо 20 найкращих ознак
X_selected = selector.fit_transform(X_scaled, y)
test_selected = selector.transform(test_scaled)


# Розділення даних на тренувальний і валідаційний набори
X_train, X_val, y_train, y_val = train_test_split(X_selected, y, test_size=0.2, random_state=42)


# Гіперпараметричний пошук для KNN
param_grid = {
    'n_neighbors': range(1, 30),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Найкраща модель
best_knn = grid_search.best_estimator_
print("Найкращі параметри:", grid_search.best_params_)


# Оцінка на валідаційному наборі
y_pred_val = best_knn.predict(X_val)
print("Точність на валідаційному наборі:", accuracy_score(y_val, y_pred_val))
print("\nЗвіт про класифікацію:\n", classification_report(y_val, y_pred_val))
print("\nМатриця плутанини:\n", confusion_matrix(y_val, y_pred_val))


# Прогнозування на тестовому наборі
y_pred_test = best_knn.predict(test_selected)

# Створення файлу для відправки
submission = pd.DataFrame({
    'Id': range(1, len(y_pred_test) + 1),
    'Solution': y_pred_test
})
submission.to_csv('submission.csv', index=False)
print("Файл submission.csv створено.")




