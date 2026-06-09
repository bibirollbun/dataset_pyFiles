import pandas as pd
import numpy as np

train = pd.read_csv('/kaggle/input/forest-cover-type-prediction/train.csv')
test = pd.read_csv('/kaggle/input/forest-cover-type-prediction/test.csv')

train.head().T


train.describe().T


# SoilType 7 и 15 всегда равны 0. Удалим их из даты как малоинформативные

train = train.drop(columns = ["Soil_Type7", "Soil_Type15"])
test = test.drop(columns = ["Soil_Type7", "Soil_Type15"])


# Добавим признак Евклидово расстояния до воды
train['Distance_To_Water'] = np.sqrt(
    train['Horizontal_Distance_To_Hydrology']**2 +
    train['Vertical_Distance_To_Hydrology']**2
)


import matplotlib.pyplot as plt
import seaborn as sns

# Посмотрим на матрицы корреляции
num_features = [
    'Elevation',
    'Aspect',
    'Slope',
    'Horizontal_Distance_To_Hydrology',
    'Vertical_Distance_To_Hydrology',
    'Horizontal_Distance_To_Roadways',
    'Hillshade_9am',
    'Hillshade_Noon',
    'Hillshade_3pm',
    'Horizontal_Distance_To_Fire_Points',
    'Distance_To_Water'
]

corr_matrix = train[num_features].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Matrix")
plt.show()


train = train.drop('Distance_To_Water', axis = 1)


plt.figure(figsize=(10, 6))
sns.violinplot(
    x='Cover_Type',
    y='Horizontal_Distance_To_Hydrology',
    data=train,
    inner='quartile',
    cut=0
)
plt.xlabel('Cover Type')
plt.ylabel('Horizontal Distance to Water')
plt.title('Horizontal Distance to Water by Cover Type')
plt.show()

plt.figure(figsize=(10, 6))
sns.violinplot(
    x='Cover_Type',
    y='Vertical_Distance_To_Hydrology',
    data=train,
    inner='quartile',
    cut=0
)
plt.xlabel('Cover Type')
plt.ylabel('Vertical Distance to Water')
plt.title('Vertical Distance to Water by Cover Type')
plt.show()


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

X = train.drop(['Id','Cover_Type'], axis=1)
y = train['Cover_Type']
X_test = test.drop('Id', axis=1)

# cat_cols = [c for c in X.columns if 'Wilderness_Area' or 'Soil_Type' in c]

# num_cols = [
#     'Elevation','Aspect','Slope',
#     'Horizontal_Distance_To_Hydrology',
#     'Vertical_Distance_To_Hydrology',
#     'Horizontal_Distance_To_Roadways',
#     'Hillshade_9am','Hillshade_Noon','Hillshade_3pm',
#     'Horizontal_Distance_To_Fire_Points',
# ]


X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

rf_params = {
    'n_estimators': [200, 300, 400, 500, 600],
}


rf_search = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv = 5, n_jobs=-1, scoring='accuracy', verbose=1)
rf_search.fit(X_train, y_train)
rf_val_preds = rf_search.predict(X_val)
rf_val_acc = accuracy_score(y_val, rf_val_preds)
print(f"Random Forest validation accuracy: {rf_val_acc:.4f}")
print("Best params:", rf_search.best_params_)

rf = rf_search.best_estimator_
rf.fit(X, y)
test_preds = rf.predict(X_test)

submission = pd.DataFrame({
    'Id': test['Id'],
    'Cover_Type': test_preds
})
submission.to_csv('submission.csv', index=False)
print("submission.csv saved")








