import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


train_data = pd.read_csv(r'/kaggle/input/titanic/train.csv')
test_data = pd.read_csv(r'/kaggle/input/tutorial-competition-for-beginners/test.csv')


train_data.isnull().sum()


sns.boxplot(train_data)


train_data = train_data.ffill().bfill()


def normalizer(data):
    numeric_data = data.select_dtypes(include=['number'])  # Yalnız rəqəmsal sütunları seç
    Q1 = numeric_data.quantile(0.25)
    Q3 = numeric_data.quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    lower_bound = Q1 - 1.5 * IQR
    for col in numeric_data.columns:
        numeric_data[col] = numeric_data[col].clip(upper=upper_bound[col], lower=lower_bound[col])
    return numeric_data



train_data = normalizer(train_data)


X = train_data.drop(columns='Survived', axis=1)
y = train_data['Survived']


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)


model = RandomForestClassifier()


param_grid = {
    'n_estimators': [50, 100, 200],  # Ağac sayı
    'max_depth': [None, 10, 20, 30],  # Maksimum dərinlik
    'min_samples_split': [2, 5, 10],  # Bölünmə üçün minimal nümunə sayı
    'min_samples_leaf': [1, 2, 4],  # Yarpaq düyündə minimal nümunə sayı
    'max_features': ['sqrt', 'log2']  # Seçiləcək xüsusiyyətlər
}



grid = GridSearchCV(estimator=model,param_grid=param_grid, cv=5)


grid.fit(X_train,y_train)


best_model = grid.best_estimator_


preds = best_model.predict(X_test)


acc = accuracy_score(y_test,preds)
creport = classification_report(y_test,preds)


print(f'Deqiqlik:\n{acc*100:.0f}%\nTesnifat Hesabati:\n{creport}')


y_pred_proba = best_model.predict_proba(X_test)[:, 1]  # AUC üçün ehtimallar


from sklearn.metrics import roc_auc_score, accuracy_score

# AUC
auc_score = roc_auc_score(y_test, y_pred_proba)
print("AUC Score:", auc_score)

# Dəqiqlik (accuracy)
accuracy = accuracy_score(y_test, preds)
print("Accuracy:", accuracy)



submission = pd.DataFrame({"PassengerId": test_data["PassengerId"], "Survived": preds})
submission.to_csv("submission.csv", index=False)





