import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


traindata = pd.read_csv('/kaggle/input/playground-series-s3e7/train.csv')
testdata = pd.read_csv('/kaggle/input/playground-series-s3e7/test.csv')


traindata.info()
traindata.describe()


num_vars = [
    'no_of_adults','no_of_children','no_of_weekend_nights','no_of_week_nights',
    'lead_time','no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled',
    'avg_price_per_room','no_of_special_requests'
]


fig, axes = plt.subplots(3,3, figsize = (12,6))
for i in range(len(num_vars)):
    sns.boxplot(traindata[num_vars[i]], ax = axes[i//3, i%3])
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(3, 3, figsize=(15, 10))
for i, var in enumerate(num_vars):
    axes[i//3, i%3].hist(traindata[var].dropna(), bins=30, color='skyblue', edgecolor='black')
    axes[i//3, i%3].set_title(var)
plt.tight_layout()
plt.show()


scaler = StandardScaler()
traindata[num_vars] = scaler.fit_transform(traindata[num_vars])
testdata[num_vars] = scaler.transform(testdata[num_vars])


cat_vars = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']


fig, ax = plt.subplots(1, 3, figsize=(18, 5))
for i, var in enumerate(cat_vars):
    sns.countplot(data=traindata, x=var, hue='booking_status', ax=ax[i])
    ax[i].set_title(var)
    ax[i].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()


X = traindata.drop(columns=['id', 'booking_status'])
y = traindata['booking_status']


X = pd.get_dummies(X, drop_first=True)
testdata_proc = pd.get_dummies(testdata.drop(columns=['id']), drop_first=True)


testdata_proc = testdata_proc.reindex(columns=X.columns, fill_value=0)


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)


logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_val)


print("Результаты Логистической регрессии")
print(confusion_matrix(y_val, y_pred_logreg))
print(classification_report(y_val, y_pred_logreg))
print(f"Точность: {accuracy_score(y_val, y_pred_logreg):.4f}\n")


coeff_df = pd.DataFrame(logreg.coef_[0], index=X.columns, columns=['Коэффициент'])
coeff_df.sort_values(by='Коэффициент', ascending=False).plot(kind='bar', figsize=(12, 6), title="Логистическая регрессия: важность признаков")
plt.tight_layout()
plt.show()


selector = RFE(logreg, n_features_to_select=10)
selector = selector.fit(X_train, y_train)
selected_features = X_train.columns[selector.support_]
print("Выбранные признаки (RFE):", selected_features)


rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_val)


print("Результаты случайного леса:")
print(confusion_matrix(y_val, y_pred_rf))
print(classification_report(y_val, y_pred_rf))
print(f"Accuracy: {accuracy_score(y_val, y_pred_rf):.4f}\n")


feat_importances = rf.feature_importances_
sns.barplot(x=feat_importances, y=X_train.columns)
plt.xlabel("Оценка важности")
plt.ylabel("Признаки")
plt.title("Random Forest: важность признаков")
plt.tight_layout()
plt.show()


best_model = rf if accuracy_score(y_val, y_pred_rf) > accuracy_score(y_val, y_pred_logreg) else logreg


test_preds = best_model.predict(testdata_proc)


submission = pd.DataFrame({
    'id': testdata['id'],
    'booking_status': test_preds
})
submission.to_csv('submission.csv', index=False)

