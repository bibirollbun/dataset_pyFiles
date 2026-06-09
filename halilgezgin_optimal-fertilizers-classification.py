import numpy as np
import pandas as pd


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from xgboost import plot_importance
import matplotlib.pyplot as plt
import seaborn as sns


train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv", index_col="id")
train.head()


train = pd.get_dummies(train, columns=['Soil Type'], drop_first=True)
train.head()


train = pd.get_dummies(train, columns=['Crop Type'], drop_first=True)
train.head()


X = train.drop('Fertilizer Name', axis=1)
X.head()


y = train['Fertilizer Name']
y.head()


le = LabelEncoder()
y = le.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)


numeric_cols = train.select_dtypes(include=['int64', 'float64']).columns

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])


xbc = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xbc.fit(X_train, y_train)


y_pred = xbc.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))


plot_importance(xbc)
plt.show()


cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Tahmin')
plt.ylabel('Gerçek')
plt.show()


proba = xbc.predict_proba(X_test)
top_3 = np.argsort(proba, axis=1)[:, -3:][:, ::-1]
class_labels = xbc.classes_
top_3_labels = [[class_labels[i] for i in row] for row in top_3]


correct = 0
for true_label, predicted in zip(y_test, top_3_labels):
    if true_label in predicted:
        correct += 1

top_3_accuracy = correct / len(y_test)
print("Top-3 Accuracy:", top_3_accuracy)


test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv", index_col="id")
test.head()


numeric_cols_1 = test.select_dtypes(include=['int64', 'float64']).columns

test[numeric_cols] = scaler.fit_transform(test[numeric_cols_1])
test.head()


test = pd.get_dummies(test, columns=['Soil Type'], drop_first=True)
test = pd.get_dummies(test, columns=['Crop Type'], drop_first=True)
test.head()


proba = xbc.predict_proba(test)
top_3 = np.argsort(proba, axis=1)[:, -3:][:, ::-1]
class_labels = xbc.classes_
top_3_labels = [[class_labels[i] for i in row] for row in top_3]
top_3_labels


class_labels


le.classes_


top_preds_labels = np.vectorize(lambda i: le.classes_[i])(top_3_labels)
top_preds_labels


ids = list(range(750000, 750000 + len(top_preds_labels)))
fertilizer_names = [' '.join(row) for row in top_preds_labels]
df = pd.DataFrame({
    'ID': ids,
    'Fertilizer Name': fertilizer_names
})
df


test.shape


df.to_csv("predicting_optimal_fertilizers_result.csv", index=False)




