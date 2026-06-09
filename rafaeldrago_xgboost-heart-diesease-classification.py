import pandas as pd
import numpy as np
import os
import warnings 

from sklearn.preprocessing import LabelEncoder

import seaborn as sns
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import accuracy_score, fbeta_score, f1_score, recall_score, r2_score, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score

warnings.filterwarnings('ignore')

df = pd.read_csv('/kaggle/input/heart-disease-prediction-dataquest/heart_train.csv')
test = pd.read_csv('/kaggle/input/heart-disease-prediction-dataquest/heart_test.csv')


df.head()


df.describe(include='all')


df.info()


mt = df.select_dtypes(include='number').corr()
corr_matrix = mt

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f')
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.show()


# Count the target class values
class_counts = df['HeartDisease'].value_counts()
labels = ['No Disease', 'Heart Disease']
colors = ['#66b3ff', '#ff9999']

# Create the pie chart
plt.figure(figsize=(8, 8))
plt.pie(class_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
plt.title('Target Class Distribution - HeartDisease')
plt.axis('equal')  # Ensures pie is drawn as a circle
plt.show()



x_values = df[['RestingBP', 'Cholesterol', 'Oldpeak', 'MaxHR']]
fig, axis = plt.subplots(nrows=2, ncols=2, figsize=(16, 10))
y_value = 'Age'

for ax, x_value in zip(axis.flat, x_values):
    sns.scatterplot(data=df, x=x_value, y=y_value, hue='HeartDisease', ax=ax)
    ax.set_title(f'{x_value.capitalize()} and {y_value.capitalize()}')
plt.tight_layout()
plt.show()


fig, axis = plt.subplots(nrows=2, ncols=2, figsize=(16, 10))

for ax, x_value in zip(axis.flat, x_values):
    sns.kdeplot(data=df, x=x_value, hue='HeartDisease', fill=True, common_norm=False, alpha=0.5, ax=ax)
    ax.set_title(f'{x_value.capitalize()}')
plt.tight_layout()
plt.show()


x_values = df.select_dtypes(include=['number'])
fig, axis = plt.subplots(nrows=3, ncols=2, figsize=(16, 14))
for ax, x_value in zip(axis.flat, x_values):
    sns.histplot(data=df, x=x_value, hue="HeartDisease", kde=True, ax=ax, bins=20, alpha=0.6)
    ax.set_title(f'Histogram of {x_value.capitalize()}')
plt.tight_layout()
plt.show()


cols_encoder = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

def label_encode_columns(data, columns):
    for col in columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
    return data

df = label_encode_columns(df,cols_encoder)
test = label_encode_columns(test,cols_encoder)


df


test


def remove_outliers(df):
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:

            Q1 = df[col].quantile(0.20)
            Q3 = df[col].quantile(0.80)
            IQR = Q3 - Q1

            lower_limit = Q1 - 1.5 * IQR
            upper_limit = Q3 + 1.5 * IQR

            df[col] = df[col].apply(
                lambda x: x if pd.isnull(x) or (lower_limit <= x <= upper_limit) else None
            )
    
    return df

df = remove_outliers(df)
test = remove_outliers(test)


X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

model = xgb.XGBClassifier()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)


 
y_pred = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')

# ROC metrics 
prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, prob)
roc_auc = roc_auc_score(y_test, prob)

# Precision-Recall metrics
precision, recall_pr, _ = precision_recall_curve(y_test, prob)
avg_precision = average_precision_score(y_test, prob)

# Axes
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 12))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0, 0])
axes[0, 0].set_title('Confusion Matrix')

# ROC Curve
axes[0, 1].plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})', color='blue')
axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random guess')
axes[0, 1].set_xlabel('False Positive Rate')
axes[0, 1].set_ylabel('True Positive Rate')
axes[0, 1].set_title('ROC Curve')
axes[0, 1].legend(loc='lower right')
axes[0, 1].grid(True)

# Feature Importance
xgb.plot_importance(model, ax=axes[1, 0])
axes[1, 0].set_title('Feature Importance')
axes[1, 0].grid(True)

# Precision-Recall Curve
axes[1, 1].plot(recall_pr, precision, label=f'AP = {avg_precision:.4f}', color='green')
axes[1, 1].set_xlabel('Recall')
axes[1, 1].set_ylabel('Precision')
axes[1, 1].set_title('Precision-Recall Curve')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()

# Results
print(f"Accuracy: {accuracy:.4f}")
print(f"F2-score: {f2:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Recall: {recall:.4f}")



y_test = test
y_test = model.predict(test).flatten()

submission = pd.DataFrame({
    'id': range(1, 185),
    'target': y_test
})
submission.to_csv('submission', index=False)

