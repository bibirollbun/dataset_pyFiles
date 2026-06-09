import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import warnings
warnings.filterwarnings('ignore')


df_train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')

# First 5 data
print(df_train.head())

# Data Info
print(df_train.info())

# Data Description
print(df_train.describe())


# Null values in the dataset
df_train.isnull().sum()


numerical_cols = ['Time_spent_Alone','Social_event_attendance','Going_outside','Friends_circle_size','Post_frequency']
categorical_cols = ['Stage_fear','Drained_after_socializing']


# Correlation
plt.figure(figsize=(8, 6))
sns.heatmap(df_train[numerical_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


# Numerical Columns Analysis
for col in numerical_cols:
    plt.figure()
    sns.boxplot(x=df_train[col])
    plt.title(f'{col}')
    plt.show()


fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for i, col in enumerate(numerical_cols):
    # Histogram
    axes[i].hist(df_train[col], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[i].set_title(f'{col} Distribution')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')
    
    mean_val = df_train[col].mean()
    axes[i].axvline(mean_val, color='red', linestyle='--', label=f'Avg.: {mean_val:.2f}')
    axes[i].legend()


# Categorical Column Analysis
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.pie(df_train['Stage_fear'].value_counts().values, labels = df_train['Stage_fear'].value_counts().index, autopct='%1.1f%%')
ax1.set_title('Stage Fear')

ax2.pie(df_train['Drained_after_socializing'].value_counts().values, labels = df_train['Drained_after_socializing'].value_counts().index, autopct='%1.1f%%',)
ax2.set_title('Drained After Socializing')

plt.show()


# Cross Tabulation
for col in categorical_cols:
    print(col.upper())
    cross_tab = pd.crosstab(df_train[col], df_train['Personality'])
    print(cross_tab)
    print("=========")


plt.figure(figsize=(10, 5))
sns.countplot(data=df_train, x='Personality', order=df_train['Personality'].value_counts().index, palette = 'Spectral')
plt.title('Distribution of Personalities')
plt.xticks(rotation=45)
plt.show()


averages = df_train.groupby('Personality')[numerical_cols].mean().round(2)
display(averages)


# Filling null values
df_train['Time_spent_Alone'].fillna(df_train['Time_spent_Alone'].median(), inplace = True)
df_train['Going_outside'].fillna(df_train['Going_outside'].median(), inplace = True)
df_train['Friends_circle_size'].fillna(df_train['Friends_circle_size'].median(), inplace = True)

df_train['Social_event_attendance'].fillna(df_train['Social_event_attendance'].mean(), inplace=True)
df_train['Post_frequency'].fillna(df_train['Post_frequency'].mean(), inplace=True)

df_train['Stage_fear'].fillna(df_train['Stage_fear'].mode()[0], inplace=True)
df_train['Drained_after_socializing'].fillna(df_train['Drained_after_socializing'].mode()[0], inplace=True)


print("Remaining null values:", df_train.isnull().sum().sum())


df_train['Social_Activity_Score'] = (
    df_train['Social_event_attendance'] +
    df_train['Going_outside'] +
    df_train['Friends_circle_size'] +
    df_train['Post_frequency']
) / 4

df_train.head()


# Label Encoding for categorical values
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for col in categorical_cols:
    df_train[col] = le.fit_transform(df_train[col])


X = df_train.drop(columns = ['id', 'Personality'])
y = df_train['Personality']


from sklearn.preprocessing import LabelEncoder

le2 = LabelEncoder()
y_encoded = le2.fit_transform(y)  # Extrovert: 1, Introvert: 0


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size = 0.2, stratify = y_encoded, random_state = 42)


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
fold_accuracies = []
models = []
feature_importances = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded)):
    print(f"\n Fold {fold + 1}")
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )

    model.fit(X_train, y_train)
    models.append(model)
    feature_importances.append(model.feature_importances_)
    preds = model.predict(X_val)

    acc = accuracy_score(y_val, preds)
    fold_accuracies.append(acc)

    print(f"Fold {fold + 1} Accuracy: {acc:.4f}")
    print(classification_report(y_val, preds, target_names=le.classes_))

print(f"\n Average CV Accuracy: {np.mean(fold_accuracies):.4f}")


avg_importance = np.mean(feature_importances, axis=0)

importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': avg_importance
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.gca().invert_yaxis()
plt.title("Average Feature Importance (Across Folds)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()


df_test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
df_test.head()


# Filling missing values
df_test['Time_spent_Alone'].fillna(df_train['Time_spent_Alone'].median(), inplace=True)
df_test['Social_event_attendance'].fillna(df_train['Social_event_attendance'].mean(), inplace=True)
df_test['Going_outside'].fillna(df_train['Going_outside'].median(), inplace=True)
df_test['Friends_circle_size'].fillna(df_train['Friends_circle_size'].median(), inplace=True)
df_test['Post_frequency'].fillna(df_train['Post_frequency'].mean(), inplace=True)

for col in ['Stage_fear', 'Drained_after_socializing']:
    df_test[col] = le.fit_transform(df_test[col])

df_test['Social_Activity_Score'] = (
    df_test['Social_event_attendance'] +
    df_test['Going_outside'] +
    df_test['Friends_circle_size'] +
    df_test['Post_frequency']
) / 4

df_test.head()


X_test = df_test.drop(columns = ['id'])

fold_preds = np.zeros((len(X_test), len(models)))
for i, m in enumerate(models):
    fold_preds[:, i] = m.predict(X_test)

from scipy.stats import mode
y_test_pred = mode(fold_preds, axis=1)[0].flatten()


y_test_labels = le2.inverse_transform(y_test_pred.astype(int))

submission = pd.DataFrame({
    'id': df_test['id'],
    'Personality': y_test_labels
})

submission.to_csv("submission.csv", index=False)


print(submission['Personality'].unique())
print(submission.head())




