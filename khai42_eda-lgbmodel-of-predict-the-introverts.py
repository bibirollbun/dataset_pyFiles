import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from scipy.stats import gaussian_kde
from lightgbm import LGBMClassifier, early_stopping, plot_importance
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


train =pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test =pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
sample_submission=pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')


train.head(3)


test.head(3)


train.dtypes


missing_data = train.isnull().sum().to_frame(name='Missing Count')
missing_data['Missing %'] = ((missing_data['Missing Count'] / len(train)) * 100).round(2)
missing_data = missing_data.sort_values(by='Missing %', ascending=False)
missing_data


eda_df = train.drop(columns=['id'])
msno.matrix(eda_df)
plt.show()


eda_test = train.drop(columns=['id'])
msno.matrix(eda_df)
plt.show()


numeric_features = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside','Friends_circle_size', 'Post_frequency']
categorical_features = ['Stage_fear', 'Drained_after_socializing', 'Personality']

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12,7))
axes = axes.flatten()
for i, col in enumerate(numeric_features):
    sns.histplot(data=train, x=col, bins=30, kde=False, ax=axes[i])
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Count')

if len(numeric_features) < len(axes):
    for j in range(len(numeric_features), len(axes)):
        fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14,4))
axes = axes.flatten()
for i, col in enumerate(categorical_features):
    sns.countplot(data=train, x=col, ax=axes[i])
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Count')

plt.tight_layout()
plt.show()


personality_counts = train['Personality'].value_counts()
plt.figure(figsize=(5,5))
plt.pie(personality_counts, labels=personality_counts.index, autopct='%1.1f%%')
plt.title('Personality Distribution')
plt.show()


cleaned_df = train.copy()
for col in numeric_features:
    cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
cleaned_df = cleaned_df[numeric_features + ['Personality']].dropna()
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
axes = axes.flatten()

for i, col in enumerate(numeric_features):
    sns.kdeplot(data=cleaned_df, x=col, hue="Personality", fill=True, common_norm=False, ax=axes[i])
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Density')

if len(numeric_features) < len(axes):
    for j in range(len(numeric_features), len(axes)):
        fig.delaxes(axes[j])
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
axes = axes.flatten()

for i, col in enumerate(numeric_features):
    sns.boxplot(data=train, x="Personality", y=col, ax=axes[i])
    axes[i].set_title(f'Boxplot of {col} by Personality')
    axes[i].set_xlabel("Personality")
    axes[i].set_ylabel(col)

if len(numeric_features) < len(axes):
    for j in range(len(numeric_features), len(axes)):
        fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

sampled_df = train[numeric_features + ['Personality']].dropna().sample(n=1000, random_state=42)
sns.pairplot(sampled_df, hue="Personality", corner=True, plot_kws={"alpha": 0.6})
plt.suptitle("Pairplot of Features Colored by Personality", y=1.02)
plt.show()


train_df_encoded = train.copy()
train_df_encoded['Personality_Encoded'] = train_df_encoded['Personality'].map({'Introvert': 0, 'Extrovert': 1})
correlation_with_target = train_df_encoded[numeric_features + ['Personality_Encoded']].corr()['Personality_Encoded'].drop('Personality_Encoded')

plt.figure(figsize=(8, 4))
sns.barplot(x=correlation_with_target.index, y=correlation_with_target.values)
plt.title("Correlation of Numeric Features with Personality")
plt.xticks(rotation=45)
plt.ylabel("Correlation")
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
axes = axes.flatten()

for i, col in enumerate(numeric_features):
    sns.violinplot(data=train, x="Personality", y=col, ax=axes[i])
if len(numeric_features) < len(axes):
    for j in range(len(numeric_features), len(axes)):
        fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
axes = axes.flatten()

for i, col in enumerate(numeric_features):
    ax = axes[i]

    train_values = pd.to_numeric(train[col], errors='coerce').dropna()
    test_values = pd.to_numeric(test[col], errors='coerce').dropna()

    if len(train_values) > 1 and len(test_values) > 1:
        x_range = np.linspace(min(train_values.min(), test_values.min()),
                              max(train_values.max(), test_values.max()), 1000)
        train_kde = gaussian_kde(train_values)
        test_kde = gaussian_kde(test_values)

        ax.plot(x_range, train_kde(x_range), label='Train', linewidth=2)
        ax.plot(x_range, test_kde(x_range), label='Test', linewidth=2)

    ax.set_title(f'{col}: Train vs Test')
    ax.legend()

if len(numeric_features) < len(axes):
    for j in range(len(numeric_features), len(axes)):
        fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
axes = axes.flatten()

for i, col in enumerate(numeric_features):
    temp_df = pd.concat([
        train[[col]].assign(source='Train'),
        test[[col]].assign(source='Test')
    ])
    temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce')

    sns.boxplot(data=temp_df, x='source', y=col, ax=axes[i])
    axes[i].set_title(f'Boxplot of {col}: Train vs Test')

if len(numeric_features) < len(axes):
    for j in range(len(numeric_features), len(axes)):
        fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


label_encoders = {}
for col in ['Stage_fear', 'Drained_after_socializing', 'Personality']:
    le = LabelEncoder()
    if col != 'Personality':
        train[col] = le.fit_transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))
    else:
        train[col] = le.fit_transform(train[col])
    label_encoders[col] = le

train.fillna(train.median(numeric_only=True), inplace=True)
test.fillna(train.median(numeric_only=True), inplace=True)


X = train.drop(columns=['id', 'Personality'])
y = train['Personality']
X_test = test.drop(columns=['id'])

lgbm_params = {
    'max_depth': 9,
    'learning_rate': 0.23345819420207964,
    'n_estimators': 66,
    'subsample': 0.5277670904459099,
    'colsample_bytree': 0.7764751780253709,
    'reg_alpha': 0.023963329809817968,
    'reg_lambda': 0.24662127682644475
}

n_splits = 10
oof_preds = np.zeros_like(y)
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

for train_idx, val_idx in skf.split(X, y):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = LGBMClassifier(**lgbm_params)
    model.fit(X_train, y_train)
    oof_preds[val_idx] = model.predict(X_val)

val_accuracy = accuracy_score(y, oof_preds)
print("Validation Accuracy:", val_accuracy)


final_model = LGBMClassifier(**lgbm_params)
final_model.fit(X, y)
preds = final_model.predict(X_test)
preds_labels = label_encoders['Personality'].inverse_transform(preds)


plt.figure(figsize=(10, 6))
ax = plot_importance(final_model, max_num_features=10, importance_type='gain')
ax.grid(False)
plt.title("Feature Importances")
plt.tight_layout()
plt.show()


submission = sample_submission.copy()
submission['Personality'] = preds_labels
submission.to_csv("submission.csv", index=False)
submission.head()




