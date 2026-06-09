# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



train_df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
train_df.shape


missing_train = train_df.isnull().sum()
missing_train[missing_train > 0]


test_df = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
test_df.shape


missing_test = test_df.isnull().sum()
missing_test[missing_test > 0]


# original dataset
original_df = pd.read_csv("/kaggle/input/extrovert-vs-introvert-behavior-data/personality_datasert.csv")
original_df.shape


missing_original = original_df.isnull().sum()
missing_original[missing_original > 0]


original_df.columns


train_df['data'] = 'train'
test_df['data'] = 'test'
original_df['data']= 'original'

df = pd.concat([train_df, original_df, test_df])
df = df.reset_index(drop=True)
df.shape


df.info()


missing_df = df.isnull().sum()
missing_df[missing_df > 0]


plt.figure(figsize=(10,6))
sns.heatmap(df.isna().transpose(),
            cmap="YlGnBu",
            cbar_kws={'label': 'Missing Data'})


df.describe()


num_cols = df.select_dtypes(include='number').columns.tolist()
num_cols


bin_size = [11, 10, 7, 15, 10]
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))
axes = axes.flatten()

for ax, col, bins in zip(axes, num_cols[1:], bin_size):
    ax.hist(df[col], bins=bins, align='mid', edgecolor='white')
    ax.set_title(col.replace("_", " ").title())
    ax.grid(axis='y')
for ax in axes[len(num_cols[1:]):]:
    ax.set_axis_off()

fig.tight_layout()
plt.show()


corr_mat = df[num_cols[1:]].corr()
sns.heatmap(corr_mat, annot=True, cmap="coolwarm", center=0)
plt.title("Correlation matrix of Numerical Variable")


num_plots = len(num_cols[1:])
rows, cols = 2, 3

fig, axes = plt.subplots(rows, cols, figsize=(15, 6))
axes = axes.flatten()  

for i, col in enumerate(num_cols[1:]):
    sns.boxplot(data=df, x='Personality', y=col, palette='Set2', linewidth=1.2, fliersize=4, ax=axes[i])
    axes[i].set_title(f'{col} by Personality', fontsize=14, fontweight='semibold', color='#2E4057')
    axes[i].set_xlabel('Personality', fontsize=12)
    axes[i].set_ylabel(col, fontsize=12)
    axes[i].grid(axis='y', linestyle='--', alpha=0.4)

# Turn off unused subplots
for j in range(i + 1, len(axes)):
    axes[j].set_axis_off()

fig.tight_layout()
plt.show()



#  'Stage_fear', 'Drained_after_socializing'
stage_fear_vc = df['Stage_fear'].value_counts()
das_vc = df['Drained_after_socializing'].value_counts()
fig, (ax1, ax2) = plt.subplots(ncols=2)
ax1.bar(stage_fear_vc.index, stage_fear_vc.values)
ax1.set_title("Stage Fear")
ax2.bar(das_vc.index, das_vc.values)
ax2.set_title("Drained After Socializing")
fig.tight_layout()
plt.show()


target_vc = df['Personality'].value_counts()
fig, ax = plt.subplots()
ax.bar(target_vc.index, target_vc.values)
ax.set_title("Personality")
plt.show()


df['Personality'].value_counts(normalize=True)


# Numerical columns
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)


df.loc[:, num_cols].isnull().sum()


# Categorical columns
cat_cols = ['Stage_fear', 'Drained_after_socializing']
for col in cat_cols:
    df[col].fillna('missing', inplace=True)

# sanity check
df.loc[:, cat_cols].isnull().sum()


from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


enc = OrdinalEncoder()
df[["Stage_fear", "Drained_after_socializing"]] = enc.fit_transform(df[["Stage_fear", "Drained_after_socializing"]])
df.loc[:5, ["Stage_fear", "Drained_after_socializing"]]


enc.categories_


le = LabelEncoder()
df['Personality'] = le.fit_transform(df['Personality'])


train_set = ['train', 'original']
train_df = df[df['data'].isin(train_set)].drop(columns=['data', 'id'], errors='ignore')
test_df = df[df['data'] == 'test'].drop(columns=['data', 'Personality'], errors='ignore')
print(f"Train data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")


X = train_df.drop('Personality', axis=1)
y = train_df['Personality']
print(f"X shape: {X.shape}\ny shape: {y.shape}")


from sklearn.model_selection import train_test_split, cross_val_score,StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from catboost import CatBoostClassifier


cat_model = CatBoostClassifier(
    iterations=300,
    learning_rate=0.1,
    depth=4,
    random_seed=5771,
    verbose=False
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=5771)

# Cross-validation score
cv_acc_score = cross_val_score(cat_model, X, y, cv=cv, scoring="accuracy")
cv_f1_score = cross_val_score(cat_model, X, y, cv=cv, scoring='f1_weighted')

print(f"CV Accuracy Scores: {cv_acc_score}")
print(f"Mean CV Accuracy: {cv_acc_score.mean()}")
print(f"Std CV Accuracy: {cv_acc_score.std()}")
print(f"CV Weighted F1 Score: {cv_f1_score}")
print(f"Mean CV Weighted F1 Score: {cv_f1_score.mean()}")
print(f"Std CV Weighted F1 Score: {cv_f1_score.std()}")


# Training on dataset and predicting on the same.
cat_model.fit(X, y)
y_pred = cat_model.predict(X)

test_acc = accuracy_score(y, y_pred)
test_f1 = f1_score(y, y_pred, average='weighted')

print(f"Accuracy: {test_acc}")
print(f"Weighted F1 score: {test_f1}")
print(f"Classification Report:\n{classification_report(y, y_pred, digits=3)}")


# Confusion Matrix
cm = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cat_model.classes_)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()


imp_feat = cat_model.get_feature_importance()
feat_names = X.columns if hasattr(X, 'columns') else [f"Feature {i}" for i in range(X.shape[1])]

df_feat_imp = (
    pd.DataFrame({'Feature': feat_names, 'Importance': imp_feat})
    .sort_values(by='Importance', ascending=False)
)

print("\nCatBoost Feature Importance:")
print(df_feat_imp)

plt.figure(figsize=(10, 6))
sns.barplot(data=df_feat_imp, x='Importance', y='Feature', palette='viridis')
plt.title('CatBoost Top Feature Importance')
plt.tight_layout()
plt.show()


test_df.columns


# dropping the 'id' column from test_df
test_features = test_df.drop(columns=['id'], errors='ignore')

predictions = cat_model.predict(test_features)

predictions = le.inverse_transform(predictions)

submission = pd.DataFrame({
    'id': test_df['id'],  # if 'id' exists in test_df
    'Personality': predictions
})

# Save submission
submission.to_csv('submission.csv', index=False)







