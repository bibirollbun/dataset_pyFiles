import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, classification_report

import warnings
warnings.filterwarnings('ignore')


train_path = '/kaggle/input/playground-series-s5e12/train.csv'
test_path = '/kaggle/input/playground-series-s5e12/test.csv'

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)


def dataset_summary(datasets):
    summary = []

    for name, df, path in datasets:
        size_on_disk = os.path.getsize(path) / (1024 * 1024)  # MB
        size_in_memory = df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
        rows, cols = df.shape

        summary.append({
            "Dataset": name,
            "Size on Disk (MB)": round(size_on_disk, 2),
            "Size in Memory (MB)": round(size_in_memory, 2),
            "# of Rows": rows,
            "# of Cols": cols
        })

    return pd.DataFrame(summary)


datasets = [
    ("train", train, train_path),
    ("test", test, test_path)
]

dataset_summary(datasets)


train.head()


test.head()


train.info()


train.isnull().sum()


train.duplicated().sum()


num_cols = train.select_dtypes(include='number').columns

for col in num_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(train[col], kde=True, bins=50)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()


for col in num_cols:
    plt.figure(figsize=(6,2))
    sns.boxplot(x=train[col])
    plt.title(f'Boxplot of {col}')
    plt.show()


plt.figure(figsize=(6,4))
sns.countplot(x=train['diagnosed_diabetes'])
plt.title('Target Variable Distribution')
plt.show()


cat_cols = train.select_dtypes(include='object').columns

for col in cat_cols:
    plt.figure(figsize=(6,4))
    sns.countplot(y=train[col], order=train[col].value_counts().index)
    plt.title(f'Distribution of {col}')
    plt.xlabel('Count')
    plt.ylabel(col)
    plt.show()


for col in cat_cols:
    plt.figure(figsize=(6,4))
    
    sns.countplot(
        data=train,
        y=col,
        hue='diagnosed_diabetes',
        order=train[col].value_counts().index
    )
    
    plt.title(f'{col} vs Diagnosed Diabetes')
    plt.xlabel('Count')
    plt.ylabel(col)
    plt.legend(title='Diabetes')
    plt.tight_layout()
    plt.show()


num_df = train.select_dtypes(include='number').drop(columns=['id'])

corr = num_df.corr()

target_corr = (
    corr[['diagnosed_diabetes']]
    .sort_values(by='diagnosed_diabetes', ascending=False)
)

plt.figure(figsize=(10,8))
sns.heatmap(
    target_corr,
    annot=True,
    cmap='coolwarm',
    fmt=".2f"
)

plt.title('Correlation of Numeric Features with Diagnosed Diabetes')
plt.tight_layout()
plt.show()



X = train.drop(columns=['diagnosed_diabetes'])
y = train['diagnosed_diabetes']

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


cat_cols = X_train.select_dtypes(include='object').columns.tolist()

model = CatBoostClassifier(
    iterations=10000,          
    learning_rate=0.03,        
    depth=8,
    loss_function='Logloss',
    eval_metric='AUC',
    random_seed=42,
    od_type='Iter',           
    od_wait=300,             
    verbose=200
)

model.fit(
    X_train, y_train,
    cat_features=cat_cols,
    eval_set=(X_valid, y_valid),
    use_best_model=True
)



y_pred_proba = model.predict_proba(X_valid)[:, 1]
auc = roc_auc_score(y_valid, y_pred_proba)

print(f"Validation ROC-AUC: {auc:.4f}")


y_pred = (y_pred_proba >= 0.5).astype(int)
print(classification_report(y_valid, y_pred))


test_pred_proba = model.predict_proba(test)[:, 1]

submission = pd.DataFrame({
    'id': test['id'],
    'diagnosed_diabetes': test_pred_proba
})

submission.to_csv('submission.csv', index=False)


feature_importance = model.get_feature_importance()

feat_imp_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': feature_importance
}).sort_values(by='importance', ascending=False)

feat_imp_df.head(25)




