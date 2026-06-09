import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef


train = pd.read_csv("/kaggle/input/playground-series-s4e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s4e8/test.csv")


test.shape, train.shape


train.head(10)


train.columns.tolist()


train.info()


test.isna().sum(), test.isna().mean() * 100


train.nunique()


train.iloc[:, 1:]


len(train[train['class'] == 'e']), len(train[train['class'] == 'p'])


train.iloc[:, 1:].describe()


plt.figure(figsize=(10, 6))
sns.histplot(train['cap-diameter'], kde=True, bins=30)
plt.title('Распределение числовой переменной')
plt.show()


plt.figure(figsize=(10, 6))
sns.histplot(train['stem-height'], kde=True, bins=30)
plt.title('Распределение числовой переменной')
plt.show()


plt.figure(figsize=(10, 6))
sns.histplot(np.log(train['stem-width']), kde=True, bins=30)
plt.title('Распределение числовой переменной')
plt.show()


num_cols = ['cap-diameter', 'stem-height', 'stem-width']
cat_cols = [c for c in train.columns if c not in num_cols + ['id', 'class']]


for col in cat_cols:
    train[col] = train[col].fillna('Missing').astype(str)
    test[col] = test[col].fillna('Missing').astype(str)


target_map = {'e': 0, 'p': 1}
inv_target_map = {0: 'e', 1: 'p'}
train['class_encoded'] = train['class'].map(target_map)


X = train.drop(['id', 'class', 'class_encoded'], axis=1)
y = train['class_encoded']
X_test = test.drop(['id'], axis=1)


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


model = CatBoostClassifier(
    iterations=2000,          
    learning_rate=0.05,
    eval_metric='MCC',         
    cat_features=cat_cols,     
    task_type="GPU",           
    devices='0',
    early_stopping_rounds=100,
    verbose=100,
    random_seed=42
)


model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    use_best_model=True
)


val_preds = model.predict(X_val)
score = matthews_corrcoef(y_val, val_preds)
print(f"Validation MCC Score: {score:.5f}")


test_preds = model.predict(X_test)
submission = pd.DataFrame({
    'id': test['id'],
    'class': [inv_target_map[x] for x in test_preds]
})

submission.to_csv('submission.csv', index=False)

