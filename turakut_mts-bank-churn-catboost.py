import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns



train = pd.read_csv('/kaggle/input/bank-churn-competition-by-ipii-hs-ex-mts/train.csv')


print(f"Размер данных: {train.shape}")  # (строки, столбцы)
print("\nПервые 5 строк:")
display(train.head())
print("\nИнформация о данных:")
train.info()
print("\nОписательная статистика:")
display(train.describe())



plt.figure(figsize=(6, 4))
sns.countplot(data=train, x='Exited')
plt.title("Распределение ушедших клиентов (0 = остался, 1 = ушел)")
plt.show()

print(train['Exited'].value_counts(normalize=True))


num_cols = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary', 'Tenure']
train[num_cols].hist(bins=30, figsize=(12, 8))
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(data=train[['CreditScore', 'Age', 'Balance']])
plt.title("Распределение числовых признаков (без выбросов)")
plt.show()


train = pd.read_csv('/kaggle/input/bank-churn-competition-by-ipii-hs-ex-mts/train.csv')
test = pd.read_csv('/kaggle/input/bank-churn-competition-by-ipii-hs-ex-mts/test.csv')

def add_features(df):
    
    df['BalanceToSalary'] = df['Balance'] / (df['EstimatedSalary'] + 1e-6)
    df['CreditScoreToAge'] = df['CreditScore'] / df['Age']
    df['IsHighBalance'] = (df['Balance'] > df['Balance'].median()).astype(int)
    
    df['InactiveWithProducts'] = ((df['IsActiveMember'] == 0) & 
                                (df['NumOfProducts'] > 1)).astype(int)
    return df


train = add_features(train)
test = add_features(test)


X = train.drop(['Exited', 'CustomerId', 'Surname'], axis=1)
y = train['Exited']
X_test = test.drop(['CustomerId', 'Surname'], axis=1)

# Категориальные признаки 
cat_features = ['Geography', 'Gender', 'IsHighBalance', 'InactiveWithProducts']


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Параметры p.s. которые я подобрал
params = {
    'iterations': 1200,
    'learning_rate': 0.035,
    'depth': 6,
    'l2_leaf_reg': 3,
    'border_count': 128,
    'random_strength': 0.5,
    'bagging_temperature': 0.8,
    'eval_metric': 'AUC',
    'loss_function': 'Logloss',
    'random_seed': 42,
    'verbose': 200  # Вывод инфы
}


model = CatBoostClassifier(**params)
model.fit(
    X_train, y_train,
    cat_features=cat_features,  
    eval_set=(X_valid, y_valid),
    early_stopping_rounds=100,
    use_best_model=True
)


test_probs = model.predict_proba(X_test)[:, 1]


submission = pd.DataFrame({
    'id': test['id'],
    'Exited': test_probs
})
submission.to_csv('submission.csv', index=False)


valid_probs = model.predict_proba(X_valid)[:, 1]
print(f"Validation ROC-AUC: {roc_auc_score(y_valid, valid_probs):.6f}")

