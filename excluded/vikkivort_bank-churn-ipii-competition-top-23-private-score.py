import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('/kaggle/input/bank-churn-mts/train.csv')


df_test = pd.read_csv('/kaggle/input/bank-churn-mts/test.csv')


df.head()


df.info()


df.describe()


df['CustomerId'].nunique()


df['Surname'].nunique()


df[(df['CustomerId'] == 15796834.0) & (df['Surname'] == 'Rizzo')]


df[(df['CustomerId'] == 15772573.0) & (df['Surname'] == 'Hao')]


df = df.drop_duplicates(subset=['CustomerId', 'Surname'])
df


df_enc = pd.get_dummies(df, columns = ['Gender', 'Geography'], dtype = int)
df_enc


features = ['Age', 'NumOfProducts',	'HasCrCard','IsActiveMember','Gender_Male', 'Gender_Female', 'Geography_France', 'Geography_Germany', 'Geography_Spain']


cnt = len(df_enc)
for ft in features:
  dp = df_enc.groupby(f'{ft}').agg(count=pd.NamedAgg(column=f"{ft}", aggfunc="count")).reset_index()
  dp = dp.sort_values(by = [f"{ft}"])
  if ft == 'Age':
      x = dp[f"{ft}"]  
  else:
      x = dp[f"{ft}"].apply(lambda x: str(x))
  y = dp['count']/cnt
  plt.bar(x,y)
  plt.xlabel(f'{ft}')
  plt.xticks(rotation = 45)
  plt.ylabel('Доля наблюдений')
  plt.title(f'Распределение фактора {ft}, train')
  plt.show()


df_test = df_test.drop_duplicates(subset=['CustomerId', 'Surname'])


df_test_enc = pd.get_dummies(df_test, columns = ['Gender', 'Geography'], dtype = int)
df_test_enc


cnt = len(df_test_enc)
for ft in features:
  dp = df_test_enc.groupby(f'{ft}').agg(count=pd.NamedAgg(column=f"{ft}", aggfunc="count")).reset_index()
  dp = dp.sort_values(by = [f"{ft}"])
  if ft == 'Age':
      x = dp[f"{ft}"]  
  else:
      x = dp[f"{ft}"].apply(lambda x: str(x))
  y = dp['count']/cnt
  plt.bar(x,y)
  plt.xlabel(f'{ft}')
  plt.xticks(rotation = 45)
  plt.ylabel('Доля наблюдений')
  plt.title(f'Распределение фактора {ft}, test')
  plt.show()


features1 = ['CreditScore', 'Tenure', 'Balance', 'EstimatedSalary']


cnt = len(df_enc)
for ft in features1:
  sns.histplot(df_enc, x = ft, kde = True)
  plt.title(f'Распределение фактора {ft}, train')
  plt.show()


cnt = len(df_test_enc)
for ft in features1:
  sns.histplot(df_test_enc, x = ft, kde = True)
  plt.title(f'Распределение фактора {ft}, test')
  plt.show()


import seaborn as sns

df_enc1 = df_enc[['CustomerId', 'CreditScore',	'Age',	'Balance',	'NumOfProducts',	'HasCrCard',	'IsActiveMember',	'EstimatedSalary', 'Gender_Male', 'Gender_Female', 'Geography_France', 'Geography_Germany', 'Geography_Spain', 'Exited']]
corr = df_enc1.corr()
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
plt.title('Correlation heatmap')


print(corr)


import lightgbm as lgb
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split


X = df_enc[['CreditScore', 'Tenure','Balance', 'Age', 'NumOfProducts', 'IsActiveMember', 'HasCrCard','EstimatedSalary', 'Gender_Male', 'Gender_Female', 'Geography_France', 'Geography_Germany', 'Geography_Spain']]
y = df_enc['Exited']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 41)


train_data = lgb.Dataset(X_train, label = y_train, categorical_feature=['IsActiveMember', 'HasCrCard', 'Gender_Male', 'Gender_Female', 'Geography_France', 'Geography_Germany', 'Geography_Spain'])
test_data = lgb.Dataset(X_test, label = y_test, reference=train_data)


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

params = {
    'random_state': [42],
    'seed': [42],
    'learning_rate': [0.08, 0.09, 0.095],
    'n_estimators': [70, 80, 85],
    'num_leaves': [15, 31],
    'max_depth': [3, 5],
    'reg_lambda': [1.2],
}

# 'reg_lambda': [1.1, 1.2],
#     'reg_alpha': [0.1],
#     'colsample_bytree': [0.7, 0.8],

model1 = lgb.LGBMClassifier()

grid = GridSearchCV(model1, params, cv=cv, scoring='roc_auc')
grid.fit(X_train, y_train)

print("Лучшие параметры:", grid.best_params_)


train_data = lgb.Dataset(X_train, label = y_train, categorical_feature=['IsActiveMember', 'HasCrCard', 'Gender_Male', 'Gender_Female', 'Geography_France', 'Geography_Germany', 'Geography_Spain'])
test_data = lgb.Dataset(X_test, label = y_test, reference=train_data)


params = {
    'objective': 'binary',
    'metric': 'auc',
    'seed': 42,
    'boosting_type': 'gbdt',
    'max_depth': 3,         
    'reg_lambda': 1.2,
    'num_leaves': 15,
    'n_estimators': 80,
    'learning_rate': 0.095,
    'colsample_bytree':0.8
}


# Обучение
model = lgb.train(params, train_data, valid_sets=[test_data])

# Предсказание
y_pred = model.predict(X_test)
print("ROC_AUC:", roc_auc_score(y_test, y_pred))


from sklearn.metrics import roc_curve, auc


# вычисляем ROC кривую
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

# вычисляем AUC
roc_auc = auc(fpr, tpr)

# строим график
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.show()


X = df_test_enc[['CreditScore', 'Tenure', 'Balance', 'Age', 'NumOfProducts', 'IsActiveMember', 'HasCrCard','EstimatedSalary', 'Gender_Male', 'Gender_Female', 'Geography_France', 'Geography_Germany', 'Geography_Spain']]


y_pred_t = model.predict(X)


y_pred_t = pd.DataFrame(y_pred_t)


y_pred_t


result = pd.concat([df_test_enc['id'], y_pred_t], axis=1)
result['Exited'] = result[0]
result = result[['id', 'Exited']]


result.to_csv('submission.csv')

