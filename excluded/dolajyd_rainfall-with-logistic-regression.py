import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


train_df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
ss_df = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')


train_df.info()


df = train_df.drop_duplicates(subset=['id'])


df.shape


test_df.info()


train_df.head()


test_df.head()


train_df.describe().T


test_df.describe().T


test_df['winddirection'] = test_df.winddirection.fillna(train_df.winddirection.median())


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve


X = train_df.drop(['id', 'rainfall'], axis=1)
y = train_df['rainfall']


test_df_1 = test_df.drop('id', axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y, )


logreg = LogisticRegression()

param_grid_lr = {'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}

grid_search = GridSearchCV(logreg, param_grid=param_grid_lr, cv=3, scoring='roc_auc')


grid_search.fit(X_train, y_train)


print('Best parameters: {}'.format(grid_search.best_params_))

print('Best Cross-Validation Score: {:.2f}'.format(grid_search.best_score_))


y_pred = grid_search.predict_proba(X_test)[:,1]

print('ROC AUC: {:.6f}'.format(roc_auc_score(y_test, y_pred)))


fpr, tpr, thresholds = roc_curve(y_test, y_pred)

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr, tpr, label='Logistic Regression')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Logistic Regression ROC Curve')

plt.show()


test_df_pred = grid_search.predict_proba(test_df_1)[:,1]


ss_df.head()


result = pd.DataFrame({'id':ss_df.id, 'rainfall':test_df_pred})
result.to_csv("logreg.csv", index=False)




