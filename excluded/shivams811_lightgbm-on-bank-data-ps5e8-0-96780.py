import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier



bank_train_ds = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
bank_test_ds = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')


bank_test_ds.shape


bank_train_ds.head()


bank_train_ds.info()


bank_train_ds.describe()


bank_train_ds.isna().sum()


bank_train_ds['y'].value_counts(normalize=True).plot(kind='bar', title='Target Variable Distribution',)



categorical_cols = ['job', 'education', 'contact', 'month', 'poutcome', 'marital', 'default', 'housing', 'loan']
for col in categorical_cols:
    bank_train_ds[col].value_counts().plot(kind='bar', title=f'Distribution of {col}')
    plt.show()



for col in categorical_cols:
    pd.crosstab(bank_train_ds[col], bank_train_ds['y'], normalize='index').plot(kind='bar', stacked=True, title=f'{col} vs y')
    plt.show()



plt.figure(figsize=(10,8))
# corr = bank_train_ds[numerical_cols].corr()
sns.heatmap(bank_train_ds.corr(numeric_only=True), annot=True, cmap='coolwarm')



bank_train_ds['default']=np.where(bank_train_ds['default']=='yes', 1,0)
bank_train_ds['housing']=np.where(bank_train_ds['housing']=='yes', 1,0)
bank_train_ds['loan']=np.where(bank_train_ds['loan']=='yes', 1,0)

bank_test_ds['default']=np.where(bank_test_ds['default']=='yes', 1,0)
bank_test_ds['housing']=np.where(bank_test_ds['housing']=='yes', 1,0)
bank_test_ds['loan']=np.where(bank_test_ds['loan']=='yes', 1,0)


month_map = {
    'jan': 0, 'feb': 1, 'mar': 2, 'apr': 3,
    'may': 4, 'jun': 5, 'jul': 6, 'aug': 7,
    'sep': 8, 'oct': 9, 'nov': 10, 'dec': 11
}
bank_train_ds['month'] = bank_train_ds['month'].map(month_map)
bank_test_ds['month'] = bank_test_ds['month'].map(month_map)


education_map = {
    'unknown': 0,
    'primary': 1,
    'secondary': 2,
    'tertiary': 3
}

bank_train_ds['education'] = bank_train_ds['education'].map(education_map)
bank_test_ds['education'] = bank_test_ds['education'].map(education_map)


marital_map = {
    'divorced': 0,
    'married': 1,
    'single': 2
}

bank_train_ds['marital'] = bank_train_ds['marital'].map(marital_map)
bank_test_ds['marital'] = bank_test_ds['marital'].map(marital_map)


bank_train_ds.head()


bank_train_ds = pd.get_dummies(bank_train_ds, columns=['job', 'contact', 'poutcome'], drop_first=True)

bank_test_ds = pd.get_dummies(bank_test_ds, columns=['job', 'contact', 'poutcome'], drop_first=True)


bank_train_ds.shape


X = bank_train_ds.drop(columns=['y', 'id'])
y = bank_train_ds['y']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)


st=StandardScaler()
X_train = st.fit_transform(X_train)
X_test = st.transform(X_test)

ids = bank_test_ds['id']
bank_test_ds = bank_test_ds.drop(columns=['id'])
bank_test_ds = st.transform(bank_test_ds)


from xgboost import XGBClassifier

xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)
y_probs = xgb_model.predict_proba(X_test)[:, 1]

# Print ROC AUC Score
roc_auc = roc_auc_score(y_test, y_probs)
print("ROC AUC Score:", roc_auc)
print("Training ROC AUC Score:", roc_auc_score(y_train, xgb_model.predict_proba(X_train)[:, 1]))



model = LGBMClassifier(n_estimators= 500, max_depth= 7, random_state= 2)
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_test)[:, 1]
score = roc_auc_score(y_test, y_pred)
print("ROC-AUC: ", score)


predictions = model.predict_proba(bank_test_ds)

submission = pd.DataFrame({
    'id': ids,       
    'y': predictions[:,1]        
})
submission.to_csv("submission.csv", index=False)


submission.shape

