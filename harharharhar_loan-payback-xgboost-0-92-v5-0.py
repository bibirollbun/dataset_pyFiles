import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample


train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')


train.head()


train.info()


train_clean = train.drop(['id'],axis=1)
test_clean = test.drop(['id'],axis=1)


columns = ['gender','marital_status','education_level','employment_status','loan_purpose','grade_subgrade']
ohe = OneHotEncoder(sparse_output = False).set_output(transform='pandas')
feature_array = ohe.fit_transform(train_clean[columns])
train_encoded = pd.concat([train_clean, feature_array], axis=1).drop(columns=columns)
train_encoded.head()


feature_array_test = ohe.fit_transform(test_clean[columns])
test_encoded = pd.concat([test_clean, feature_array_test], axis=1).drop(columns=columns)
test_encoded.head()


corr = train_encoded.corr()


corr


plt.figure(figsize=(16,10))
mask = np.triu(np.ones_like(corr))
sns.heatmap(corr, cmap='Blues', vmin=0, linewidths=0.9,mask=mask)


correlation_with_target = train_encoded.corr()['loan_paid_back'].sort_values(ascending=False)


threshold = 0.15
strong_correlations = correlation_with_target[abs(correlation_with_target) > threshold]

plt.figure(figsize=(10, 6))
sns.barplot(x=strong_correlations.index, y=strong_correlations.values, palette='viridis')
plt.title(f'Strong Correlations (> {threshold}) with loan_paid_back')
plt.xticks(rotation=90)
plt.ylabel('Correlation Coefficient')
plt.xlabel('Features')
plt.show()



majority_class = train_encoded[train_encoded['loan_paid_back'] == 1]
minority_class = train_encoded[train_encoded['loan_paid_back'] == 0]
minority_upsampled = resample(minority_class,
                              replace=True,        
                              n_samples=len(majority_class),  
                              random_state=42)
balanced_data = pd.concat([majority_class, minority_upsampled])


X_balanced = balanced_data.drop('loan_paid_back', axis=1)
y_balanced = balanced_data['loan_paid_back']


scaler = StandardScaler()
X_balanced = scaler.fit_transform(X_balanced)
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=101)
test_sca = scaler.transform(test_encoded)


xgboost_model = XGBClassifier(
    n_estimators=7500,
    learning_rate=0.01,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="auc",
    early_stopping_rounds=75,
    alpha=1.0,  # L1 regularization term
    
)

xgboost_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=100
)


xgboost_preds = xgboost_model.predict_proba(test_sca)[:, 1]
submission = pd.DataFrame({
    'id': test['id'],
    'loan_paid_back': xgboost_preds
})

submission.to_csv('/kaggle/working/submission_loan_xgb.csv', index=False)
submission.head()


