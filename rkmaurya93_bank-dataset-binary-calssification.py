import pandas as pd
import numpy as np


df=pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv",index_col=0)


df.head()


df.shape


categorical_features=df.select_dtypes(include='object').columns


categorical_features


df[categorical_features].head()


df[categorical_features].nunique()


df['poutcome'].value_counts()


# Encoding the Poutcome column:
from category_encoders import TargetEncoder


te_poutcome=TargetEncoder(cols=['poutcome'])
df['poutcome']=te_poutcome.fit_transform(df['poutcome'],df['y'])


df['poutcome']


df['poutcome'].value_counts()


df['month'].value_counts()


from category_encoders import TargetEncoder

te_month = TargetEncoder(smoothing=0.3)
df['month'] = te_month.fit_transform(df['month'], df['y'])


df['month'].value_counts()


df['contact'].value_counts()


# Apply target encoding on the contact column
te_contact=TargetEncoder()
df['contact']=te_contact.fit_transform(df['contact'],df['y'])


df['loan'].value_counts()


# Apply label encoding
from sklearn.preprocessing import LabelEncoder


le=LabelEncoder()
df['loan']=le.fit_transform(df['loan'])


df['housing'].value_counts()


df['housing']=le.fit_transform(df['housing'])


df['default'].value_counts()


df['default']=le.fit_transform(df['default'])


df['education'].value_counts()


te_education=TargetEncoder()


df['education']=te_education.fit_transform(df['education'],df['y'])


df['marital'].value_counts()


te_marital=TargetEncoder()
df['marital']=te_marital.fit_transform(df['marital'],df['y'])


te_job=TargetEncoder()


df['job']=te_job.fit_transform(df['job'],df['y'])


df.head()


df.head()


import seaborn as sns


import matplotlib.pyplot as plt 


df['age'].describe()


sns.boxplot(df['age'])


sns.displot(df['age'],kind='kde')


df['balance'].describe()


sns.boxplot(df['balance'])


sns.displot(df['balance'],kind='kde')


df['duration'].describe()


sns.boxplot(df['duration'])


sns.displot(df['duration'],kind='kde')


df.corr()


corr_matrix=df.corr()


plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix,annot=True,cmap='coolwarm',fmt='.2f',linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()


from sklearn.model_selection import train_test_split


x=df.drop('y',axis=1)
y=df['y']


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


x_train.head()


y_train.head()


from sklearn.preprocessing import StandardScaler


ss=StandardScaler()


x_train_transformed=ss.fit_transform(x_train)
x_test_transformed=ss.transform(x_test)


x_train_transformed.shape


from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ✅ Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(x_train, y_train)
rf_preds = rf.predict_proba(x_test)[:, 1]

# ✅ Train Neural Network
nn = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
nn.fit(x_train_transformed, y_train)
nn_preds = nn.predict_proba(x_test_transformed)[:, 1]

# ✅ Train XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb.fit(x_train, y_train)
xgb_preds = xgb.predict_proba(x_test)[:, 1]

# ✅ Train LightGBM
lgbm = LGBMClassifier(random_state=42)
lgbm.fit(x_train, y_train)
lgbm_preds = lgbm.predict_proba(x_test)[:, 1]



# ✅ Stack predictions horizontally
stacked_preds = np.vstack([rf_preds, nn_preds, xgb_preds, lgbm_preds]).T

# ✅ Train meta learner (Logistic Regression)
meta_model = LogisticRegression()
meta_model.fit(stacked_preds, y_test)

# ✅ Final predictions
final_preds = meta_model.predict(stacked_preds)

# ✅ Evaluate
accuracy = accuracy_score(y_test, final_preds)
print("Stacked Model Accuracy:", round(accuracy * 100, 2), "%")


from xgboost import XGBClassifier

meta_model_xg_boost = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
meta_model_xg_boost.fit(stacked_preds, y_test)
final_preds = meta_model.predict(stacked_preds)

from sklearn.metrics import accuracy_score
print("Stacked Accuracy with XGBoost Meta:", round(accuracy_score(y_test, final_preds) * 100, 2), "%")


df_test=pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv',index_col=0)


df_test.head()


df_test['job']=te_job.transform(df_test['job'])
df_test['marital']=te_marital.transform(df_test['marital'])
df_test['education']=te_education.transform(df_test['education'])
df_test['contact']=te_contact.transform(df_test['contact'])
df_test['month']=te_month.transform(df_test['month'])



df_test['poutcome']=te_poutcome.transform(df_test['poutcome'])


df.head()


df_test['default']=le.transform(df_test['default'])
df_test['housing']=le.transform(df_test['housing'])
df_test['loan']=le.transform(df_test['loan'])



df_test.head()


df_test_scaled=ss.transform(df_test)


# Step 1: Get base model probabilities
rf_probs = rf.predict_proba(df_test_scaled)[:, 1]
nn_probs = nn.predict_proba(df_test_scaled)[:, 1]
xgb_probs = xgb.predict_proba(df_test_scaled)[:, 1]
lgbm_probs = lgbm.predict_proba(df_test_scaled)[:, 1]

# Step 2: Stack them horizontally
stacked_probs = np.vstack([rf_probs, nn_probs, xgb_probs, lgbm_probs]).T

# Step 3: Predict probabilities using meta-model
final_probs_lr = meta_model.predict_proba(stacked_probs)
probs_class_1_lr = final_probs_lr[:, 1]


probs_class_1_lr


final_probs_xgb = meta_model_xg_boost.predict_proba(stacked_probs)
probs_class_1_xgb = final_probs_xgb[:, 1]


probs_class_1_xgb 


submission_df = pd.DataFrame({
    'y': probs_class_1_lr  # or final_preds
}, index=df_test.index)

submission_df.to_csv('submission_probs.csv')




