import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)

import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
sns.set()

import warnings
warnings.filterwarnings('ignore')



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


train.head()


test.head()


train.shape, test.shape


train.info()


test.info()


train.describe().T


test.describe().T


train.columns


train.isna().sum()[train.isna().sum()>0]


test.isna().sum()[test.isna().sum()>0]


train.dtypes


test.dtypes


train = train.drop(columns='id', axis=1)
train.shape, train.head()


test = test.drop(columns = 'id', axis=1)
test.shape, test.head()


train.nunique()


for col in train.columns:
  val = train[col].unique().tolist()
  print(col)
  print(sorted(val))
  print('*'*30)


for col in test.columns:
  val = test[col].unique().tolist()
  print(col)
  print(sorted(val))
  print('*'*30)


test[test['winddirection'].isna()]


test['winddirection'].mean(), test['winddirection'].median(), test['winddirection'].nunique()


test['winddirection'].unique()



test.dtypes


test.corr()['winddirection']


test.groupby('dewpoint')['winddirection'].mean()


dewpoint_grouped = test.groupby('dewpoint')['winddirection'].median()


target_dewpoint = 22.0
if target_dewpoint in dewpoint_grouped.index:
  print (dewpoint_grouped[target_dewpoint])
else:
  print ("Not matched")



temp_grouped = test.groupby('temparature')['winddirection'].median()
temp_grouped


target_temp = 30.6
if target_temp in temp_grouped.index:
  print (temp_grouped[target_temp])
else:
  print ("Not matched")



test[test['winddirection'].isna()]


#test['winddirection'] = test['winddirection'].fillna(test['dewpoint'].map(dewpoint_grouped))
test['winddirection'] = test['winddirection'].fillna(103.0)



test.iloc[517]


test.isna().sum()


plt.figure(figsize=(25,10))
sns.boxplot(train)
plt.show()


def box_dist_plot(df, col):
  plt.figure(figsize=(14,8))
  plt.subplot(1,2,1)
  sns.boxplot(df[col])

  plt.subplot(1,2,2)
  sns.distplot(df[col], kde=True)
  plt.title(col)
  plt.show()



for col in train.columns:
  box_dist_plot(train, col)


sns.pairplot(train)


train['rainfall'].value_counts()


train['rainfall'].value_counts(normalize=True)


X = train.drop(columns='rainfall', axis=1)
y = train[['rainfall']]


X.head()


y.head()


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y, random_state=42)


y_train.value_counts(), y_test.value_counts()


X_train.shape


X_train.head()


from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


pd.DataFrame(X_train_scaled).head()


from sklearn.model_selection import GridSearchCV, StratifiedKFold


from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, log_loss,
    matthews_corrcoef, cohen_kappa_score, brier_score_loss,
    confusion_matrix, classification_report, roc_curve
)



overall_result = pd.DataFrame(columns=['model', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'log_loss', 'matthews_corrcoef', 'cohen_kappa', 'brier_score'])

def evaluate_model(model, y_train, y_train_pred, y_proba=None):
    results = {
      'model': model,
      'accuracy': accuracy_score(y_train, y_train_pred),
      'precision': precision_score(y_train, y_train_pred),
      'recall': recall_score(y_train, y_train_pred),
      'f1': f1_score(y_train, y_train_pred),
      'roc_auc': roc_auc_score(y_train, y_train_pred),
      'log_loss': log_loss(y_train, y_train_pred),
      'matthews_corrcoef': matthews_corrcoef(y_train, y_train_pred),
      'cohen_kappa': cohen_kappa_score(y_train, y_train_pred),
      'brier_score': brier_score_loss(y_train, y_proba)}

    # Calculate PR AUC
    precision, recall, _ = precision_recall_curve(y_train, y_proba)
    results['pr_auc'] = auc(recall, precision)
    return results



from sklearn.linear_model import LogisticRegression


def logisticRegModel(X_train, y_train, X_test, y_test):
  lr = LogisticRegression()
  lr.fit(X_train, y_train)
  y_train_pred = lr.predict(X_train)
  y_test_pred = lr.predict(X_test)
  y_train_proba = lr.predict_proba(X_train)[:,1]
  y_test_proba = lr.predict_proba(X_test)[:,1]
  return y_train_pred, y_test_pred, y_train_proba, y_test_proba


y_train_pred, y_test_pred, y_train_proba, y_test_proba = logisticRegModel(X_train_scaled, y_train, X_test_scaled, y_test)


lr_train_results = evaluate_model('LogisticRegression-Train', y_train, y_train_pred, y_train_proba)
lr_test_results = evaluate_model('LogisticRegression-Test', y_test, y_test_pred, y_test_proba)


lr_train_results


lr_test_results


overall_result = pd.concat([overall_result, pd.DataFrame([lr_train_results])], ignore_index=True)
overall_result = pd.concat([overall_result, pd.DataFrame([lr_test_results])], ignore_index=True)


overall_result


from sklearn.tree import DecisionTreeClassifier


def decisionTreeModel(X_train, y_train, X_test, y_test):
  dt = DecisionTreeClassifier()
  dt.fit(X_train, y_train)
  y_train_pred = dt.predict(X_train)
  y_test_pred = dt.predict(X_test)
  y_train_proba = dt.predict_proba(X_train)[:,1]
  y_test_proba = dt.predict_proba(X_test)[:,1]
  return y_train_pred, y_test_pred, y_train_proba, y_test_proba


y_train_pred, y_test_pred, y_train_proba, y_test_proba = decisionTreeModel(X_train_scaled, y_train, X_test_scaled, y_test)


dt_train_results = evaluate_model('DecisionTreeClassifier-Train',y_train, y_train_pred, y_train_proba)
dt_test_results = evaluate_model('DecisionTreeClassifier-Test',y_test, y_test_pred, y_test_proba)


dt_train_results


dt_test_results


# Training the model with original data
y_train_pred, y_test_pred, y_train_proba, y_test_proba = decisionTreeModel(X_train, y_train, X_test, y_test)


dt_train_results = evaluate_model('DecisionTreeClassifier-Train',y_train, y_train_pred, y_train_proba)
dt_test_results = evaluate_model('DecisionTreeClassifier-Test',y_test, y_test_pred, y_test_proba)


dt_train_results


dt_test_results


overall_result = pd.concat([overall_result, pd.DataFrame([dt_train_results])], ignore_index=True)
overall_result = pd.concat([overall_result, pd.DataFrame([dt_test_results])], ignore_index=True)


overall_result


from sklearn.ensemble import RandomForestClassifier


def randomForestModel(X_train, y_train, X_test, y_test):
  rf = RandomForestClassifier()
  rf.fit(X_train, y_train)
  y_train_pred = rf.predict(X_train)
  y_test_pred = rf.predict(X_test)
  y_train_proba = rf.predict_proba(X_train)[:,1]
  y_test_proba = rf.predict_proba(X_test)[:,1]
  return y_train_pred, y_test_pred, y_train_proba, y_test_proba



y_train_pred, y_test_pred, y_train_proba, y_test_proba = randomForestModel(X_train, y_train, X_test, y_test)


rf_train_results = evaluate_model('RandomForestClassifier-Train',y_train, y_train_pred, y_train_proba)
rf_test_results = evaluate_model('RandomForestClassifier-Test',y_test, y_test_pred, y_test_proba)


rf_train_results


rf_test_results


overall_result = pd.concat([overall_result, pd.DataFrame([rf_train_results])], ignore_index=True)
overall_result = pd.concat([overall_result, pd.DataFrame([rf_test_results])], ignore_index=True)


overall_result


from sklearn.ensemble import GradientBoostingClassifier


def gradientBoostingModel(X_train, y_train, X_test, y_test):
  gb = GradientBoostingClassifier()
  gb.fit(X_train, y_train)
  y_train_pred = gb.predict(X_train)
  y_test_pred = gb.predict(X_test)
  y_train_proba = gb.predict_proba(X_train)[:,1]
  y_test_proba = gb.predict_proba(X_test)[:,1]
  return y_train_pred, y_test_pred, y_train_proba, y_test_proba


y_train_pred, y_test_pred, y_train_proba, y_test_proba = gradientBoostingModel(X_train, y_train, X_test, y_test)


gb_train_results = evaluate_model('GradientBoostingClassifier-Train',y_train, y_train_pred, y_train_proba)
gb_test_results = evaluate_model('GradientBoostingClassifier-Test',y_test, y_test_pred, y_test_proba)


overall_result = pd.concat([overall_result, pd.DataFrame([gb_train_results])], ignore_index=True)
overall_result = pd.concat([overall_result, pd.DataFrame([gb_test_results])], ignore_index=True)


overall_result


#pip install xgboost


from xgboost import XGBClassifier


def xgboostModel(X_train, y_train, X_test, y_test):
  xgb = XGBClassifier()
  xgb.fit(X_train, y_train)
  y_train_pred = xgb.predict(X_train)
  y_test_pred = xgb.predict(X_test)
  y_train_proba = xgb.predict_proba(X_train)[:,1]
  y_test_proba = xgb.predict_proba(X_test)[:,1]
  return y_train_pred, y_test_pred, y_train_proba, y_test_proba


xgb_train_results = evaluate_model('XGBClassifier-Train',y_train, y_train_pred, y_train_proba)
xgb_test_results = evaluate_model('XGBClassifier-Test',y_test, y_test_pred, y_test_proba)


overall_result = pd.concat([overall_result, pd.DataFrame([xgb_train_results])], ignore_index=True)
overall_result = pd.concat([overall_result, pd.DataFrame([xgb_test_results])], ignore_index=True)


overall_result.tail()


#pip install lightgbm


from lightgbm import LGBMClassifier


def lightgbmModel(X_train, y_train, X_test, y_test):
  lgb = LGBMClassifier()
  lgb.fit(X_train, y_train)
  y_train_pred = lgb.predict(X_train)
  y_test_pred = lgb.predict(X_test)
  y_train_proba = lgb.predict_proba(X_train)[:,1]
  y_test_proba = lgb.predict_proba(X_test)[:,1]
  return y_train_pred, y_test_pred, y_train_proba, y_test_proba


lgb_train_results = evaluate_model('LGBMClassifier-Train',y_train, y_train_pred, y_train_proba)
lgb_test_results = evaluate_model('LGBMClassifier-Test',y_test, y_test_pred, y_test_proba)


overall_result = pd.concat([overall_result, pd.DataFrame([lgb_train_results])], ignore_index=True)
overall_result = pd.concat([overall_result, pd.DataFrame([lgb_test_results])], ignore_index=True)


overall_result.tail()


#pip install catboost


# pip uninstall numpy catboost -y


# pip install numpy catboost


from catboost import CatBoostClassifier


def catboostModel(X_train, y_train, X_test, y_test):
  cb = CatBoostClassifier(verbose=0)
  cb.fit(X_train, y_train)
  y_train_pred = cb.predict(X_train)
  y_test_pred = cb.predict(X_test)
  y_train_proba = cb.predict_proba(X_train)[:,1] # Selecting probabilities for the positive class (class 1)
  y_test_proba = cb.predict_proba(X_test)[:,1] # Selecting probabilities for the positive class (class 1)
  return y_train_pred, y_test_pred, y_train_proba, y_test_proba


y_train_pred, y_test_pred, y_train_proba, y_test_proba = catboostModel(X_train, np.array(y_train), X_test, np.array(y_test))


cb_train_results = evaluate_model('CatBoostClassifier-Train',y_train, y_train_pred, y_train_proba)
cb_test_results = evaluate_model('CatBoostClassifier-Test',y_test, y_test_pred, y_test_proba)


overall_result = pd.concat([overall_result, pd.DataFrame([cb_train_results])], ignore_index=True)
overall_result = pd.concat([overall_result, pd.DataFrame([cb_test_results])], ignore_index=True)


overall_result


from sklearn.svm import SVC


def svcModel(X_train, y_train, X_test, y_test):
  svc = SVC(probability=True) # Set probability=True to enable probability estimates
  svc.fit(X_train, y_train)
  y_train_pred = svc.predict(X_train)
  y_test_pred = svc.predict(X_test)
  y_train_proba = svc.predict_proba(X_train)[:,1] # Use predict_proba to get probabilities
  y_test_proba = svc.predict_proba(X_test)[:,1] # Use predict_proba to get probabilities
  return y_train_pred, y_test_pred, y_train_proba, y_test_proba


y_train_pred, y_test_pred, y_train_proba, y_test_proba = svcModel(X_train_scaled, y_train, X_test_scaled, y_test)


svc_train_results = evaluate_model('SVC-Train',y_train, y_train_pred, y_train_proba)
svc_test_results = evaluate_model('SVC-Test',y_test, y_test_pred, y_test_proba)


overall_result = pd.concat([overall_result, pd.DataFrame([svc_train_results])], ignore_index=True)
overall_result = pd.concat([overall_result, pd.DataFrame([svc_test_results])], ignore_index=True)


overall_result


from sklearn.naive_bayes import GaussianNB


def gaussianNBModel(X_train, y_train, X_test, y_test):
  gnb = GaussianNB()
  gnb.fit(X_train, y_train)
  y_train_pred = gnb.predict(X_train)
  y_test_pred = gnb.predict(X_test)
  y_train_propa = gnb.predict_proba(X_train)[:,1] # Use predict_proba to get probabilities
  y_test_proba = gnb.predict_proba(X_test)[:,1] # Use predict_proba to get probabilities
  return y_train_pred, y_test_pred, y_train_proba, y_test_proba


y_train_pred, y_test_pred, y_train_proba, y_test_proba = gaussianNBModel(X_train_scaled, y_train, X_test_scaled, y_test)


gnb_train_results = evaluate_model('GaussianNB-Train',y_train, y_train_pred, y_train_proba)
gnb_test_results = evaluate_model('GaussianNB-Test',y_test, y_test_pred, y_test_proba)


overall_result = pd.concat([overall_result, pd.DataFrame([gnb_train_results])], ignore_index=True)
overall_result = pd.concat([overall_result, pd.DataFrame([gnb_test_results])], ignore_index=True)


overall_result.tail()


from sklearn.naive_bayes import BernoulliNB


def bernoulliNBModel(X_train, y_train, X_test, y_test):
  bnb = BernoulliNB()
  bnb.fit(X_train, y_train)
  y_train_pred = bnb.predict(X_train)
  y_test_pred = bnb.predict(X_test)
  y_train_proba = bnb.predict_proba(X_train)[:,1] # Use predict_proba to get probabilities
  y_test_proba = bnb.predict_proba(X_test)[:,1] # Use predict_proba to get probabilities
  return y_train_pred, y_test_pred, y_train_proba, y_test_proba


y_train_pred, y_test_pred, y_train_proba, y_test_proba = bernoulliNBModel(X_train_scaled, y_train, X_test_scaled, y_test)


bnb_train_results = evaluate_model('BernoulliNB-Train',y_train, y_train_pred, y_train_proba)
bnb_test_results = evaluate_model('BernoulliNB-Test',y_test, y_test_pred, y_test_proba)


overall_result = pd.concat([overall_result, pd.DataFrame([bnb_train_results])], ignore_index=True)
overall_result = pd.concat([overall_result, pd.DataFrame([bnb_test_results])], ignore_index=True)


overall_result.tail()


from sklearn.neighbors import KNeighborsClassifier


def knnModel(X_train, y_train, X_test, y_test):
  knn = KNeighborsClassifier()
  knn.fit(X_train, y_train)
  y_train_pred = knn.predict(X_train)
  y_test_pred = knn.predict(X_test)
  y_train_proba = knn.predict_proba(X_train)[:,1] # Use predict_proba to get probabilities
  y_test_proba = knn.predict_proba(X_test)[:,1] # Use predict_proba to get probabilities
  return y_train_pred, y_test_pred, y_train_proba, y_test_proba


y_train_pred, y_test_pred, y_train_proba, y_test_proba = knnModel(X_train_scaled, y_train, X_test_scaled, y_test)


knn_train_results = evaluate_model('KNeighborsClassifier-Train',y_train, y_train_pred, y_train_proba)
knn_test_results = evaluate_model('KNeighborsClassifier-Test',y_test, y_test_pred, y_test_proba)


overall_result = pd.concat([overall_result, pd.DataFrame([knn_train_results])], ignore_index=True)
overall_result = pd.concat([overall_result, pd.DataFrame([knn_test_results])], ignore_index=True)


overall_result.tail()


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def ldaModel(X_train, y_train, X_test, y_test):
  lda = LinearDiscriminantAnalysis()
  lda.fit(X_train, y_train)
  y_train_pred = lda.predict(X_train)
  y_test_pred = lda.predict(X_test)
  y_train_proba = lda.predict_proba(X_train)[:,1] # Use predict_proba to get probabilities
  y_test_proba = lda.predict_proba(X_test)[:,1] # Use predict_proba to get probabilities
  return y_train_pred, y_test_pred, y_train_proba, y_test_proba


y_train_pred, y_test_pred, y_train_proba, y_test_proba = ldaModel(X_train_scaled, y_train, X_test_scaled, y_test)


lda_train_results = evaluate_model('LinearDiscriminantAnalysis-Train',y_train, y_train_pred, y_train_proba)
lda_test_results = evaluate_model('LinearDiscriminantAnalysis-Test',y_test, y_test_pred, y_test_proba)


overall_result = pd.concat([overall_result, pd.DataFrame([lda_train_results])], ignore_index=True)
overall_result = pd.concat([overall_result, pd.DataFrame([lda_test_results])], ignore_index=True)


overall_result.tail()


from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


def qdaModel(X_train, y_train, X_test, y_test):
  qda = QuadraticDiscriminantAnalysis()
  qda.fit(X_train, y_train)
  y_train_pred = qda.predict(X_train)
  y_test_pred = qda.predict(X_test)
  y_train_proba = qda.predict_proba(X_train)[:,1] # Use predict_proba to get probabilities
  y_test_proba = qda.predict_proba(X_test)[:,1] # Use predict_proba to get probabilities
  return y_train_pred, y_test_pred, y_train_proba, y_test_proba


y_train_pred, y_test_pred, y_train_proba, y_test_proba = qdaModel(X_train_scaled, y_train, X_test_scaled, y_test)


qda_train_results = evaluate_model('QuadraticDiscriminantAnalysis-Train',y_train, y_train_pred, y_train_proba)
qda_test_results = evaluate_model('QuadraticDiscriminantAnalysis-Test',y_test, y_test_pred, y_test_proba)


overall_result = pd.concat([overall_result, pd.DataFrame([qda_train_results])], ignore_index=True)
overall_result = pd.concat([overall_result, pd.DataFrame([qda_test_results])], ignore_index=True)


overall_result.tail()


from sklearn.ensemble import AdaBoostClassifier


def abstModel(X_train, y_train, X_test, y_test):
  abst = AdaBoostClassifier()
  abst.fit(X_train, y_train)
  y_train_pred = abst.predict(X_train)
  y_test_pred = abst.predict(X_test)
  y_train_proba = abst.predict_proba(X_train)[:,1] # Use predict_proba to get probabilities
  y_test_proba = abst.predict_proba(X_test)[:,1] # Use predict_proba to get probabilities
  return y_train_pred, y_test_pred, y_train_proba, y_test_proba


y_train_pred, y_test_pred, y_train_proba, y_test_proba = abstModel(X_train_scaled, y_train, X_test_scaled, y_test)


abst_train_results = evaluate_model('AdaBoostClassifier-Train',y_train, y_train_pred, y_train_proba)
abst_test_results = evaluate_model('AdaBoostClassifier-Test',y_test, y_test_pred, y_test_proba)


overall_result = pd.concat([overall_result, pd.DataFrame([abst_train_results])], ignore_index=True)
overall_result = pd.concat([overall_result, pd.DataFrame([abst_test_results])], ignore_index=True)


overall_result.tail()


from sklearn.linear_model import SGDClassifier


def sgdModel(X_train, y_train, X_test, y_test):
  sgd = SGDClassifier(loss='log_loss', max_iter=1000)
  sgd.fit(X_train, y_train)
  y_train_pred = sgd.predict(X_train)
  y_test_pred = sgd.predict(X_test)
  y_train_proba = sgd.predict_proba(X_train)[:,1] # Use predict_proba to get probabilities
  y_test_proba = sgd.predict_proba(X_test)[:,1] # Use predict_proba to get probabilities
  return y_train_pred, y_test_pred, y_train_proba, y_test_proba


y_train_pred, y_test_pred, y_train_proba, y_test_proba = sgdModel(X_train_scaled, y_train, X_test_scaled, y_test)


sgd_train_results = evaluate_model('SGDClassifier-Train',y_train, y_train_pred, y_train_proba)
sgd_test_results = evaluate_model('SGDClassifier-Test',y_test, y_test_pred, y_test_proba)


overall_result = pd.concat([overall_result, pd.DataFrame([sgd_train_results])], ignore_index=True)
overall_result = pd.concat([overall_result, pd.DataFrame([sgd_test_results])], ignore_index=True)


overall_result.tail()


from sklearn.neural_network import MLPClassifier


def mlpModel(X_train, y_train, X_test, y_test):
  mlp = MLPClassifier(hidden_layer_sizes=(100,),activation = 'relu', max_iter=1000, random_state = 42)
  mlp.fit(X_train, y_train)
  y_train_pred = mlp.predict(X_train)
  y_test_pred = mlp.predict(X_test)
  y_train_proba = mlp.predict_proba(X_train)[:,1] # Use predict_proba to get probabilities
  y_test_proba = mlp.predict_proba(X_test)[:,1] # Use predict_proba to get probabilities
  return y_train_pred, y_test_pred, y_train_proba, y_test_proba


y_train_pred, y_test_pred, y_train_proba, y_test_proba = mlpModel(X_train_scaled, y_train, X_test_scaled, y_test)


mlp_train_results = evaluate_model('MLPClassifier-Train',y_train, y_train_pred, y_train_proba)
mlp_test_results = evaluate_model('MLPClassifier-Test',y_test, y_test_pred, y_test_proba)


overall_result = pd.concat([overall_result, pd.DataFrame([mlp_train_results])], ignore_index=True)
overall_result = pd.concat([overall_result, pd.DataFrame([mlp_test_results])], ignore_index=True)


overall_result.tail()


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping


def deepNNModel(X_train, y_train, X_test, y_test):
  model = Sequential()
  model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
  model.add(BatchNormalization())
  model.add(Dropout(0.25))
  model.add(Dense(32, activation='relu'))
  model.add(BatchNormalization())
  model.add(Dropout(0.5))
  model.add(Dense(1, activation='sigmoid'))
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
  history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

  y_train_pred = model.predict(X_train)
  y_test_pred = model.predict(X_test)

  # Convert probabilities to class labels using a threshold (e.g., 0.5)
  y_train_pred = (y_train_pred > 0.5).astype(int)
  y_test_pred = (y_test_pred > 0.5).astype(int)

  # For binary classification with sigmoid, 'predict' gives probabilities
  y_train_proba = model.predict(X_train)
  y_test_proba = model.predict(X_test)

  return y_train_pred, y_test_pred, y_train_proba, y_test_proba


y_train_pred, y_test_pred, y_train_proba, y_test_proba = deepNNModel(X_train_scaled, y_train, X_test_scaled, y_test)


dnn_train_results = evaluate_model('DeepNN-Train',y_train, y_train_pred, y_train_proba)
dnn_test_results = evaluate_model('DeepNN-Test',y_test, y_test_pred, y_test_proba)


overall_result = pd.concat([overall_result, pd.DataFrame([dnn_train_results])], ignore_index=True)
overall_result = pd.concat([overall_result, pd.DataFrame([dnn_test_results])], ignore_index=True)


overall_result.tail()


overall_result





from sklearn.model_selection import RandomizedSearchCV


# Define Parameter Distribution
param_dist = {
    'learning_rate': np.logspace(-3, -1, 100),  # log scale from 0.001 to 0.1
    'depth': [4, 6, 8, 10],
    'l2_leaf_reg': [1, 3, 5, 7, 9],
    'random_strength': [0.1, 0.5, 1.0, 2.0],
    'bagging_temperature': [0, 1, 5, 10],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bylevel': [0.6, 0.7, 0.8, 0.9, 1.0]
}


# CatBoost with early stopping
cat_model = CatBoostClassifier(
    iterations = 1000, # will be reduced by early stopping
    eval_metric='Accuracy',
    od_type='Iter',
    od_wait=30,  # stop if no improvement for 30 rounds
)


# Random Search
random_search = RandomizedSearchCV(
    estimator=cat_model,
    param_distributions=param_dist,
    n_iter=50,
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1)



# Fit to training data
random_search.fit(X_train, y_train['rainfall'], eval_set=(X_test, y_test['rainfall']))
# Changed y_train to y_train['rainfall'] and X_test to X_test, y_test to y_test['rainfall']


results = pd.DataFrame(random_search.cv_results_)
results.head()



results['mean_test_score'] = results['mean_test_score'].round(6)
results.head()



# Top parameters visualization
plt.figure(figsize=(12, 6))
top_results = results.sort_values('mean_test_score', ascending=False).head(10)
sns.barplot(x='mean_test_score', y=top_results.index, data=top_results)
plt.title('Top 10 Parameter Combinations')
plt.xlabel('Mean CV Accuracy')
plt.ylabel('Parameter Combination Index')
plt.tight_layout()
plt.show()


# Get the best parameters
best_params = random_search.best_params_
print(f"Best parameters found: {best_params}")


# Get the best model directly
best_model = random_search.best_estimator_
best_model


# Get the best cross-validation score
best_score = random_search.best_score_
print(f"Best cross-validation score: {best_score:.4f}")


# Evaluate the best model on test data
test_score = best_model.score(X_test, y_test)
print(f"Test accuracy with best parameters: {test_score:.4f}")


final_model = CatBoostClassifier(**best_params)
final_model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=30, verbose=0)


y_test_predict = final_model.predict(X_test)


plt.figure(figsize=(10, 6))
sns.heatmap(confusion_matrix(y_test, y_test_predict), annot=True, fmt='d', cmap='Blues')


test.head()


test_predict = best_model.predict(test)


test_predict


test_dataset = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
test_dataset.head()


result = pd.DataFrame([test_dataset['id'], test_predict]).T


result


result.to_csv('submission.csv', index=False)




