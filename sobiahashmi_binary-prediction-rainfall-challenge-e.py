import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, minmax_scale
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,roc_auc_score


df_train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
display(df_train.head())

print("Shape of the Train dataset: ", df_train.shape)


df_test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
display(df_test.head())
print("Shape of the Test dataset: ", df_test.shape)


df_train.info()


df_train.isnull().sum()


df_test.isnull().sum()


df_test['winddirection'] = df_test['winddirection'].fillna(df_test['winddirection'].mean())


df_test.isnull().sum()


df_train.describe()


df_train.corr()


df_train.hist(figsize=(12, 8), bins=30)
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 6))
sns.boxplot(data=df_train)
plt.xticks(rotation=90)
plt.show()


plt.figure(figsize=(10, 6))
sns.heatmap(df_train.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()



df_train.shape


df_train.columns


X = df_train.drop(['rainfall'], axis = 1)
y = df_train['rainfall']


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


X_train.shape, X_test.shape, y_train.shape, y_test.shape


model_XGBClassifier = XGBClassifier()
model_XGBClassifier.fit(X_train,y_train)


model_xg = XGBClassifier()

param_grid = {
    'n_estimators': [100, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2],
    'reg_alpha': [0, 0.01, 0.1],
    'reg_lambda': [0.1, 1, 10]
}

grid_search = GridSearchCV(
    estimator=model_xg,
    param_grid=param_grid,
    scoring='accuracy',
    n_jobs=-1,
    cv=5
)


grid_search.fit(X_train, y_train)


print("Best Parameters:",grid_search.best_params_)
print("Best Score:",grid_search.best_score_)


model_xg_hp = XGBClassifier(colsample_bytree= 0.8, gamma= 0.2, learning_rate= 0.01, max_depth= 5, min_child_weight= 5, n_estimators= 500, reg_alpha= 0.1, reg_lambda= 10, subsample= 0.6)


model_xg_hp.fit(X_train, y_train)


model_rf = RandomForestClassifier()
model_rf.fit(X_train,y_train)


%%time
model_rf_hp = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [100, 300, 500],  # Number of trees
    'max_depth': [10, 20, None],  # Tree depth (None = fully grown trees)
    'min_samples_split': [2, 5, 10],  # Minimum samples to split a node
    'min_samples_leaf': [1, 2, 5],  # Minimum samples per leaf
    'max_features': ['sqrt', 'log2'],  # Feature selection for each tree
    'bootstrap': [True, False]  # Bootstrapping (True for bagging)
}

grid_search = GridSearchCV(
    estimator=model_rf_hp,
    param_grid=param_grid,
    scoring='roc_auc',
    n_jobs=-1,
    cv=5
)

grid_search.fit(X_train, y_train) 


print("Best Parameters:",grid_search.best_params_)
print("Best Score:",grid_search.best_score_)


model_rf_hp = RandomForestClassifier(bootstrap= True, max_depth= 10, max_features= 'sqrt', min_samples_leaf = 2, min_samples_split= 2, n_estimators= 500)
model_rf_hp = model_rf_hp.fit(X_train, y_train)


model_svm = SVC()
model_svm.fit(X_train,y_train)


model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)


model_gbc = GradientBoostingClassifier()
model_gbc.fit(X_train, y_train)


model_lightGBM = LGBMClassifier(random_state=42)
model_lightGBM.fit(X_train, y_train)


model_catBoost = CatBoostClassifier(random_state=42)
model_catBoost.fit(X_train, y_train)


ridge_clf = RidgeClassifier(alpha=8.84, class_weight=None, copy_X=True, fit_intercept=True,
                max_iter=None, positive=False, random_state=None, solver='auto',
                tol=0.0001)
ridge_clf.fit(X_train, y_train)


model_xg_pred = model_XGBClassifier.predict(df_test)
model_rf_pred = model_rf.predict(df_test)
model_xg_hp_pred = model_xg_hp.predict(df_test)
model_rf_hp_pred = model_rf_hp.predict(df_test)
model_svm_pred = model_svm.predict(df_test)
model_lr_pred = model_lr.predict(df_test)
model_gbc_pred = model_gbc.predict(df_test)


model_lightGBM_pred = model_lightGBM.predict(df_test)
model_catBoost_pred = model_catBoost.predict(df_test)
model_ridge_pred = ridge_clf.predict(df_test)


display(y_test.shape,df_test.shape)


df_train = df_train.head(730)



print("_____XGBClassifier Prediction_____")
Accuracy = accuracy_score(df_train['rainfall'], model_xg_pred)
print(f"Accuracy: {Accuracy:.2f}")
auc_score = roc_auc_score(df_train['rainfall'], model_xg_pred)
print(f"ROC-AUC Score: {auc_score:.2f}")

print("_____XGBClassifier Prediction with Hyperparameters_____")
Accuracy = accuracy_score(df_train['rainfall'], model_xg_hp_pred)
print(f"Accuracy: {Accuracy:.2f}")
auc_score = roc_auc_score(df_train['rainfall'], model_xg_hp_pred)
print(f"ROC-AUC Score: {auc_score:.2f}")

print("_____Random Forest Prediction_____")
Accuracy = accuracy_score(df_train['rainfall'], model_rf_pred)
print(f"Accuracy: {Accuracy:.2f}")
auc_score = roc_auc_score(df_train['rainfall'], model_rf_pred)
print(f"ROC-AUC Score: {auc_score:.2f}")

print("_____Random Forest Prediction with Hyperparameters_____")
Accuracy = accuracy_score(df_train['rainfall'], model_rf_pred)
print(f"Accuracy: {Accuracy:.2f}")
auc_score = roc_auc_score(df_train['rainfall'], model_rf_pred)
print(f"ROC-AUC Score: {auc_score:.2f}")

print("_____Support Vector Machine Prediction_____")
Accuracy = accuracy_score(df_train['rainfall'], model_svm_pred)
print(f"Accuracy: {Accuracy:.2f}")
auc_score = roc_auc_score(df_train['rainfall'], model_svm_pred)
print(f"ROC-AUC Score: {auc_score:.2f}")

print("_____Logistic Regression Prediction_____")
Accuracy = accuracy_score(df_train['rainfall'], model_lr_pred)
print(f"Accuracy: {Accuracy:.2f}")
auc_score = roc_auc_score(df_train['rainfall'], model_lr_pred)
print(f"ROC-AUC Score: {auc_score:.2f}")

print("_____Gradient Boosting Classifier Prediction_____")
Accuracy = accuracy_score(df_train['rainfall'], model_gbc_pred)
print(f"Accuracy: {Accuracy:.2f}")
auc_score = roc_auc_score(df_train['rainfall'], model_gbc_pred)
print(f"ROC-AUC Score: {auc_score:.2f}")


print("_____LightGBM Prediction_____")
Accuracy = accuracy_score(df_train['rainfall'], model_lightGBM_pred)
print(f"Accuracy: {Accuracy:.2f}")
auc_score = roc_auc_score(df_train['rainfall'], model_lightGBM_pred)
print(f"ROC-AUC Score: {auc_score:.2f}")

print("_____CatBoost Prediction_____")
Accuracy = accuracy_score(df_train['rainfall'], model_catBoost_pred)
print(f"Accuracy: {Accuracy:.2f}")
auc_score = roc_auc_score(df_train['rainfall'], model_catBoost_pred)
print(f"ROC-AUC Score: {auc_score:.2f}")

print("_____Ridge Classifier_____")
Accuracy = accuracy_score(df_train['rainfall'], model_ridge_pred)
print(f"Accuracy: {Accuracy:.2f}")
auc_score = roc_auc_score(df_train['rainfall'], model_ridge_pred)
print(f"ROC-AUC Score: {auc_score:.2f}")


from sklearn.metrics import roc_curve, auc

y_pred_proba = model_lightGBM.predict_proba(X_test)[:, 1]  # Get probabilities
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="blue", label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


df_xg = pd.DataFrame({"id":df_test["id"], "rainfall":model_xg_pred})
df_xg.head()

df_rf = pd.DataFrame({"id":df_test["id"], "rainfall":model_rf_pred})
df_rf.head()

df_xg_hp = pd.DataFrame({"id":df_test["id"], "rainfall":model_xg_hp_pred})
df_xg_hp.head()

df_rf_hp = pd.DataFrame({"id":df_test["id"], "rainfall":model_rf_hp_pred})
df_rf_hp.head()

df_svm = pd.DataFrame({"id":df_test["id"], "rainfall":model_svm_pred})
df_svm.head()

df_lr = pd.DataFrame({"id":df_test["id"], "rainfall":model_lr_pred})
df_lr.head()

df_gbc = pd.DataFrame({"id":df_test["id"], "rainfall":model_gbc_pred})
df_gbc.head()


df_lightGBM = pd.DataFrame({"id":df_test["id"], "rainfall":model_lightGBM_pred})
df_lightGBM.head()

df_catBoost = pd.DataFrame({"id":df_test["id"], "rainfall":model_catBoost_pred})
df_catBoost.head()

df_ridge = pd.DataFrame({"id":df_test["id"], "rainfall":model_ridge_pred})
df_ridge.head()


df_xg.to_csv("XGBClassifier_Prediction.csv", index = False)
df_rf.to_csv("RandomForest_Prediction.csv", index = False)
df_xg_hp.to_csv("XGB_Prediction_with_hp.csv", index = False)
df_rf_hp.to_csv("Random_Forest_with_hp.csv", index = False)
df_svm.to_csv("SVM_Prediction.csv", index = False)
df_lr.to_csv("Logistic_Regression_Prediction.csv", index = False)
df_gbc.to_csv("Gradient_Boosting_Prediction.csv", index = False)


df_lightGBM.to_csv("LightGBM_Prediction.csv", index = False)
df_catBoost.to_csv("CatBoost_Prediction.csv", index = False)
df_ridge.to_csv("Ridge_Prediction.csv", index = False)

