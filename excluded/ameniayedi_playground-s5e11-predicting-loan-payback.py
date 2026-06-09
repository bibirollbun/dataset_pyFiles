import pandas as pd
import numpy as np
from scipy.stats import randint, uniform

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_palette("colorblind")

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier

from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, classification_report


import warnings 
warnings.filterwarnings("ignore")


train_df=pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test_df=pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
ss_df=pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")


print("Training dataset shape :",train_df.shape)
print("Testing dataset shape :",test_df.shape)
print("Submission dataset shape :",ss_df.shape)


train_df.head()


test_df.head()


#Show informations about the training dataset
train_df.info()


#Display the statistical summary of the training dataset
train_df.describe()


#Show informations about the testing dataset
test_df.info()


#Display the statistical summary of the testing dataset
test_df.describe()


train_df.columns


train_df.drop(columns=["id"], inplace=True)


#Selecting numeric columns in the training dataset
num_cols=train_df.select_dtypes(include=["number"]).columns.to_list()
print(num_cols)


#Selecting categorical columns in the training dataset
cat_cols=train_df.select_dtypes(exclude=["number"]).columns.to_list()
print(cat_cols)


#Checking for missing values in the training dataset
train_df.isna().sum()


#Checking for missing values in the testing datset
test_df.isna().sum()


# Checking for duplicates in the training dataset
print("The number of duplicated observations in the train dataset is equal to", train_df.duplicated().sum())

# Checking for duplicates in the testing dataset
print("The number of duplicated observations in the test dataset is equal to", test_df.duplicated().sum())


train_df.nunique().sort_values()


sns.histplot(train_df['annual_income'],kde=True)
plt.show()


sns.histplot(train_df['debt_to_income_ratio'],kde=True)
plt.show()


sns.histplot(train_df['credit_score'],kde=True)
plt.show()


sns.histplot(train_df['loan_amount'],kde=True)
plt.show()


sns.histplot(train_df['interest_rate'],kde=True)
plt.show()


sns.pairplot(train_df, vars=num_cols[0:5], diag_kind='kde',hue="loan_paid_back", plot_kws={'alpha': 0.6, 's': 20})
plt.suptitle('Pairplot of All Numeric Variables', y=1.02)
plt.show()


sns.histplot(train_df['loan_paid_back'])
plt.show()


graph = sns.FacetGrid(train_df, col ='loan_paid_back') 
graph.map(sns.distplot, 'credit_score') 
plt.show()


train_df["gender"].value_counts() #categorical value


sns.countplot(data=train_df, x="gender", hue="loan_paid_back")
plt.title("Gender Distribution")
plt.show()


train_df["marital_status"].value_counts() #categorical value


sns.countplot(data=train_df, x="marital_status", hue="loan_paid_back")
plt.title("Distribution of Marital Status")
plt.show()


train_df["education_level"].value_counts().sort_values() #categorical value


sns.countplot(data=train_df, x="education_level", order=train_df["education_level"].value_counts().sort_values(ascending=False).index, hue="loan_paid_back")
plt.title("Distribution of Education Levels")
plt.show()


train_df["employment_status"].value_counts().sort_values() #categorical value


sns.countplot(data=train_df, x="employment_status", order=train_df["employment_status"].value_counts().sort_values(ascending=False).index, hue="loan_paid_back")
plt.title("Distribution of Employment Status")
plt.xticks(rotation=90)
plt.show()


train_df["loan_purpose"].value_counts().sort_values() #categorical value


sns.countplot(data=train_df, x="loan_purpose", order=train_df["loan_purpose"].value_counts().sort_values(ascending=False).index, hue="loan_paid_back")
plt.title("Distribution of Loan Purposes")
plt.xticks(rotation=90)
plt.show()


train_df["grade_subgrade"].value_counts()


sns.countplot(data=train_df, x="grade_subgrade", order=train_df["grade_subgrade"].value_counts().sort_values(ascending=False).index, hue="loan_paid_back")
plt.title("Distribution of Grade Subgrade")
plt.xticks(rotation=90)
plt.show()


plt.figure(figsize=(10,7))
sns.heatmap(train_df[num_cols].corr(), annot=True, fmt=".2f")
plt.show()


for col in num_cols[0:5]:
    plt.figure(figsize=(12,6))
    sns.boxplot(data=train_df, x=col)
    plt.title(f'Boxplot of {col}')
    plt.show()


train_df["grade_subgrade"]=train_df["grade_subgrade"].str[0]
test_df["grade_subgrade"]=test_df["grade_subgrade"].str[0]


sns.countplot(data=train_df, x="grade_subgrade", order=train_df["grade_subgrade"].value_counts().sort_values(ascending=False).index, hue="loan_paid_back")
plt.title("Distribution of Grade Subgrade")
plt.show()


le=LabelEncoder()
for col in cat_cols:   
    train_df[col]=le.fit_transform(train_df[col])
    test_df[col]=le.transform(test_df[col])


train_df.head()


test_df.head()


scaler=StandardScaler()
train_df[num_cols[0:5]]=scaler.fit_transform(train_df[num_cols[0:5]])
test_df[num_cols[0:5]]=scaler.transform(test_df[num_cols[0:5]])


X=train_df.drop(columns=["loan_paid_back"])
y=train_df["loan_paid_back"]


X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)


logreg=LogisticRegression()
logreg.fit(X_train, y_train)
y_probs = logreg.predict_proba(X_test)[:, 1]
logreg_auc_score = round(roc_auc_score(y_test, y_probs),3)


print("AUC ROC for logistic regression model:",logreg_auc_score)


catboost=CatBoostClassifier()
catboost.fit(X_train, y_train)
y_probs = catboost.predict_proba(X_test)[:, 1]
catboost_auc_score = round(roc_auc_score(y_test, y_probs),3)


print("AUC ROC for catboost model:",catboost_auc_score)


lgbm=LGBMClassifier()
lgbm.fit(X_train, y_train)
y_probs = lgbm.predict_proba(X_test)[:, 1]
lgbm_auc_score = round(roc_auc_score(y_test, y_probs),2)


print("AUC ROC for lightGBM classifier model:",lgbm_auc_score)


xgb=XGBClassifier()
xgb.fit(X_train, y_train)
y_probs = xgb.predict_proba(X_test)[:, 1]
xgb_auc_score = round(roc_auc_score(y_test, y_probs),3)


print("AUC ROC for XGBoost classifier model:",xgb_auc_score)


rfc=RandomForestClassifier()
rfc.fit(X_train, y_train)
y_probs = rfc.predict_proba(X_test)[:, 1]
rfc_auc_score = round(roc_auc_score(y_test, y_probs),3)


print("AUC ROC for random forest model:",rfc_auc_score)


ensemble=StackingClassifier(
    estimators=[("catboost",catboost),
                ("lgbm",lgbm),
                ("xgb",xgb),
                ("rfc",rfc)],
    final_estimator=LogisticRegression(),
    stack_method="predict_proba",
    cv=5)


ensemble.fit(X_train, y_train)


y_pred = ensemble.predict(X_test)
print(f"VotingClassifier Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"ROC AUC Score: {round(roc_auc_score(y_test, y_pred),3)}")
print(f"Classification Report: \n{classification_report(y_test, y_pred)}")


models = ["logreg","catboost","lgbm","xgb","rfc","stacked_models"]
auc_roc= [logreg_auc_score, catboost_auc_score, lgbm_auc_score, xgb_auc_score, rfc_auc_score, round(roc_auc_score(y_test, y_pred),3)]


model_auc_roc = pd.DataFrame({"Model": models, "AUC_ROC": auc_roc})
model_auc_roc.sort_values("AUC_ROC",ignore_index=True)


sns.barplot(x="AUC_ROC",y="Model",data=model_auc_roc.sort_values(by="AUC_ROC"))
plt.xlim(min(auc_roc)-0.01, max(auc_roc)+0.01) 
plt.ylabel("Model")
plt.title("AUC_ROC Scores")
plt.show()


catboost=CatBoostClassifier(verbose=0, random_state=42)
catboost_params = {
    'depth': randint(4, 10),
    'learning_rate': uniform(0.01, 0.5),
    'iterations': randint(300, 900),
    'l2_leaf_reg': uniform(1, 10)
}

catboost_search = RandomizedSearchCV(
    catboost, catboost_params, n_iter=20, scoring='roc_auc',cv=5, 
    n_jobs=-1, random_state=42, verbose=1
)


catboost_search.fit(X_train, y_train)
best_catboost = catboost_search.best_estimator_


y_probs = best_catboost.predict_proba(X_test)[:, 1]
catboost_auc_score = round(roc_auc_score(y_test, y_probs),3)
print("AUC ROC for best catboost model:",catboost_auc_score)


lgbm = LGBMClassifier(
    verbose=-1,
    allow_writing_files=False,
    random_state=42
)

lgbm_params = {
    'n_estimators': randint(200, 900),
    'num_leaves': randint(20, 150),
    'max_depth': randint(-1, 15),
    'learning_rate': uniform(0.01, 0.5),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4)
}

lgbm_search = RandomizedSearchCV(
    lgbm, lgbm_params, n_iter=20, scoring='roc_auc', cv=5,
    n_jobs=-1, random_state=42, verbose=1
)



lgbm_search.fit(X_train, y_train)
best_lgbm = lgbm_search.best_estimator_


y_probs = best_lgbm.predict_proba(X_test)[:, 1]
lgbm_auc_score = round(roc_auc_score(y_test, y_probs),3)
print("AUC ROC for best lgbm model:",lgbm_auc_score)


xgb = XGBClassifier(
    use_label_encoder=False,
    eval_metric='auc',
    verbosity=0,
    random_state=42
)

xgb_params = {
    'n_estimators': randint(200, 900),
    'max_depth': randint(3, 30),
    'learning_rate': uniform(0.01, 0.5),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'gamma': uniform(0, 0.5)
}

xgb_search = RandomizedSearchCV(
    xgb, xgb_params, n_iter=20, scoring='roc_auc', cv=5,
    n_jobs=-1, random_state=42, verbose=1
)


xgb_search.fit(X_train, y_train)
best_xgb = xgb_search.best_estimator_


y_probs = best_xgb.predict_proba(X_test)[:, 1]
xgb_auc_score = round(roc_auc_score(y_test, y_probs),3)
print("AUC ROC for best xgb model:",xgb_auc_score)


rfc = RandomForestClassifier()
rfc_params = {
    'n_estimators': randint(100, 600),
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 5),
    'max_features': ['sqrt', 'log2', None]
}

rfc_search = RandomizedSearchCV(
    rfc, rfc_params, n_iter=20, scoring='roc_auc', cv=5,
    n_jobs=-1, random_state=42, verbose=1
)


rfc_search.fit(X_train, y_train)
best_rfc = rfc_search.best_estimator_


y_probs = best_rfc.predict_proba(X_test)[:, 1]
rfc_auc_score = round(roc_auc_score(y_test, y_probs),3)
print("AUC ROC for best xgb model:",rfc_auc_score)


ensemble=StackingClassifier(
    estimators=[("catboost",best_catboost),
                ("lgbm",best_lgbm),
                ("xgb",best_xgb),
                ("rfc",best_rfc)],
    final_estimator=LogisticRegression(),
    stack_method="predict_proba",
    cv=5
)


ensemble.fit(X_train, y_train)


y_pred = ensemble.predict(X_test)
print(f"VotingClassifier Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"ROC AUC Score: {round(roc_auc_score(y_test, y_pred),3)}")
print(f"Classification Report: \n{classification_report(y_test, y_pred)}")


y_pred=ensemble.predict(test_df.drop(columns=['id']))


submission = pd.DataFrame({'id': test_df['id'], 'loan_paid_back': y_pred})
submission.to_csv('submission.csv', index=False)
print("Submission file saved")
display(submission.head())




