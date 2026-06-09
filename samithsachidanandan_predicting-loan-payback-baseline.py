import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

import warnings
warnings.filterwarnings('ignore')

from pandas.plotting import scatter_matrix
from sklearn import set_config
set_config(display='diagram')


from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, roc_auc_score


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform






train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')


train.head()


test.head()


train.shape


train.info()


train.dtypes


print("Target column statistics (loan_paid_back):")

train['loan_paid_back'].describe()


train.isnull().sum()


print("Duplicated Rows:",train.duplicated().sum())


train.describe().T


num_cols = train.select_dtypes(include=[np.number]).columns.drop('id')
train[num_cols].hist(figsize=(15, 8), bins=10)
plt.suptitle("Numerical Feature Distributions")
plt.show()


plt.figure(figsize=(10, 8))
sns.heatmap(train[num_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()



attributes = ["annual_income", "debt_to_income_ratio", "credit_score",
     "loan_amount","interest_rate"]
scatter_matrix(train[attributes], figsize=(12, 8))
plt.show()


num_cols = num_cols.drop('loan_paid_back') 
fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(20, 10))
ax = ax.flatten()

index = 0
for col in num_cols:
    sns.boxplot(y=col, data=train, ax=ax[index])
    ax[index].set_title(f"Boxplot of {col}")
    index += 1


for j in range(index, len(ax)):
    fig.delaxes(ax[j])

plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)
plt.show()


plt.figure(figsize=(16, 8))

plt.subplot(2, 3, 1)
sns.histplot(train['annual_income'], kde=True)
plt.title('Annual Income')

plt.subplot(2, 3, 2)
sns.histplot(train['debt_to_income_ratio'], kde=True)
plt.title('Debt to Income Ratio')

plt.subplot(2, 3, 3)
sns.histplot(train['credit_score'], kde=True)
plt.title('Credit Score')

plt.subplot(2, 3, 4)
sns.histplot(train['loan_amount'], kde=True)
plt.title('Loan Amount')

plt.subplot(2, 3, 5)
sns.histplot(train['interest_rate'], kde=True)
plt.title('Interest Rate')

plt.tight_layout()
plt.show()



sns.countplot(x=train["loan_paid_back"])
plt.title("Loan Paid Back - Distribution")
plt.show()

train["loan_paid_back"].value_counts(normalize=True)


fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(20, 10))
ax = ax.flatten()

for i, col in enumerate(num_cols):
    sns.kdeplot(
        data=train,
        x=col,
        hue="loan_paid_back",
        fill=True,
        ax=ax[i]
    )
    ax[i].set_title(f"{col} vs Loan Paid Back")


for j in range(i+1, len(ax)):
    fig.delaxes(ax[j])

plt.tight_layout(pad=2.0)
plt.show()


cat_cols = train.select_dtypes(include=['object']).columns
fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(20, 10))
ax = ax.flatten()

for i, col in enumerate(cat_cols):
    sns.countplot(
        data=train,
        x=col,
        hue="loan_paid_back",
        fill=True,
        ax=ax[i]
    )
    ax[i].set_title(f"{col} vs Loan Paid Back")


for j in range(i+1, len(ax)):
    fig.delaxes(ax[j])

plt.tight_layout(pad=2.0)
plt.show()


plt.figure(figsize=(7,5))
sns.scatterplot(data=train, x="loan_amount", y="interest_rate", hue="loan_paid_back")
plt.title("Loan Amount vs Interest Rate")
plt.show()


sns.boxplot(data=train, x="loan_paid_back", y="debt_to_income_ratio")
plt.title("DTI vs Loan Paid Back")
plt.show()


plt.figure(figsize=(7,4))
sns.barplot(data=train, x="education_level", y="loan_amount", hue="loan_paid_back")
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(7,4))
sns.barplot(data=train, x="employment_status", y="loan_amount", hue="loan_paid_back")
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(8,4))
sns.countplot(data=train, x="grade_subgrade", hue="loan_paid_back")
plt.xticks(rotation=90)
plt.show()


sns.pairplot(train, hue='loan_paid_back', vars = ['annual_income','loan_amount','interest_rate' ])
plt.show()


num_cols




upper_limit_credit_score = train['credit_score'].mean() + 3 *train['credit_score'].std()
lower_limit_credit_score = train['credit_score'].mean() - 3 *train['credit_score'].std()
upper_limit_interest_rate = train['interest_rate'].mean() + 3 *train['interest_rate'].std()
lower_limit_interest_rate = train['interest_rate'].mean() - 3 *train['interest_rate'].std()


train['credit_score'] = np.where(train['credit_score'] > upper_limit_credit_score,upper_limit_credit_score,np.where(train['credit_score']<lower_limit_credit_score, lower_limit_credit_score,train['credit_score']))


train['interest_rate'] = np.where(train['interest_rate'] > upper_limit_interest_rate,upper_limit_interest_rate,np.where(train['interest_rate']<lower_limit_interest_rate, lower_limit_interest_rate,train['interest_rate']))


plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)
sns.histplot(train['credit_score'], kde=True)
plt.title('Credit Score')

plt.subplot(1, 2, 2)
sns.histplot(train['interest_rate'], kde=True)
plt.title('Interest Rate')

plt.tight_layout()
plt.show()



train['annual_income'].skew()


train['debt_to_income_ratio'].skew()


train['loan_amount'].skew()


features = [ 'annual_income','debt_to_income_ratio','loan_amount' ]



for feature in features:
    Q1 = train[feature].quantile(0.25)
    Q3 = train[feature].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    
    train[feature] = np.where(train[feature] >upper, upper, np.where(train[feature]<lower,lower, train[feature]) )


fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(20, 10))
ax = ax.flatten()

index = 0
for col in num_cols:
    sns.boxplot(y=col, data=train, ax=ax[index])
    ax[index].set_title(f"Boxplot of {col}")
    index += 1


for j in range(index, len(ax)):
    fig.delaxes(ax[j])

plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)
plt.show()


cat_cols


def engineer_features(df):
    
    df = df.copy()
    
    df['monthly_income'] = df['annual_income'] / 12
    df['monthly_payment'] = (df['loan_amount'] * df['interest_rate'] / 100) / 12
    df['payment_to_income_ratio'] = df['monthly_payment'] / df['monthly_income']
    
    
    df['total_debt'] = df['loan_amount'] * df['debt_to_income_ratio']
    df['monthly_debt'] = df['total_debt'] / 12
    df['remaining_income'] = df['monthly_income'] - df['monthly_debt']
    
    
    df['credit_efficiency'] = df['credit_score'] / df['debt_to_income_ratio']
    df['loan_to_income_ratio'] = df['loan_amount'] / df['annual_income']
    
    
    df['income_bracket'] = pd.cut(df['annual_income'], 
                                   bins=[0, 25000, 50000, 75000, 100000, np.inf],
                                   labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    
    
    df['credit_category'] = pd.cut(df['credit_score'],
                                    bins=[0, 580, 670, 740, 800, np.inf],
                                    labels=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'])
    
    
    df['loan_size'] = pd.cut(df['loan_amount'],
                              bins=[0, 5000, 10000, 20000, np.inf],
                              labels=['Small', 'Medium', 'Large', 'Very Large'])
    
    
    df['rate_category'] = pd.cut(df['interest_rate'],
                                  bins=[0, 10, 15, 20, np.inf],
                                  labels=['Low', 'Medium', 'High', 'Very High'])
    
    
    df['risk_score'] = (df['debt_to_income_ratio'] * df['interest_rate']) / df['credit_score']
    df['income_credit_interaction'] = df['annual_income'] * df['credit_score']
    df['debt_credit_interaction'] = df['debt_to_income_ratio'] * df['credit_score']
    

    df['credit_score_squared'] = df['credit_score'] ** 2
    df['debt_ratio_squared'] = df['debt_to_income_ratio'] ** 2
    df['income_log'] = np.log1p(df['annual_income'])
    df['loan_amount_log'] = np.log1p(df['loan_amount'])
    
   
    df['gender_marital'] = df['gender'] + '_' + df['marital_status']
    df['education_employment'] = df['education_level'] + '_' + df['employment_status']
    
   
    df['high_risk_flag'] = ((df['debt_to_income_ratio'] > 0.4) | 
                            (df['credit_score'] < 650) | 
                            (df['interest_rate'] > 15)).astype(int)
    
    df['excellent_credit_flag'] = (df['credit_score'] >= 750).astype(int)
    df['high_income_flag'] = (df['annual_income'] >= 50000).astype(int)
    df['has_advanced_degree'] = (df['education_level'] == "Master's").astype(int)
    
 
    gender_income_mean = df.groupby('gender')['annual_income'].transform('mean')
    df['income_vs_gender_avg'] = df['annual_income'] / gender_income_mean
    

    edu_income_mean = df.groupby('education_level')['annual_income'].transform('mean')
    df['income_vs_edu_avg'] = df['annual_income'] / edu_income_mean
    
  
        
    return df



train_df = engineer_features(train)




test_df = engineer_features(test)


numeric_cols = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()

numeric_cols = [col for col in numeric_cols if col != 'loan_paid_back']


categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()



print("*"*180)
print("Numeric:", numeric_cols)

print("*"*180)

print("Categorical:", categorical_cols)
print("*"*180)




train_df['education_employment'].value_counts()


y_train = train_df['loan_paid_back']
X_train = train_df.drop('loan_paid_back', axis=1)

X_test = test_df.copy() 


preprocessor = ColumnTransformer([
    ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols),
    ('scale', MinMaxScaler(), numeric_cols)
])


tune_pipeline = Pipeline([
    ('prep', preprocessor),
    ('model', xgb.XGBClassifier(
        eval_metric='logloss',
        use_label_encoder=False
    ))
])


# param_dist = {
#     'model__n_estimators': randint(300, 800),
#     'model__max_depth': randint(3, 10),
#     'model__learning_rate': uniform(0.01, 0.2),
#     'model__subsample': uniform(0.6, 0.4),
#     'model__colsample_bytree': uniform(0.6, 0.4),
#     'model__min_child_weight': randint(1, 10),
#     'model__gamma': uniform(0, 5)
# }


import torch

param_dist = {
    "objective"             : "binary:logistic",
    "eval_metric"           : "auc",
    "device"                : "cuda:0", 
    "learning_rate"         : 0.075,        
    "n_estimators"          : 630,
    "max_depth"             : 7,
    "subsample"             : 0.88,
    "colsample_bytree"      : 0.66,
    "reg_lambda"            : 0.75,
    "reg_alpha"             : 0.001,
    "verbosity"             : 0,
    "random_state"          : 42,
    "enable_categorical"    : False,
    
    "min_child_weight"      : 5,
    "gamma"                 : 1.0
}



 # models = [
 #     ('XGB', xgb.XGBClassifier()),
 #     ('RF', RandomForestClassifier()),
 #     ('LR', LogisticRegression()),
 #     ]

models = [
    ('XGB', xgb.XGBClassifier(**param_dist))
]






def run_oof_cv_models(models, preprocessor, X, y, n_splits=5):
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = {}
    oof_preds = {}
    model_folds = {}

    for name, model in models:
        print(f"\n{'-'*60}")
        print(f"Running OOF CV for Model: {name}")
        print(f"{'-'*60}")

        oof = np.zeros(len(X))
        fold_models = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"\nFold {fold+1}/{n_splits}")

            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

            # Pipeline with preprocessing
            pipe = Pipeline([
                ('prep', preprocessor),
                ('model', model)
            ])

            pipe.fit(X_train_fold, y_train_fold)
            fold_models.append(pipe)

            
            val_preds = pipe.predict_proba(X_val_fold)[:, 1]
            oof[val_idx] = val_preds

            fold_auc = roc_auc_score(y_val_fold, val_preds)
            print(f"Fold {fold+1} ROC AUC={fold_auc:.6f}")

        oof_preds[name] = oof
        model_folds[name] = fold_models

        overall_auc = roc_auc_score(y, oof)
        results[name] = overall_auc
        print(f"\n{name} OOF CV ROC AUC={overall_auc:.6f}")

    return results, oof_preds, model_folds


results, oof_predictions, trained_models = run_oof_cv_models(
    models=models,
    preprocessor=preprocessor,
    X=X_train,
    y=y_train,
    n_splits=5
)


print("\nTraining final XGBoost model on full dataset...")

final_xgb = Pipeline([
    ('prep', preprocessor),
    ('model', xgb.XGBClassifier(**param_dist))
])

final_xgb.fit(X_train, y_train)


test_preds = final_xgb.predict(X_test)

print("\nFinal XGBoost model trained successfully!") 
print(f"Number of original features: {len(features)}") 
unique, counts = np.unique(test_preds, return_counts=True) 
print("Prediction distribution:", dict(zip(unique, counts)))

os.makedirs("saved_models", exist_ok=True)

joblib.dump(final_xgb, "saved_models/final_xgb_tuned.joblib")

print("Final tuned model saved!")




# os.makedirs("saved_models", exist_ok=True)

# final_models = {}        
# test_predictions = {}    

# print("\nTraining ALL final models on full dataset...\n")

# for name, model in models:
#     print(f"{'='*60}")
#     print(f"Training model: {name}")
#     print(f"{'='*60}")
    
    
#     pipe = Pipeline([
#         ('prep', preprocessor),
#         ('model', model)
#     ])
    
  
#     pipe.fit(X_train, y_train)
    
   
#     filepath = f"saved_models/{name}_final_model.joblib"
#     joblib.dump(pipe, filepath)
#     print(f"Saved model: {filepath}")
    
    
#     preds = pipe.predict(X_test)
#     test_predictions[name] = preds
    
  
#     unique, counts = np.unique(preds, return_counts=True)
#     print("Prediction distribution:", dict(zip(unique, counts)))
    

# print("\n All models trained and saved successfully!")




print("Submission shape:", submission.shape)
print("Test predictions shape:", len(test_preds))


submission = submission.copy()
submission['loan_paid_back'] = test_preds

submission.to_csv('submission.csv', index=False)
print("\n Submission saved to 'submission.csv'")




