import pandas as pd
import numpy as np


train=pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')


def data(df):
    print(df.head(2))
    print("\nDataType : ",df.dtypes)
    print("\nShape of the Dataset : ",df.shape)
    print("\n",df.info())
    print("\nTotal Null Values Present : \n",df.isna().sum())
    print("\nDescriptive : \n",df.describe())
    print("\n------------------------")


data(train)


data(test)


import matplotlib.pyplot as plt
import seaborn as sns


ax = sns.barplot(
    x='employment_status',
    y='loan_paid_back',
    data=train,
    palette='viridis' 
)
plt.title('Loan Repayment Rate by Employment Status', fontsize=16)
plt.xlabel('Employment Status', fontsize=12)
plt.ylabel('Proportion of Loans Paid Back (Repayment Rate)', fontsize=12)


ax = sns.countplot(
    y='employment_status', 
    data=train, 
    order=train['employment_status'].value_counts().index, 
    palette='plasma'
)

# Now set the properties on the captured axis object 'ax'
ax.set_title('2. Count of Applicants by Employment Status', fontsize=14)
ax.set_xlabel('Count', fontsize=12)
ax.set_ylabel('Employment Status', fontsize=12)
plt.show()


x = sns.histplot(
    train['annual_income'] / 1000, 
    kde=True, 
    color='skyblue' 
)
ax.set_title('1. Distribution of Annual Income (in Thousands)', fontsize=14)
ax.set_xlabel('Annual Income (in $1,000s)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)

plt.show()


X=train.drop(columns=['id','loan_paid_back'],axis=1)
Y=train['loan_paid_back']


test_id=test['id']
test_feat=test.drop(columns='id',axis=1)


continuous_cols = [
    "annual_income", "debt_to_income_ratio",
    "credit_score", "loan_amount", "interest_rate"
]


#Winsorization
for col in continuous_cols:
    lower, upper  = X[col].quantile(0.01), X[col].quantile(0.99)
    X[col] = X[col].clip(lower, upper)
    test_feat[col] = test_feat[col].clip(lower, upper)


categorical_cols = [
    'gender', 
    'marital_status', 
    'employment_status', 
    'education_level', 
    'loan_purpose', 
    'grade_subgrade'
]


for col in categorical_cols:
    X[col] = X[col].astype(str)
    test_feat[col] = test_feat[col].astype(str)

test_feat = test_feat[X.columns]

cat_features = [X.columns.get_loc(col) for col in categorical_cols]


from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier,Pool
from sklearn.model_selection import StratifiedKFold


#X_train,X_test,y_train,y_test=train_test_split(X_fe,y,test_size=0.2,random_state=42,stratify=y)


params = {
    "iterations": 1000,
    "depth": 5,
    "learning_rate": 0.22775461488,
    "l2_leaf_reg": 7.46314929,
    "bagging_temperature": 0.0350283198,
    "border_count": 252,
    "random_strength": 1.59045421e-05,
    "eval_metric": "AUC",
    "loss_function": "Logloss",
    "random_seed": 42,
    "verbose": False,
    "task_type": "GPU",
    "devices": "0"             
}


cv = StratifiedKFold(n_splits=15, shuffle=True, random_state=42)

auc_scores = []
all_preds = []
oof_preds = np.zeros(len(Y))

# ---------------------
# Cross-validation with GPU
# ---------------------
for fold, (train_idx, val_idx) in enumerate(cv.split(X, Y), 1):
    print(f"\nðŸš€ Training Fold {fold} on GPU...")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    Y_train, Y_val = Y.iloc[train_idx], Y.iloc[val_idx]

    train_pool = Pool(X_train, Y_train, cat_features=cat_features)
    val_pool = Pool(X_val, Y_val, cat_features=cat_features)

    model = CatBoostClassifier(**params, scale_pos_weight = len(Y[Y == 0]) / len(Y[Y == 1]))
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    Y_pred_proba = model.predict_proba(val_pool)[:, 1]
    auc = roc_auc_score(Y_val, Y_pred_proba)
    auc_scores.append(auc)
    oof_preds[val_idx] = Y_pred_proba

    # Predict on test data (GPU)
    test_pool = Pool(test_feat, cat_features=cat_features)
    test_proba = model.predict_proba(test_pool)[:, 1]
    all_preds.append(test_proba)

    print(f"Fold {fold} AUC: {auc:.5f}")

#
# ---------------------
print(f"\nâœ… Mean AUC-ROC across folds: {np.mean(auc_scores):.5f}")


final = np.mean(all_preds, axis = 0)


submission=pd.DataFrame({'id':test_id,'loan_paid_back':final})
submission.to_csv('submission.csv',index=False)


submission.head(3)

