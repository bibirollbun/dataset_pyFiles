import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')


train_df = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')


train_df.describe()


test_df.describe()


train_df.head()


print("Missing values in train:", train_df.isnull().sum().sum())
print("Missing values in test:", test_df.isnull().sum().sum())


print("Train Columns:", train_df.columns.tolist())


numcols = ['id', 'annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount',
                 'interest_rate']
catcols = [col for col in test_df.columns if col not in numcols]

print("Numerical Columns:", numcols)
print("\nCategorical Columns:", catcols)


#Finding the cardinality of the categorical columns to determine how to process this data

for col in catcols:
    print(f"Cardinality of {col}: ", train_df[col].nunique() )


for col in catcols:
    print(f"\nOptions for {col}:" ,train_df[col].unique())


for c in catcols:
    p = train_df.groupby(c)['loan_paid_back'].mean().sort_values() * 100
    plt.figure(); plt.bar(p.index.astype(str), p.values)
    plt.title(f"{'loan_paid_back'}=1 by {c}"); plt.ylabel("% with 1")
    plt.xticks(rotation=30, ha="right"); plt.tight_layout(); plt.show()


plt.figure(figsize=(10,8))
corr_matrix = train_df[numcols+['loan_paid_back']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center = 0)
plt.title('Correlation Matrix')
plt.show()


#Useful to have graphs of the numerical features so we can decide what
#transformations are appropriate

interesting = ['annual_income', 'loan_amount', 'credit_score']
for c in interesting:
    x = train_df[c].clip(lower=0)                     # log1p needs non-negatives
    print(f"{c}: skew={x.skew():.2f}, log1p skew={np.log1p(x).skew():.2f}")
    plt.figure(); x.hist(bins=50); plt.title(c); plt.show()
    plt.figure(); np.log1p(x).hist(bins=50); plt.title("log1p "+c); plt.show()


def feature_engineering(df, emp_map=None, subgrade_map=None, prior=None):
    df = df.copy()

    # --- categorical ---
    education_mapping = {'High School':1, 'Other':2, "Bachelor's":3, "Master's":4, 'PhD':5}
    df['numericaleducation'] = df['education_level'].map(education_mapping)

    # learn maps if target is present; otherwise use provided ones
    if 'loan_paid_back' in df.columns:
        emp_map      = df.groupby('employment_status')['loan_paid_back'].mean()
        subgrade_map = df.groupby('grade_subgrade')['loan_paid_back'].mean()
        prior        = df['loan_paid_back'].mean() if prior is None else prior
    elif emp_map is None or subgrade_map is None:
        raise ValueError("Provide emp_map and subgrade_map when target is absent.")

    # keep as 0–1 (you can multiply by 100 later if you like)
    df['numericalemployment']     = df['employment_status'].map(emp_map).fillna(prior if prior is not None else 0)
    df['numericalgrade_subgrade'] = df['grade_subgrade'].map(subgrade_map).fillna(prior if prior is not None else 0)

    # One-hot loan_purpose
    d = pd.get_dummies(df['loan_purpose'], prefix='loan_purpose', dtype=int)
    df = df.join(d).drop(columns='loan_purpose')

    # --- numeric features (NEW: safe divides) ---
    inc = df['annual_income'].replace(0, np.nan)
    loan = df['loan_amount'].replace(0, np.nan)

    df['log_income']           = np.log1p(df['annual_income'].clip(lower=0))
    df['log_loan']             = np.log1p(df['loan_amount'].clip(lower=0))
    df['income_to_loan_ratio'] = df['annual_income'] / (df['loan_amount'] + 1)
    df['debt_absolute']        = df['annual_income'] * df['debt_to_income_ratio']
    df['loan_settle_value']    = df['loan_amount'] * ((df['interest_rate']/1200) + 1)
    df['affordability_ratio']  = (df['annual_income']/12) / df['loan_settle_value'].replace(0, np.nan)
    df['credit_income_ratio']  = df['credit_score'] / inc

    df['risk_score'] = (df['debt_to_income_ratio']*0.3
                        + (800 - df['credit_score'])/800*0.3
                        + df['interest_rate']/25*0.2
                        + (df['loan_amount']/inc)*0.2)

    # (NEW) drop raw categoricals so XGB sees only numeric columns
    drop_cols = ['education_level', 'employment_status', 'grade_subgrade']
    df = df.drop(columns=[c for c in drop_cols if c in df])

    return df



# Train features (learn maps inside)
train_df_eng = feature_engineering(train_df)

# Reuse maps for test (same as you had)
emp_map      = train_df.groupby('employment_status')['loan_paid_back'].mean()
subgrade_map = train_df.groupby('grade_subgrade')['loan_paid_back'].mean()
prior        = train_df['loan_paid_back'].mean()
test_df_eng  = feature_engineering(test_df, emp_map=emp_map, subgrade_map=subgrade_map, prior=prior)

# NEW: align OHE columns across train/test
train_df_eng, test_df_eng = train_df_eng.align(test_df_eng, join='left', axis=1, fill_value=0)



import numpy as np
import pandas as pd

def preprocess_data(train_df, test_df):
    features = [
        'log_income','debt_to_income_ratio','credit_score','log_loan','interest_rate',
        'income_to_loan_ratio','debt_absolute','affordability_ratio','credit_income_ratio',
        'risk_score','numericaleducation','numericalemployment','numericalgrade_subgrade'
    ]

    # sanity: make sure all features exist in train
    miss = [c for c in features if c not in train_df.columns]
    if miss: raise KeyError(f"Missing in train: {miss}")

    # test may miss some cols; add if needed
    for c in features:
        if c not in test_df.columns:
            test_df[c] = 0

    X_train = (train_df[features]
               .replace([np.inf, -np.inf], np.nan)
               .astype('float32'))
    y_train = train_df['loan_paid_back'].astype(int)

    X_test  = (test_df[features]
               .replace([np.inf, -np.inf], np.nan)
               .astype('float32'))

    return X_train, y_train, X_test

X_train, y_train, X_test = preprocess_data(train_df_eng, test_df_eng)






from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score

X = X_train
y = y_train

X_train_split, X_val, y_train_split, y_val = train_test_split(
    X, y, test_size = 0.2, random_state = 42, stratify =y
)

neg, pos = (y_train_split==0).sum(), (y_train_split==1).sum()
spw = neg/pos if pos else 1


xgb_model = XGBClassifier(
    # core
    objective="binary:logistic",    # outputs probabilities for class 1
    eval_metric="auc",              # optimize ROC AUC on val set
    tree_method="hist",             # fast, memory-efficient

    # capacity & learning
    n_estimators=2000,              # upper bound; early stopping finds the best round
    learning_rate=0.05,             # small steps → needs more trees, reduces overfit
    max_depth=4,                    # shallow–medium trees capture nonlinearity w/o memorizing
    min_child_weight=3,            # require minimum Hessian/weight → prunes tiny/rare splits

    # stochastic regularization
    subsample=0.9,                  # row sampling per tree
    colsample_bytree=0.9,           # feature sampling per tree

    # split & penalties
    #gamma=0.0,                      # min loss reduction to split; raise if you still overfit
    #reg_lambda=5.0,                 # L2 (ridge) on leaf weights
    #reg_alpha=1.0,                  # L1 (lasso) on leaf weights (feature/leaf sparsity)

    # practicality
    n_jobs=-1,
    random_state=42,
    scale_pos_weight=spw
)


xgb_model.fit(X_train_split, y_train_split, eval_set = [(X_val, y_val)],
             early_stopping_rounds=200,
             verbose = False)
xgb_val_pred = xgb_model.predict_proba(X_val)[:,1]

print(mean_absolute_error(xgb_val_pred, y_val))
print(roc_auc_score(y_val, xgb_val_pred))


import pandas as pd, matplotlib.pyplot as plt

imp = pd.Series(xgb_model.feature_importances_, index=X_val.columns)  # gain by default
imp = imp.sort_values().tail(30)  # top 30

plt.figure(figsize=(8, max(4, 0.3*len(imp))))
plt.barh(imp.index, imp.values)
plt.xlabel("Gain importance"); plt.title("XGBoost feature importance (gain)")
plt.tight_layout(); plt.show()



#Parameter tuning

candidates = [
    dict(n_estimators=3000, learning_rate=0.03, max_depth=4, min_child_weight=5,  subsample=0.8, colsample_bytree=0.8),
    dict(n_estimators=6000, learning_rate=0.02, max_depth=5, min_child_weight=10, subsample=0.7, colsample_bytree=0.7),
    dict(n_estimators=2000, learning_rate=0.05, max_depth=4, min_child_weight=3,  subsample=0.9, colsample_bytree=0.9),
]

best_auc, best_params, best_model = -1, None, None
neg, pos = (y_train_split==0).sum(), (y_train_split==1).sum()
spw = neg/pos if pos else 1

for p in candidates:
    clf = XGBClassifier(
        objective="binary:logistic", eval_metric="auc", tree_method="hist",
        n_jobs=-1, random_state=42, scale_pos_weight=spw, **p
    )
    clf.fit(X_train_split, y_train_split, 
            eval_set=[(X_val, y_val)], 
            early_stopping_rounds=200, 
            verbose=False)
    auc = roc_auc_score(y_val, clf.predict_proba(X_val)[:,1])
    if auc > best_auc: best_auc, best_params, best_model = auc, p, clf

print("\nBest:", best_auc, best_params)




#Final Predictions

xgb_model.fit(X_train, y_train)
final_predictions = xgb_model.predict_proba(X_test)[:,1]


#Submitting work

submission_df = pd.DataFrame({
    'id': test_df['id'],
    'loan_paid_back': final_predictions
})

submission_df.to_csv('/kaggle/working/submission.csv', index = False)

