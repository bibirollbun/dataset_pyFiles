import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier, Pool
pd.set_option("display.max_colwidth", 500)
from lightgbm import LGBMClassifier


train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test  = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
sub   = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')


train.shape


test.shape


train_about = pd.DataFrame({
    'DataTypes': train.dtypes,
    'MissingValues': train.isnull().sum(),
    'UniqueValues': train.nunique()
}).sort_values(by='MissingValues', ascending = True)

train_about['MissingRatio'] = round(train_about['MissingValues'] /len(train),0)

print(train_about)


test_about = pd.DataFrame({
    'DataTypes': test.dtypes,
    'MissingValues': test.isnull().sum(),
    'UniqueValues': test.nunique()
}).sort_values(by='MissingValues', ascending = True)

test_about['MissingRatio'] = round(test_about['MissingValues'] /len(test),0)

print(test_about)


def describe_columns(df):
    records = []
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str).str.lower().str.strip()
            categories = list(df[c].unique())
            if set(categories) <= {"yes", "no"}:
                col_type = "binary"
            elif c == "education":
                col_type = "ordinal"
            else:
                col_type = "categorical"
            records.append({
                "Column": c,
                "Type": col_type,
                "Categories": categories
            })
        else: 
            min_val = df[c].min()
            max_val = df[c].max()
            if  set(df[c].dropna().unique()) <= {0, 1}:
                col_type = "binary"
                cats_str = "[0, 1]"
            else:
                col_type = "numeric"
                cats_str = f"min = {min_val}, max = {max_val}"
            records.append({
                "Column": c,
                "Type": col_type,
                "Categories": cats_str
            })
    return pd.DataFrame(records)


describe_columns(train)


describe_columns(test)


def one_hot_encode(df, cols, drop_first=False):
    
    df_encoded = df.copy()
    for c in cols:
        dummies = pd.get_dummies(df_encoded[c], prefix=f'is_{c}', dtype='int8')
        df_encoded = pd.concat([df_encoded, dummies], axis=1)
        df_encoded.drop(columns=[c], inplace=True)
    return df_encoded


cat_cols = ['gender', 'marital_status', 'education_level',
            'employment_status', 'loan_purpose']

train = one_hot_encode(train, cat_cols, drop_first=False)
test = one_hot_encode(test, cat_cols, drop_first=False)


train['grade'] = train['grade_subgrade'].str[0].str.upper()
train['subgrade'] = train['grade_subgrade'].str[1].astype(int)
train = train.drop(columns=['grade_subgrade'])

grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6}
train['grade'] = train['grade'].map(grade_map).astype('int8')


test['grade'] = test['grade_subgrade'].str[0].str.upper()
test['subgrade'] = test['grade_subgrade'].str[1].astype(int)
test = test.drop(columns=['grade_subgrade'])
test['grade'] = test['grade'].map(grade_map).astype('int8')


for df in (train, test):
    if 'loan_amount' in df.columns and 'annual_income' in df.columns:
        df["loan_income_ratio"] = df["loan_amount"] / (df["annual_income"].replace(0, np.nan)) 
        df["loan_income_ratio"] = df["loan_income_ratio"].fillna(0)
    if {'debt_to_income_ratio', 'loan_amount'}.issubset(df.columns):
        df["debt_burden"] = df["debt_to_income_ratio"] * df["loan_amount"]
    if {'grade', 'interest_rate'}.issubset(df.columns):
        df["grade_rate"] = df["grade"] * df["interest_rate"]
    if 'dti_rate' not in df.columns:
        df['dti_rate'] = df['debt_to_income_ratio'] * df['interest_rate']
    if 'score_minus_grade' not in df.columns:
        df['score_minus_grade'] = df['credit_score'] - 100 * df['grade']

print("Added features:", [c for c in ["loan_income_ratio", "debt_burden", "grade_rate"] if c in train.columns])


describe_columns(train)


num_cols = [
    'annual_income',
    'debt_to_income_ratio',
    'credit_score',
    'loan_amount',
    'interest_rate',
    'grade',
    'subgrade',
    'loan_income_ratio',
    'debt_burden'
]
train[num_cols].describe().T


Q1 = train[num_cols].quantile(0.25)
Q3 = train[num_cols].quantile(0.75)
IQR = Q3 - Q1
outlier_mask = (train[num_cols] < (Q1 - 1.5 * IQR)) | (train[num_cols] > (Q3 + 1.5 * IQR))
outlier_ratio = (outlier_mask.sum() / len(train) * 100).round(0).astype(int).astype(str) + '%'
outlier_ratio.sort_values(ascending=False)


corr_cont = train[num_cols + ['loan_paid_back']].corr()

plt.figure(figsize=(8, 5))
sns.heatmap(corr_cont, cmap='coolwarm', annot=True, fmt=".2f", square=False)
plt.title('Correlation heatmap (continuous features + target)')
plt.tight_layout()
plt.show()


TARGET = 'loan_paid_back'

num_all = [c for c in train.select_dtypes(include='number').columns if c not in ['id', TARGET]]
corr_to_target = train[num_all].corrwith(train[TARGET]).dropna()
corr_to_target = corr_to_target.reindex(corr_to_target.abs().sort_values(ascending=False).index)
corr_to_target.head(20)



selected_features = [
    'is_employment_status_unemployed',
    'is_employment_status_employed',
    'is_employment_status_retired',
    'is_employment_status_self-employed',
    'debt_to_income_ratio',
    'debt_burden',
    'loan_income_ratio',
    'credit_score',
    'grade',
    'interest_rate',
    'is_employment_status_student',
    'grade_rate'
]


feat_cols = [c for c in selected_features if c in train.columns]
X = train[feat_cols].copy().fillna(0)
y = train[TARGET].astype(int).copy()
X_test = test.reindex(columns=feat_cols).copy().fillna(0)

p1 = y.mean()
class_weights = [1.0, float((1.0 - p1) / p1)]

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_cat = np.zeros(len(X))
pred_test_cat = np.zeros(len(X_test))
cat_params = dict(
    loss_function='Logloss',
    eval_metric='AUC',
    learning_rate=0.05,
    depth=5,
    l2_leaf_reg=2.0,
    iterations=7000,
    early_stopping_rounds=300,
    random_seed=42,
    class_weights=class_weights,
    verbose=False,
    allow_writing_files=False,  # Ğ½Ğµ Ñ�Ğ¾Ğ·Ğ´Ğ°Ñ‘Ñ‚ catboost_info
)

oof_lgb = np.zeros(len(X))
pred_test_lgb = np.zeros(len(X_test))
lgb_params = dict(
    n_estimators=1500,
    learning_rate=0.03,
    num_leaves=64,
    min_data_in_leaf=50,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    objective="binary",
    random_state=42,
    verbosity=-1, 
)

for tr_idx, va_idx in kf.split(X, y):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    m_cat = CatBoostClassifier(**cat_params)
    m_cat.fit(Pool(X_tr, y_tr), eval_set=Pool(X_va, y_va), use_best_model=True)
    oof_cat[va_idx]  = m_cat.predict_proba(X_va)[:, 1]
    pred_test_cat   += m_cat.predict_proba(X_test)[:, 1] / kf.n_splits

    m_lgb = LGBMClassifier(**lgb_params)
    m_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        callbacks=[lgb.log_evaluation(0)]
    )
    oof_lgb[va_idx] = m_lgb.predict_proba(X_va)[:, 1]
    pred_test_lgb  += m_lgb.predict_proba(X_test)[:, 1] / kf.n_splits

W_CAT, W_LGB = 0.75, 0.25
oof_blend = W_CAT * oof_cat + W_LGB * oof_lgb
auc_blend = roc_auc_score(y, oof_blend)
print(f"OOF AUC (blend â‰ˆ): {auc_blend:.5f}")

pred_test = W_CAT * pred_test_cat + W_LGB * pred_test_lgb


submission = pd.DataFrame({
    'id': pd.to_numeric(test['id'], errors='coerce').astype('Int64'),
    'loan_paid_back': np.clip(pred_test, 0, 1)
})

submission.to_csv('/kaggle/working/submission.csv', index=False)
print("Saved: /kaggle/working/submission.csv")
print(submission.head())


