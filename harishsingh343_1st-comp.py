import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt 
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer    
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.model_selection import KFold, cross_val_score,StratifiedKFold
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from catboost import CatBoostClassifier
!pip install category-encoders
from category_encoders import TargetEncoder
import lightgbm as lgb







df_train  = pd.read_csv(r"/kaggle/input/playground-series-s5e12/train.csv")
df_test = pd.read_csv(r'/kaggle/input/playground-series-s5e12/test.csv')
test = df_test.copy()


df_train.head()


X = df_train.drop(columns=['diagnosed_diabetes', 'id'])
y = df_train['diagnosed_diabetes']


X.shape, y.shape



numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_cols.remove('hypertension_history')
numerical_cols.remove('cardiovascular_history')
numerical_cols.remove('family_history_diabetes')

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()


print("Categorical Columns:", categorical_cols, len(categorical_cols))
print("Numerical Columns:", numerical_cols, len(numerical_cols))


for cols in categorical_cols:
    print(f"Value counts for {cols}:")
    print(X[cols].value_counts())
    print("\n")


plt.figure(figsize=(15,13))
for i, col in enumerate(categorical_cols, 1):
    plt.subplot(3,3, i)
    gen = X.groupby(col).size()
    gen.plot.bar(stacked=True)
    plt.title(col)
plt.tight_layout()
plt.show()


X.describe()


print("CORRELATION WITH TARGET VARIABLE:\n")
correlations = df_train[numerical_cols + ['diagnosed_diabetes']].corr()['diagnosed_diabetes'].sort_values(ascending=False)
print(correlations)


def bmi_category(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif 18.5 <= bmi < 24.9:
        return 'Normal weight'
    elif 25 <= bmi < 29.9:
        return 'Overweight'
    else:
        return 'Obesity'
X['bmi_category'] = X['bmi'].apply(bmi_category)
df_test['bmi_category'] = df_test['bmi'].apply(bmi_category)

X['bmi_category'].value_counts()    


df_test['medical_history_count'] = df_test[['cardiovascular_history', 'family_history_diabetes', 'hypertension_history']].sum(axis=1)
X['medical_history_count'] = X[['cardiovascular_history', 'family_history_diabetes', 'hypertension_history']].sum(axis=1)



X['ldl_hdl_ratio'] = X['ldl_cholesterol'] / (X['hdl_cholesterol']+1)
df_test['ldl_hdl_ratio'] = df_test['ldl_cholesterol'] / (df_test['hdl_cholesterol']+1)


X['cholesterol_ratio'] = X['cholesterol_total'] / (X['hdl_cholesterol']+1)
df_test['cholesterol_ratio'] = df_test['cholesterol_total'] / (df_test['hdl_cholesterol']+1)


  df_test['Hypertension_Risk'] = ((df_test['systolic_bp'] >= 130) | (df_test['diastolic_bp'] >= 80)).astype(int)
  X['Hypertension_Risk'] = ((X['systolic_bp'] >= 130) | (X['diastolic_bp'] >= 80)).astype(int)


X["age_bmi"] = X["age"] * X["bmi"]
df_test["age_bmi"] = df_test["age"] * df_test["bmi"]


numerical_cols =    X.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_cols.remove('hypertension_history')
numerical_cols.remove('cardiovascular_history')
numerical_cols.remove('family_history_diabetes')
numerical_cols.remove('medical_history_count')
numerical_cols.remove('Hypertension_Risk')


categorical_cols = X.select_dtypes(include=['object']).columns.tolist()


enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
imp = SimpleImputer(strategy="median")

X[categorical_cols] = enc.fit_transform(X[categorical_cols])
df_test[categorical_cols] = enc.transform(df_test[categorical_cols])

X[numerical_cols] = imp.fit_transform(X[numerical_cols])
df_test[numerical_cols] = imp.transform(df_test[numerical_cols])


kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(X))
pred_lgb = np.zeros(len(df_test))


df_test = df_test.drop(columns=['id'])


for fold, (trn_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\n===== FOLD {fold+1} / 5 =====")

    X_train, y_train = X.iloc[trn_idx], y.iloc[trn_idx]
    X_valid, y_valid = X.iloc[val_idx], y.iloc[val_idx]

    
    model = LGBMClassifier(
        n_estimators=1500,
        learning_rate= 0.15550266487959966,
        num_leaves= 215,
        max_depth= 3,
        min_child_samples= 73,
        subsample= 0.9082199990178259,
        colsample_bytree= 0.621473703438141,
        reg_alpha= 3.679996794053407,
        reg_lambda= 7.731770344772727,
        random_state= 42,
        verbosity=-1
    )
    model.fit(X_train, y_train,
           eval_set=[(X_valid, y_valid)], 
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=0) 
        ])

    oof_lgb[val_idx] = model.predict_proba(X_valid)[:, 1]
    pred_lgb += model.predict_proba(df_test)[:, 1] / kf.n_splits



print("\nLightGBM ROC:", roc_auc_score(y, oof_lgb))


submission_data = pd.DataFrame({
        "id": test['id'],
        "diagnosed_diabetes": pred_lgb
})
submission_data.to_csv('submission.csv', index = False)

