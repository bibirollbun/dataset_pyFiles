import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import StackingClassifier
import warnings
warnings.filterwarnings('ignore')


base_path = '/kaggle/input/playground-series-s5e11/'

train_df = pd.read_csv(f'{base_path}train.csv')
test_df = pd.read_csv(f'{base_path}test.csv')


def check_df(dataframe, head=5):
    print("-------------------- Shape --------------------")
    print(dataframe.shape)
    print("-------------------- Types --------------------")
    print(dataframe.dtypes)
    print("-------------------- Head --------------------")
    print(dataframe.head(head))
    print("-------------------- NA --------------------")
    print(dataframe.isnull().sum())

check_df(train_df)


train_df = train_df.astype({col: 'category' for col in train_df.select_dtypes('object').columns})
test_df = test_df.astype({col: 'category' for col in test_df.select_dtypes('object').columns})


train_df = train_df.drop(columns=['id'])


object_cols = [col for col in train_df.columns if train_df[col].dtype == 'category']
numeric_cols = [col for col in train_df.columns if train_df[col].dtype != 'category']


fig, axes = plt.subplots(6, 1, figsize=(8, 15))
j = 0
for col in object_cols:
    sns.countplot(data=train_df, x=col, hue="loan_paid_back", ax=axes[j])
    axes[j].set_title('Count Plot {}'.format(col))
    axes[j].set_ylabel(None)

    j += 1

plt.subplots_adjust(hspace=0.6)
plt.show()



loan_paid_back_ratio_by_grade = train_df.groupby('grade_subgrade')['loan_paid_back'].value_counts(normalize=True).unstack()
loan_paid_back_ratio_by_grade.plot(kind='bar', stacked=True, figsize=(8, 4))
plt.title('Loan Paid Back Ratio by Grade Subgrade')


fig, axes = plt.subplots(5, 1, figsize=(5, 20))
j = 0
for col in numeric_cols:
    if col != 'loan_paid_back':
        sns.violinplot(
        data=train_df,
        x="loan_paid_back",
        y=col,
        hue="loan_paid_back",
        split=True,
        inner="quart",
        ax=axes[j])
        axes[j].set_title('Violin plot {}'.format(col))
        axes[j].set_ylabel(None)

        j += 1

plt.subplots_adjust(hspace=0.6)
plt.show()


def feature_pipeline(df, functions, **kwargs):

    df = df.copy()
    for func in functions:
        df = func(df, **kwargs)
    return df


def distribute_others(df, random_state=42):
    df = df.copy()
    np.random.seed(random_state)
    if 'gender' in df.columns:
        s = df['gender'].astype(str)
        male_mask = s == 'Male'
        female_mask = s == 'Female'
        male_ratio = male_mask.mean()
        female_ratio = female_mask.mean()
        total = male_ratio + female_ratio
        if total == 0:
            probs = [0.5, 0.5]
        else:
            probs = [male_ratio / total, female_ratio / total]
        mask = s == 'Other'
        n_others = int(mask.sum())
        if n_others > 0:
            df.loc[mask, 'gender'] = np.random.choice(
                ['Male', 'Female'],
                size=n_others,
                p=probs
            )
    return df


def merge_marital_status(df):
    df = df.copy()
    if 'marital_status' in df.columns:
        df['marital_status'] = df['marital_status'].astype(str).replace({
            'Divorced': 'Single',
            'Widowed': 'Single'
        })
        df['marital_status'] = df['marital_status'].astype('category')
    return df


def encode_education(df):
    df = df.copy()
    if 'education_level' in df.columns:
        order = ['Other', 'High School', "Bachelor's", "Master's", 'PhD']
        cat_type = pd.api.types.CategoricalDtype(categories=order, ordered=True)
        df['education_level'] = df['education_level'].astype(cat_type).cat.codes
    return df


def loan_income_ratio(df):
    df = df.copy()
    if ('loan_amount' in df.columns) and ('annual_income' in df.columns):
        df['income_to_loan_ratio'] = df['annual_income'] / (df['loan_amount'] + 1)
    return df


def simplify_grade_subgrade(df):
    df = df.copy()
    if 'grade_subgrade' in df.columns:
        df['grade_subgrade'] = df['grade_subgrade'].astype(str).replace({
            'A1': 'AB', 'A2': 'AB', 'A3': 'AB', 'A4': 'AB', 'A5': 'AB',
            'B1': 'AB', 'B2': 'AB', 'B3': 'AB', 'B4': 'AB', 'B5': 'AB',
            'E1': 'EF', 'E2': 'EF', 'E3': 'EF', 'E4': 'EF', 'E5': 'EF',
            'F1': 'EF', 'F2': 'EF', 'F3': 'EF', 'F4': 'EF', 'F5': 'EF',
            'C1': 'C', 'C2': 'C', 'C3': 'C', 'C4': 'C', 'C5': 'C',
            'D1': 'D', 'D2': 'D', 'D3': 'D', 'D4': 'D', 'D5': 'D'
            
        })
        mapping = {'AB': 3, 'C': 2, 'D': 1, 'EF': 0}
        df['grade_subgrade'] = df['grade_subgrade'].map(mapping).fillna(-1).astype(int)
    return df


def employed_merge(df):
    df = df.copy()
    if 'employment_status' in df.columns:
        df['employment_status'] = df['employment_status'].astype(str).replace(
            'Self-employed', 'Employed'
        )
        df['employment_status'] = df['employment_status'].astype('category')
    return df


def risk_score(df):
    df = df.copy()
    required = ['debt_to_income_ratio', 'income_to_loan_ratio', 'interest_rate']
    if all(col in df.columns for col in required):
        df['risk_score'] = (
            pd.to_numeric(df['debt_to_income_ratio'], errors='coerce') * 0.4 +
            pd.to_numeric(df['income_to_loan_ratio'], errors='coerce') * 0.3 +
            pd.to_numeric(df['interest_rate'], errors='coerce') * 0.3
        )
    else:
        df['risk_score'] = np.nan
    return df


def cut(df):
    df = df.copy()
    if 'debt_to_income_ratio' in df.columns:
        try:
            df['dti_qcut'] = pd.qcut(
                df['debt_to_income_ratio'], q=5, duplicates='drop'
            ).astype(str)
        except Exception:
            df['dti_qcut'] = pd.qcut(
                df['debt_to_income_ratio'].rank(method='first'), q=5, duplicates='drop'
            ).astype(str)
        df['dti_qcut'] = df['dti_qcut'].astype('category')

    if 'annual_income' in df.columns:
        try:
            df['income_qcut'] = pd.qcut(
                df['annual_income'], q=4, duplicates='drop'
            ).astype(str)
        except Exception:
            df['income_qcut'] = pd.qcut(
                df['annual_income'].rank(method='first'), q=4, duplicates='drop'
            ).astype(str)
        df['income_qcut'] = df['income_qcut'].astype('category')
    return df


def prepare_for_model(df):
    df = df.copy()
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype('category')
    return df



functions = [
    distribute_others,
    merge_marital_status,
    encode_education,
    loan_income_ratio,
    simplify_grade_subgrade,
    employed_merge,
    risk_score,
    cut,
    prepare_for_model
    ]

train_fe = feature_pipeline(train_df, functions)
test_fe = feature_pipeline(test_df, functions)


def encode_features(df, label_columns=[], one_hot_columns=[]):

    df = df.copy()

    for col in label_columns:
        if col in df.columns:
            le = LabelEncoder()
            df[col + '_labeled'] = le.fit_transform(df[col].astype(str))
            df.drop(columns=[col], inplace=True)

    for col in one_hot_columns:
        if col in df.columns:
            ohe = OneHotEncoder(sparse_output=False, drop='first')
            encoded = ohe.fit_transform(df[col].values.reshape(-1, 1))
            encoded_df = pd.DataFrame(
                encoded,
                columns=[f"{col}_{c}" for c in ohe.categories_[0][1:]], 
                index=df.index
            )
            df = pd.concat([df, encoded_df], axis=1)
            df.drop(columns=[col], inplace=True)

    return df

label_columns = ['income_qcut', 'dti_qcut']
one_hot_columns = ['gender', 'marital_status', 'employment_status', 'loan_purpose']

train_fe = encode_features(train_fe, label_columns, one_hot_columns)
test_fe  = encode_features(test_fe, label_columns, one_hot_columns)



from sklearn.preprocessing import StandardScaler, MinMaxScaler

feature_cols = [
    'annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', 
    'interest_rate', 'income_to_loan_ratio', 
    'risk_score']

scaler_std = StandardScaler()
scaler_minmax = MinMaxScaler()

train_std = scaler_std.fit_transform(train_fe[feature_cols])
train_scaled = scaler_minmax.fit_transform(train_std)
train_scaled_df = train_fe.copy()
train_scaled_df[feature_cols] = train_scaled

test_std = scaler_std.transform(test_fe[feature_cols])
test_scaled = scaler_minmax.transform(test_std)
test_scaled_df = test_fe.copy()
test_scaled_df[feature_cols] = test_scaled


X = train_fe.drop(columns=['id','loan_paid_back'], errors='ignore')
y = train_fe['loan_paid_back']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Validation samples: {X_val.shape[0]}")


xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss',
    n_jobs=-1
)

cat_model = CatBoostClassifier(
    iterations=100, 
    depth=6,
    learning_rate=0.1,
    random_state=42,
    verbose=False,
    thread_count=-1 
)

rf_model = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

print("Training base models...")

print("Training XGBoost...", end=" ")
xgb_model.fit(X_train, y_train)
xgb_val_pred = xgb_model.predict_proba(X_val)[:, 1]
print("✓")

print("Training CatBoost...", end=" ")
cat_model.fit(X_train, y_train)
cat_val_pred = cat_model.predict_proba(X_val)[:, 1]
print("✓")

print("Training Random Forest...", end=" ")
rf_model.fit(X_train, y_train)
rf_val_pred = rf_model.predict_proba(X_val)[:, 1]
print("✓")

meta_features_val = np.column_stack([xgb_val_pred, cat_val_pred, rf_val_pred])

meta_model = LogisticRegression(random_state=42, C=1.0)
meta_model.fit(meta_features_val, y_val)

print("Meta-model trained successfully!")

print("\nBase Model Performance (Validation AUC):")
print(f"XGBoost: {roc_auc_score(y_val, xgb_val_pred):.4f}")
print(f"CatBoost: {roc_auc_score(y_val, cat_val_pred):.4f}")
print(f"Random Forest: {roc_auc_score(y_val, rf_val_pred):.4f}")

# Ensemble performance
ensemble_val_pred = meta_model.predict_proba(meta_features_val)[:, 1]
print(f"Stacking Ensemble: {roc_auc_score(y_val, ensemble_val_pred):.4f}")


X_test = test_fe.drop(columns="id")

xgb_test_pred_prob = xgb_model.predict_proba(X_test)[:, 1]

xgb_test_pred = (xgb_test_pred_prob >= 0.5).astype(int)

sub = pd.DataFrame({
    "id": test_fe["id"],
    "loan_paid_back": xgb_test_pred_prob 
})

sub.to_csv("submission.csv", index=False)


xgb_importance = xgb_model.feature_importances_
xgb_indices = np.argsort(xgb_importance)[-15:]
fig, ax = plt.subplots(figsize=(8,6))
ax.barh(range(len(xgb_indices)), xgb_importance[xgb_indices])
ax.set_yticks(range(len(xgb_indices)))
ax.set_yticklabels([X_train.columns[i] for i in xgb_indices])
ax.set_title('XGBoost - Top 15 Feature Importance')
ax.set_xlabel('Importance')
ax.set_ylabel('Feature')

plt.tight_layout()
plt.show()

