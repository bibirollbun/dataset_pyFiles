import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import StackingClassifier, StackingRegressor, VotingClassifier
from sklearn.linear_model import LogisticRegression
import optuna
from lightgbm import LGBMClassifier


train_df = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')


train_df.head()


train_df.info()


train_df.describe()


train_df.isna().sum()


train_df.duplicated().sum()


train_df['loan_paid_back'].value_counts()


cat_cols = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade']
num_cols = ['annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate']



num_cols = ['annual_income', 'debt_to_income_ratio', 'credit_score',
            'loan_amount', 'interest_rate']

print("\n--- Numerical Columns Analysis ---")
for col in num_cols:
    print(f"\nðŸ”¹ {col}")
    
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    sns.histplot(train_df[col], kde=True)
    plt.title(f"Distribution of {col}")
    
    plt.subplot(1,2,2)
    sns.boxplot(x=train_df[col])
    plt.title(f"Boxplot of {col}")
    
    plt.show()

cat_cols = ['gender', 'marital_status', 'education_level',
            'employment_status', 'loan_purpose', 'grade_subgrade']

print("\n--- Categorical Columns Analysis ---")
for col in cat_cols:
    print(f"\nðŸ”¹ {col}")
    print(train_df[col].value_counts(normalize=True).head())
    
    plt.figure(figsize=(8,4))
    sns.countplot(data=train_df, x=col, order=train_df[col].value_counts().index)
    plt.title(f"Countplot of {col}")
    plt.xticks(rotation=45)
    plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(train_df[num_cols + ['loan_paid_back']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

target = 'loan_paid_back'

for col in num_cols:
    plt.figure(figsize=(6,4))
    sns.kdeplot(data=train_df, x=col, hue='loan_paid_back', fill=True)

    plt.title(f"{col} vs {target}")
    plt.show()



def create_frequency_features(df, df_test):
    """
    Add frequency and binning features efficiently.

    - For each categorical column, create <col>_freq = how often each value appears in train data.
    - For numeric columns, split values into 5, 10, 15 quantile bins.
    """
    # Pre-allocate DataFrames for new features to avoid fragmentation
    freq_features_train = pd.DataFrame(index=df.index)
    freq_features_test = pd.DataFrame(index=df_test.index)
    bin_features_train = pd.DataFrame(index=df.index)
    bin_features_test = pd.DataFrame(index=df_test.index)

    for col in cols:
        # --- Frequency encoding ---
        freq = df[col].value_counts()
        df[f"{col}_freq"] = df[col].map(freq)
        freq_features_test[f"{col}_freq"] = df_test[col].map(freq).fillna(freq.mean())

        # --- Quantile binning for numeric columns ---
        if col in num:
            for q in [5, 10, 15]:
                try:
                    train_bins, bins = pd.qcut(df[col], q=q, labels=False, retbins=True, duplicates="drop")
                    bin_features_train[f"{col}_bin{q}"] = train_bins
                    bin_features_test[f"{col}_bin{q}"] = pd.cut(df_test[col], bins=bins, labels=False, include_lowest=True)
                except Exception:
                    bin_features_train[f"{col}_bin{q}"] = 0
                    bin_features_test[f"{col}_bin{q}"] = 0

    # Concatenate all new features at once
    df = pd.concat([df, freq_features_train, bin_features_train], axis=1)
    df_test = pd.concat([df_test, freq_features_test, bin_features_test], axis=1)

    return df, df_test


cols = train_df.drop(columns=[target,"id"]).columns.tolist()
num = [c for c in cols if train_df[c].dtype not in ["object","category","bool"]]
print("num: ", num)
print("cols: ", cols)
train_df, test_df = create_frequency_features(train_df, test_df)


remove = [ "interest_rate", 
         "employment_status_freq", "credit_score_bin5",  "loan_amount_bin5",
          "debt_to_income_ratio_bin5"]
train_df, test_df = train_df.drop(columns = remove), test_df.drop(columns = remove)



# for col in num_cols:
#     print(f"{col} skewness: {train_df[col].skew()}")
#     print(f"{col} skewness: {test_df[col].skew()}")

train_df['annual_income_log'] = np.log1p(train_df['annual_income'])
train_df['debt_to_income_ratio_log'] = np.log1p(train_df['debt_to_income_ratio'])
train_df.drop(columns=['annual_income', 'debt_to_income_ratio'], inplace=True)

test_df['annual_income_log'] = np.log1p(test_df['annual_income'])
test_df['debt_to_income_ratio_log'] = np.log1p(test_df['debt_to_income_ratio'])
test_df.drop(columns=['annual_income', 'debt_to_income_ratio'], inplace=True)




education_order = [
    "Other",
    "High School",
    "Bachelor's",
    "Master's",
    "PhD"
]
grade_order = [
    'A1','A2','A3','A4','A5',
    'B1','B2','B3','B4','B5',
    'C1','C2','C3','C4','C5',
    'D1','D2','D3','D4','D5',
    'E1','E2','E3','E4','E5',
    'F1','F2','F3','F4','F5',
]
ordinal_cols = ['education_level', 'grade_subgrade']
ordinal_categories = [education_order, grade_order]

nominal_cols = ['gender', 'marital_status', 'employment_status', 'loan_purpose']

preprocessor = ColumnTransformer([
    ('ord', OrdinalEncoder(categories=ordinal_categories), ordinal_cols),
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'), nominal_cols)
], remainder='passthrough')


X_train = train_df.drop(columns=['loan_paid_back', 'id'])
y_train = train_df['loan_paid_back']
X_test = test_df.drop(columns=['id'])


X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)


print("X_train_encoded shape:", X_train_encoded.shape)
print("y_train shape:", y_train.shape)



xgb_params = {'n_estimators': 1250, 'max_depth': 3, 'learning_rate': 0.13510868166273501, 'subsample': 0.8794864404235258, 'colsample_bytree': 0.976543804857637, 'gamma': 0.2829961549131938, 'min_child_weight': 2, 'reg_alpha': 6.433167562715905, 'reg_lambda': 7.292109939548801}
lgb_params = { 'verbosity':-1, 'n_estimators': 1942, 'max_depth': 3, 'learning_rate': 0.12093539056257775, 'subsample': 0.9643697539245966, 'colsample_bytree': 0.6138587381273723, 'min_child_weight': 4, 'reg_alpha': 3.9731738630617075, 'reg_lambda': 9.182017682059731}
cat_params = { 'verbose':0, 'iterations': 2623, 'depth': 3, 'learning_rate': 0.16286923955599264, 'l2_leaf_reg': 0.7407588360827997, 'random_strength': 0.19727874622488412, 'bagging_temperature': 6.525351017328033, 'border_count': 250}



xgb_model = XGBClassifier(
    **xgb_params
)

lgb_model = LGBMClassifier(**lgb_params)
cat_model = CatBoostClassifier(**cat_params)


kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

roc_scores = []


for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_encoded, y_train)):
    print(f"\n----- Fold {fold + 1} -----")

    X_train, X_val = X_train_encoded[train_idx], X_train_encoded[val_idx]
    y_trn, y_val = y_train[train_idx], y_train[val_idx]

   
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

   
    lgb_model.fit(X_train_scaled, y_trn)
    cat_model.fit(X_train_scaled, y_trn)
    


    y_pred_proba = (lgb_model.predict_proba(X_val_scaled)[:, 1]+ cat_model.predict_proba(X_val_scaled)[:, 1])/2
    roc = roc_auc_score(y_val, y_pred_proba)
    roc_scores.append(roc)


    print(f"ROC-AUC (Fold {fold + 1}): {roc:.4f}")


print("\n========================")
print(f"Average ROC-AUC: {np.mean(roc_scores):.4f}")
print("========================")


scaler = StandardScaler()
X_train_encoded = scaler.fit_transform(X_train_encoded)

X_test_encoded = scaler.transform(X_test_encoded)


lgb_model.fit(X_train_encoded, y_train)


cat_model.fit(X_train_encoded, y_train)


submission = pd.DataFrame({
    'id':test_df['id'],
    'loan_paid_back':(lgb_model.predict_proba(X_test_encoded)[:,1]+cat_model.predict_proba(X_test_encoded)[:,1])/2
})
submission.to_csv("submission.csv", index=False)

submission.head()

