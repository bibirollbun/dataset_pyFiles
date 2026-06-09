# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_df = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
train_df[:5]


train_df.shape


train_df.duplicated().sum()


train_df.isna().sum()


train_df.info()


train_df.describe().round(2)


cat_features = ['gender', 'marital_status', 'education_level', 'employment_status',
       'loan_purpose', 'grade_subgrade']

for feature in cat_features:
    unique_categories = train_df[feature].unique()
    num_unique_categories = train_df[feature].nunique()

    print(f"Number of unique {feature.title()}'s: {num_unique_categories}")
    print(f"Total {feature.title()}'s: {unique_categories}")
    print("="*100)


num_cols = train_df.select_dtypes(include=["int64", "float64"])
cat_cols = train_df.select_dtypes(include="object")


selected_cat_features = ['annual_income', 'debt_to_income_ratio', 'credit_score',
                         'loan_amount', 'interest_rate']

fig, axes = plt.subplots(2,3,figsize=(18,8))
axes = axes.flatten()

col_mapping = {
    'annual_income': 'Annual Income ($)',
    'debt_to_income_ratio': 'Debt to Income Ratio',
    'credit_score': 'Credit Score',
    'loan_amount': 'Loan Amount ($)',
    'interest_rate': 'Interest Rate (%)'
}

title_mapping = {
    'annual_income': 'Distribution of Annual Income',
    'debt_to_income_ratio': 'Distribution of Debt to Income Ratio',
    'credit_score': 'Distribution of Credit Score',
    'loan_amount': 'Distribution of Loan Amount',
    'interest_rate': 'Distribution of Interest Rate'
}

for i, col in enumerate(selected_cat_features):
    sns.histplot(
        data=train_df,
        x=col,
        ax=axes[i],
        bins=40,
        kde=True
    )

    axes[i].set_title(title_mapping[col], fontsize=15, fontweight="bold")
    axes[i].set_xlabel(col_mapping[col], fontsize=12, labelpad=10)
    axes[i].set_ylabel("Frequency", fontsize=12)
    axes[i].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))

    if col in ["annual_income", "loan_amount"]:
        axes[i].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))

for j in range(len(['annual_income', 'debt_to_income_ratio', 'credit_score',
                    'loan_amount', 'interest_rate']), len(axes)):
    plt.delaxes(axes[j])

plt.tight_layout()
plt.show()


selected_num_features = ['annual_income', 'debt_to_income_ratio', 'credit_score',
                         'loan_amount', 'interest_rate']

fig, axes = plt.subplots(2,3,figsize=(18,8))
axes = axes.flatten()

col_mapping = {
    'annual_income': 'Annual Income ($)',
    'debt_to_income_ratio': 'Debt to Income Ratio',
    'credit_score': 'Credit Score',
    'loan_amount': 'Loan Amount ($)',
    'interest_rate': 'Interest Rate (%)'
}

title_mapping = {
    'annual_income': 'Distribution of Annual Income',
    'debt_to_income_ratio': 'Distribution of Debt to Income Ratio',
    'credit_score': 'Distribution of Credit Score',
    'loan_amount': 'Distribution of Loan Amount',
    'interest_rate': 'Distribution of Interest Rate'
}

for i, col in enumerate(selected_num_features):
    sns.boxplot(
        data=train_df,
        x=col,
        ax=axes[i]
    )

    axes[i].set_title(title_mapping[col], fontsize=15, fontweight="bold")
    axes[i].set_xlabel(col_mapping[col], fontsize=12, labelpad=10)
    axes[i].set_ylabel("Frequency", fontsize=12)

    if col in ["annual_income", "loan_amount"]:
        axes[i].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))

for j in range(len(['annual_income', 'debt_to_income_ratio', 'credit_score',
                    'loan_amount', 'interest_rate']), len(axes)):
    plt.delaxes(axes[j])

plt.tight_layout()
plt.show()


# def outlier_removal(df, columns):
#     df_clean = df.copy()

#     for column in columns:
#         Q1 = np.percentile(df_clean[column], 25)
#         Q1 = np.percentile(df_clean[column], 50)
#         Q3 = np.percentile(df_clean[column], 75)

#         IQR = Q3 - Q1
#         LF = Q1 - (1.5*IQR)
#         UF = Q3 + (1.5*IQR)

#         print(f"Column: {column.replace("_","").title()}")
#         print(f"IQR: {IQR:.2F} | Lower Fence: {LF:.2f} | Upper Fence: {UF:.2f}")
#         print(f"Original dataset shape: {df_clean.shape}")

#         df_clean = df_clean[(df_clean[column] >= LF) & (df_clean[column] <= UF)]

#         print(f"Dataset shape after outlier removal: {df_clean.shape}") 
#         print("-" * 50)

#     return df_clean


# train_df['income_per_score'] = train_df['annual_income'] / train_df['credit_score']
# train_df['loan_to_income'] = train_df['loan_amount'] / train_df['annual_income']


from scipy.stats import pearsonr

selected_num_features = ['annual_income', 'debt_to_income_ratio', 'credit_score',
                         'loan_amount', 'interest_rate']

corr = {
    feature: pearsonr(x=train_df[feature], y=train_df["loan_paid_back"])[0]
    for feature in selected_num_features
    }

corr_df = pd.DataFrame(
    data=corr.items(), 
    columns=["Feature", "Pearson Correlation"]
    ).sort_values(by="Pearson Correlation", ascending=False).set_index(keys="Feature")

corr_df


plt.figure(figsize=(7,7))

sns.heatmap(
    data=corr_df[["Pearson Correlation"]],
    annot=True,
    cmap="rocket",
    annot_kws={"fontsize": 10, "fontweight": "bold"}
)
plt.title("Pearson Correlation", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.show()


from scipy.stats import chi2_contingency
selected_cat_features = ['gender', 'marital_status', 'education_level', 
                         'employment_status', 'loan_purpose', 'grade_subgrade']

chi2_results = {}

for col in selected_cat_features:
    tbl = pd.crosstab(train_df[col], train_df["loan_paid_back"])

    chi2_stat, p_val, _, _ = chi2_contingency(tbl)

    decision = f"Reject Null Strong evidence of association -> Keep the feature" if p_val < 0.05 else "No strong evidence of association -> Consider dropping feature"

    chi2_results[col] = {
        "chi2_stat": chi2_stat,
        "p_val": p_val,
        "decision": decision
    }

chi2_df = pd.DataFrame(data=chi2_results).T.sort_values(by="p_val")
pd.set_option('display.max_colwidth', None)
chi2_df


train_df["grade_letter"] = train_df["grade_subgrade"].str[0]
train_df["grade_number"] = train_df["grade_subgrade"].str[1].astype(int)
train_df.drop(columns="grade_subgrade", inplace=True)


from sklearn.model_selection import train_test_split

np.random.seed(42)

X = train_df.drop(columns=["loan_paid_back"])
y = train_df["loan_paid_back"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)


print(f"Original Dataset:")
print(f"{y.value_counts(normalize=True)}")
print("Spilt without stratify")
print(f"\nTraining Set: {y_train.value_counts(normalize=True)}")
print(f"\nValidation Set: {y_val.value_counts(normalize=True)}")
print(f"Difference: {y_train.value_counts(normalize=True) - y_val.value_counts(normalize=True)}")


from sklearn.model_selection import train_test_split

np.random.seed(42)

X = train_df.drop(columns=["loan_paid_back"])
y = train_df["loan_paid_back"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)


print("Spilt with stratify")
print(f"\nTraining Set: {y_train.value_counts(normalize=True)}")
print(f"\nValidation Set: {y_val.value_counts(normalize=True)}")
print(f"Difference: {(y_train.value_counts(normalize=True) - y_val.value_counts(normalize=True)).round(6)}")


from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

education_order = ["High School", "Other", "Bachelor's", "Master's", "PhD"]
grade_letter_order = ['F', 'E', 'D', 'C', 'B', 'A']
onehot_cat_features = ['gender', 'employment_status', 'loan_purpose', "marital_status"]
ordinal_cat_features = ["education_level", "grade_letter"]

ordinal_encoder = OrdinalEncoder(
    categories=[education_order, grade_letter_order],
    unknown_value=-1,
    handle_unknown="use_encoded_value"
)

one_hot = OneHotEncoder(
    drop="first",
    handle_unknown="ignore"
)

transformer = ColumnTransformer(transformers=[
        ("ordinal_encoder", ordinal_encoder, ordinal_cat_features),
        ("one_hot", one_hot, onehot_cat_features)
    ],
    remainder="passthrough"
)

X_train_encoded = transformer.fit_transform(X_train)
X_val_encoded = transformer.transform(X_val)

feature_names = transformer.get_feature_names_out()
feature_names = [col.replace("remainder__", "").replace("one_hot__", "").replace("ordinal_encoder__", "") for col in feature_names]

X_train = pd.DataFrame(X_train_encoded, columns=feature_names, index=X_train.index)
X_val = pd.DataFrame(X_val_encoded, columns=feature_names, index=X_val.index)


from sklearn.preprocessing import StandardScaler

columns_to_scale = ['education_level', 'grade_letter', 'gender_Male', 'gender_Other',
       'marital_status_Married', 'marital_status_Single',
       'marital_status_Widowed', 'employment_status_Retired',
       'employment_status_Self-employed', 'employment_status_Student',
       'employment_status_Unemployed', 'loan_purpose_Car',
       'loan_purpose_Debt consolidation', 'loan_purpose_Education',
       'loan_purpose_Home', 'loan_purpose_Medical', 'loan_purpose_Other',
       'loan_purpose_Vacation', 'id', 'annual_income', 'debt_to_income_ratio',
       'credit_score', 'loan_amount', 'interest_rate', 'grade_number']

scaler = StandardScaler()

X_train[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
X_val[columns_to_scale] = scaler.transform(X_val[columns_to_scale])


from sklearn.ensemble import RandomForestClassifier

np.random.seed(43)

baseline_model = RandomForestClassifier(
                                    n_estimators=300,
                                    min_samples_split=20,
                                    min_samples_leaf=4,
                                    max_features="sqrt",
                                    max_depth=20
                                    )

baseline_model.fit(X_train, y_train)

baseline_model_y_preds = baseline_model.predict(X_val)


results = pd.DataFrame(
    {
        "actual": y_val,
        "predicted": baseline_model_y_preds
    }
)

print(results[:10])


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def evaluate_preds(y_true, y_pred):
    """
    Performs evaluation comparison on y_tru lables vs. y_pred labels.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    metric_dict = {
        "accuracy": round(accuracy,3),
        "precision": round(precision,3),
        "recall": round(recall, 3),
        "f1": round(f1, 3)
    }

    print(f"Accuracy: {accuracy * 100:.3f}%")
    print(f"Precision: {round(precision,3)}")
    print(f"Recall: {round(recall,3)}")
    print(f"F1: {round(f1,3)}")

    return metric_dict


baseline_metric_evaluation = evaluate_preds(y_val, baseline_model_y_preds)
baseline_metric_evaluation


# Get probability predictions for Kaggle submission (ROC-AUC evaluation)
baseline_y_proba = baseline_model.predict_proba(X_val)

baseline_y_proba_positive = baseline_y_proba[:,1]

print("First 10 probability predictions:")
print(baseline_y_proba_positive[:10])


from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, thresholds = roc_curve(y_val, baseline_y_proba_positive)

plt.plot(fpr, tpr, color="olive", label="ROC")
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("ROC Curve", fontsize=14, fontweight="bold")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()


# Calculate AUC score
auc = roc_auc_score(y_val, baseline_y_proba_positive)
print(f"AUC Score: {auc:.6f}")


# from sklearn.model_selection import RandomizedSearchCV

# np.random.seed(42)

# rs_grid = {
#     "n_estimators": [100, 200, 300, 500],
#     "max_depth": [20, 30, None],
#     "min_samples_split": [5, 10, 20],
#     "min_samples_leaf": [2, 4],
#     "max_features": ["sqrt", "log2"]
# }

# model = RandomForestClassifier(n_jobs=-1)

# rs_model = RandomizedSearchCV(estimator=model,
#                               param_distributions=rs_grid,
#                               verbose=2,
#                               cv=5,
#                               n_iter=100,
#                               scoring="roc_auc")

# rs_model.fit(X_train, y_train)


# rs_model.best_params_


from lightgbm import LGBMClassifier

np.random.seed(42)

lgbm_model = LGBMClassifier(
    n_estimators=2500,
    learning_rate=0.15,
    num_leaves=180,
    max_depth=2,
    colsample_bytree=0.5,
    subsample=0.85,
    reg_alpha=5.0,
    reg_lambda=5.0,
    min_child_samples=20,
    random_state=42,
    n_jobs=-1,
    metric='auc',
    objective='binary',
    boosting_type='gbdt',
    verbosity=-1
)

lgbm_model.fit(X_train, y_train)

lgbm_y_preds = lgbm_model.predict(X_val)


lgbm_metric_evaluation = evaluate_preds(y_val, lgbm_y_preds)
lgbm_metric_evaluation


lgbm_y_proba = lgbm_model.predict_proba(X_val)

lgbm_y_proba_positive = lgbm_y_proba[:,1]

print("First 10 probability predictions:")
print(lgbm_y_proba_positive[:10])


from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_val, lgbm_y_proba_positive)

plt.plot(fpr, tpr, color="olive", label="ROC")
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("ROC Curve", fontsize=14, fontweight="bold")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()


from sklearn.metrics import roc_auc_score

auc = roc_auc_score(y_val, lgbm_y_proba_positive)
print(f"AUC Score: {auc:.6f}")


# from skopt import BayesSearchCV
# from skopt.space import Real, Integer
# from lightgbm import LGBMClassifier

# param_space = {
#     'n_estimators': Integer(1800, 2500),
#     'learning_rate': Real(0.05, 0.15, prior='log-uniform'),
#     'num_leaves': Integer(120, 180),
#     'max_depth': Integer(2, 5),
#     'min_child_samples': Integer(5, 20),
#     'subsample': Real(0.85, 0.98),
#     'colsample_bytree': Real(0.5, 0.75), 
#     'reg_alpha': Real(5.0, 20.0, prior='log-uniform'),
#     'reg_lambda': Real(5.0, 20.0, prior='   log-uniform'),
# }

# lgbm_model_tuned = LGBMClassifier(objective='binary', metric='auc', random_state=42, verbosity=-1)

# bayes_search = BayesSearchCV(
#     lgbm_model_tuned,
#     param_space,
#     n_iter=50,
#     cv=5,
#     scoring='roc_auc',
#     n_jobs=-1,
#     random_state=42
# )

# bayes_search.fit(X_train, y_train)


# bayes_search.best_params_


# bayes_search.best_score_


# bayes_y_preds = bayes_search.predict(X_val)
# bayes_y_preds[:5]


# bayes_metric_evaluation = evaluate_preds(y_val, bayes_y_preds)
# bayes_metric_evaluation


# bayes_y_proba = bayes_search.predict_proba(X_val)

# bayes_y_proba_positive = bayes_y_proba[:,1]

# print("First 10 probability predictions:")
# print(bayes_y_proba_positive[:10])


# from sklearn.metrics import roc_auc_score

# auc = roc_auc_score(y_val, bayes_y_proba_positive)
# print(f"AUC Score: {auc:.6f}")


test_df = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
test_df[:5]


# test_df['income_per_score'] = test_df['annual_income'] / test_df['credit_score']
# test_df['loan_to_income'] = test_df['loan_amount'] / test_df['annual_income']


test_df["grade_letter"] = test_df["grade_subgrade"].str[0]
test_df["grade_number"] = test_df["grade_subgrade"].str[1].astype(int)
test_df.drop(columns="grade_subgrade", inplace=True)


test_encoded = transformer.transform(test_df)
test_df_encoded = pd.DataFrame(data=test_encoded, columns=feature_names)
test_df_encoded[:5]


cols_to_scale = [c for c in columns_to_scale if c in test_df_encoded.columns]
test_df_encoded[cols_to_scale] = scaler.transform(test_df_encoded[cols_to_scale])


kaggle_predictions = lgbm_model.predict_proba(test_df_encoded)[:, 1]
kaggle_predictions[:10]


submission = pd.DataFrame({
    "id": test_df["id"],
    "loan_paid_back": kaggle_predictions
})
submission.to_csv('submission.csv', index=False)


submission[:10]




