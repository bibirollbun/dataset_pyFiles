#kaggle competitions download -c playground-series-s5e11


from google.colab import drive
drive.mount('/content/drive')
path = "/content/drive/MyDrive/Vamshi/K.Vamshi/Professional/Vamshi Work/ML/PredictLoanPayback/Data/"


import os
# Check current path
#print(os.getcwd())

# Printing the Data Files
#path = os.getcwd()+"/Data/"

os.listdir(path)



import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
import scipy.stats as stats



df_train = pd.read_csv(path+"train.csv")
df_test = pd.read_csv(path+"test.csv")


df_train.info()


# Checking for null values in train Data
(df_train.isnull().mean()*100).sort_values(ascending=False)


# Check for null values in train data
(df_test.isnull().mean()*100).sort_values(ascending=False)


df_train.head()


df_train.shape


df_train.columns


df_test.head()


df_test.columns


df_test.shape


# Ensure correct dtypes
df_train['loan_paid_back'] = df_train['loan_paid_back'].astype(int)

num_cols = df_train.select_dtypes(include = ['number']).columns
cat_cols = df_train.select_dtypes(exclude = ['number']).columns

# 2) Numeric summary stats (describe + skew/kurtosis)
num_Desc_Of_DF = df_train[num_cols].describe().T
num_Desc_Of_DF['skew'] = df_train[num_cols].skew()
num_Desc_Of_DF['kurtosis'] = df_train[num_cols].kurtosis()
num_Desc_Of_DF



# Basic Checks for each column
print("Basic Checks for Column Annual Income")
print("No of Values which are negative or Zero: ",(df_train['annual_income']<=0).sum())
print("Min Value :",df_train['annual_income'].min()," and Max Value :",df_train['annual_income'].max())
print("-"*50)
print("Basic Checks for Column debt_to_income_ratio")
print("No of Values which are negative or Zero :",(df_train['debt_to_income_ratio']<=0).sum())
print("Min Value :",df_train['debt_to_income_ratio'].min()," and Max Value :",df_train['debt_to_income_ratio'].max())
print("-"*50)
print("Basic Checks for Column credit_score")
print("No of Values which are negative or Zero:",(df_train['credit_score']<=0).sum())
print("Min Value :",df_train['credit_score'].min()," and Max Value :",df_train['credit_score'].max())
print("-"*50)
print("Basic Checks for Column loan_amount")
print("No of Values which are negative or Zero:",(df_train['loan_amount']<=0).sum())
print("Min Value :",df_train['loan_amount'].min()," and Max Value :",df_train['loan_amount'].max())
print("-"*50)
print("Basic Checks for Column interest_rate")
print("No of Values which are negative or Zero: ",(df_train['interest_rate']<=0).sum())
print("Min Value :",df_train['interest_rate'].min()," and Max Value :",df_train['interest_rate'].max())




# Outlier detection by IQR and z-score for numeric columns
outlier_stats = []
for c in num_cols:
    if c == 'id': continue
    series = df_train[c]
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5*iqr
    upper = q3 + 1.5*iqr
    iqr_outliers = ((series < lower) | (series > upper)).sum()
    z_outliers = (np.abs(stats.zscore(series)) > 3).sum()
    outlier_stats.append({'column': c, 'iqr_outliers': int(iqr_outliers), 'z_outliers': int(z_outliers),
                          'min': series.min(), 'max': series.max()})
outlier_df = pd.DataFrame(outlier_stats)
outlier_df


# Histogram and Boxplot
i=0
fig, axes = plt.subplots(len(num_cols)-2, 2, figsize=(12, 4*(len(num_cols)-2)))
for c in num_cols:
    if c == 'id' or c == 'loan_paid_back': continue

    # Left: Histogram
    sns.histplot(df_train[c], bins=50, kde=True, ax=axes[i,0])
    axes[i,0].set_title(f"Histogram: {c}")
    axes[i,0].set_xlabel(c)
    axes[i,0].set_ylabel("Count")

    # Right: Boxplot
    axes[i,1].boxplot(df_train[c], vert=False)
    axes[i,1].set_title(f"Boxplot: {c}")
    axes[i,1].set_xlabel(c)
    i+=1

plt.tight_layout()
plt.show()



for c in cat_cols:

    # Count of paid back loans per category
    ct = (
        df_train.groupby(c)['loan_paid_back']
        .sum()                         # count of 1s
        .reset_index(name='count_paid')
        .sort_values('count_paid', ascending=False)
    )

    print(f"Paid-back count by category: {c}")
    print(ct)

    plt.figure(figsize=(8,3))
    sns.barplot(
        data=ct,
        x=c,
        y='count_paid',
        hue=c,
        dodge=False
    )
    plt.xticks(rotation=45, ha='right')
    plt.title(f"Paid-back count by {c}")
    plt.ylabel("Count of Paid Back")
    plt.tight_layout()
    plt.show()



for c in cat_cols:

    # Compute paid & not-paid counts
    ct = (df_train.groupby(c)['loan_paid_back']
        .agg(paid=lambda x: (x == 1).sum(),not_paid=lambda x: (x == 0).sum())
        .reset_index()
    )

    # Melt into long format
    ct_melted = ct.melt(
        id_vars=c,
        value_vars=['paid', 'not_paid'],
        var_name='status',
        value_name='count'
    )

    print(f"\nPaid and Not Paid Counts for: {c}")
    print(ct_melted)

    plt.figure(figsize=(9,4))
    sns.barplot(data=ct_melted,x=c,y='count',hue='status',dodge=True)
    plt.xticks(rotation=45, ha='right')
    plt.title(f"Paid vs Not Paid by {c}")
    plt.ylabel("Count")
    plt.xlabel(c)
    plt.tight_layout()
    plt.show()



# 8) Correlation matrix for numeric columns (pearson)
num_corr = df_train[num_cols].corr()
print("Numeric Correlation Matrix")
print(num_corr)

plt.figure(figsize=(8,8))
sns.heatmap(num_corr,annot=True,fmt=".2f",cmap='viridis',cbar=True)
plt.title("Correlation Matrix (Pearson)", fontsize=12)
plt.tight_layout()
plt.show()


# Before Applying Feature Engineering
# Split the data into Train and Test

from sklearn.model_selection import train_test_split

df_train_Sample = df_train.sample(300000)

target = 'loan_paid_back'
train_df, test_df = train_test_split(df_train_Sample, test_size=0.3, stratify=df_train_Sample[target])



# Display Head

train_df.head()



num_Desc_Of_DF


from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):

    """
    Production-grade custom transformer for engineered features.
    Applies high-impact transformations used in lending/risk modeling.
    """

    def __init__(self, id_column="id"):
        self.id_column = id_column

    def fit(self, X, y=None):
        return self  # nothing to fit

    def transform(self, X):
        X = X.copy()

        # Drop ID column if exists
        if self.id_column in X.columns:
            X = X.drop(columns=[self.id_column])

        # Log transforms
        X["log_annual_income"] = np.log1p(X["annual_income"])
        X["log_loan_amount"] = np.log1p(X["loan_amount"])

        # Ratio-based features
        X["loan_to_income"] = X["loan_amount"] / X["annual_income"]
        X["log_loan_to_income"] = np.log1p(X["loan_to_income"])
        X["credit_score_pct"] = X["credit_score"] / 850

        # Credit score band (business bins)
        bins = [0, 579, 669, 739, 799, 850]
        labels = ["Poor", "Fair", "Good", "Very Good", "Excellent"]

        X["credit_score_band"] = pd.cut(
            X["credit_score"],
            bins=bins,
            labels=labels,
            include_lowest=True
        )

        return X



# Split X and Y

X_train = train_df.drop(columns=[target])
y_train = train_df[target]

X_test = test_df.drop(columns=[target])
y_test = test_df[target]



# Create transformer instance
fe_transformer = FeatureEngineeringTransformer()

# Fit only on train data
fe_transformer.fit(X_train)

# Transform both
X_train_fe = fe_transformer.transform(X_train)
X_test_fe = fe_transformer.transform(X_test)



X_train_fe.head()


cat_cols = X_train_fe.select_dtypes(exclude = ['number']).columns
print(cat_cols)


for col in cat_cols:
  print(X_train_fe[col].value_counts())
  print("-"*50)


X_train_fe.head()


from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# Define your columns
one_hot_cols = ['gender', 'marital_status', 'employment_status', 'loan_purpose']
ordinal_cols = ['education_level', 'grade_subgrade','credit_score_band']

# Define the order for ordinal columns
education_order = ['High School', "Bachelor's", "Master's", 'PhD', 'Other']
print(education_order)
# grade_subgrade order A1 < A2 ... F5
grade_order = sorted(df_train['grade_subgrade'].unique().tolist())
print(grade_order)
# Define the order for ordinal columns
credit_Score_Band_order = ["Poor","Fair","Good","Very Good","Excellent"]
print(credit_Score_Band_order)

# Transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('One', OneHotEncoder(drop='first', handle_unknown='ignore'), one_hot_cols),
        ('Ord', OrdinalEncoder(categories=[education_order, grade_order,credit_Score_Band_order]), ordinal_cols)
    ],
    remainder='passthrough'    # leave numerical columns unchanged
)

# Fit + transform train
X_train_pre_proc = preprocessor.fit_transform(X_train_fe)
X_train_pre_proc = pd.DataFrame(X_train_pre_proc, columns=preprocessor.get_feature_names_out())



X_train_pre_proc.head()


X_train_pre_proc.columns


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

scaled_models = {
    "LogisticRegression": LogisticRegression(max_iter=2000,solver="lbfgs",n_jobs=-1),
    "SVC": SVC(),
    "KNN": KNeighborsClassifier(),
    "MLP": MLPClassifier(max_iter=500)
}



from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

tree_models = {
    "RandomForest": RandomForestClassifier(n_estimators=300,random_state=42,n_jobs=-1),
    "XGBoost": XGBClassifier(tree_method="hist", eval_metric="logloss"),
    "LightGBM": LGBMClassifier()
}



from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer

# Define scoring dictionary for cross-validation
scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = {}

for name, model in scaled_models.items():
    print("\nExecuting (Scaled Model):", name)

    pipe = Pipeline([
        ("fe", FeatureEngineeringTransformer()),
        ("encode", preprocessor),
        ("scale", StandardScaler()),
        ("model", model)
    ])

    cv_scores = cross_validate(
        pipe,
        X_train,
        y_train,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        return_train_score=False
    )

    results[name] = {
        "Accuracy": cv_scores["test_accuracy"].mean(),
        "Precision": cv_scores["test_precision"].mean(),
        "Recall": cv_scores["test_recall"].mean(),
        "F1": cv_scores["test_f1"].mean(),
        "ROC_AUC": cv_scores["test_roc_auc"].mean()
    }

for name, model in tree_models.items():
    print("\nExecuting (Tree Model):", name)

    pipe = Pipeline([
        ("fe", FeatureEngineeringTransformer()),
        ("encode", preprocessor),
        ("model", model)
    ])

    cv_scores = cross_validate(
        pipe,
        X_train,
        y_train,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        return_train_score=False
    )

    results[name] = {
        "Accuracy": cv_scores["test_accuracy"].mean(),
        "Precision": cv_scores["test_precision"].mean(),
        "Recall": cv_scores["test_recall"].mean(),
        "F1": cv_scores["test_f1"].mean(),
        "ROC_AUC": cv_scores["test_roc_auc"].mean()
    }

# Convert results into DataFrame
results_df = pd.DataFrame(results).T
results_df



from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV

lgb_params = {
    "model__num_leaves": [31, 63, 127],
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__n_estimators": [300, 500, 800],
    "model__max_depth": [-1, 7, 10, 15],
    "model__subsample": [0.7, 0.8, 1.0],
    "model__colsample_bytree": [0.7, 0.8, 1.0]
}

lgb_pipe = Pipeline([
    ("fe", FeatureEngineeringTransformer()),
    ("encode", preprocessor),
    ("model", LGBMClassifier())
])

lgb_search = RandomizedSearchCV(
    lgb_pipe,
    lgb_params,
    cv=5,
    scoring="roc_auc",
    n_iter=20,
    n_jobs=-1,
    random_state=42
)

lgb_search.fit(X_train, y_train)
best_lgb = lgb_search.best_estimator_



final_pipe = best_lgb  # or best_xgb or best_lr

final_pipe.fit(X_train, y_train)

pred = final_pipe.predict(X_test)
proba = final_pipe.predict_proba(X_test)[:,1]



from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, pred)
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues");
plt.show()


from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(y_test, proba)

plt.plot(fpr, tpr, label=f"AUC={auc(fpr,tpr):.3f}")
plt.plot([0,1],[0,1], '--')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve")
plt.legend()
plt.show()



from sklearn.metrics import precision_recall_curve, average_precision_score

prec, rec, _ = precision_recall_curve(y_test, proba)

plt.plot(rec, prec, label=f"AP={average_precision_score(y_test, proba):.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("PR Curve")
plt.legend()
plt.show()




df_ks = pd.DataFrame({"proba": proba, "y": y_test})
df_ks = df_ks.sort_values("proba")

cum_bad = np.cumsum(df_ks["y"]) / df_ks["y"].sum()
cum_good = np.cumsum(1 - df_ks["y"]) / (1 - df_ks["y"]).sum()

ks = np.max(np.abs(cum_bad - cum_good))
print("KS Statistic:", ks)



from sklearn.calibration import calibration_curve

prob_true, prob_pred = calibration_curve(y_test, proba, n_bins=10)

plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0,1], [0,1], '--')
plt.title("Calibration Curve")
plt.xlabel("Predicted")
plt.ylabel("Observed")
plt.show()



# Now PREDICTING ON ORIGINAL TEST DATA

pred = final_pipe.predict(df_test)
proba = final_pipe.predict_proba(df_test)[:,1]


submission = pd.DataFrame({
    "id": df_test["id"],
    "loan_paid_back": pred
})

submission.to_csv("submission.csv", index=False)




