# Check version
import sklearn
print(sklearn.__version__) # scikit-learn should be > 1.4 to use package imblearn for handling class imbalance issue 

if float(sklearn.__version__[:3]) < 1.4:
    # If you don't need cesium for this notebook, remove it first:
    %pip uninstall -y cesium
    
    # Pin a compatible stack
    %pip install --upgrade \
      "numpy==1.26.4" \
      "scikit-learn==1.4.2" \
      "imbalanced-learn==0.12.3" \
      "category-encoders==2.7.0" \
      "sklearn-compat==0.1.3"

    print("Restart the kernel and run again.")
else:
    # # (Optional) tidy up any rogue constraints
    # %pip check
    print("Ready to go! Enjoy.")


# For data manipulation
import numpy as np
import pandas as pd

# For Plots
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# For Data Modeling
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif

## Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


## Model Evaluation
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

## For class imbalance
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import warnings
warnings.filterwarnings('ignore')


train_file_path = "/kaggle/input/playground-series-s5e8/train.csv"
test_file_path = "/kaggle/input/playground-series-s5e8/test.csv"
submission_file_path = "/kaggle/input/playground-series-s5e8/sample_submission.csv"
original_file_path = "/kaggle/input/bank-marketing-dataset-full/bank-full.csv"


train_df = pd.read_csv(train_file_path)
test_df = pd.read_csv(test_file_path)
submission_df = pd.read_csv(submission_file_path)
original_df = pd.read_csv(original_file_path, sep=";")


print(f"There are {len(train_df)} examples in the training dataset.")
print(f"There are {len(train_df.columns) - 1} features.")
train_df


print(f"There are {len(test_df)} examples in the training dataset.")
print(f"There are {len(test_df.columns)} features.")
test_df


submission_df


print("Check Training data: " + "-"* 30)
print(train_df.info())

print("Check Test data: " + "-"* 30)
print(test_df.info())


#  Check duplicates
print("No duplicates in train_df" if len(train_df[train_df.duplicated()]) == 0 else "Has duplicates in train_df")
print("No duplicates in test_df" if len(test_df[test_df.duplicated()]) == 0 else "Has duplicates in test_df")

# Check If test_df has same examples as train_df
train_df_dropped = train_df.iloc[:, :-1]
mask = test_df.apply(tuple, axis=1).isin(train_df_dropped.apply(tuple, axis=1))
print("test_df and train_df have no same example." if len(test_df[mask]) == 0 else "test_df and train_df have same examples.")


# Data Cleaning


print(train_df[train_df["pdays"]==0])
print(test_df[test_df["pdays"]==0])


def sanity_check(data):
    # Rule 1 violation: previous = 0 but pdays != -1
    rule1_violations = data[(data['previous'] == 0) & (data['pdays'] != -1)]
    
    # Rule 2 violation: previous > 0 but pdays < 0
    rule2_violations = data[(data['previous'] > 0) & (data['pdays'] < 0)]
    
    # Special impossible case: previous = 0 and pdays = 0
    impossible_case = data[(data['previous'] == 0) & (data['pdays'] == 0)]
    
    print("Rule 1 violations (previous = 0 but pdays != -1):", len(rule1_violations))
    print("Rule 2 violations (previous > 0 but pdays < 0):", len(rule2_violations))
    print("Impossible case (previous = 0 and pdays = 0):", len(impossible_case))

    return rule1_violations, rule2_violations, impossible_case


rule1_violations, rule2_violations, impossible_case = sanity_check(train_df)


# rule1_violations


# rule2_violations


# impossible_case


rule1_violations, rule2_violations, impossible_case = sanity_check(test_df)


rule1_violations, rule2_violations, impossible_case = sanity_check(original_df)


# original_df[original_df["pdays"]==-1]["y"].value_counts()
# original_df[original_df["previous"]==0]["y"].value_counts()


def fix_issue(data):
    # Case 1: previous = 0 but pdays != -1 â†’ fix pdays
    data.loc[(data['previous'] == 0) & (data['pdays'] != -1), 'pdays'] = -1
    
    # Case 2: previous > 0 but pdays < 0 â†’ fix previous
    data.loc[(data['previous'] > 0) & (data['pdays'] < 0), 'previous'] = 0
    
    # Case 3: previous = 0 and pdays = 0 â†’ fix pdays
    data.loc[(data['previous'] == 0) & (data['pdays'] == 0), 'pdays'] = -1


print("Fix training data:")
fix_issue(train_df)
rule1_violations, rule2_violations, impossible_case = sanity_check(train_df)

print("Fix test data:")
fix_issue(test_df)
rule1_violations, rule2_violations, impossible_case = sanity_check(test_df)


# discrete_variables = ["campaign", "pdays", "previous"]
# continuous_variables = ["age", "balance", "duration"]
# nominal_variables = ["job", "marital", "education", "default", "housing", "loan", "contact", "poutcome", "y"]
# time_variables = ["day", "month"]


# Define visualization functions for Univariate EDA

# # For discrete / categorical variables: bar plot
# def show_barplot(variable, sort="index",df=train_df):
#     """
#     This function is used to make bar plots.

#     Parameters:
#     - variable - str - The variable name
#     - sort - str - either "index" or "value", used to control the sorting method
#         - sort="index" for discrete variables
#         - sort="value" for nominal variables
#     - df - DataFrame - The table where the data stores
#     """
#     try:
#         if not isinstance(variable, str):
#             raise TypeError("Parameter 'variable' has to be a string.")
            
#         if sort == "index":
#             counts = df[variable].value_counts().sort_index()
#         elif sort == "value":
#             counts = df[variable].value_counts().sort_values(ascending=False)
#         else:
#             raise ValueError("Parameter 'sort' has only 2 choices: either 'index' or 'value'")
            
#         plt.figure(figsize=(15, 8))
#         plt.bar(counts.index, counts.values, color="skyblue", edgecolor="black")
#         plt.xlabel(variable)
#         plt.ylabel("Frequency")
#         plt.title(f"Bar Plot of {variable}")
#         plt.xticks(counts.index,rotation=45)
#         plt.show()
#     except Exception as e:
#         print("Error occurred", e)

# # For continuous variables: histogram, boxplot, Q-Q plot
# def show_continuous_hist(variable, bins, df=train_df):
#     plt.figure(figsize=(10,8))
#     plt.hist(df[variable], bins=bins, color="skyblue", edgecolor="black")
#     plt.xlabel(variable)
#     plt.ylabel("Frequency")
#     plt.title(f"Histogram of {variable}")
#     plt.show()

# def show_continuous_boxplot(variable, df=train_df):
#     plt.figure(figsize=(10,8))
#     plt.boxplot(df[variable], labels=[variable])
#     plt.ylabel("Value")
#     plt.title(f"Box plot of {variable}")
#     plt.show()

# def show_continuous_qqplot(variable, df=train_df):
#     plt.figure(figsize=(10,8))
#     sm.qqplot(df[variable], line="45")
#     plt.title(f"Q-Q plot of {variable}")
#     plt.show()


# Define visualization functions for Univariate EDA
# Use axes to manage multiple plots

# For discrete / categorical variables: bar plot
def show_barplot(ax, variable, sort="index",df=train_df):
    """
    This function is used to make bar plots.

    Parameters:
    - ax - the subplot figure
    - variable - str - The variable name
    - sort - str - either "index" or "value", used to control the sorting method
        - sort="index" for discrete variables
        - sort="value" for nominal variables
    - df - DataFrame - The table where the data stores
    """
    try:
        if not isinstance(variable, str):
            raise TypeError("Parameter 'variable' has to be a string.")
            
        if sort == "index":
            counts = df[variable].value_counts().sort_index()
        elif sort == "value":
            counts = df[variable].value_counts().sort_values(ascending=True)
        else:
            raise ValueError("Parameter 'sort' has only 2 choices: either 'index' or 'value'")
            
        ax.barh(counts.index, counts.values, color="skyblue", edgecolor="black")
        ax.set_ylabel(variable)
        ax.set_xlabel("Frequency")
        ax.set_title(f"Bar Plot of {variable}")
        # ax.set_yticks(counts.index)
    except Exception as e:
        print("Error occurred", e)

# For continuous variables: histogram, boxplot, Q-Q plot
def show_continuous_hist(ax, variable, bins, df=train_df):
    ax.hist(df[variable], bins=bins, color="skyblue", edgecolor="black")
    ax.set_xlabel(variable)
    ax.set_ylabel("Frequency")
    ax.set_title(f"Histogram of {variable}")

def show_continuous_boxplot(ax, variable, df=train_df):
    ax.boxplot(df[variable], labels=[variable])
    ax.set_ylabel("Value")
    ax.set_title(f"Box plot of {variable}")

def show_continuous_qqplot(ax, variable, df=train_df):
    sm.qqplot(df[variable], line="45", ax=ax) 
    ax.set_title(f"Q-Q plot of {variable}")


def show_numeric(which_variable, df=train_df):
    print(df[which_variable].describe())
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    show_barplot(axes[0, 0], which_variable, sort="index", df=df)
    show_continuous_hist(axes[0, 1], which_variable, bins=15, df=df)
    show_continuous_boxplot(axes[1, 0], which_variable, df=df)
    show_continuous_qqplot(axes[1, 1], which_variable, df=df)

def show_categorical(which_variable, df=train_df):
    print(df[which_variable].describe())
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    show_barplot(ax, which_variable, sort="value", df=df)


# Continuous variable: age
show_numeric("age")


# Continuous variable: balance
show_numeric("balance")


# Continuous variable: duration
show_numeric("duration")


show_numeric("campaign")


# print(train_df["pdays"].value_counts())
# train_df[(train_df["pdays"]<=200) & (train_df["pdays"]>0)]
plt.hist(train_df[train_df["pdays"]>=0]["pdays"])


show_numeric("pdays")


show_numeric("previous")


train_df[train_df["previous"]==200]


# show_numeric("previous", df=original_df) # original_df has 1 outlier
original_df[original_df["previous"]==275]


show_categorical("job")


show_categorical("marital")


show_categorical("education")


show_categorical("default")


show_categorical("housing")


show_categorical("loan")


show_categorical("contact")


show_categorical("poutcome")


y_transform = {1:"yes", 0:"no"}
train_df["y_transformed"] = train_df["y"].map(y_transform)
show_categorical("y_transformed")


def numeric_categorical(categorical_x, numerical_y, df=train_df):
    sns.boxplot(data=df, x=categorical_x, y=numerical_y, palette='Set2')
    plt.title('', fontsize=12, fontweight='bold')
    plt.xlabel(categorical_x)
    plt.ylabel(numerical_y)
    plt.xticks(rotation=45)


plt.figure(figsize=(20, 15))
plt.subplot(2,3,1)
numeric_categorical("y", "age")
plt.subplot(2,3,2)
numeric_categorical("y", "campaign")
plt.subplot(2,3,3)
numeric_categorical("y", "pdays")
plt.subplot(2,3,4)
numeric_categorical("y", "previous")
plt.subplot(2,3,5)
numeric_categorical("y", "balance")
plt.subplot(2,3,6)
numeric_categorical("y", "duration")
plt.show()


# What if we ignore previous=200 for better comparison?
df_temp = train_df[train_df["previous"] < 200]
numeric_categorical("y", "previous", df=df_temp)


from scipy.stats import chi2_contingency
def check_categorical_relationship(which_variable, df=train_df):
    table = pd.crosstab(df[which_variable], df["y"])
    #print(table)
    chi2, p, dof, expected = chi2_contingency(table)
    print("p-value:", p) # If p<0.05, we say that the two varibales are dependent.
    if p < 0.05:
        print(f"Variable {which_variable} and target y are dependent.")
    else:
        print(f"Variable {which_variable} and target y are independent.")

def show_categorical_bar(which_variable, df=train_df):
    freq_table = pd.crosstab(df['y'], df[which_variable], normalize='index')
    #print(freq_table)
    freq_long = freq_table.reset_index().melt(id_vars='y', var_name=which_variable, value_name='frequency')
    #print(freq_long)
    freq_long['y_label'] = freq_long['y'].map({0: 'no', 1: 'yes'})
    sns.barplot(x='y_label', y='frequency', hue=which_variable, data=freq_long, dodge=True)
    plt.ylabel("Proportion")
    plt.xlabel("Whether the client subscribed to a term deposit")
    plt.title(f"Proportion of {which_variable} by y")
    plt.legend(
    title=which_variable,
    bbox_to_anchor=(1.05, 1),
    loc='upper left')


nominal_variables = ["job", "marital", "education", "default", "housing", "loan", "contact", "poutcome"]
for i in range(len(nominal_variables)):
    check_categorical_relationship(nominal_variables[i])


plt.figure(figsize=(20, 20))
for i in range(len(nominal_variables)):
    plt.subplot(4, 2, i+1)
    show_categorical_bar(nominal_variables[i])
plt.tight_layout()
plt.show()


show_categorical("day")
show_categorical("month")


def show_time_bar(which_variable, df=train_df):
    freq_table = pd.crosstab(df['y'], df[which_variable], normalize='index')
    #print(freq_table)
    freq_long = freq_table.reset_index().melt(id_vars='y', var_name=which_variable, value_name='frequency')
    #print(freq_long)
    freq_long['y_label'] = freq_long['y'].map({0: 'no', 1: 'yes'})
    sns.barplot(x='y_label', y='frequency', hue=which_variable, data=freq_long, dodge=True)
    plt.ylabel("Proportion")
    plt.xlabel("Whether the client subscribed to a term deposit")
    plt.title(f"Proportion of {which_variable} by y")
    plt.legend(
    title=which_variable,
    loc="upper center",          
    bbox_to_anchor=(0.5, -0.15),
    ncol=6)


time_variables = ["day", "month"]
for name in time_variables:
    check_categorical_relationship(name)


plt.figure(figsize=(15, 15))
for i in range(len(time_variables)):
    plt.subplot(2, 1, i+1)
    show_time_bar(time_variables[i])
plt.tight_layout()
plt.show()


train_data = train_df.drop(columns=["id", "y_transformed"], errors="ignore")
test_data = test_df.drop(columns=["id"], errors="ignore")


# Config
TARGET = "y"
POS_LABELS = {"yes", 1, "1"} # which values are positive class
# USE_SMOTE = True
USE_KBEST = True
KBEST_K = 30
RANDOM_STATE = 42


class FeatureEngineering(BaseEstimator, TransformerMixin):
    """
    Bank Marketing feature engineering:
    - pdays -> was_previously_contacted (/ days_since_last_contact) (keep raw pdays)
    - previous -> had_previous_contact (keep raw previous)
    - balance -> has_negative_balance (+ balance_bin if add_bins=True)
    - day -> day_sin, day_cos
    - month -> month_sin, month_cos
    """
    def __init__(self, add_bins=True):
        self.add_bins = add_bins

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # pdays â†’ two features
        if "pdays" in X.columns:
            X["was_previously_contacted"] = (pd.to_numeric(X["pdays"], errors="coerce") > 0).astype(int)
            #X["days_since_last_contact"]  = pd.to_numeric(X["pdays"], errors="coerce").replace(-1, np.nan)
            #X.drop(columns=["pdays"], inplace=True, errors="ignore")

        # previous â†’ binary flag
        if "previous" in X.columns:
            X["had_previous_contact"] = (pd.to_numeric(X["previous"], errors="coerce") > 0).astype(int)

        # balance â†’ sign info (+ optional coarse bins)
        if "balance" in X.columns:
            bal = pd.to_numeric(X["balance"], errors="coerce")
            X["has_negative_balance"] = (bal < 0).astype(int)
            if self.add_bins:
                X["balance_bin"] = pd.cut(
                    bal, bins=[-np.inf, 0, 500, 1500, 5000, np.inf],
                    labels=["<=0", "0-500", "500-1500", "1500-5000", ">5000"]
                )
                
        # day -> day_sin, day_cos
        if "day" in X:
            X["day_sin"] = np.sin(2 * np.pi * X["day"] / 31)
            X["day_cos"] = np.cos(2 * np.pi * X["day"] / 31)
            X.drop(columns=["day"], inplace=True)

        # month -> month_sin, month_cos
        if "month" in X:
            month_map = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
                         'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
            X["month"] = X["month"].map(month_map)
            X["month_sin"] = np.sin(2 * np.pi * X["month"] / 12)
            X["month_cos"] = np.cos(2 * np.pi * X["month"] / 12)
            X.drop(columns=["month"], inplace=True)
            
        return X


class OutlierHandler(BaseEstimator, TransformerMixin):
    """IQR clipping for numeric columns (more robust for heavy tails)."""
    def __init__(self, factor=2.5):
        self.factor = factor
        self.bounds_ = {}

    def fit(self, X, y=None):
        Xnum = pd.DataFrame(X).select_dtypes(include=[np.number])
        self.bounds_ = {}
        for col in Xnum.columns:
            q1, q3 = Xnum[col].quantile(0.25), Xnum[col].quantile(0.75)
            iqr = q3 - q1
            low = q1 - self.factor * iqr
            up  = q3 + self.factor * iqr
            self.bounds_[col] = (low, up)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col, (low, up) in self.bounds_.items():
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors="coerce")
                X[col] = X[col].clip(lower=low, upper=up)
        return X


def identify_feature_types(df: pd.DataFrame, target_col: str):
    """Return numeric and categorical columns (excluding target)."""
    tmp = df.copy()
    if target_col in tmp.columns:
        tmp = tmp.drop(columns=[target_col])
    num_cols = tmp.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = tmp.select_dtypes(exclude=[np.number]).columns.tolist()
    return num_cols, cat_cols


# ==========================
# Main pipeline
# ==========================
def train_and_evaluate(
    name: str, 
    pipeline, 
    X_train, y_train, 
    X_test, y_test, 
    threshold: float = 0.5
):
    """
    Fit, evaluate and print results for a given model pipeline.
    Returns the fitted pipeline, predicted probabilities, AUC.
    """
    pipeline.fit(X_train, y_train)

    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    auc = roc_auc_score(y_test, proba)

    print(f"\n=== {name} ===")
    print(f"ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, pred, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, pred))

    return {"name": name,
           "pipeline": pipeline,
            "proba": proba,
            "auc": auc
           } #pipeline, proba, auc
    
def run_bank_marketing_pipeline(
    df: pd.DataFrame,
    target_col: str = TARGET,
    pos_labels = POS_LABELS,
    use_smote: bool = True,
    use_kbest: bool = USE_KBEST,
    kbest_k: int = KBEST_K,
    random_state: int = RANDOM_STATE,
    threshold: float = 0.5 # decision threshold for all models
):
    data = df.copy()
    y = data[target_col]
    X = data.drop(columns=[target_col])
    

    # split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # one pass of FE to lock feature spaces before ColumnTransformer
    fe_preview = FeatureEngineering(add_bins=True)
    X_train_fe = fe_preview.fit_transform(X_train)
    X_test_fe = fe_preview.transform(X_test)

    num_cols, cat_cols = identify_feature_types(
        pd.concat([X_train_fe, y_train], axis=1), target_col=target_col
    ) # Just pass in X_train_fe is enough
    

    # numeric: (KNN impute â†’) IQR clip â†’ Robust scale
    numeric_pipeline = Pipeline(steps=[
        # ("imputer", KNNImputer(n_neighbors=5)), # Handle missing value
        ("outliers", OutlierHandler(factor=2.5)),
        ("scaler", RobustScaler())
    ])

    # categorical: (mode impute â†’) OneHot
    categorical_pipeline = Pipeline(steps=[
        # ("imputer", SimpleImputer(strategy="most_frequent")), # Handle missing value
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols),
        ],
        verbose_feature_names_out=False
    )

    # assemble steps (FE â†’ preprocess â†’ optional KBest â†’ optional SMOTE â†’ model)
    base_steps = [
        ("feature_engineering", FeatureEngineering(add_bins=True)),
        ("preprocessor", preprocessor),
    ]
    if use_kbest:
        base_steps.append(("kbest", SelectKBest(score_func=f_classif, k=kbest_k)))

    # use SMOTE only in training via ImbPipeline
    pipe_cls = ImbPipeline if use_smote else Pipeline

    # Helper function
    def make_pipe(estimator):
        steps = base_steps.copy()
        if use_smote:
            steps.append(("smote", SMOTE(random_state=random_state)))
        steps.append(("clf", estimator))
        return pipe_cls(steps)

    # Class imbalance ratio
    pos_count = np.sum(y_train == 1)
    neg_count = np.sum(y_train == 0)
    scale_pos_weight = (neg_count / pos_count) if (pos_count > 0) else 1.0
    # scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    # ===== model specs (define once; easy to extend)
    models = [] # Store functions rather than initialized models

    # # Logistic Regression
    # models.append((
    #     "Logistic Regression",
    #     lambda: LogisticRegression(
    #         max_iter=1000,
    #         class_weight="balanced",   # LR uses class_weight; OK with/without SMOTE
    #         random_state=random_state
    #     )
    # ))

    # # Random Forest
    # models.append((
    #     "Random Forest",
    #     lambda: RandomForestClassifier(
    #         n_estimators=300,
    #         max_depth=None,
    #         n_jobs=-1,
    #         class_weight=("balanced" if not use_smote else None),
    #         random_state=random_state
    #     )
    # ))

    # XGBoost
    def _xgb_est():
        params = dict(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="auc",
            n_jobs=-1,
            random_state=random_state
        )
        if not use_smote:
            params["scale_pos_weight"] = scale_pos_weight
        return XGBClassifier(**params)
    models.append(("XGBoost", _xgb_est)) 

    # LightGBM
    def _lgbm_est():
        params = dict(
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.9,
            colsample_bytree=0.9,
            n_jobs=-1,
            random_state=random_state
        )
        if not use_smote:
            params["class_weight"] = "balanced"
        return LGBMClassifier(**params)
    models.append(("LightGBM", _lgbm_est))

    # ===== run loop
    results = []
    for name, est_fn in models:
        try:
            est = est_fn() # Lazy Instantiation
            pipe = make_pipe(est)
            res = train_and_evaluate(
                name, pipe, X_train, y_train, X_test, y_test,
                threshold=threshold
            )
            results.append(res)
        except Exception as e:
            print(f"[Warn] {name} failed: {e}")


    # ===== return a tidy dict - result: (name, pipeline, proba, auc)
    out = {
        "results": results,                         # list of dicts from train_and_evaluate
        "best_by_auc": max(results, key=lambda r: r["auc"]) if results else None,
        "pipelines": {r["name"]: r["pipeline"] for r in results},
        "aucs": {r["name"]: r["auc"] for r in results}
    }
    return out


out = run_bank_marketing_pipeline(train_data, target_col="y", use_smote=False) 
print("AUCs:", out["aucs"])


# Access the best model
best_model_info = out["best_by_auc"]
print("Best model by AUC:", best_model_info["name"], "AUC:", best_model_info["auc"])
best_pipeline = best_model_info["pipeline"]


# # Make submissions
which_pipeline = best_pipeline #results["lr_pipeline"]
y_proba = which_pipeline.predict_proba(test_data)[:, 1]
submission_df["y"] = y_proba
submission_df


submission_df.to_csv('submission.csv', index=False)

