import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

from xgboost import XGBClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import mutual_info_classif

import warnings

warnings.filterwarnings('ignore')


experiments = pd.DataFrame(columns=["Name", "FeaturesAdded", "Result"])

def log_experiment(name, features_added, auc, df=experiments):
    entry = pd.DataFrame(
        [[name, features_added, auc]],
        columns=["Name", "FeaturesAdded", "Result"]
    )
    return pd.concat([df, entry], ignore_index=True)
    
def load_data(): 
    """
    Just load the csv files
    """
    df_train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv', index_col="id")
    df_test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv', index_col="id")
    df_external = pd.read_csv('/kaggle/input/bank-marketing-dataset-full/bank-full.csv', sep=";")

    df_external["y"] = df_external["y"].map({"no": 0, "yes": 1})
    
    return df_train, df_test, df_external

# Find discrete columns
def create_buckets(df):
    """
    Will create 4 Buckets and put the columns that correspond
    • Numerical (Continuous)
    • Numerical (Discrete)
    • Categorical (Nominal)
    """
    continuous_cols = df.select_dtypes(include="float64").columns.tolist()
    discrete_cols   = df.select_dtypes(include="int64").columns.tolist()
    nominal_cols    = df.select_dtypes(include="object").columns.tolist()

    return {
        "continuous": continuous_cols,
        "discrete": discrete_cols,
        "nominal": nominal_cols,
    }


%%time
df_train, df_test, df_external = load_data()
df = pd.concat([df_train, df_external])
buckets = create_buckets(df)

continuous_cols = buckets ["continuous"]
discrete_cols = buckets["discrete"]
nominal_cols = buckets["nominal"]

print("Continuous:", continuous_cols, "\nDiscrete:", discrete_cols, "\nNominal:", nominal_cols)


%%time
# Explore a bit the different cols
plt.figure(figsize=(15,4))

# Plot each column
for i, col in enumerate(discrete_cols, 1):
    plt.subplot(3,3,i)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')

plt.tight_layout()
plt.show()


%%time
# Explore a bit the different cols

# Handle 'job' separately
plt.figure(figsize=(20, 4))
sns.countplot(data=df, x="job", order=df["job"].value_counts().index)
plt.title("Distribution of job")
plt.tight_layout()
plt.show()

# Now plot the rest in a 3x3 grid
other_nominal_cols = [col for col in nominal_cols if col != "job"]

plt.figure(figsize=(15,4))
for i,col in enumerate(other_nominal_cols,1):
    plt.subplot(3,3,i)
    sns.countplot(data=df, x=col, order=df[col].value_counts().index)
    plt.title(f'Distribution of {col}')

plt.tight_layout()
plt.show()


def format_data(df, nominal_cols, discrete_cols, enc=None, fit_encoder=True):
    df = df.copy()

    for col in ["default", "housing", "loan"]:
        df[col] = df[col].map({"no": 0, "yes": 1}).astype("Int64")

    df[nominal_cols] = df[nominal_cols].fillna("missing").astype(str)

    ordinal_cols = ["job", "month", "education", "poutcome"]

    if fit_encoder:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        df[ordinal_cols] = enc.fit_transform(df[ordinal_cols])
    else:
        df[ordinal_cols] = enc.transform(df[ordinal_cols])

    ohe_cols = ["marital", "contact"]
    df = pd.get_dummies(df, columns=ohe_cols, dummy_na=True)

    updated_nominal = [c for c in nominal_cols if c not in ordinal_cols + ohe_cols]
    updated_discrete = discrete_cols + ordinal_cols + [
        c for c in df.columns if any(c.startswith(ohe) for ohe in ohe_cols)
    ]

    df = df.astype(np.float32)

    if fit_encoder:
        return df, updated_nominal, updated_discrete, enc
    else:
        return df, updated_nominal, updated_discrete


%%time
X, nominal_cols, discrete_cols, enc = format_data(df, nominal_cols, discrete_cols, fit_encoder=True)
y = X.pop("y")

X.select_dtypes("object").columns


def score_dataset(X, y, model=None):
    if model is None:
        model = XGBClassifier(
            verbose=0,
            eval_metric='AUC',
            random_seed=42
        )
    
    scores = cross_val_score(
        model, X, y,
        cv=3,
        scoring="roc_auc",
    )
    return scores.mean()


%%time
def mutual_info_ranking(X, y):
    mi_scores = mutual_info_classif(
        X, y,
        discrete_features=False,
        random_state=42
    )
    return pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

mi_ranking = mutual_info_ranking(X, y)
print(mi_ranking)


%%time
model = XGBClassifier(verbose=0)
score = score_dataset(X, y, model)

experiments.loc[len(experiments)] = ["Default", "Cleaned Dataset", score]
experiments


# Mid tier features like balance, education and month
def engineer_features(X: pd.DataFrame):
    Xf = X.copy()
    added = []

    if 'duration' in Xf:
        Xf['duration_log'] = np.log1p(Xf['duration'])
        p99 = np.nanpercentile(Xf['duration'], 99)
        Xf['duration_cap'] = Xf['duration'].clip(upper=p99)
        Xf['long_call'] = (Xf['duration'] >= p99).astype('int8')
        added += ['duration_log', 'duration_cap', 'long_call']

        if 'contact_cellular' in Xf:
            Xf['dur_x_cellular'] = Xf['duration_log'] * Xf['contact_cellular']
            added.append('dur_x_cellular')
        if 'contact_unknown' in Xf:
            Xf['dur_x_unknown'] = Xf['duration_log'] * Xf['contact_unknown']
            added.append('dur_x_unknown')   

    if 'pdays' in Xf:
        Xf['pdays_no_prev'] = (Xf['pdays'] == -1).astype('int8')
        added.append('pdays_no_prev')

    if 'balance' in Xf:
        balance_nonan = Xf['balance'].fillna(Xf['balance'].median())
        Xf['balance_bin'] = pd.qcut(balance_nonan, q=4, labels=False)
        Xf['balance_high'] = (Xf['balance_bin'] == 3).astype('int8')
        added += ['balance_bin', 'balance_high']

        for col in ['poutcome_success', 'poutcome_failure', 'poutcome_other', 'poutcome_unknown']:
            if col in Xf:
                name = f'balance_high_x_{col}'
                Xf[name] = Xf['balance_high'] * Xf[col]
                added.append(name)

    return Xf, added

X_featured, _ = engineer_features(X)
score = score_dataset(X_featured, y, model)
experiments.loc[len(experiments)] = ["Default", "Features Eng", score]
experiments


pip install optuna-integration[xgboost]


# Prepare same data as before
df_train, df_test, df_external = load_data()
df = pd.concat([df_train, df_external])
buckets = create_buckets(df)

continuous_cols = buckets ["continuous"]
discrete_cols = buckets["discrete"]
nominal_cols = buckets["nominal"]

X_featured, nominal_cols, discrete_cols, enc = format_data(
    df, nominal_cols, discrete_cols, fit_encoder=True
)
y = X_featured.pop("y")
X_featured, added = engineer_features(X_featured)
train_cols = X_featured.columns


import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

def objective(trial):
    params = {
        "device": "cuda",
        "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-2, 10.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
        "eval_metric": "auc",
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": -1,
    }
    model = XGBClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_featured, y, cv=cv, scoring="roc_auc")
    return scores.mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30, timeout=900)
print("Best params:", study.best_params)
print("Best AUC:", study.best_value)


boost = XGBClassifier(**study.best_params)
boost.fit(X_featured, y)


X_test, _, _ = format_data(df_test, nominal_cols, discrete_cols, enc=enc, fit_encoder=False)
X_test, _ = engineer_features(X_test)
X_test = X_test.reindex(columns=train_cols, fill_value=0)


predictions = boost.predict_proba(X_test)[:, 1]

output = pd.DataFrame({'id': df_test.index, 'y': predictions})
output.to_csv('ps5e8_prediction.csv', index=False)
print("Your submission was successfully saved!")

