# Packages 
# Data Processing 
import numpy as np 
import pandas as pd 
# Visualization 
import matplotlib.pyplot as plt 
plt.rcParams['figure.dpi'] = 200 
import seaborn as sns 
# Statistics 
import math 
from scipy import stats 
from scipy.stats import norm 
# File Path 
import os 
for dirname, _, filenames in os.walk('/kaggle/input'): 
    for filename in filenames: 
        print(os.path.join(dirname, filename))


# Version check
print(f"numpy version: {np.__version__}")
print(f"pandas version: {pd.__version__}")

# Ignore Warning
import warnings
warnings.filterwarnings("ignore")

# setting
path_root = "/kaggle/input/playground-series-s5e12/"
seed = 394
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)


df_train = pd.read_csv(path_root + "train.csv")
print("Train shape:",df_train.shape)

df_test = pd.read_csv(path_root + "test.csv")
print("Test shape:", df_test.shape)


df_train.columns


# features
list_not_features = ['id', 'diagnosed_diabetes']
list_features = [c for c in df_train.columns if not c in list_not_features]

# categorical features
list_categorical_features = df_train.select_dtypes(include=["object", "category"]).columns.tolist()
list_categorical_features = list(set(list_features).intersection(set(list_categorical_features)))
list_numeric_features = list(set(list_features) - set(list_categorical_features))

print(f"Numeric features({len(list_numeric_features)}): {list_numeric_features}")
print(f"Categorical features({len(list_categorical_features)}): {list_categorical_features}")


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline


# copy df
df_train_adv = df_train[list_features].copy()
df_test_adv = df_test[list_features].copy()

# labeling
df_train_adv["is_test"] = 0
df_test_adv["is_test"] = 1

# concat
df_adv = pd.concat([df_train_adv, df_test_adv], axis=0, ignore_index=True)
X = df_adv[list_features]
y = df_adv["is_test"]


# preprocess pipeline

numeric_transformer = Pipeline(steps = [
    ('passthrough', 'passthrough')
])

categorical_transformer = Pipeline(steps = [
    ('onehot', OneHotEncoder(sparse_output = False, handle_unknown = 'ignore'))
])

preprocessor = ColumnTransformer(
    transformers = [
        ('num', numeric_transformer, list_numeric_features),
        ('cat', categorical_transformer, list_categorical_features)
    ]
)


from sklearn.ensemble import RandomForestClassifier


# model definition
clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    n_jobs=-1,
    random_state=seed,
    class_weight="balanced_subsample",
)


# model pipeline
model = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", clf),
    ]
)


from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score


X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=seed,
)


model.fit(X_train, y_train)


# auc score

valid_pred_proba = model.predict_proba(X_valid)[:, 1]
adv_auc = roc_auc_score(y_valid, valid_pred_proba)

print(f"[Adversarial ROC AUC] {adv_auc:.4f}")


# roc curve

fpr, tpr, thresholds = roc_curve(y_valid, valid_pred_proba)
plt.figure(figsize=(4, 4), facecolor="white")
plt.plot(fpr, tpr, label=f"AUC = {adv_auc:.4f}", linewidth=2)

plt.plot([0, 1], [0, 1], "k--", alpha=0.6)

plt.title("Adversarial Validation ROC Curve", fontsize=12)
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.legend()
plt.grid(alpha=0.3)

plt.show()


preprocessor_fitted = model.named_steps["preprocess"]
model_fitted = model.named_steps["model"]

# dummy feature names
if len(list_categorical_features) > 0:
    ohe = preprocessor_fitted.named_transformers_["cat"].named_steps["onehot"]
    list_ohe_feature_names = ohe.get_feature_names_out(list_categorical_features).tolist()
else:
    list_ohe_feature_names = []

# feature names
list_all_feature_names = list_numeric_features + list_ohe_feature_names
print(list_all_feature_names)


# feature importances
importances = model_fitted.feature_importances_

df_feature_importance = (
    pd.DataFrame(
        {"feature": list_all_feature_names, "importance": importances}
    )
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)

display(df_feature_importance.head(10))


top_k = 10
list_top_features = df_feature_importance.head(top_k)["feature"].tolist()


display(df_train[list_top_features].describe().round(3).T)
display(df_test[list_top_features].describe().round(3).T)


def hist_train_and_test(col, bins=40, alpha=0.5):

    plt.figure(figsize=(6, 4), facecolor="white")

    sns.histplot(df_train[col], stat="probability",kde=True, bins=bins, edgecolor=None, alpha=alpha, label="train")
    sns.histplot(df_test[col], stat="probability",kde=True, bins=bins, edgecolor=None, alpha=alpha, label="test")

    plt.title(f"Distribution of: {col}", fontsize = 12)
    plt.xlabel("")
    
    plt.tight_layout()
    plt.legend()
    plt.show()


hist_train_and_test("physical_activity_minutes_per_week", bins=50)


hist_train_and_test("triglycerides")

