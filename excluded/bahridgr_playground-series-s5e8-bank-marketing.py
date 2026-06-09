import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
import optuna 
from lightgbm import early_stopping, log_evaluation

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)


TRAIN_PATH = "/kaggle/input/playground-series-s5e8/train.csv"
TEST_PATH  = "/kaggle/input/playground-series-s5e8/test.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

train_df = train_df.drop('id', axis=1)
test_id = test_df['id']
test_df = test_df.drop('id', axis=1)


def check_data(dataframe):
    print("########################## HEAD ##########################")
    display(dataframe.head())
    print("########################## ISNULL(?) ##########################")
    display(dataframe.isna().sum())
    print("########################## INFO ##########################")
    display(dataframe.info())
    print("########################## SHAPE ##########################")
    display(dataframe.shape)
    print("########################## DESCRİBE ##########################")
    display(dataframe.describe([0.1, 0.25, 0.5, 0.75, 0.90, 0.99]).T)

check_data(train_df)


# 1) Target variable distribution
plt.figure(figsize=(5,4))
sns.countplot(x="y", data=train_df)
plt.title("Target (y) Distribution")
plt.show()

print(train_df["y"].value_counts(normalize=True))

# 2) The relationship between categorical variables and the target
print(' ######## The relationship between categorical variables and the target #######')
categorical_cols = train_df.select_dtypes(include=["object"]).columns.tolist()
print("Categorical Columns:", categorical_cols)

n_cat = len(categorical_cols)
n_cols = 2
n_rows = (n_cat + 1) // n_cols  # yukarı yuvarla

fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4*n_rows))
axes = axes.flatten()

for i, col in enumerate(categorical_cols):
    sns.countplot(x=col, hue="y", data=train_df, ax=axes[i])
    axes[i].set_title(f"{col} vs Target (y)")
    axes[i].tick_params(axis='x', rotation=45)

# Boş kalan subplot'ları silelim
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# 3) Correlation heatmap
print(' ######## Correlation heatmap ########')
numeric_cols = train_df.select_dtypes(include=["int64","float64"]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in ["id", "y"]]

plt.figure(figsize=(12,8))
corr = train_df[numeric_cols + ["y"]].corr()
sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap (Numeric Features)")
plt.show()

# 4) Distributions of numerical variables
print(' ####### Distributions of numerical variables ########')
num_features = ["age", "duration", "campaign", "previous"]
valid_num_features = [col for col in num_features if col in train_df.columns]

n_num = len(valid_num_features)
n_cols = 2
n_rows = (n_num + 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
axes = axes.flatten()

for i, col in enumerate(valid_num_features):
    sns.histplot(data=train_df, x=col, hue="y", kde=True, bins=30, ax=axes[i])
    axes[i].set_title(f"Distribution of {col} by Target (y)")

# Boş subplot'ları sil
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


train_fe = train_df.copy()
test_fe = test_df.copy()

# 1) Age groups (sample derivative variable)
def age_group(age):
    if age < 30:
        return "young"
    elif 30 <= age < 50:
        return "middle"
    else:
        return "senior"

train_fe["age_group"] = train_fe["age"].apply(age_group)
test_fe["age_group"] = test_fe["age"].apply(age_group)

# 2) Categorical encoding (One-Hot Encoding)
categorical_cols = train_fe.select_dtypes(include=["object"]).columns.tolist()
categorical_cols = [c for c in categorical_cols if c != "y"]  # target hariç

train_fe = pd.get_dummies(train_fe, columns=categorical_cols, drop_first=True)
test_fe  = pd.get_dummies(test_fe,  columns=categorical_cols, drop_first=True)

# 3) Do Train and Test have the same columns? (some categories may not be in test)
train_cols = set(train_fe.columns)
test_cols  = set(test_fe.columns)

# Add missing columns
for col in train_cols - test_cols:
    test_fe[col] = 0
for col in test_cols - train_cols:
    train_fe[col] = 0

# Synchronize column order
train_fe = train_fe.sort_index(axis=1)
test_fe  = test_fe.sort_index(axis=1)

# Control
print("Train shape after FE:", train_fe.shape)
print("Test shape after FE:", test_fe.shape)


train_fe.head()


test_fe.head()


# Target Variables and Features
X = train_fe.drop(["y"], axis=1)
y = train_fe["y"]
X_test = test_df.drop(["id", "y"], axis=1, errors="ignore")

# CV 
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "XGBoost": xgb.XGBClassifier(
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6
    ),
    "LightGBM": lgb.LGBMClassifier(
        random_state=42,
        n_estimators=500,
        learning_rate=0.05,
        max_depth=-1,
        verbose=-1
    ),
    "CatBoost": CatBoostClassifier(
        verbose=0,
        random_state=42,
        iterations=500,
        learning_rate=0.05,
        depth=6
    )
}

results = {}

for name, model in models.items():
    cv_scores = []
    for train_idx, valid_idx in kf.split(X, y):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_valid)[:,1]
        auc = roc_auc_score(y_valid, y_pred)
        cv_scores.append(auc)
    
    results[name] = {
        "mean_auc": np.mean(cv_scores),
        "std_auc": np.std(cv_scores)
    }

# Show the results a Dataframe
results_df = pd.DataFrame(results).T.sort_values(by="mean_auc", ascending=False)
print(results_df)


import warnings
warnings.filterwarnings("ignore")  # sklearn, lightgbm warningleri kapatır

import lightgbm as lgb
import optuna
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from lightgbm import early_stopping, log_evaluation

# Target ve Features
X = train_fe.drop(["y"], axis=1)
y = train_fe["y"]

# Optuna objective function
def objective(trial):
    params = {
        "n_estimators": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "num_leaves": trial.suggest_int("num_leaves", 20, 3000),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 300),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": 42,
        "n_jobs": -1,
        "device": "gpu"   # GPU kullanımı aktif
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    
    for train_idx, valid_idx in cv.split(X, y):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="auc",
            callbacks=[early_stopping(50), log_evaluation(-1)]  # logları tamamen sustur
        )
        
        y_pred = model.predict_proba(X_valid)[:, 1]
        aucs.append(roc_auc_score(y_valid, y_pred))
    
    mean_auc = np.mean(aucs)
    print(f"Trial {trial.number}: AUC={mean_auc:.5f}")  # sadece trial ve auc yazdır
    return mean_auc

# Start Optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30, show_progress_bar=True)

best_params = study.best_params
print("\n✅ Best Parameters:", best_params)
print("✅ Best AUC:", study.best_value)



best_params


final_model = lgb.LGBMClassifier(**best_params)
final_model.fit(X, y)


import joblib

joblib.dump(final_model, "lgb_best_model.pkl")


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')
plot_importance(final_model, X_train)


test_fe= test_fe.drop('y', axis=1)


y_pred = final_model.predict(test_fe)


submission = pd.DataFrame({
    "id": test_id,
    "y": y_pred
})

submission.to_csv("submission.csv", index=False)




