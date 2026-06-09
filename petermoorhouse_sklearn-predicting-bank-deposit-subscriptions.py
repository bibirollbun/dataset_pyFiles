import warnings
warnings.filterwarnings("ignore")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree


train_df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
train_df.head()


FEATURES = train_df.columns.tolist()
FEATURES.remove("id")
FEATURES.remove("y")
TARGET = "y"

NOMINAL_FEATURES = ["job", "marital", "contact", "poutcome"]
ORDINAL_FEATURES = ["month", "education"]

CATEGORICAL_FEATURES = NOMINAL_FEATURES + ORDINAL_FEATURES
BINARY_FEATURES = ["default", "housing", "loan"]
NUMERICAL_FEATURES = [f for f in FEATURES if f not in (CATEGORICAL_FEATURES + BINARY_FEATURES)]

X = train_df[FEATURES].copy()
y = train_df[TARGET]


encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoder.fit(X[CATEGORICAL_FEATURES])


def preprocess_features(X_scoped):
    
    ohe_array = encoder.transform(X_scoped[CATEGORICAL_FEATURES])
    ohe_column_names = encoder.get_feature_names_out(CATEGORICAL_FEATURES)
    ohe_df = pd.DataFrame(ohe_array, columns=ohe_column_names, index=X_scoped.index)
    
    bin_df = X_scoped[BINARY_FEATURES].replace({"no": 0, "yes": 1})
    num_df = X_scoped[NUMERICAL_FEATURES]
    
    return pd.concat([num_df, bin_df, ohe_df], axis=1)


X = preprocess_features(X)


pseudo_df = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
X_pseudo = pseudo_df[FEATURES].copy()
X_pseudo = preprocess_features(X_pseudo)

chris_df = pd.read_csv("/kaggle/input/train-more-xgb-nn-lb-0-9774/submission_ensemble_train_more.csv")
y_pseudo = chris_df[TARGET]
N_ORIGINAL_LABELS = len(y_pseudo)
print(f"{N_ORIGINAL_LABELS} ensemble labels loaded.")

pseudo_full = X_pseudo.copy()
pseudo_full[TARGET] = y_pseudo

THRESHHOLD_UPPER = 0.99
THRESHHOLD_LOWER = 0.01

confidence_mask = (pseudo_full[TARGET] >= THRESHHOLD_UPPER) | (pseudo_full[TARGET] <= THRESHHOLD_LOWER)
pseudo_confident = pseudo_full[confidence_mask].copy()

pseudo_confident[TARGET] = (pseudo_confident[TARGET] >= 0.5).astype(int)

X_pseudo = pseudo_confident.drop(columns=[TARGET])
y_pseudo = pseudo_confident[TARGET]
N_PSEUDO_LABELS = len(y_pseudo)
print(f"{N_PSEUDO_LABELS} pseudo labels created ({(100*N_PSEUDO_LABELS/N_ORIGINAL_LABELS):.1f} % of predictions).")

pseudo_confident.head()


original_df = pd.read_csv("/kaggle/input/bank-marketing-dataset-full/bank-full.csv", delimiter=";")
original_df['y'] = original_df.y.map({'yes':1,'no':0})

X_original = original_df[FEATURES].copy()
X_original = preprocess_features(X_original)
y_original = original_df[TARGET]


model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X, y)

plt.figure(figsize=(14, 6))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=["No", "Yes"],
    filled=True,
    rounded=True,
    impurity=False,
    label="root",
    fontsize=10,
    precision=2,
)

plt.title("Inspecting Decision Logic for a Shallow Classifier")
plt.show()


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1756)


gbt_models = []
gbt_val_roc_aucs = []
gbt_oof_preds = np.zeros(len(X))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"Training fold {fold + 1}...")
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    X_train = np.concatenate([X_train, X_original, X_pseudo])
    y_train = np.concatenate([y_train, y_original, y_pseudo])
    
    model = HistGradientBoostingClassifier(class_weight="balanced", learning_rate=0.05, max_iter=1024, min_samples_leaf=128, random_state=42)
    model.fit(X_train, y_train)
    
    gbt_models.append(model)
    
    val_proba = model.predict_proba(X_val)[:, 1]
    gbt_oof_preds[val_idx] = val_proba
    
    val_auc = roc_auc_score(y_val, val_proba)
    gbt_val_roc_aucs.append(val_auc)
    print(f"  Fold ROC AUC: {val_auc:.4f}")

print(f"\nMean ROC AUC across folds: {np.mean(gbt_val_roc_aucs):.4f}")


lr_models = []
lr_val_roc_aucs = []
lr_oof_preds = np.zeros(len(X))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"Training fold {fold + 1}...")
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    X_train = np.concatenate([X_train, X_original, X_pseudo])
    y_train = np.concatenate([y_train, y_original, y_pseudo])
    
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    lr_models.append(model)
    
    val_proba = model.predict_proba(X_val)[:, 1]
    lr_oof_preds[val_idx] = val_proba
    
    val_auc = roc_auc_score(y_val, val_proba)
    lr_val_roc_aucs.append(val_auc)
    print(f"  Fold ROC AUC: {val_auc:.4f}")

print(f"\nMean ROC AUC across folds: {np.mean(lr_val_roc_aucs):.4f}")


gbt_cv_auc = roc_auc_score(y, gbt_oof_preds)
print(f"Hist. GBT CV ROC AUC: {gbt_cv_auc:.4f}")

lr_cv_auc = roc_auc_score(y, lr_oof_preds)
print(f"Logistic Regression CV ROC AUC: {lr_cv_auc:.4f}")

corr = np.corrcoef(gbt_oof_preds, lr_oof_preds)[0, 1]
print(f"Correlation between Hist. GBT and LogisticRegression OOF predictions: {corr:.4f}")


X_meta_full = np.stack((gbt_oof_preds, lr_oof_preds), axis=1)
y_meta_full = y

meta_model = LogisticRegression(max_iter=1000)
meta_model.fit(X_meta_full, y_meta_full)

meta_val_preds = meta_model.predict_proba(X_meta_full)[:, 1]


l2model_cv_auc = roc_auc_score(y_meta_full, meta_val_preds)
print(f"Overall CV ROC AUC: {l2model_cv_auc:.4f}")


weights = meta_model.coef_[0]
intercept = meta_model.intercept_[0]

features = ["Hist. GBT", "Logistic Regression"]

plt.bar(features, weights)
plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
plt.title("L2 Model Coefficients")
plt.ylabel("Weight")
plt.show()

print("Intercept:", intercept)


test_df = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
X_test = test_df[FEATURES].copy()
X_test = preprocess_features(X_test)

l1_gbt_preds = np.zeros(len(test_df))
for model in gbt_models:
    l1_gbt_preds += model.predict_proba(X_test)[:, 1]
l1_gbt_preds /= len(gbt_models)

l1_lr_preds = np.zeros(len(test_df))
for model in lr_models:
    l1_lr_preds += model.predict_proba(X_test)[:, 1]
l1_lr_preds /= len(lr_models)

l2_inputs = np.stack((l1_gbt_preds, l1_lr_preds), axis=1)
test_preds = meta_model.predict_proba(l2_inputs)[:, 1]


submission = pd.DataFrame({
    "id": test_df["id"],
    "y": test_preds
})

submission.to_csv("submission.csv", index=False)
submission.head()

