#Import necessary libraries
import pandas as pd
import polars as pl
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, make_scorer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


ss_df = pd.read_csv('/kaggle/input/amex-default-prediction/sample_submission.csv', engine = 'python')
train_labels = pd.read_csv('/kaggle/input/amex-default-prediction/train_labels.csv', engine = 'python')
train_df1 = pd.read_csv('/kaggle/input/amex-default-prediction/train_data.csv', nrows=200_000) #contains header and first 200_000 data rows
train_df2 = pd.read_csv('/kaggle/input/amex-default-prediction/train_data.csv', skiprows=range(1,200_000+1), nrows = 200_000) #contains header and data rows from 200_001 till and including 400_000
train_df3 = pd.read_csv('/kaggle/input/amex-default-prediction/train_data.csv', skiprows=range(1,400_000+1), nrows=58913) #contains header and data rows from 400_001 till and including 458_913
train_df1_2 = pd.concat([train_df1,train_df2], ignore_index = True)
train_df = pd.DataFrame(pd.concat([train_df1_2, train_df3], ignore_index=True))
cat_features = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']


# test_df1 = pd.read_csv('/kaggle/input/amex-default-prediction/test_data.csv', nrows=200_000)
# test_df2 = pd.read_csv('/kaggle/input/amex-default-prediction/test_data.csv', skiprows=range(1,200_000+1), nrows=200_000)
# test_df3 = pd.read_csv('/kaggle/input/amex-default-prediction/test_data.csv', skiprows=range(1,400_000+1), nrows=200_000)
# test_df4 = pd.read_csv('/kaggle/input/amex-default-prediction/test_data.csv', skiprows=range(1,600_000+1), nrows=324_621)
# test_df1_2 = pd.concat([test_df1, test_df2], ignore_index = True)
# test_df3_4 = pd.concat([test_df3, test_df4], ignore_index = True)
# test_df = pd.concat([test_df1_2, test_df3_4], ignore_index = True)


def denoise_numeric(df):

    df = df.copy()

    # numeric columns only
    num_cols = df.select_dtypes(include=['float', 'int']).columns

    for col in num_cols:

        # Remove infinities
        #df[col].replace([np.inf, -np.inf], np.nan, inplace=True)

        # Clip extreme outliers (standard Kaggle practice)
        q1, q99 = df[col].quantile([0.01, 0.99])
        df[col] = df[col].clip(q1, q99)

        # Apply rounding / bucketing to reduce noise
        # floor(x * 100) → keeps 2 decimal places
        # You can change 100 to 1000 if you want more resolution.
        df[col] = np.floor(df[col] * 100) / 100.0

        # NA values remain NA — untouched

    return df





train_merged = train_df.merge(train_labels, on='customer_ID', how='left')
train_merged.shape
train_merged['target'].isna().sum()


def CleanMissing(df, threshold):
    # Drop the columns with missing fraction > criteria
    cols_to_drop = df.columns[df.isna().mean() > threshold].tolist()
    df_clear = df.drop(columns=cols_to_drop)
    return df_clear



# Drop the columns with missing fraction > 0.5
cols_to_drop = train_merged.columns[train_merged.isna().mean() > 0.5].tolist()

# Add customer_ID and S_2
cols_to_drop += ['customer_ID', 'S_2']

# Create the modified dataframe
train_merged = train_merged.drop(columns=cols_to_drop)
cat_features = [c for c in cat_features if c not in cols_to_drop]


def amex_metric(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:

    def top_four_percent_captured(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
        four_pct_cutoff = int(0.04 * df['weight'].sum())
        df['weight_cumsum'] = df['weight'].cumsum()
        df_cutoff = df.loc[df['weight_cumsum'] <= four_pct_cutoff]
        return (df_cutoff['target'] == 1).sum() / (df['target'] == 1).sum()
        
    def weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
        df['random'] = (df['weight'] / df['weight'].sum()).cumsum()
        total_pos = (df['target'] * df['weight']).sum()
        df['cum_pos_found'] = (df['target'] * df['weight']).cumsum()
        df['lorentz'] = df['cum_pos_found'] / total_pos
        df['gini'] = (df['lorentz'] - df['random']) * df['weight']
        return df['gini'].sum()

    def normalized_weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        y_true_pred = y_true.rename(columns={'target': 'prediction'})
        return weighted_gini(y_true, y_pred) / weighted_gini(y_true, y_true_pred)

    g = normalized_weighted_gini(y_true, y_pred)
    d = top_four_percent_captured(y_true, y_pred)

    return 0.5 * (g + d)

def amex_metric_xgb(preds, dtrain):
    y_true = pd.DataFrame({"target": dtrain.get_label()})
    y_pred = pd.DataFrame({"prediction": preds})
    return "amex", amex_metric(y_true, y_pred)


#test_df = denoise_numeric(test_df)


train_labels['customer_ID'].unique().size


cols_to_keep = X_train.columns
X_test = test_df[cols_to_keep].copy()

for c in cat_features:
    X_test[c] = X_test[c].astype('category')


# -----------------------
# 1. Align test columns
# -----------------------
cols_to_keep = X_train.columns
X_test = test_df[cols_to_keep]

for c in cat_features:
    X_test[c] = X_test[c].astype("category")

# -----------------------
# 2. Predict on test
# -----------------------
test_df['prediction'] = best_xgb.predict_proba(X_test)[:, 1]

# -----------------------
# 3. Aggregate per customer_ID
# -----------------------
final_pred = (
    test_df.groupby('customer_ID')['prediction']
    .tail(1)
    .reset_index()
)

# -----------------------
# 4. Merge with sample submission to ensure correct order
# -----------------------
submission = ss_df[['customer_ID']].merge(
    final_pred,
    on='customer_ID',
    how='right'
)

# -----------------------
# 5. Safety: fill missing
# -----------------------
submission['prediction'].fillna(0.0, inplace=True)

# -----------------------
# 6. Save file
# -----------------------
submission.to_csv("submission.csv", index=False)



train_last = (
    train_df
    .groupby("customer_ID")
    .last()
)



train_last


train_last_merged = train_last.merge(train_labels, on='customer_ID', how='left')
train_last_merged.shape
train_last_merged['target'].isna().sum()


# Drop the columns with missing fraction > 0.5
cols_to_drop = train_last_merged.columns[train_last_merged.isna().mean() > 0.5].tolist()

# Add customer_ID and S_2
cols_to_drop += ['customer_ID', 'S_2']

# Create the modified dataframe
train_last_merged = train_last_merged.drop(columns=cols_to_drop)
cat_features = [c for c in cat_features if c not in cols_to_drop]


X = train_last_merged.drop(columns=['target']).copy()
for c in cat_features:
    X[c] = X[c].astype('category')
y = train_last_merged['target'].copy()
sum(y)/len(y)


X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=42, stratify = y)


#denoise the data



import optuna

trial_history = []

def objective(trial):

    params = {
        "objective": "binary:logistic",
        "tree_method": "hist",
        "device": "cuda",
        "eval_metric": "auc",
        "missing": np.nan,
        "enable_categorical": True,
        "n_estimators": 400,

        # ---- Parameters to tune ----
        "max_depth": trial.suggest_int("max_depth", 8, 80, step=8),
        "min_child_weight": trial.suggest_int("min_child_weight", 10, 100, step=10),
        "learning_rate": trial.suggest_categorical("learning_rate", [0.05, 0.1]),
        "subsample": trial.suggest_categorical("subsample", [0.3, 0.4, 0.5, 0.8, 1.0]),
        "colsample_bytree": trial.suggest_categorical("colsample_bytree", [0.5, 0.8, 1.0]),
        

        # ---- Regularization tuning ----
        "lambda": trial.suggest_float("lambda", 1e-3, 10, log=True),
        "alpha": trial.suggest_float("alpha", 1e-3, 5, log=True),
        "gamma": trial.suggest_float("gamma", 0, 10),
    }

    model = xgb.XGBClassifier(**params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        early_stopping_rounds=30,
        verbose=False
    )

    # predictions
    preds = model.predict_proba(X_valid)[:, 1]

    # Compute AMEX
    y_true_df = pd.DataFrame({'target': y_valid}).reset_index(drop=True)
    y_pred_df = pd.DataFrame({'prediction': preds}).reset_index(drop=True)

    score = amex_metric(y_true_df, y_pred_df)
    
    # ----------------------------
    # SAVE TRIAL RESULTS SAFELY
    # ----------------------------
    trial_history.append({
        "params": params.copy(),              # <-- IMPORTANT FIX
        "best_iteration": model.best_iteration,
        "amex_score": score
    })
    
    return score  # Optuna will maximize the AMEX metric


# Run study
study = optuna.create_study(
    study_name="xgb_amex_gpu",
    direction="maximize",
    storage="sqlite:///optuna_xgb.db",
    load_if_exists=True
)

study.optimize(objective, n_trials=300)  # adds 300 more

# Convert to DataFrame
trial_df = pd.DataFrame(trial_history)

# Flatten params dict into columns (optional but very useful!)
trial_df = trial_df.join(trial_df["params"].apply(pd.Series)).drop(columns=["params"])

trial_df.sort_values("amex_score", ascending=False, inplace=True)

print("Best AMEX:", study.best_trial.value)
print("Best Params:", study.best_trial.params)



# Convert to DataFrame
trial_df = pd.DataFrame(trial_history)

# Flatten params dict into columns (optional but very useful!)
trial_df = trial_df.join(trial_df["params"].apply(pd.Series)).drop(columns=["params"])

trial_df.sort_values("amex_score", ascending=False, inplace=True)

print("Best AMEX:", study.best_trial.value)
print("Best Params:", study.best_trial.params)


import seaborn as sns
import matplotlib.pyplot as plt

for col in ['max_depth', 'min_child_weight', 'learning_rate', 'subsample', 
            'colsample_bytree', 'n_estimators', 'lambda', 'alpha', 'gamma']:
    
    plt.figure(figsize=(6,4))
    sns.scatterplot(data=trial_df, x=col, y='amex_score')
    plt.title(f'{col} vs AMEX')
    plt.show()



# The following values of bes


# import optuna

# trial_history2 = []

# def objective(trial):

#     params = {
#         "objective": "binary:logistic",
#         "tree_method": "hist",
#         "eval_metric": "auc",
#         "missing": np.nan,
#         "enable_categorical": True,

#         # ---- tuned params ----
#         "max_depth": 40,
#         "min_child_weight": 30,
#         "learning_rate":0.05,
#         "subsample": 1.0,
#         "colsample_bytree": 0.5,
#         "n_estimators": 400,
#         "alpha": 0,

#         # ---- Regularization tuning ----
#         "lambda": trial.suggest_float("lambda", 1e-3, 7, log=True),
#         "gamma": trial.suggest_float("gamma", 0, 4),
#     }

#     model = xgb.XGBClassifier(**params)

#     model.fit(
#         X_train, y_train,
#         eval_set=[(X_valid, y_valid)],
#         early_stopping_rounds=30,
#         verbose=False
#     )

#     # predictions
#     preds = model.predict_proba(X_valid)[:, 1]

#     # Compute AMEX
#     y_true_df = pd.DataFrame({'target': y_valid}).reset_index(drop=True)
#     y_pred_df = pd.DataFrame({'prediction': preds}).reset_index(drop=True)

#     score = amex_metric(y_true_df, y_pred_df)
    
#     # ----------------------------
#     # SAVE TRIAL RESULTS SAFELY
#     # ----------------------------
#     trial_history2.append({
#         "params": params.copy(),              # <-- IMPORTANT FIX
#         "best_iteration": model.best_iteration,
#         "amex_score": score
#     })
    
#     return score  # Optuna will maximize the AMEX metric


# # Run study
# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=50)

# # Convert to DataFrame
# trial2_df = pd.DataFrame(trial_history2)

# # Flatten params dict into columns (optional but very useful!)
# trial2_df = trial2_df.join(trial2_df["params"].apply(pd.Series)).drop(columns=["params"])

# trial2_df.sort_values("amex_score", ascending=False, inplace=True)

# print("Best AMEX:", study.best_trial.value)
# print("Best Params:", study.best_trial.params)



from xgboost.callback import EarlyStopping

best_params = {
    "objective": "binary:logistic",
    "tree_method": "hist",
    "device": "cuda",
    "missing": np.nan,
    "enable_categorical": True,

    # tuned params
    "max_depth": 80,
    "min_child_weight": 40,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.5,
    "n_estimators": 1000,   # LARGE ON PURPOSE
    "lambda": 0.007670118758441345,
    "alpha": 0.09157175965761555,
    "gamma": 1.028865262384579,
}

best_xgb = xgb.XGBClassifier(**best_params)

best_xgb.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="auc",  # REQUIRED for early stopping
    callbacks=[
        EarlyStopping(
            rounds=50,           # patience
            save_best=True,      # keep best trees
            maximize=True
        )
    ],
    verbose=False
)



preds = best_xgb.predict_proba(X_valid)[:,-1]


def top_four_percent_captured(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
    df = (pd.concat([y_true, y_pred], axis='columns')
          .sort_values('prediction', ascending=False))
    df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
    four_pct_cutoff = int(0.04 * df['weight'].sum())
    df['weight_cumsum'] = df['weight'].cumsum()
    df_cutoff = df.loc[df['weight_cumsum'] <= four_pct_cutoff]
    return (df_cutoff['target'] == 1).sum() / (df['target'] == 1).sum()
    
def weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
    df = (pd.concat([y_true, y_pred], axis='columns')
          .sort_values('prediction', ascending=False))
    df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
    df['random'] = (df['weight'] / df['weight'].sum()).cumsum()
    total_pos = (df['target'] * df['weight']).sum()
    df['cum_pos_found'] = (df['target'] * df['weight']).cumsum()
    df['lorentz'] = df['cum_pos_found'] / total_pos
    df['gini'] = (df['lorentz'] - df['random']) * df['weight']
    return df['gini'].sum()

def normalized_weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
    y_true_pred = y_true.rename(columns={'target': 'prediction'})
    return weighted_gini(y_true, y_pred) / weighted_gini(y_true, y_true_pred)




y_pred = pd.DataFrame({'prediction': preds}).reset_index(drop=True)
y_true = pd.DataFrame({'target': y_valid}).reset_index(drop=True)


g = normalized_weighted_gini(y_true, y_pred)
d = top_four_percent_captured(y_true, y_pred)

print(g, d, 0.5 * (g + d))


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 1. Predictions from your trained model
preds = best_xgb.predict_proba(X_valid)[:, 1]

# 2. Choose threshold
threshold = 0.5  # or any value you want (tuned threshold)

# 3. Convert probabilities → binary labels
y_pred_label = (preds >= threshold).astype(int)

# 4. Compute confusion matrix
cm = confusion_matrix(y_valid, y_pred_label)

# 5. Plot
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title(f"Confusion Matrix (threshold={threshold})")
plt.show()



preds = best_xgb.predict_proba(X_valid)[:, 1]
y_true = y_valid  # 0/1 labels


import numpy as np
from sklearn.metrics import f1_score

thresholds = np.linspace(0.0, 1.0, 200)
scores = []

for t in thresholds:
    y_pred_label = (preds >= t).astype(int)
    scores.append(f1_score(y_true, y_pred_label))

best_idx = np.argmax(scores)
best_threshold = thresholds[best_idx]
best_f1 = scores[best_idx]

print("Best threshold:", best_threshold)
print("Best F1:", best_f1)



from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

precision, recall, pr_thresholds = precision_recall_curve(y_true, preds)

plt.figure(figsize=(7,5))
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve")
plt.grid()
plt.show()



from sklearn.metrics import roc_curve, auc

fpr, tpr, roc_thresholds = roc_curve(y_true, preds)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7,5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()



from sklearn.metrics import precision_score, recall_score, f1_score

precisions = []
recalls = []
f1s = []

for t in thresholds:
    y_pred_label = (preds >= t).astype(int)
    precisions.append(precision_score(y_true, y_pred_label))
    recalls.append(recall_score(y_true, y_pred_label))
    f1s.append(f1_score(y_true, y_pred_label))

plt.figure(figsize=(8,6))
plt.plot(thresholds, precisions, label="Precision")
plt.plot(thresholds, recalls, label="Recall")
plt.plot(thresholds, f1s, label="F1 Score")
plt.axvline(best_threshold, color='black', linestyle='--', label=f"Best t={best_threshold:.3f}")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Threshold Tuning Curve")
plt.legend()
plt.grid()
plt.show()



from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_pred_best = (preds >= best_threshold).astype(int)

cm = confusion_matrix(y_true, y_pred_best)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title(f"Confusion Matrix (Best threshold = {best_threshold:.3f})")
plt.show()



from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Convert probabilities to labels
y_pred_best = (preds >= best_threshold).astype(int)

acc = accuracy_score(y_true, y_pred_best)
prec = precision_score(y_true, y_pred_best)
rec = recall_score(y_true, y_pred_best)
f1 = f1_score(y_true, y_pred_best)

print(f"Best Threshold: {best_threshold:.4f}")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")



import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

thresholds = np.linspace(0.0, 1.0, 101)

precision_list = []
recall_list = []
f1_list = []

for t in thresholds:
    y_pred = (preds > t).astype(int)
    precision_list.append(precision_score(y_valid, y_pred))
    recall_list.append(recall_score(y_valid, y_pred))
    f1_list.append(f1_score(y_valid, y_pred))

plt.figure(figsize=(8, 6))
plt.plot(thresholds, precision_list, label="Precision")
plt.plot(thresholds, recall_list, label="Recall")
plt.plot(thresholds, f1_list, label="F1 Score")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Threshold Tuning Curve")
plt.legend()
plt.grid(True)
plt.show()



# test_last = (
#     test_df
#     .groupby("customer_ID")
#     .apply(lambda x: x.sort_values("S_2").iloc[-1])
#     .reset_index(drop=True)
# )



train_df['customer_ID'].unique().sum()


# -----------------------
# 1. Align test columns
# -----------------------
cols_to_keep = X_train.columns
X_test = test_last[cols_to_keep].copy()

for c in cat_features:
    X_test[c] = X_test[c].astype("category")

# -----------------------
# 2. Predict on test
# -----------------------
test_last['prediction'] = best_xgb.predict_proba(X_test)[:, 1]

# -----------------------
# 3. Merge with sample submission (correct order)
# -----------------------
submission = ss_df[['customer_ID']].merge(
    test_last[['customer_ID', 'prediction']],
    on='customer_ID',
    how='left'
)

# -----------------------
# 4. Safety
# -----------------------
submission['prediction'].fillna(0.0, inplace=True)

# -----------------------
# 5. Save file
# -----------------------
submission.to_csv("submission.csv", index=False)



train_df['customer_ID']


df = train_df.copy()
uninclude_cols = ["customer_ID", "S_2"]

# -----------------------------------
# Explicit categorical columns
# -----------------------------------
cat_cols = [
    'B_30', 'B_38', 'D_114', 'D_116', 'D_117', 
    'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68'
]

# Everything else except customer_ID, S_2, categorical = numeric
num_cols = [c for c in df.columns if c not in cat_cols and c not in uninclude_cols]

# Convert numericals safely
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# -----------------------------------
# Aggregation specs
# -----------------------------------
num_aggs = ['mean', 'std', 'min', 'max', 'last']

cat_aggs = [
    ('last', 'last'),
    ('mode', lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan),
    ('nunique', 'nunique')
]

# Build agg dict
agg_dict = {}

for col in num_cols:
    agg_dict[col] = num_aggs

for col in cat_cols:
    agg_dict[col] = cat_aggs

# -----------------------------------
# Groupby aggregation
# -----------------------------------
df_sorted = df.sort_values(["customer_ID", "S_2"])
df_agg = df_sorted.groupby("customer_ID").agg(agg_dict)



# flatten names
df_agg.columns = [f"{c[0]}_{c[1]}" for c in df_agg.columns]
df_agg.reset_index(inplace=True)

# need to make category datatype for the appropriate aggregated columns
agg_cat_cols = [
    c for c in df_agg.columns
    if any(cat in c for cat in cat_cols)
    and not c.endswith("_nunique")
]

for col in agg_cat_cols:
    df_agg[col] = df_agg[col].astype("category")

print(df_agg.shape)
print([c for c in df_agg.columns if "mode" in c])



df_agg_merged = df_agg.merge(train_labels, on = "customer_ID", how ="left")


# Train validation split
X = df_agg_merged.drop(columns=["target", "customer_ID"])
y = df_agg_merged ["target"]

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)



missing_frac = X_train.isna().mean().sort_values(ascending=False)


#This model is only to find the importance of features

base_params = {
    "objective": "binary:logistic",
    "tree_method": "hist",
    "missing": np.nan,
    "enable_categorical": True,
    "device": "cuda",

    "learning_rate": 0.05,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "auc"
}

base_model = xgb.XGBClassifier(
    **base_params,
    n_estimators=500
)

base_model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    early_stopping_rounds=50,
    verbose=True
)



importance = (
    pd.Series(base_model.get_booster().get_score(importance_type="gain"))
    .rename("gain")
)

importance = importance / importance.sum()



feature_stats = pd.DataFrame({
    "missing_frac": missing_frac,
    "gain": importance
}).fillna(0)

feature_stats.sort_values(
    ["gain", "missing_frac"],
    ascending=[False, True],
    inplace=True
)


# DROP_MISSING = 0.9
# MIN_GAIN = 0.001  # 0.1%

# features_to_keep = feature_stats[
#     ~(
#         (feature_stats["missing_frac"] > DROP_MISSING) &
#         (feature_stats["gain"] < MIN_GAIN)
#     )
# ].index.tolist()


# X_train = X_train[features_to_keep]
# X_valid = X_valid[features_to_keep]


# feature_stats must already exist
# Columns expected:
#   - missing_frac
#   - gain

feature_stats = feature_stats.copy()






from optuna.exceptions import TrialPruned
import gc, torch, tqdm

def expand_cat_features(cat_cols, all_columns):
    expanded = []
    for c in cat_cols:
        expanded += [col for col in all_columns if col.startswith(f"{c}_")]
    return expanded

from optuna.exceptions import TrialPruned

def objective(trial):

    drop_missing = trial.suggest_float("drop_missing", 0.6, 0.95)
    min_gain = trial.suggest_float("min_gain", 1e-5, 1e-2, log=True)

    # Feature selection
    selected_features = feature_stats[
        ~(
            (feature_stats["missing_frac"] > drop_missing) &
            (feature_stats["gain"] < min_gain)
        )
    ].index.tolist()

    # Expand categorical features properly
    cat_features_expanded = expand_cat_features(cat_cols, X_train.columns)

    # Force keep expanded categorical features
    selected_features = list(set(selected_features).union(cat_features_expanded))

    if len(selected_features) < 50:
        raise TrialPruned()

    X_tr = X_train[selected_features]
    X_va = X_valid[selected_features]
    

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        tree_method="hist",
        device="cuda",
        enable_categorical=True,
        learning_rate=0.05,
        max_depth=24,
        subsample=0.8,
        colsample_bytree=0.8,
        n_estimators=100,
        eval_metric="auc"
    )

    model.fit(
        X_tr, y_train,
        eval_set=[(X_va, y_valid)],
        early_stopping_rounds=20,
        verbose=True
    )

    preds = model.predict_proba(X_va)[:, 1]

    # -------------------------
    # GPU CLEANUP (CRITICAL)
    # -------------------------
    del model
    gc.collect()
    torch.cuda.empty_cache()
    y_valid_df = pd.DataFrame(y_valid, columns = ["target"])
    preds_df = pd.DataFrame(preds, columns = ["prediction"])
    return amex_metric(y_valid_df, preds_df)


import gc
import torch

# Remove references to Optuna internals
del study

# Clean Python memory
gc.collect()

# Release CUDA memory
torch.cuda.empty_cache()



import optuna, tqdm, time

study = optuna.create_study(direction="maximize")

study.optimize(
    objective,
    n_trials=10,
    show_progress_bar=True
)

best_drop_missing = study.best_params["drop_missing"]
best_min_gain = study.best_params["min_gain"]

print(best_drop_missing, best_min_gain)



# Build mapping once
cat_feature_map = {}

for c in cat_cols:
    cat_feature_map[c] = [
        col for col in X_train.columns
        if col.startswith(f"{c}_")
    ]



forced_cat_features = []

for c in cat_cols:
    forced_cat_features.extend(cat_feature_map.get(c, []))

selected_features = list(set(selected_features).union(forced_cat_features))



selected_features = [
    f for f in selected_features
    if f in X_train.columns
]



X_tr = X_train[selected_features]
X_va = X_valid[selected_features]



selected_features = feature_stats[
    ~(
        (feature_stats["missing_frac"] > best_drop_missing) &
        (feature_stats["gain"] < best_min_gain)
    )
].index.tolist()

selected_features = list(set(selected_features).union(cat_cols))

X_tr = X_train[selected_features]
X_va = X_valid[selected_features]

print("Final feature count:", len(selected_features))



# SHAP based keep/drop logic
ref_model = xgb.XGBClassifier(
    objective="binary:logistic",
    tree_method="hist",
    device="cuda",
    enable_categorical=True,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    n_estimators=300,
    eval_metric="auc"
)

ref_model.fit(
    X_train[selected_features],
    y_train,
    eval_set=[(X_valid[selected_features], y_valid)],
    early_stopping_rounds=50,
    verbose=False
)


import optuna

trial_history = []

def objective(trial):

    params = {
        "objective": "binary:logistic",
        "tree_method": "hist",
        "device": "cuda",
        "eval_metric": "auc",
        "missing": np.nan,
        "enable_categorical": True,
        "n_estimators": 500,

        # ---- Parameters to tune ----
        "max_depth": trial.suggest_int("max_depth", 8, 80, step=8),
        "min_child_weight": trial.suggest_int("min_child_weight", 10, 100, step=10),
        "learning_rate": trial.suggest_categorical("learning_rate", [0.05, 0.1]),
        "subsample": trial.suggest_categorical("subsample", [0.3, 0.4, 0.5, 0.8, 1.0]),
        "colsample_bytree": trial.suggest_categorical("colsample_bytree", [0.5, 0.8, 1.0]),
        

        # ---- Regularization tuning ----
        "lambda": trial.suggest_float("lambda", 1e-3, 10, log=True),
        "alpha": trial.suggest_float("alpha", 1e-3, 5, log=True),
        "gamma": trial.suggest_float("gamma", 0, 10),
    }

    model = xgb.XGBClassifier(**params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        early_stopping_rounds=30,
        verbose=False
    )

    # predictions
    preds = model.predict_proba(X_valid)[:, 1]

    # Compute AMEX
    y_true_df = pd.DataFrame({'target': y_valid}).reset_index(drop=True)
    y_pred_df = pd.DataFrame({'prediction': preds}).reset_index(drop=True)

    score = amex_metric(y_true_df, y_pred_df)
    
    # ----------------------------
    # SAVE TRIAL RESULTS SAFELY
    # ----------------------------
    trial_history.append({
        "params": params.copy(),              # <-- IMPORTANT FIX
        "best_iteration": model.best_iteration,
        "amex_score": score
    })
    
    return score  # Optuna will maximize the AMEX metric


# Run study
study = optuna.create_study(
    study_name="xgb_amex_aggregated",
    direction="maximize",
    storage="sqlite:///optuna_xgb.db",
    load_if_exists=True
)

study.optimize(objective, n_trials=50)  # adds 300 more

# Convert to DataFrame
trial_df = pd.DataFrame(trial_history)

# Flatten params dict into columns (optional but very useful!)
trial_df = trial_df.join(trial_df["params"].apply(pd.Series)).drop(columns=["params"])

trial_df.sort_values("amex_score", ascending=False, inplace=True)

print("Best AMEX:", study.best_trial.value)
print("Best Params:", study.best_trial.params)



best_params = {
    "objective": "binary:logistic",
    "tree_method": "hist",
    "missing": np.nan,
    "enable_categorical": True,
    "device": "cuda",

    "max_depth": 80,
    "min_child_weight": 40,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.5,

    "lambda": 0.007670118758441345,
    "alpha": 0.09157175965761555,
    "gamma": 1.028865262384579,
}



final_model = xgb.XGBClassifier(
    **best_params,
    n_estimators=500  # intentionally large
)

final_model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric=amex_metric_xgb,
    verbose=True
)



# Drop the columns with missing fraction > 0.5
cols_to_drop = df_agg.columns[df_agg.isna().mean() > 0.83].tolist()
# Create the modified dataframe
df_agg= df_agg.drop(columns=cols_to_drop)

#denoise
import warnings
warnings.filterwarnings("ignore")
df_agg = denoise_numeric(df_agg)


agg_cat_cols = [col for col in df_agg.columns if any(col.startswith(cat) for cat in cat_cols)]


df_agg_merged


for c in agg_cat_cols:
    df_agg_merged[c] = df_agg_merged[c].astype('category')


X = df_agg_merged.drop(['customer_ID', 'target'], axis=1)
y = df_agg_merged['target']


X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=42, stratify = y)


import optuna

trial_history = []

def objective(trial):

    params = {
        "objective": "binary:logistic",
        "tree_method": "hist",
        "eval_metric": "auc",
        "missing": np.nan,
        "enable_categorical": True,

        # ---- Parameters to tune ----
        "max_depth": trial.suggest_int("max_depth", 8, 48, step=8),
        "min_child_weight": trial.suggest_int("min_child_weight", 20, 50, step=10),
        "learning_rate": trial.suggest_categorical("learning_rate", [0.05, 0.1]),
        "subsample": trial.suggest_categorical("subsample", [0.5, 0.8, 1.0]),
        "colsample_bytree": trial.suggest_categorical("colsample_bytree", [0.5, 0.8, 1.0]),
        "n_estimators": trial.suggest_categorical("n_estimators", [200, 400]),

        # ---- Regularization tuning ----
        "lambda": trial.suggest_float("lambda", 1e-3, 10, log=True),
        "alpha": trial.suggest_float("alpha", 1e-3, 5, log=True),
        "gamma": trial.suggest_float("gamma", 0, 10),
    }

    model = xgb.XGBClassifier(**params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        early_stopping_rounds=30,
        verbose=False
    )

    # predictions
    preds = model.predict_proba(X_valid)[:, 1]

    # Compute AMEX
    y_true_df = pd.DataFrame({'target': y_valid}).reset_index(drop=True)
    y_pred_df = pd.DataFrame({'prediction': preds}).reset_index(drop=True)

    score = amex_metric(y_true_df, y_pred_df)
    
    # ----------------------------
    # SAVE TRIAL RESULTS SAFELY
    # ----------------------------
    trial_history.append({
        "params": params.copy(),              # <-- IMPORTANT FIX
        "best_iteration": model.best_iteration,
        "amex_score": score
    })
    
    return score  # Optuna will maximize the AMEX metric


# Run study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# Convert to DataFrame
trial_df = pd.DataFrame(trial_history)

# Flatten params dict into columns (optional but very useful!)
trial_df = trial_df.join(trial_df["params"].apply(pd.Series)).drop(columns=["params"])

trial_df.sort_values("amex_score", ascending=False, inplace=True)

print("Best AMEX:", study.best_trial.value)
print("Best Params:", study.best_trial.params)



import seaborn as sns
import matplotlib.pyplot as plt

for col in ['max_depth', 'min_child_weight', 'learning_rate', 'subsample', 
            'colsample_bytree', 'n_estimators', 'lambda', 'alpha', 'gamma']:
    
    plt.figure(figsize=(6,4))
    sns.scatterplot(data=trial_df, x=col, y='amex_score')
    plt.title(f'{col} vs AMEX')
    plt.show()







best_params = {
        "objective": "binary:logistic",
        "tree_method": "hist",
        "missing": np.nan,
        "enable_categorical": True,

        # ---- tuned params ----
        "max_depth": 48,
        "min_child_weight": 30,
        "learning_rate":0.05,
        "subsample": 1.0,
        "colsample_bytree": 0.8,
        "n_estimators": 10000,
        "lambda": 1.2899531383219924, 
        "alpha": 1.3441815317991856, 
        "gamma": 3.066108845170049,
        "eval_metric": "auc"
    }

best_xgb = xgb.XGBClassifier(**best_params)

best_xgb.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        early_stopping_rounds=150,

        verbose=True
    )


import matplotlib.pyplot as plt


import numpy as np
from sklearn.metrics import f1_score

thresholds = np.linspace(0.0, 1.0, 200)
scores = []

for t in thresholds:
    y_pred_label = (preds >= t).astype(int)
    scores.append(f1_score(y_true, y_pred_label))

best_idx = np.argmax(scores)
best_threshold = thresholds[best_idx]
best_f1 = scores[best_idx]

print("Best threshold:", best_threshold)
print("Best F1:", best_f1)



from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

precision, recall, pr_thresholds = precision_recall_curve(y_true, preds)

plt.figure(figsize=(7,5))
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve")
plt.grid()
plt.show()


from sklearn.metrics import roc_curve, auc

fpr, tpr, roc_thresholds = roc_curve(y_true, preds)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7,5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()


from sklearn.metrics import precision_score, recall_score, f1_score

precisions = []
recalls = []
f1s = []

for t in thresholds:
    y_pred_label = (preds >= t).astype(int)
    precisions.append(precision_score(y_true, y_pred_label))
    recalls.append(recall_score(y_true, y_pred_label))
    f1s.append(f1_score(y_true, y_pred_label))

plt.figure(figsize=(8,6))
plt.plot(thresholds, precisions, label="Precision")
plt.plot(thresholds, recalls, label="Recall")
plt.plot(thresholds, f1s, label="F1 Score")
plt.axvline(best_threshold, color='black', linestyle='--', label=f"Best t={best_threshold:.3f}")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Threshold Tuning Curve")
plt.legend()
plt.grid()
plt.show()


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_pred_best = (preds >= best_threshold).astype(int)

cm = confusion_matrix(y_true, y_pred_best)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title(f"Confusion Matrix (Best threshold = {best_threshold:.3f})")
plt.show()



from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Convert probabilities to labels
y_pred_best = (preds >= best_threshold).astype(int)

acc = accuracy_score(y_true, y_pred_best)
prec = precision_score(y_true, y_pred_best)
rec = recall_score(y_true, y_pred_best)
f1 = f1_score(y_true, y_pred_best)

print(f"Best Threshold: {best_threshold:.4f}")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")





y_pred = pd.DataFrame({'prediction': preds}).reset_index(drop=True)
y_true = pd.DataFrame({'target': y_valid}).reset_index(drop=True)


g = normalized_weighted_gini(y_true, y_pred)
d = top_four_percent_captured(y_true, y_pred)

print(g, d, 0.5 * (g + d))

