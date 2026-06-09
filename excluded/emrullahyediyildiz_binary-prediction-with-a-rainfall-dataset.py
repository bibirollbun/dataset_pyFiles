import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Trainings- und Testdaten laden
train_df = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print(train_df.head())


train_df.head()


train_df.info()


test_df.info()


train_df.describe()


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(5,4))
sns.countplot(x="rainfall", data=train_df, palette="coolwarm")
plt.title("Klassenverteilung: Regen vs. Kein Regen")
plt.xlabel("Rainfall")
plt.ylabel("Anzahl")
plt.show()



plt.figure(figsize=(7,5))
sns.kdeplot(data=train_df, x="temparature", hue="rainfall", fill=True, common_norm=False, palette="coolwarm")
plt.title("Temperatur-Verteilung bei Regen vs. Kein Regen")
plt.show()



plt.figure(figsize=(10,8))
corr = train_df.corr()
sns.heatmap(corr, annot=False, cmap="Blues", cbar=True)
plt.title("Korrelationsmatrix")
plt.show()





# Features und Ziel trennen
X = train_df.drop(columns=["rainfall", "id", "day"])
y = train_df["rainfall"]

# Random Forest Modell
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X, y)

# Feature Importances extrahieren
importances = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

# Top 10 Features
top10 = importances.head(10)

# Plot
plt.figure(figsize=(8,5))
sns.barplot(data=top10, x="Importance", y="Feature", palette="Blues_r")
plt.title("Top 10 Feature Importances (Random Forest)")
plt.show()



# Features und Ziel
X = train_df.drop(columns=["rainfall", "id", "day"])
y = train_df["rainfall"]


from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import time

def algo_test(X, y):
    models = {
        "LogisticRegression": LogisticRegression(max_iter=3000, solver="lbfgs"),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42, n_estimators=200),
        "LightGBM": LGBMClassifier(random_state=42, n_estimators=200)
    }
    
    results = []
    
    for name, model in models.items():
        start = time.time()
        
        # Stratified 
        oof_pred = cross_val_predict(model, X, y, cv=5, method="predict_proba")[:,1]
        oof_class = (oof_pred >= 0.5).astype(int)
        
        roc  = roc_auc_score(y, oof_pred)
        acc  = accuracy_score(y, oof_class)
        f1   = f1_score(y, oof_class)
        rmse = mean_squared_error(y, oof_pred) ** 0.5
        mae  = mean_absolute_error(y, oof_pred)
        
        results.append({
            "Model": name,
            "ROC AUC": roc,
            "Accuracy": acc,
            "F1": f1,
            "RMSE": rmse,
            "MAE": mae,
            "Time (s)": time.time() - start
        })
    
    df_results = pd.DataFrame(results).sort_values(by="ROC AUC", ascending=False)
    return df_results.reset_index(drop=True)


# Features und Ziel
X = train_df.drop(columns=["rainfall", "id", "day"])
y = train_df["rainfall"]

# Modellvergleich
df_results = algo_test(X, y)
df_results



# Optional: Optuna installieren (in Kaggle i. d. R. schon da)
# !pip -q install optuna

import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np
import warnings
optuna.logging.set_verbosity(optuna.logging.WARNING)


# Features/Ziel
X = train_df.drop(columns=["rainfall", "id", "day"])
y = train_df["rainfall"].astype(int)

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": 42,
        "eval_metric": "logloss",
        "tree_method": "hist",
        "n_jobs": -1,
        "early_stopping_rounds": 100   # ğŸ‘ˆ hier im Konstruktor!
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(X))
    for tr_idx, va_idx in skf.split(X, y):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        model = XGBClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False   # kein Warnings/Infos Spam
        )
        oof[va_idx] = model.predict_proba(X_va)[:, 1]

    return roc_auc_score(y, oof)


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=40, show_progress_bar=False)

best_params = study.best_params
best_auc = study.best_value
print("Best AUC:", round(best_auc, 5))
best_params



from xgboost import XGBClassifier
import pandas as pd

# Daten vorbereiten
X = train_df.drop(columns=["rainfall", "id", "day"])
y = train_df["rainfall"].astype(int)

best_params = {
    "n_estimators": 1123,
    "learning_rate": 0.0345079084789159286,
    "max_depth": 2,
    "subsample": 0.6110257230025689,
    "colsample_bytree": 0.89334087038330878,
    "min_child_weight": 8.1485911494943497,
    "reg_alpha": 0.000270701118246441534,
    "reg_lambda": 0.00340698030394567493,
}

final_params = {
    **best_params,
    "random_state": 42,
    "eval_metric": "logloss",
    "tree_method": "hist",
    "n_jobs": -1,
}

final_model = XGBClassifier(**final_params)
final_model.fit(X, y)



# Testdaten & Sample Submission laden (Competition-Pfad in Kaggle)
test_df  = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
sub_df   = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")

X_test = test_df.drop(columns=["id", "day"])
pred_proba = final_model.predict_proba(X_test)[:, 1]

sub_df["rainfall"] = pred_proba
sub_df.to_csv("submission.csv", index=False)
print("âœ… submission.csv gespeichert")
sub_df.head()



from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

X_tr, X_va, y_tr, y_va = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

val_model = XGBClassifier(**final_params)
val_model.fit(X_tr, y_tr)

va_proba = val_model.predict_proba(X_va)[:, 1]
va_pred  = (va_proba >= 0.5).astype(int)

cm = confusion_matrix(y_va, va_pred)
print(classification_report(y_va, va_pred, digits=3))

plt.figure(figsize=(4.5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Rain (0)", "Rain (1)"],
            yticklabels=["No Rain (0)", "Rain (1)"])
plt.title("Confusion Matrix (Threshold = 0.50)")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.show()



from sklearn.metrics import f1_score
best_t, best_f1 = 0.5, -1
for t in np.linspace(0.1, 0.9, 33):
    f1 = f1_score(y_va, (va_proba >= t).astype(int))
    if f1 > best_f1:
        best_f1, best_t = f1, t
print(f"Best F1 = {best_f1:.4f} @ threshold = {best_t:.2f}")



final_model.save_model("xgb_rain_model.json")





