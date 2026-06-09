import os
import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import cohen_kappa_score
import lightgbm as lgb

from lightgbm import LGBMRegressor
import lightgbm as lgb
import optuna
import numpy as np

import optuna


BASE = "/kaggle/input/inputs-txt-and-img-2"

train_text = pd.read_parquet(f"{BASE}/train_text.parquet")
train_img  = pd.read_parquet(f"{BASE}/train_img.parquet")

test_text = pd.read_parquet(f"{BASE}/test_text.parquet")
test_img  = pd.read_parquet(f"{BASE}/test_img.parquet")


folder =  "/kaggle/input/petfinder-adoption-prediction"
train = pd.read_csv(f"{folder}/train/train.csv")
test = pd.read_csv(f"{folder}/test/test.csv")


# Aseguramos que TODO esté indexado por PetID
if "PetID" in train.columns:
    train = train.set_index("PetID")
if "PetID" in test.columns:
    test = test.set_index("PetID")

for df in [train_text, train_img, test_text, test_img]:
    if "PetID" in df.columns:
        df.set_index("PetID", inplace=True)

# Unificamos: base + texto + imagen
train_full = (
    train
    .join(train_text, how="left")
    .join(train_img, how="left")
)

test_full = (
    test
    .join(test_text, how="left")
    .join(test_img, how="left")
)

print("train_full:", train_full.shape)
print("test_full :", test_full.shape)



target_col = "AdoptionSpeed"

# Por seguridad, si AdoptionSpeed quedó como índice (no debería), lo recuperamos
if target_col not in train_full.columns and target_col in train.columns:
    train_full[target_col] = train[target_col]

# Nos quedamos sólo con columnas numéricas para el modelo
numeric_cols = train_full.select_dtypes(include=[np.number]).columns.tolist()

# Removemos el target de las features
if target_col in numeric_cols:
    numeric_cols.remove(target_col)

X = train_full[numeric_cols]
y = train_full[target_col].astype(int)

X_test = test_full[numeric_cols]

print("X shape     :", X.shape)
print("X_test shape:", X_test.shape)
print("Target dist :")
print(y.value_counts())



# Se suprimen warnings para evitar que saturen la salida del notebook.
import warnings
warnings.filterwarnings("ignore")

callbacks = [
    lgb.early_stopping(100),
    lgb.log_evaluation(0)   # silencioso total
]



from sklearn.model_selection import StratifiedKFold

N_FOLDS = 5
RANDOM_STATE = 42

kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)



# Parametro que nos permite optimizar el optuna 
# o utilizar los hiperparametros optimizados anteriormente
ejectuar_optuna = 0



if ejectuar_optuna == 1: 
    def objective_reg(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 300, 1500),
    
            # fijos
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbosity": -1,
        }
    
        oof_pred_reg = np.zeros(len(X), dtype=float)
        fold_scores = []
    
        for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y), 1):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
    
            model = LGBMRegressor(**params)
    
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric="rmse",
                callbacks=[
                    lgb.early_stopping(100),
                    lgb.log_evaluation(0)
                ],
            )
    
            y_val_pred = model.predict(X_val)
            oof_pred_reg[val_idx] = y_val_pred
    
            # redondeo simple 0–4 para evaluar QWK durante Optuna
            y_val_pred_round = np.clip(np.rint(y_val_pred), 0, 4).astype(int)
            score = cohen_kappa_score(y_val, y_val_pred_round, weights="quadratic")
            fold_scores.append(score)
    
        return float(np.mean(fold_scores))
    
    
    def print_best_callback(study, trial):
        print(f"\n[Trial {trial.number}] value: {trial.value:.5f}")
        print(f"Best so far: {study.best_value:.5f}")
        print(f"Best params: {study.best_params}")
    
    
    study_reg = optuna.create_study(direction="maximize", study_name="lgbm_reg_petfinder")
    study_reg.optimize(
        objective_reg,
        n_trials=100,
        show_progress_bar=True,
        callbacks=[print_best_callback]
    )
    
    print("\n=== OPTUNA REGRESIÓN TERMINADO ===")
    print("Best QWK (redondeando):", study_reg.best_value)
    print("Best params (reg):")
    for k, v in study_reg.best_params.items():
        print(f"  {k}: {v}")
    
    best_params_reg = study_reg.best_params.copy()
    best_params_reg.update({
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": -1,
    })
else:
    best_params_reg = {
    "learning_rate": 0.017016522979106517,
    "num_leaves": 76,
    "max_depth": 8,
    "min_child_samples": 95,
    "subsample": 0.8512229857507079,
    "colsample_bytree": 0.8186904774386399,
    "reg_alpha": 0.6828936792548929,
    "reg_lambda": 2.2380795158933614e-07,
    "n_estimators": 1328,

    # Ajustes adicionales necesarios
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbosity": -1,
}



oof_pred_reg = np.zeros(len(X), dtype=float)
test_pred_reg = np.zeros(len(X_test), dtype=float)

for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y), 1):
    print(f"Fold {fold}/{N_FOLDS}")

    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    model = LGBMRegressor(**best_params_reg)

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[
            lgb.early_stopping(100),
            lgb.log_evaluation(0)
        ],
    )

    # OOF continuos
    y_val_pred = model.predict(X_val)
    oof_pred_reg[val_idx] = y_val_pred

    # Predicciones sobre test (promedio de folds)
    test_pred_reg += model.predict(X_test) / N_FOLDS

# QWK con redondeo simple (antes de optimizar cortes, solo para ver)
oof_round = np.clip(np.rint(oof_pred_reg), 0, 4).astype(int)
qwk_round = cohen_kappa_score(y, oof_round, weights="quadratic")
print(f"\nQWK OOF redondeando sin cortes óptimos: {qwk_round:.5f}")



def optimize_cuts(oof_pred_reg, y_true):
    def objective_cuts(trial):
        # 4 cortes dentro del rango [0, 4]
        t1 = trial.suggest_float("t1", 0.0, 4.0)
        t2 = trial.suggest_float("t2", 0.0, 4.0)
        t3 = trial.suggest_float("t3", 0.0, 4.0)
        t4 = trial.suggest_float("t4", 0.0, 4.0)

        cuts = sorted([t1, t2, t3, t4])

        # aplicar cortes
        y_pred = np.digitize(oof_pred_reg, cuts)
        y_pred = np.clip(y_pred, 0, 4)

        return cohen_kappa_score(y_true, y_pred, weights="quadratic")

    study_cuts = optuna.create_study(direction="maximize", study_name="cuts_qwk")
    study_cuts.optimize(objective_cuts, n_trials=200, show_progress_bar=True)

    return study_cuts.best_params, study_cuts.best_value


best_cuts_params, best_cuts_qwk = optimize_cuts(oof_pred_reg, y)
print("\n=== CORTES ÓPTIMOS ENCONTRADOS ===")
print("Best QWK con cortes:", best_cuts_qwk)
print("Cortes (sin ordenar):", best_cuts_params)

# ordenamos los cortes en una lista
cuts = sorted([best_cuts_params["t1"],
               best_cuts_params["t2"],
               best_cuts_params["t3"],
               best_cuts_params["t4"]])

print("Cortes ordenados:", cuts)



# aplicar cortes a OOF
oof_pred_final = np.digitize(oof_pred_reg, cuts)
oof_pred_final = np.clip(oof_pred_final, 0, 4)

final_qwk = cohen_kappa_score(y, oof_pred_final, weights="quadratic")
print(f"\nQWK OOF final con cortes óptimos: {final_qwk:.5f}")

# aplicar cortes a test
test_pred_final = np.digitize(test_pred_reg, cuts)
test_pred_final = np.clip(test_pred_final, 0, 4).astype(int)

submission_reg = pd.DataFrame({
    "PetID": test_full.index,
    "AdoptionSpeed": test_pred_final,
})

submission_reg.to_csv("submission.csv", index=False)
print("\nArchivo 'submission.csv' generado.")
print(submission_reg.head())





























