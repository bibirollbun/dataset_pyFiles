"""import kagglehub

# Download latest version
path = kagglehub.dataset_download("ianktoo/simulated-roads-accident-data")

print("Path to dataset files:", path)"""


import numpy as np 
import pandas as pd
import scipy
from sklearn.model_selection import KFold, StratifiedKFold, RandomizedSearchCV
from sklearn.base import BaseEstimator, RegressorMixin, clone
from scipy.stats import uniform, randint

from xgboost import XGBRegressor, DMatrix, train as xgb_train
from lightgbm import LGBMRegressor
from lightgbm import early_stopping as lgb_early_stopping, log_evaluation as lgb_log_evaluation
from catboost import CatBoostRegressor

from sklearn.metrics import mean_squared_error

from datetime import datetime


from xgboost import DMatrix, train as xgb_train

import warnings
warnings.filterwarnings('ignore')


#-----------------------------------------------------------------------------------------------
def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

"""
start_time = timer(None) # timing starts from this point for "start_time" variable
timer(start_time) # timing ends here for "start_time" variable"""

#-----------------------------------------------------------------------------------------------



# df_train
df_train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')

# df_test
df_test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
df_test['accident_risk'] = 0.5

# df_original
original = pd.read_csv('/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_2k.csv')
original_1 = pd.read_csv('/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_10k.csv')
original_2 = pd.read_csv('/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_100k.csv')
orig = pd.concat([original, original_1, original_2])
orig['id'] = np.arange(len(orig))+df_test['id'].max()+1
orig = orig[ df_test.columns ] 

#df_combine
combine = pd.concat([df_train, df_test , orig]) 


FEATURES = list( orig.columns[1:-1] )
TARGET = orig.columns[-1]
print(f"Features: {FEATURES}, Target: '{TARGET}'")
print('\n')

def f(X):
    return \
    0.3 * X["curvature"] + \
    0.2 * (X["lighting"] == "night").astype(int) + \
    0.1 * (X["weather"] != "clear").astype(int) + \
    0.2 * (X["speed_limit"] >= 60).astype(int) + \
    0.1 * (X["num_reported_accidents"] > 2).astype(int)

def clip(f):
    def clip_f(X):
        sigma = 0.05
        mu = f(X)
        a, b = -mu/sigma, (1-mu)/sigma
        Phi_a, Phi_b = scipy.stats.norm.cdf(a), scipy.stats.norm.cdf(b)
        phi_a, phi_b = scipy.stats.norm.pdf(a), scipy.stats.norm.pdf(b)
        return mu*(Phi_b-Phi_a)+sigma*(phi_a-phi_b)+1-Phi_b
    return clip_f

z = clip(f)(combine)
combine["y"] = z.values
FEATURES.append("y")

CATS = []
NUMS = []
for c in FEATURES:
    if combine[c].dtype == 'object':
        CATS.append(c)
    else:
        NUMS.append(c)

print("CATS:", CATS)
print("NUMS:", NUMS)
print('\n')
for c in CATS:
    combine[c],_ = combine[c].factorize()
    combine[c] = combine[c].astype('int32')
    combine[c] = combine[c].astype('int32')

df_train = combine.iloc[:len(df_train)]
df_test = combine.iloc[len(df_train):len(df_train)+len(df_test)]
orig = combine.iloc[-len(orig):]

TE = []
for c in FEATURES:
    tmp = orig.groupby(c)[TARGET].mean()
    n = f"TE_{c}"
    print(f"{n}, ",end="")
    tmp.name = n
    df_train = df_train.merge(tmp, on=c, how='left')
    df_test = df_test.merge(tmp, on=c, how='left')
    TE.append(n)
    
print('\n')
print(f"Train shape: {df_train.shape}, Test shape: {df_test.shape}, Original data shape: {orig.shape}")

X = df_train[FEATURES + TE].copy()
y = df_train[TARGET].copy()


"""
class TargetEncodedModel(BaseEstimator, RegressorMixin):
    def __init__(self, base_model, features, target_col, early_stopping_rounds=None, verbose=False):
        self.base_model = base_model
        self.features = features
        self.target_col = target_col
        self.te_maps_ = {}
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose

    def fit(self, X, y):
        X = X.copy()
        X[self.target_col] = y

        # ==== Crear target encoding ====
        self.te_maps_ = {}
        for c in self.features:
            tmp = X.groupby(c)[self.target_col].mean()
            self.te_maps_[c] = tmp
            X[f"TE_{c}"] = X[c].map(tmp)

        # ==== Clonar el modelo base ====
        self.model_ = clone(self.base_model)

        # ==== Preparar lista de features con TE ====
        all_features = self.features + [f"TE_{c}" for c in self.features]

        # ==== Detectar si el modelo base soporta early stopping ====
        fit_params = {}
        if self.early_stopping_rounds is not None:
            if "early_stopping_rounds" in self.model_.get_params().keys():
                fit_params["early_stopping_rounds"] = self.early_stopping_rounds
                fit_params["eval_set"] = [(X[all_features], y)]
                fit_params["verbose"] = self.verbose
            elif "early_stopping" in self.model_.get_params().keys():  # por si es LightGBM
                fit_params["early_stopping"] = self.early_stopping_rounds
                fit_params["eval_set"] = [(X[all_features], y)]
                fit_params["verbose"] = self.verbose

        # ==== Entrenar el modelo ====
        self.model_.fit(X[all_features], y, **fit_params)
        return self

    def transform_with_te(self, X):
        X = X.copy()
        for c in self.features:
            X[f"TE_{c}"] = X[c].map(self.te_maps_[c])
        return X

    def predict(self, X):
        X_te = self.transform_with_te(X)
        all_features = self.features + [f"TE_{c}" for c in self.features]
        return self.model_.predict(X_te[all_features])

#------------------------------------------------------------------------------------------------------------------------------------
search_spaces = {
    "XGBoost": {
        "model": XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            eval_metric="rmse",
            random_state=42
        ),
        "params": {
            # Foco: tu modelo usa 0.01 y 6 â†’ mantenemos cerca
            "base_model__n_estimators": randint(800, 1500),
            "base_model__learning_rate": uniform(0.008, 0.007),  # 0.008â€“0.015
            "base_model__max_depth": randint(4, 8),
            "base_model__subsample": uniform(0.8, 0.15),         # 0.8â€“0.95
            "base_model__colsample_bytree": uniform(0.5, 0.2),   # 0.5â€“0.7
            "base_model__min_child_weight": randint(1, 6),
        },
    },

    "LightGBM": {
        "model": LGBMRegressor(objective="regression", random_state=42),
        "params": {
            # Foco: 1000 trees, lr=0.05, depth=7, leaves=31
            "base_model__n_estimators": randint(800, 1500),
            "base_model__learning_rate": uniform(0.03, 0.03),    # 0.03â€“0.06
            "base_model__max_depth": randint(5, 9),
            "base_model__num_leaves": randint(25, 50),
            "base_model__min_child_samples": randint(10, 40),
            "base_model__subsample": uniform(0.7, 0.25),         # 0.7â€“0.95
            "base_model__colsample_bytree": uniform(0.7, 0.25),  # 0.7â€“0.95
            "base_model__reg_alpha": uniform(0.0, 0.2),
            "base_model__reg_lambda": uniform(0.0, 0.2),
        },
    },

    "CatBoost": {
        "model": CatBoostRegressor(
            loss_function="RMSE",
            task_type="CPU",   # si tienes GPU, cambia a "GPU"
            verbose=0,
            random_seed=42
        ),
        "params": {
            # Foco: 1000, lr=0.05, depth=7, l2=3
            "base_model__iterations": randint(800, 1200),
            "base_model__learning_rate": uniform(0.03, 0.03),    # 0.03â€“0.06
            "base_model__depth": randint(5, 9),
            "base_model__l2_leaf_reg": uniform(2, 2),            # 2â€“4
            "base_model__bagging_temperature": uniform(0.2, 0.6),
            "base_model__subsample": uniform(0.8, 0.15),         # 0.8â€“0.95
        },
    },
}

#------------------------------------------------------------------------------------------------------------------------------------
start_time = timer(None) # timing starts from this point for "start_time" variable

results = []

for name, conf in search_spaces.items():
    print(f"\nðŸš€ Entrenando modelo: {name} ...")

    model_te = TargetEncodedModel(
        base_model=conf["model"],
        features=FEATURES,
        target_col=TARGET
    )

    search = RandomizedSearchCV(
        estimator=model_te,
        param_distributions=conf["params"],
        n_iter=param_comb,
        scoring="neg_root_mean_squared_error",
        cv=skf.split(X,y_binned),
        verbose=2,
        n_jobs=-1,
        random_state=42
    )

    search.fit(X, y)

    rmse = -search.best_score_
    params = search.best_params_

    print(f"âœ… {name} - Best RMSE: {rmse:.5f}")
    print(f"ðŸ”§ Best Params: {params}\n")

    results.append({
        "model": name,
        "best_rmse": rmse,
        "best_params": params
    })
    
timer(start_time) # timing ends here for "start_time" variable

"""


models = {
    "XGBoost": XGBRegressor(
        n_estimators=1298,
        learning_rate=0.010764170627228987,
        max_depth=7,
        min_child_weight=3,
        subsample=0.9060286015771426,
        colsample_bytree=0.5846802961412739,
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",   
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbosity=0
    ),

    "LightGBM": LGBMRegressor(
        n_estimators=1254,
        learning_rate=0.03692681476866447,
        max_depth=8,
        num_leaves=36,
        min_child_samples=24,
        subsample=0.9273301005196954,
        colsample_bytree=0.7039915630550535,
        reg_alpha=0.09903538202225404,
        reg_lambda=0.006877704223043679,
        objective="regression",
        random_state=42,
        verbose=-1
    ),

    "CatBoost": CatBoostRegressor(
        iterations=1103,
        learning_rate=0.047113319232161985,
        depth=7,
        l2_leaf_reg=2.653081537611671,
        bagging_temperature=0.7559953194762765,
        subsample=0.8781251390038736,
        loss_function="RMSE",
        random_seed=42,
        verbose=0,
        task_type="CPU"  
    )
}


"""# ====================================================
# Cross-validation setup
# ====================================================
FOLDS = 10
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

# EstratificaciÃ³n basada en el target binned
y_binned = pd.qcut(y, q=10, labels=False, duplicates='drop')

# ====================================================
# Entrenamiento con folds
# ====================================================
results = {}
oof_predictions = {}
test_predictions = {}

start_time = timer(None)

for model_name, model in models.items():
    print(f"\n{'='*70}")
    print(f" Entrenando modelo: {model_name}")
    print(f"{'='*70}")

    oof_preds = np.zeros(len(df_train))
    test_preds = np.zeros(len(df_test))
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_binned), 1):
        print(f"\n{'#'*25}\n### Fold {fold} - {model_name} ###\n{'#'*25}")

        # ====================================================
        # Dividir datos
        # ====================================================
        X_train = df_train.iloc[train_idx][FEATURES + TE].copy()
        y_train = df_train.iloc[train_idx][TARGET] - df_train.iloc[train_idx]['y']

        X_valid = df_train.iloc[val_idx][FEATURES + TE].copy()
        y_valid = df_train.iloc[val_idx][TARGET] - df_train.iloc[val_idx]['y']
        y_valid2 = df_train.iloc[val_idx]['y'].values

        X_test = df_test[FEATURES + TE].copy()
        y_test2 = df_test['y'].values

        # ====================================================
        # Entrenamiento por modelo
        # ====================================================
        if model_name == "XGBoost":
            dtrain = DMatrix(X_train, label=y_train, enable_categorical=True)
            dval = DMatrix(X_valid, label=y_valid, enable_categorical=True)
            dtest = DMatrix(X_test, enable_categorical=True)

            params_xgb = model.get_params()
            booster = xgb_train(
                params=params_xgb,
                dtrain=dtrain,
                num_boost_round=100_000,
                evals=[(dtrain, "train"), (dval, "valid")],
                early_stopping_rounds=200,
                verbose_eval=200
            )

            val_pred = booster.predict(dval, iteration_range=(0, booster.best_iteration + 1))
            test_pred = booster.predict(dtest, iteration_range=(0, booster.best_iteration + 1))

        else:
            # ====================================================
            # LightGBM / CatBoost con early stopping adaptado
            # ====================================================
            fit_params = {
                "eval_set": [(X_valid, y_valid)],
            }

            # --- LightGBM moderno usa callbacks
            if "boosting_type" in model.get_params().keys():  
                fit_params["callbacks"] = [
                    lgb_early_stopping(200),
                    lgb_log_evaluation(200)
                ]

            # --- CatBoost o versiones antiguas de LightGBM
            elif "early_stopping_rounds" in model.fit.__code__.co_varnames:
                fit_params["early_stopping_rounds"] = 200
                fit_params["verbose"] = False

            # Entrenar modelo
            model.fit(X_train, y_train, **fit_params)

            val_pred = model.predict(X_valid)
            test_pred = model.predict(X_test)

        # ====================================================
        # Predicciones ajustadas
        # ====================================================
        oof_preds[val_idx] = val_pred + y_valid2
        test_preds += (test_pred + y_test2) / FOLDS

        # ====================================================
        # MÃ©trica
        # ====================================================
        fold_rmse = np.sqrt(mean_squared_error(df_train.iloc[val_idx][TARGET], oof_preds[val_idx]))
        fold_scores.append(fold_rmse)
        print(f"Fold {fold} RMSE: {fold_rmse:.6f}")

    # ====================================================
    # Resultados globales
    # ====================================================
    oof_rmse = np.sqrt(mean_squared_error(df_train[TARGET], oof_preds))
    print(f"\n>>> {model_name} OOF RMSE: {oof_rmse:.6f} (+/- {np.std(fold_scores):.6f})")

    results[model_name] = {
        "oof_rmse": oof_rmse,
        "fold_scores": fold_scores,
        "std": np.std(fold_scores)
    }
    oof_predictions[model_name] = oof_preds
    test_predictions[model_name] = test_preds

    timer(start_time)  # tiempo por modelo

timer(start_time)  # tiempo total
"""


"""# ====================================================
# Summary de resultados
# ====================================================
results_df = pd.DataFrame(results).T
results_df = results_df.sort_values('oof_rmse')
print("\n" + results_df.to_string())

# ====================================================
# Ensemble
# ====================================================
print("\n" + "="*80)
print("CREATING ENSEMBLE")
print("="*80)

# Weighted average inversamente proporcional al RMSE
weights = 1 / results_df['oof_rmse'].values
weights = weights / weights.sum()

print("\n Ensemble Weights:")
for model, weight in zip(results_df.index, weights):
    print(f"   {model:15s}: {weight:.4f}")

# Ensemble predictions
ensemble_oof = np.zeros(len(df_train))
ensemble_test = np.zeros(len(df_test))

for model, weight in zip(results_df.index, weights):
    ensemble_oof += oof_predictions[model] * weight
    ensemble_test += test_predictions[model] * weight

# RMSE del ensemble
ensemble_rmse = np.sqrt(mean_squared_error(df_train[TARGET], ensemble_oof))
print(f"\n Ensemble OOF RMSE: {ensemble_rmse:.6f}")

# Improvement over best model
best_rmse = results_df['oof_rmse'].iloc[0]
improvement = (best_rmse - ensemble_rmse) / best_rmse * 100
print(f" Improvement over best single model: {improvement:.2f}%")

# ====================================================
# (Opcional) Guardar predicciones en CSV
# ====================================================
preds_df = pd.DataFrame({
    "id": df_test.index,
    **{f"pred_{m}": test_predictions[m] for m in test_predictions.keys()},
    "ensemble": ensemble_test
})
preds_df.to_csv("all_model_predictions.csv", index=False)
print("\nâœ… Archivo 'all_model_predictions.csv' guardado correctamente.")
"""


"""# ====================================================
# GUARDAR PREDICCIONES INDIVIDUALES Y ENSEMBLE FINAL
# ====================================================

# === Guardar predicciones individuales de cada modelo ===
for model_name, preds in test_predictions.items():
    submission = pd.DataFrame({
        "id": df_test['id'],            # usa el mismo Ã­ndice de tu test
        "accident_risk": preds
    })
    filename = f"submission_{model_name.lower()}.csv"
    submission.to_csv(filename, index=False)
    print(f"âœ… Archivo '{filename}' creado correctamente.")

# === Guardar tambiÃ©n el ensemble final ===
ensemble_submission = pd.DataFrame({
    "id": df_test['id'],
    "accident_risk": ensemble_test
})
ensemble_submission.to_csv("submission_ensemble.csv", index=False)
print("âœ… Archivo 'submission_ensemble.csv' creado correctamente.")

# ====================================================
# GUARDAR RESUMEN DE DESEMPEÃ‘O (RMSE DE CADA MODELO)
# ====================================================

results_summary = results_df.copy()
results_summary["weight_used_in_ensemble"] = weights
results_summary.loc["ensemble"] = {
    "oof_rmse": ensemble_rmse,
    "weight_used_in_ensemble": np.nan
}

results_summary.to_csv("model_performance_summary.csv", index=True)
print("âœ… Archivo 'model_performance_summary.csv' con mÃ©tricas y pesos guardado correctamente.")

results_summary.head()"""

