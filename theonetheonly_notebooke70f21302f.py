# === Step 1: Install & Imports ===
# !pip install -q gplearn

import pandas as pd
import numpy as np
from gplearn.genetic import SymbolicTransformer
from gplearn.functions import make_function
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb

# Square function
def _square(x):
    return np.power(x, 2)
square = make_function(function=_square, name='square', arity=1)

# Cube function
def _cube(x):
    return np.power(x, 3)
cube = make_function(function=_cube, name='cube', arity=1)

# Negation
def _neg(x):
    return -x
neg = make_function(function=_neg, name='neg', arity=1)

# === Step 2: Globals and Pipeline ===
le = LabelEncoder()
gp_transformer = None
scaler = None
SHAP_THRESHOLD = 0.001
filtered_programs = []

# === Step 3: Custom Symbolic Feature Filter ===
def transform_with_programs(X_raw, programs):
    features = []
    for prog in programs:
        features.append(prog.execute(X_raw))
    return np.column_stack(features) if features else np.empty((X_raw.shape[0], 0))

# === Step 4: Data Preparation ===
def prepare_data(df, fit=False):
    global gp_transformer, scaler, filtered_programs

    df = df.copy()
    df["Sex"] = le.fit_transform(df["Sex"]) if fit else le.transform(df["Sex"])
    X_raw = df.drop(columns=["Calories", "id"], errors="ignore").astype(float)

    if fit:
        X_sample = X_raw.sample(n=10000, random_state=42)
        y_sample = np.log1p(df.loc[X_sample.index, "Calories"])

        gp_transformer = SymbolicTransformer(
            generations=50,
            population_size=2000,
            hall_of_fame=200,
            n_components=50,
            function_set=('add', 'sub', 'mul', 'div', 'log', 'sqrt', 'abs', 'max', 'min', square, cube, neg),
            parsimony_coefficient=0.005,
            max_samples=0.3,
            verbose=1,
            random_state=42,
            n_jobs=-1
        )
        gp_transformer.fit(X_sample, y_sample)

        X_sym = gp_transformer.transform(X_raw)
        X_all = np.hstack([X_raw.values, X_sym])

        from shap import Explainer
        sample_idx = np.random.choice(len(X_raw), size=1000, replace=False)
        X_train_sample = X_all[sample_idx]
        y_train_sample = np.log1p(df.loc[sample_idx, "Calories"])
        model_sample = xgb.XGBRegressor(tree_method="hist", device="cuda").fit(X_train_sample, y_train_sample)
        shap_values = Explainer(model_sample, X_train_sample)(X_train_sample)

        raw_feature_len = X_raw.shape[1]
        shap_importance = np.abs(shap_values.values).mean(axis=0)[raw_feature_len:]
        all_programs = gp_transformer._best_programs
        filtered_programs = [p for p, s in zip(all_programs, shap_importance) if s > SHAP_THRESHOLD]

    X_sym_filtered = transform_with_programs(X_raw.values, filtered_programs)
    X_full = np.hstack([X_raw.values, X_sym_filtered])

    if fit:
        scaler = StandardScaler()
        return scaler.fit_transform(X_full)
    else:
        return scaler.transform(X_full)

# === Step 5: Load and Prepare Data ===
train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
y = np.log1p(train["Calories"].clip(lower=0))

X = prepare_data(train, fit=True)
X_test = prepare_data(test, fit=False)

# === Step 6: Train/Validation Split ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# === Step 7: Train Base Model ===
model = xgb.XGBRegressor(
    n_estimators=1500,
    learning_rate=0.03,
    max_depth=10,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    device="cuda",
    eval_metric='rmsle',
    early_stopping_rounds=20,
    random_state=42,
    verbosity=1
)

model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)


from sklearn.linear_model import Ridge

# === Step 8: Train Residual Model with Ridge ===
y_pred_train = model.predict(X_train)
y_pred_val = model.predict(X_val)
residuals = y_train - y_pred_train

X_train_with_pred = np.column_stack([X_train, y_pred_train])
X_val_with_pred = np.column_stack([X_val, y_pred_val])
X_test_with_pred = np.column_stack([X_test, model.predict(X_test)])

residual_model = Ridge(alpha=0.1, solver="saga", max_iter=10000, tol=1e-4)
residual_model.fit(X_train_with_pred, residuals)

# === Step 9: Evaluation ===
y_pred_combined_val = y_pred_val + residual_model.predict(X_val_with_pred)
val_rmsle_base = mean_squared_log_error(np.expm1(y_val), np.expm1(y_pred_val)) ** 0.5
val_rmsle_combined = mean_squared_log_error(np.expm1(y_val), np.expm1(y_pred_combined_val)) ** 0.5
print(f"ðŸ“Š Base model RMSLE: {val_rmsle_base:.4f}")
print(f"ðŸ“Š Residual combined RMSLE: {val_rmsle_combined:.4f}")


final_preds = np.clip(np.expm1(model.predict(X_test) + residual_model.predict(X_test_with_pred)), 0, None)
submission = pd.DataFrame({"id": test["id"], "Calories": final_preds})
submission.to_csv("submission.csv", index=False)
print("âœ… submission_ensemble_residual.csv saved.")


