import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df=pd.read_csv('train.csv')



df


df.columns


df.drop(columns='id')


df.describe()


df.shape


df.info()


df.isnull().sum()




# visualize missing values
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.show()


import math


df=df.drop(columns='id')
dt=df.select_dtypes(include=['float64', 'int64']).columns
n_cols=3
n_rows=math.ceil(len(dt)/n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
axes = axes.flatten()   # make axes iterable easily

for i, col in enumerate(dt):
    sns.histplot(df[col], kde=True, ax=axes[i])
    axes[i].set_title(col)

# hide empty plots if any
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


dt=df.select_dtypes(include=['object']).columns
n_cols=3
n_rows=math.ceil(len(dt)/n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
axes = axes.flatten()   # make axes iterable easily

for i, col in enumerate(dt):
    sns.countplot(y=df[col], ax=axes[i])
    axes[i].set_title(col)
    plt.tight_layout()



import matplotlib.pyplot as plt
import seaborn as sns

# Compute correlation for numeric columns only
corr = df.corr(numeric_only=True)

plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=0.5)
plt.title('Correlation Heatmap', fontsize=14)
plt.show()



df.columns


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score



gaussian_normal_features = ['credit_score','interest_rate','loan_amount','annual_income','debt_to_income_ratio']
log_normal_features = []

numeric_features = gaussian_normal_features + log_normal_features
categorical_features = df.select_dtypes(include=['object','category']).columns.tolist()

target = 'loan_paid_back'
X = df.drop(columns=[target])
y = df[target]



y.dtype


from sklearn.pipeline import make_pipeline

# Pipeline for Gaussian features
gaussian_pipeline = make_pipeline(StandardScaler())

# Pipeline for Log-normal features
log_pipeline = make_pipeline(
    FunctionTransformer(np.log1p, validate=False),
    StandardScaler()
)

# Categorical encoder
categorical_encoder = OneHotEncoder(handle_unknown='ignore')

# Combine all transformations
preprocessor = ColumnTransformer(
    transformers=[
        ('gaussian', gaussian_pipeline, gaussian_normal_features),
        ('lognorm', log_pipeline, log_normal_features),
        ('cat', categorical_encoder, categorical_features)
    ]
)



def ensure_writable_dense(X):
    # If it's sparse, convert to dense
    if hasattr(X, "toarray"):
        X2 = X.toarray()
    else:
        X2 = np.asarray(X)

    # If numpy array is read-only, copy to make it writable
    if not X2.flags.writeable:
        X2 = X2.copy()
    return X2


from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    StackingRegressor
)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
to_writable = FunctionTransformer(ensure_writable_dense, validate=False)
# âœ… Best tuned parameters
best_params_gradient = {
    'n_estimators': 343,
    'learning_rate': 0.060171308139773526,
    'max_depth': 5,
    'subsample': 0.9806501205847764,
    'min_samples_split': 6,
    'min_samples_leaf': 8,
    'max_features': None
}

best_params_lgb = {
    'n_estimators': 900,
    'learning_rate': 0.04345602488949582,
    'max_depth': 10,
    'num_leaves': 98,
    'subsample': 0.6694807067755107,
    'colsample_bytree': 0.6596462198156509,
    'min_child_samples': 42,
    'reg_alpha': 5.5422453146249175,
    'reg_lambda': 7.623232593483509e-08
}


best_params_xgb={'n_estimators': 588, 'learning_rate': 0.08369528331356997, 'max_depth': 6, 'min_child_weight': 9, 'subsample': 0.8542974340603399, 'colsample_bytree': 0.5484543433489398, 'gamma': 0.021145386348096484, 'reg_alpha': 7.552063336722299, 'reg_lambda': 4.2109384280932805}
# âœ… Define base models


best_params_cat={'iterations': 979, 'learning_rate': 0.06104252134078646, 'depth': 7, 'l2_leaf_reg': 1.0090474078785214, 'bagging_temperature': 0.5801037994774763, 'border_count': 255, 'random_strength': 0.00015900427357667946}

best_params_rcf={'n_estimators': 362, 'max_depth': 11, 'min_samples_split': 9, 'min_samples_leaf': 5, 'max_features': 'sqrt', 'bootstrap': True}

grad_model = GradientBoostingRegressor(
    **best_params_gradient,
    random_state=42
)

lgbm_model = LGBMRegressor(
    **best_params_lgb,
    random_state=42,
    n_jobs=-1
)

cat_model = CatBoostRegressor(
    **best_params_cat,
    random_seed=42
)

rf_model = RandomForestRegressor(
    **best_params_rcf,
    random_state=42,
    n_jobs=-1
)

ada_model = AdaBoostRegressor(
    n_estimators=300,
    learning_rate=0.05,
    random_state=42
)

xgb_model = XGBRegressor(
    **best_params_xgb,
    random_state=42,
    n_jobs=-1,
    eval_metric='rmse'
)



from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor  # âœ… Import LightGB
# Define XGBoost Regressor model
best_params_xgb={'n_estimators': 343, 'learning_rate': 0.060171308139773526, 'max_depth': 5, 'subsample': 0.9806501205847764, 'min_samples_split': 6, 'min_samples_leaf': 8, 'max_features': None}

model2= GradientBoostingRegressor(
    **best_params_xgb,
       random_state=42,
        verbose=1
        # â�Œ no n_jobs parameter (GBR is inherently single-threaded in sklearn)
    )

best_params={'n_estimators': 900, 'learning_rate': 0.04345602488949582, 'max_depth': 10, 'num_leaves': 98, 'subsample': 0.6694807067755107, 'colsample_bytree': 0.6596462198156509, 'min_child_samples': 42, 'reg_alpha': 5.5422453146249175, 'reg_lambda': 7.623232593483509e-08}
lgbm_model = LGBMRegressor(
**best_params,
 random_state=42,
 n_jobs=-1
)
# Combine preprocessing + model into pipeline
clf = Pipeline([
    ('preprocessor', preprocessor),
    ('model', lgbm_model)
])



import optuna
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
import numpy as np
# grad

def objective(trial):
    # Suggest hyperparameters
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 600),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 2, 8),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    }

    # Create model
    model = GradientBoostingRegressor(random_state=42, **params)

    # Create pipeline
    clf = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # Cross-validation using negative RMSE
    scores = cross_val_score(
        clf, X, y,
        scoring='neg_root_mean_squared_error',
        cv=5,
        n_jobs=-1
    )

    return scores.mean()

study = optuna.create_study(direction='maximize')  # because higher (less negative) is better
study.optimize(objective, n_trials=100, n_jobs=1, show_progress_bar=True)
print("Best RMSE (CV):", -study.best_value)
print("Best Params:", study.best_params)






import optuna
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
import numpy as np

# Define the objective function
def objective(trial):
    # Suggest hyperparameters for LightGBM
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', -1, 12),  # -1 means no limit
        'num_leaves': trial.suggest_int('num_leaves', 15, 255),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42,
        'n_jobs': -1
    }

    # Create LightGBM model
    model = LGBMRegressor(**params)

    # Build pipeline (preprocessor + model)
    clf = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # 5-fold cross-validation (negative RMSE)
    scores = cross_val_score(
        clf, X, y,
        scoring='neg_root_mean_squared_error',
        cv=5,
        n_jobs=-1
    )

    # Return mean score (maximize â†’ less negative RMSE)
    return scores.mean()


# Create Optuna study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, n_jobs=1, show_progress_bar=True)

# Print results
print("âœ… Best CV RMSE:", -study.best_value)
print("ğŸ�† Best Parameters:", study.best_params)



import optuna
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, make_scorer
import numpy as np

# --- Optuna objective function ---
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
        "random_state": 42,
        "n_jobs": -1,
        "eval_metric": "rmse"
    }

    model = XGBRegressor(**params)

    # Use your preprocessing pipeline to keep consistency
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # K-Fold CV (you can adjust folds)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores = cross_val_score(
        pipe, X, y,
        cv=kf,
        scoring=make_scorer(mean_squared_error, squared=False)
    )

    return rmse_scores.mean()

# --- Run the Optuna study ---
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)  # 50 trials, or 1 hour

# --- Print the best parameters ---
print("Best RMSE:", study.best_value)
print("Best Params:", study.best_params)

# --- Optional: Save study for reuse ---
study.trials_dataframe().to_csv("xgb_optuna_trials.csv", index=False)



from catboost import CatBoostRegressor

def objective_cat(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 200, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "depth": trial.suggest_int("depth", 3, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "random_strength": trial.suggest_float("random_strength", 0, 10),
        "random_seed": 42,
        "verbose": 0
    }

    model = CatBoostRegressor(**params)
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores = cross_val_score(pipe, X, y, cv=kf,
                                  scoring=make_scorer(mean_squared_error, squared=False))
    return rmse_scores.mean()

study_cat = optuna.create_study(direction="minimize")
study_cat.optimize(objective_cat, n_trials=100)
print("Best CatBoost RMSE:", study_cat.best_value)
print("Best CatBoost Params:", study_cat.best_params)



from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.pipeline import Pipeline
import optuna

def objective_ada(trial):
    # --- Base Decision Tree Regressor params ---
    base_estimator = DecisionTreeRegressor(
        max_depth=trial.suggest_int("max_depth", 2, 15),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 30),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
        splitter=trial.suggest_categorical("splitter", ["best", "random"]),
        criterion=trial.suggest_categorical("criterion", ["squared_error", "friedman_mse"]),
        random_state=42
    )

    # --- AdaBoost params ---
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 1.0, log=True),
        "loss": trial.suggest_categorical("loss", ["linear", "square", "exponential"]),
        "random_state": 42
    }

    model = AdaBoostRegressor(
        estimator=base_estimator,   # âœ… new arg name (since sklearn 1.2)
        **params
    )

    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # --- Cross-validation ---
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores = cross_val_score(
        pipe, X, y, cv=kf,
        scoring=make_scorer(mean_squared_error, squared=False)
    )

    return rmse_scores.mean()

# --- Run Optuna study ---
study_ada = optuna.create_study(direction="minimize")
study_ada.optimize(objective_ada, n_trials=100)

# --- Results ---
print("Best AdaBoost RMSE:", study_ada.best_value)
print("Best AdaBoost Params:", study_ada.best_params)



from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)


print(f"Test RMSE: {rmse:.3f}")
print(f"RÂ² Score: {r2:.3f}")



y_pred = clf.predict(X_train)

rmse = mean_squared_error(y_train, y_pred, squared=False)
r2 = r2_score(y_train, y_pred)

print(f"Train RMSE: {rmse:.3f}")
print(f"RÂ² Score: {r2:.3f}")


from sklearn.model_selection import cross_val_score

# Cross-validation using RÂ² score
cv_scores = cross_val_score(clf, X, y, cv=5, scoring='r2')
print(f"CV RÂ²: {cv_scores.mean():.3f} Â± {cv_scores.std():.3f}")

# Optionally, for RMSE:
cv_rmse = cross_val_score(clf, X, y, cv=5, scoring='neg_root_mean_squared_error')
print(f"CV RMSE: {-cv_rmse.mean():.3f} Â± {cv_rmse.std():.3f}")



import pandas as pd

# 1ï¸�âƒ£ Load test data
df2 = pd.read_csv('test.csv')


ids = df2['id']

# 3ï¸�âƒ£ Predict using the trained pipeline (clf)
y_pred_test = clf.predict(df2)

# 4ï¸�âƒ£ Make a result DataFrame
submission = pd.DataFrame({
    'id': ids,
    'loan_paid_back': y_pred_test
})

# 5ï¸�âƒ£ Save to CSV
name='submission_gaussian_all_cols_rmselight_2.csv'
submission.to_csv(name, index=False)

print(f"âœ… Predictions saved to {name}")



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from sklearn.linear_model import Ridge


# âœ… Meta model
meta_model = Ridge(alpha=0.0019983495638599144, random_state=42)

# âœ… Stacking Regressor (all ensemble models â†’ Ridge)
stack_model = StackingRegressor(
    estimators=[
        ('grad', grad_model),
        ('lgb', lgbm_model),
        ('cat', cat_model),
        ('rf', rf_model),
        ('ada', ada_model),
        ('xgb', xgb_model)
    ],
    final_estimator=meta_model,
    n_jobs=-1,
    passthrough=False
)

# âœ… Combine preprocessing + stack model into pipeline
clf = Pipeline([
    ('preprocessor', preprocessor), 
      ('to_writable', to_writable),  # your preprocessing step (e.g. ColumnTransformer)
    ('model', stack_model)
])

# âœ… Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# âœ… Train model
clf.fit(X_train, y_train)

# âœ… Evaluate on Test
y_pred_test = clf.predict(X_test)
rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)
r2_test = r2_score(y_test, y_pred_test)
print(f"ğŸ§ª Test RMSE: {rmse_test:.3f}")
print(f"ğŸ§ª Test RÂ²: {r2_test:.3f}")

# âœ… Evaluate on Train
y_pred_train = clf.predict(X_train)
rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
r2_train = r2_score(y_train, y_pred_train)
print(f"ğŸ�‹ï¸� Train RMSE: {rmse_train:.3f}")
print(f"ğŸ�‹ï¸� Train RÂ²: {r2_train:.3f}")

# âœ… Predict on unseen test.csv 
df2 = pd.read_csv('test.csv')
ids = df2['id']

y_pred_final = clf.predict(df2)

# âœ… Save submission
submission = pd.DataFrame({
    'id': ids,
    'loan_paid_back': y_pred_final
})

name = 'submission_stacked_allmodels_ridge.csv'
submission.to_csv(name, index=False)
print(f"âœ… Predictions saved to {name}")



import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import joblib
from sklearn.base import clone

# âœ… Read test data
df2 = pd.read_csv('test.csv')
ids = df2['id']

# âœ… Fit only the preprocessor on full training data
X_train_proc = preprocessor.fit_transform(X)
X_test_proc = preprocessor.transform(df2)

# âœ… Ensure dense & writable
X_train_proc = ensure_writable_dense(X_train_proc)
X_test_proc = ensure_writable_dense(X_test_proc)

# âœ… Define base models
base_models = {    
    
    'lgb': lgbm_model,
    'cat': cat_model,
    'rf': rf_model,
    'ada': ada_model,
    'xgb': xgb_model,
    'grad': grad_model,
}

# --- K-Fold setup ---
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# --- Create DataFrames for meta-features ---
train_meta = pd.DataFrame(np.zeros((X_train_proc.shape[0], len(base_models))),
                          columns=base_models.keys())
test_meta = pd.DataFrame(np.zeros((X_test_proc.shape[0], len(base_models))),
                         columns=base_models.keys())

# --- Generate out-of-fold (OOF) meta-features ---
for name, m in base_models.items():
    print(f"â�³ Training base model with K-Fold: {name}")
    
    oof_preds = np.zeros(X_train_proc.shape[0])
    test_preds_folds = np.zeros((X_test_proc.shape[0], kf.n_splits))
    model = clone(m)
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_proc, y)):
        print(f"   ğŸ”¹ Fold {fold + 1}/{kf.n_splits}")
        X_tr, X_val = X_train_proc[train_idx], X_train_proc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Fit model on fold
        model.fit(X_tr, y_tr)
        
        # Predict OOF for validation fold
        oof_preds[val_idx] = model.predict(X_val)
        
        # Predict on test set
        test_preds_folds[:, fold] = model.predict(X_test_proc)
    
    # Average test predictions across folds
    test_preds_mean = test_preds_folds.mean(axis=1)
    
    # Store in meta DataFrames
    train_meta[name] = oof_preds
    test_meta[name] = test_preds_mean
    
    # Optionally save each trained model (last fold)
    # joblib.dump(model, f"{name}_model.pkl")

# Add target to meta-train
train_meta['target'] = y.values

# âœ… Save meta features
train_meta.to_csv("meta_train.csv", index=False)
test_meta.to_csv("meta_test.csv", index=False)

print("âœ… K-Fold meta-features saved: meta_train.csv, meta_test.csv")






train_meta


import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Load meta-features
train_meta = pd.read_csv("meta_train.csv")
test_meta = pd.read_csv("meta_test.csv")

X_meta = train_meta.drop(columns=['target'])
y_meta = train_meta['target']
ids = df2['id']
X_test_meta = test_meta

# Try different meta-models quickly
meta_models = {
    "Ridge": Ridge(alpha=10),
    "Ridge2": Ridge(alpha=0.01),
    "Ridge3": Ridge(alpha=0.5),
    "Ridge": Ridge(alpha=10),
    "Lasso": Lasso(alpha=0.01),
    "Lasso2": Lasso(alpha=10),
    "Lasso3": Lasso(alpha=0.5),
    "XGB": XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42),
    "LGB": LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42),
    "Cat": CatBoostRegressor(iterations=300, learning_rate=0.05, depth=4, silent=True)
}

for name, meta in meta_models.items():
    meta.fit(X_meta, y_meta)
    
    y_pred = meta.predict(X_test_meta)
    rmse = mean_squared_error(y_meta, meta.predict(X_meta), squared=False)
    r2 = r2_score(y_meta, meta.predict(X_meta))
    print(f" Meta-model: {name} | RMSE: {rmse:.4f} | RÂ²: {r2:.4f}")

    # Optional: Save predictions
    pd.DataFrame({
        "id": ids,
        "loan_paid_back": y_pred
    }).to_csv(f"submission_meta_{name}.csv", index=False)









def objective(trial):
    # Suggest alpha (log scale â†’ better coverage)
    alpha = trial.suggest_float('alpha', 1e-3, 100.0, log=True)

    meta_model = Ridge(alpha=alpha, random_state=42)

    stack_model = StackingRegressor(
        estimators=[
            ('xgb', xgb_model),
            ('lgb', lgbm_model)
        ],
        final_estimator=meta_model,
        n_jobs=-1,
        passthrough=False
    )

    clf = Pipeline([
        ('preprocessor', preprocessor),
        ('model', stack_model)
    ])

    # Use 5-fold CV RMSE
    scores = cross_val_score(
        clf, X, y,
        cv=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )

    return scores.mean()

# âœ… Run Optuna
study = optuna.create_study(direction='maximize')  # maximize negative RMSE
study.optimize(objective, n_trials=100, show_progress_bar=True)

# âœ… Best results
print("Best CV RMSE:", -study.best_value)
print("Best alpha:", study.best_params['alpha'])


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import numpy as np

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define models
models = {
    "XGBRegressor": XGBRegressor(
        n_estimators=500,
        learning_rate=0.01,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='rmse',
        n_jobs=-1  # âœ… parallelize XGBoost
    ),
    "RandomForest": RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        random_state=42,
        n_jobs=-1  # âœ… parallelize RandomForest
    ),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
        # â�Œ no n_jobs parameter (GBR is inherently single-threaded in sklearn)
    ),
    "AdaBoost": AdaBoostRegressor(
        n_estimators=200,
        learning_rate=0.1,
        random_state=42,
       
    )
}

results = {}

# Evaluate each model
for name, model in models.items():
    print(f"Evaluating {name}...")
    clf = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    # Cross-validation
    cv_r2 = cross_val_score(clf, X, y, cv=5, scoring='r2').mean()
    cv_rmse = -cross_val_score(clf, X, y, cv=5, scoring='neg_root_mean_squared_error').mean()

    results[name] = {
        "Test_RMSE": rmse,
        "Test_R2": r2,
        "CV_R2": cv_r2,
        "CV_RMSE": cv_rmse
    }

# Display results
import pandas as pd
results_df = pd.DataFrame(results).T.sort_values(by="CV_R2", ascending=False)
print(results_df)





