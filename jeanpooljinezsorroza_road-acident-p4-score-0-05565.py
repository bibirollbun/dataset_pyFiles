import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV

from sklearn.metrics import mean_squared_error

from datetime import datetime

import warnings
warnings.filterwarnings('ignore')


df_train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv', index_col="id")
df_test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv', index_col="id")


# notebook: House Prices--advance_preprocesor_search_parameters.ipynb
def grab_col_names(df, target=None, cat_th=10, car_th=20):
    # Categóricas (objetos, categorías o booleanas)
    cat_cols = [col for col in df.columns if df[col].dtype in ["O", "category", "bool"]]
    
    # Numéricas pero categóricas (discretas con pocas categorías)
    num_but_cat = [col for col in df.columns 
                   if df[col].nunique() < cat_th and df[col].dtype in ["int64", "float64"]]
    
    # Categóricas cardinales (muchas categorías únicas)
    cat_but_car = [col for col in df.columns 
                   if df[col].nunique() > car_th and df[col].dtype in ["O", "category"]]
    
    # Unir categóricas verdaderas + numéricas discretas
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    
    # Numéricas reales (sin incluir num_but_cat)
    num_cols = [col for col in df.columns if df[col].dtype in ["int64", "float64"]]
    num_cols = [col for col in num_cols if col not in num_but_cat]
    
    # Excluir target si está en alguna lista
    if target:
        for col_list in [cat_cols, num_cols, cat_but_car, num_but_cat]:
            if target in col_list:
                col_list.remove(target)
                
    cat_cols = [col for col in cat_cols if col not in num_but_cat]
    
    # Resumen
    print("-" * 20)
    print(f"Observations: {df.shape[0]}")
    print(f"Variables: {df.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")
    print("-" * 20)
    print('Cat_cols:\n',cat_cols)
    print('num_cols:\n',num_cols)
    print('cat_but_car:\n',cat_but_car)
    print('num_but_cat:\n',num_but_cat)
    print("-" * 20)
    
    return cat_cols, num_cols, cat_but_car, num_but_cat
#-----------------------------------------------------------------------------------------------
def add_engineered_features(df):
    
    df = df.copy()
    
    bool_cols = df.select_dtypes(include='bool').columns.tolist()
    for col in bool_cols:
        df[col] = df[col].astype(int)
        
    df['curvature_speed'] = df['curvature'] * df['speed_limit']
    df['risk_intensity'] = df['curvature'] * df['speed_limit'] / 50
    df['curvature_speed_per_lane'] = df['curvature_speed'] / (df['num_lanes'] + 1)
    df['curvature_speed_lane'] = df['curvature_speed'] * df['num_lanes']    
    df['curvature_squared'] = df['curvature'] ** 2
    df['curvature_cubed'] = df['curvature'] ** 3
    df['high_risk_combo'] = (
        (df['curvature'] > 0.5) &
        (df['speed_limit'] >= 60)
    ).astype(int)
    df['accidents_curvature'] = df['num_reported_accidents'] * df['curvature']
    df['curvature_bin'] = pd.cut(df['curvature'], bins=[0, 0.3, 0.6, 1.0], labels=[0, 1, 2])
    df['lanes_curvature'] = df['num_lanes'] * df['curvature']
    df['accidents_speed'] = df['num_reported_accidents'] * df['speed_limit']
    df['accident_speed_risk'] = df['num_reported_accidents'] * df['speed_limit'] / 100
    df['speed_squared'] = df['speed_limit'] ** 2
    df['road_type_speed'] = df['road_type'] + "_" + df['speed_limit'].astype(str)
    df['speed_category'] = pd.cut(df['speed_limit'], bins=[0, 30, 50, 100], labels=[0, 1, 2])
    df['lighting_time'] = df['lighting'] + "_" + df['time_of_day']
    df['is_night'] = (df['lighting'].str.lower() == 'night').astype(int)
    
    return df    
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
def random_search(estimator, param_distributions, n_iter, cv, X, y):

    random_search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring='neg_root_mean_squared_error',
        cv=cv,
        n_jobs=-1,
        verbose=2,
    )
    
    random_search.fit(X, y)

    best_model = random_search.best_estimator_
    
    print("Mejores parámetros:", random_search.best_params_)
    print("Mejor score:", random_search.best_score_)
    print("Mejor modelo:", best_model)
    return best_model


print('NULL Values in df_train:\n',df_train.isnull().sum())
print('Duplicated values in df_train',df_train.duplicated().sum())
df_train = df_train.drop_duplicates()
print('Duplicated values in df_train',df_train.duplicated().sum())


df_train_fe = add_engineered_features(df_train)
df_test_fe = add_engineered_features(df_test)


# Prepare data
X = df_train_fe.drop(['accident_risk'], axis=1)
y = df_train_fe.accident_risk
X_test = df_test_fe


cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(X)


for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))

print(f"\n Final feature count: {X.shape[1]}")
print(f" Feature types: {X.dtypes.value_counts().to_dict()}")


# Cross-validation setup
folds  = 5
param_comb = 30
skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 42)

# Stratify based on binned target
y_binned = pd.qcut(y, q=10, labels=False, duplicates='drop')


models = {
    'LightGBM': LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1
    ),
    'CatBoost': CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=7,
        l2_leaf_reg=3,
        random_state=42,
        verbose=0
    ),
    'XGBoost': XGBRegressor(
        n_estimators=5000,
        learning_rate=0.02,
        max_depth=7,
        subsample=0.9,
        colsample_bytree=0.6,
        reg_alpha=0.1,
        reg_lambda=0.1,
        tree_method='hist',
        random_state=42,
        verbosity=0
    )
}


"""
# Train models and collect predictions
results = {}
oof_predictions = {}
test_predictions = {}

for name, model in models.items():
    print(f"\n{'='*60}")
    print(f"Training {name}...")
    print(f"{'='*60}")
    
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_binned), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        oof_preds[val_idx] = model.predict(X_val)
        test_preds += model.predict(X_test) / folds
        
        # Score
        fold_rmse = np.sqrt(mean_squared_error(y_val, oof_preds[val_idx]))
        fold_scores.append(fold_rmse)
        print(f"   Fold {fold}: RMSE = {fold_rmse:.6f}")
        
     # Overall OOF score
    oof_rmse = np.sqrt(mean_squared_error(y, oof_preds))
    results[name] = {
        'oof_score': oof_rmse,
        'fold_scores': fold_scores,
        'std': np.std(fold_scores)
    }
    oof_predictions[name] = oof_preds
    test_predictions[name] = test_preds
    
    print(f"   {'─'*50}")
    print(f"   OOF RMSE: {oof_rmse:.6f} (+/- {np.std(fold_scores):.6f})")"""


"""# Results summary
results_df = pd.DataFrame(results).T
results_df = results_df.sort_values('oof_score')
print("\n" + results_df.to_string())

# Create ensemble
print("\n" + "="*80)
print("CREATING ENSEMBLE")
print("="*80)

# Weighted average based on performance
weights = 1 / results_df['oof_score'].values
weights = weights / weights.sum()

print("\n Ensemble Weights:")
for model, weight in zip(results_df.index, weights):
    print(f"   {model:15s}: {weight:.4f}")

# Ensemble predictions
ensemble_oof = np.zeros(len(X))
ensemble_test = np.zeros(len(X_test))

for model, weight in zip(results_df.index, weights):
    ensemble_oof += oof_predictions[model] * weight
    ensemble_test += test_predictions[model] * weight

ensemble_rmse = np.sqrt(mean_squared_error(y, ensemble_oof))
print(f"\n Ensemble OOF RMSE: {ensemble_rmse:.6f}")

# Comparison
improvement = (results_df['oof_score'].iloc[0] - ensemble_rmse) / results_df['oof_score'].iloc[0] * 100
print(f" Improvement over best single model: {improvement:.2f}%")"""


"""# === Guardar predicciones individuales de cada modelo ===
for model_name, preds in test_predictions.items():
    submission = pd.DataFrame({
        "id": X_test.index,
        "accident_risk": preds
    })
    filename = f"submission_{model_name.lower()}.csv"
    submission.to_csv(filename, index=False)
    print(f"✅ Archivo '{filename}' creado correctamente.")

# === Guardar también el ensemble final ===
ensemble_submission = pd.DataFrame({
    "id": X_test.index,
    "accident_risk": ensemble_test
})
ensemble_submission.to_csv("submission_ensemble.csv", index=False)
print("✅ Archivo 'submission_ensemble.csv' creado correctamente.")
"""

