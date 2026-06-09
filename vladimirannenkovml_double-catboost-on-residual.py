import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
import optuna
from scipy import stats


def get_proc(x: str):
    if "Intel" in x:
        return x.split("-")[0]
    elif "AMD" in x:
        return " ".join(x.split(" ")[1:-1])
    else:
        return " ".join(x.split(" ")[1:])
    
def get_inherit(x: str):
    if "RTX" in x:
        return " ".join(x.split(" ")[0:2])
    elif "RX" in x:
        return " ".join(x.split(" ")[0:2])
    elif "Apple" in x:
        return x
    else:
        return x.split(" ")[0]

def get_res(x: str):
    if x in ['2560x1440', '2560x1600', '2880x1800']:
        return "QHD"
    else: 
        return x

def get_ppi(x: str):
    x, y = x.split("x")
    return np.sqrt(int(x)**2 * int(y)**2)

def prepare_data(raw_train_df):
    df = raw_train_df.copy()
    
    weak_features = ['ID', 'model', 'bluetooth', 'psu_watts', 'wifi', 'warranty_months', 'storage_drive_count']
    df = df.drop(columns=[col for col in weak_features if col in df.columns])
    
    df["age"] = 2026 - df["release_year"]
    df = df.drop(columns=["release_year"])
    
    df["bace_cpu"] = df["cpu_model"].apply(lambda x: get_proc(x))
    df = df.drop(columns=["cpu_model"])
    
    df["inherit"] = df["gpu_model"].apply(lambda x: get_inherit(x))
    df = df.drop(columns=["gpu_model"])
    
    df["PPI"] = df["resolution"].apply(lambda x: get_ppi(x)) / df["display_size_in"]
    df["resolution"] = df["resolution"].apply(lambda x: get_res(x))
    
    df["cpu_speed"] = df["cpu_base_ghz"] * 0.4 + df["cpu_boost_ghz"] * 0.6
    df["cpu_perf_per_core"] = df["cpu_speed"] * df["cpu_cores"]
    df["gpu_power"] = df["gpu_tier"] * df["vram_gb"]
    df["total_system_power"] = df["cpu_tier"] * 50 + df["gpu_tier"] * 100
    
    df["is_mobile"] = (df["device_type"] == "Laptop").astype(int)
    df = df.drop(columns=["device_type"])
    
    return df

def transform_target(y):
    """Transform the target variable to remove skewness"""
    return np.log1p(y)

def inverse_transform(y_transformed):
    """Reverse conversion"""
    return np.expm1(y_transformed)

#optimization
def fast_objective(trial, X_train, y_train, cat_features):
    params = {
        'iterations': 1500,
        'depth': trial.suggest_int('depth', 4, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.3),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'random_strength': trial.suggest_float('random_strength', 0.1, 1.0),
        'border_count': trial.suggest_int('border_count', 32, 200),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_state': 42,
        'verbose': False,
        'early_stopping_rounds': 50
    }
    
    # 2-fold CV
    kf = KFold(n_splits=2, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in kf.split(X_train):
        X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model = CatBoostRegressor(**params)
        model.set_params(cat_features=cat_features)
        model.fit(X_cv_train, y_cv_train, eval_set=(X_cv_val, y_cv_val), verbose=False)
        
        y_pred = model.predict(X_cv_val)
        rmse = np.sqrt(mean_squared_error(y_cv_val, y_pred))
        scores.append(rmse)
    
    return np.mean(scores)

#Residual 
class ResidualBoosting:
    def __init__(self, model1_params, model2_params, cat_features):
        self.model1 = CatBoostRegressor(**model1_params)
        self.model1.set_params(cat_features=cat_features)
        
        self.model2 = CatBoostRegressor(**model2_params)
        self.model2.set_params(cat_features=cat_features)
        
        self.is_fitted = False
    
    def fit(self, X_train, y_train, X_val, y_val):
        self.model1.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
            verbose=100
        )
        
        y_pred1_train = self.model1.predict(X_train)
        y_pred1_val = self.model1.predict(X_val)
        
        residuals_train = y_train - y_pred1_train
        residuals_val = y_val - y_pred1_val
        
        self.model2.fit(
            X_train, residuals_train,
            eval_set=(X_val, residuals_val),
            early_stopping_rounds=50,
            verbose=100
        )
        
        self.is_fitted = True
    
    def predict(self, X):
        
        pred1 = self.model1.predict(X)
        pred2 = self.model2.predict(X)
        
        return pred1 + pred2

def improved_training_with_residuals(X_train, X_val, y_train, y_val, cat_features):

    study = optuna.create_study(direction='minimize')
    
    study.optimize(lambda trial: fast_objective(trial, X_train, y_train, cat_features), 
                  n_trials=10, show_progress_bar=True)
    
    print(f"The best CV RMSE: {study.best_value:.4f}")

    base_params = {
        'iterations': 1500,
        'random_state': 42,
        'verbose': False,
        'early_stopping_rounds': 50
    }
    
    model1_params = base_params.copy()
    model1_params.update(study.best_params)
    model1_params['verbose'] = True
    
    model2_params = base_params.copy()
    model2_params.update({
        'depth': 4,
        'learning_rate': 0.05,
        'l2_leaf_reg': 5.0,
        'iterations': 1000
    })
    model2_params['verbose'] = True
    

    residual_model = ResidualBoosting(model1_params, model2_params, cat_features)
    residual_model.fit(X_train, y_train, X_val, y_val)
    
    y_val_pred_residual = residual_model.predict(X_val)
    rmse_residual = np.sqrt(mean_squared_error(y_val, y_val_pred_residual))
    
    print(f"\nRMSE с residual boosting: {rmse_residual:.4f}")
    
    return residual_model, rmse_residual


raw_train_df = pd.read_csv("computer_prices_all.csv")

focused_train_df = prepare_data(raw_train_df)

#  categorials
focused_categorical = ['brand', 'os', 'form_factor', 'cpu_brand', 'gpu_brand', 
                      'storage_type', 'display_type', 'resolution', 'bace_cpu', 'inherit']

focused_numerical = ['cpu_tier', 'cpu_cores', 'cpu_threads', 'cpu_base_ghz', 'cpu_boost_ghz',
                    'gpu_tier', 'vram_gb', 'ram_gb', 'storage_gb', 'display_size_in',
                    'refresh_hz', 'battery_wh', 'charger_watts', 'weight_kg', 'age',
                    'PPI', 'cpu_speed', 'cpu_perf_per_core', 'gpu_power', 
                    'total_system_power', 'is_mobile']

available_cat = [col for col in focused_categorical if col in focused_train_df.columns]
available_num = [col for col in focused_numerical if col in focused_train_df.columns]

X = focused_train_df[available_cat + available_num]
y = focused_train_df['price']


y_transformed = transform_target(y)

X_train, X_val, y_train, y_val = train_test_split(
    X, y_transformed, test_size=0.2, random_state=42
)

print(f"Train: {X_train.shape}, Validation: {X_val.shape}")


residual_model, final_rmse_transformed = improved_training_with_residuals(
    X_train, X_val, y_train, y_val, available_cat
)


y_val_pred_final = residual_model.predict(X_val)
y_val_pred_original = inverse_transform(y_val_pred_final)
y_val_original = inverse_transform(y_val)

final_rmse_original = np.sqrt(mean_squared_error(y_val_original, y_val_pred_original))

print(f"RMSE после residual boosting: {final_rmse_original:.4f}")

val_results = pd.DataFrame({
    'actual': y_val_original,
    'predicted': y_val_pred_original,
    'error': y_val_pred_original - y_val_original,
    'error_pct': ((y_val_pred_original - y_val_original) / y_val_original) * 100
})

# Статистика ошибок
print(f"\n=== СТАТИСТИКА ОШИБОК ===")
print(f"RMSE: {final_rmse_original:.4f}")
print(f"MAE: {np.mean(np.abs(val_results['error'])):.2f}")
print(f"Mean absolute error: {np.mean(np.abs(val_results['error_pct'])):.2f}%")
print(f"Median relative error: {np.median(np.abs(val_results['error_pct'])):.2f}%")




feature_importance = pd.DataFrame({
    'feature': available_cat + available_num,
    'importance': residual_model.model1.get_feature_importance()
}).sort_values('importance', ascending=False)

print("Top 15 features")
print(feature_importance.head(15))

residual_model.model1.save_model('residual_model_1.cbm')
residual_model.model2.save_model('residual_model_2.cbm')

def predict_price(new_data, residual_model=residual_model):

    predictions_transformed = residual_model.predict(new_data)
    return inverse_transform(predictions_transformed)

sample_pred = predict_price(X_val.head(1))
print(f"Predict price: ${sample_pred[0]:.2f}")

print(f"Real price: ${y_val_original.iloc[0]:.2f}")


test_df = pd.read_csv("computer_prices_test.csv")
ids = pd.DataFrame({"ID": test_df["ID"]})   
X_test =  prepare_data(test_df)
available_cat = [col for col in focused_categorical if col in focused_train_df.columns]
available_num = [col for col in focused_numerical if col in focused_train_df.columns]

X = X_test[available_cat + available_num]

y_pred_res_transformed = residual_model.predict(X)
y_pred_res = inverse_transform(y_pred_res_transformed)

res_df = pd.DataFrame({"ID": ids["ID"], "price": y_pred_res})
res_df.to_csv('submission1.csv', index=False)


