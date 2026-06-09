# Core Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Machine Learning Libraries

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# ML Models
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor

import seaborn as sns

# Suppress Warnings
import warnings
warnings.filterwarnings('ignore')


def rmse(y_true, y_pred):
   return np.sqrt(mean_squared_error(y_true, y_pred))

def find_best_ensemble_weights(predictions_list, y_true, num_iterations=100):
    num_models = len(predictions_list)
    best_rmse = float('inf')
    best_weights = None

    for _ in range(num_iterations):
        # Generate random weights from a Dirichlet distribution
        weights = np.random.dirichlet(np.ones(num_models))
        rounded_weights = tuple(np.round(weights, 2))

        # Ensure the number of weights matches the number of prediction arrays
        if len(rounded_weights) != num_models:
            raise ValueError("Number of weights does not match the number of prediction arrays.")

        # Weighted average of predictions
        ensemble_preds = np.zeros_like(predictions_list[0])
        for i in range(num_models):
            ensemble_preds += rounded_weights[i] * predictions_list[i]

        # Calculate RMSE
        ensemble_rmse = np.sqrt(mean_squared_error(y_true, ensemble_preds))

        if ensemble_rmse < best_rmse:
            best_rmse = ensemble_rmse
            best_weights = rounded_weights

    print(f"\nBest weights: {', '.join([f'Model {i+1}: {w}' for i, w in enumerate(best_weights)])}")
    print(f"Best ensemble RMSE: {best_rmse:.4f}")

    return best_weights, best_rmse

def extract_feature_importances(models, feature_names):
    all_importances = pd.DataFrame(index=feature_names)
    
    if 'XGBoost' in models:
        all_importances['XGBoost Weight'] = models['XGBoost'].feature_importances_
        xgb_gain = models['XGBoost'].get_booster().get_score(importance_type='gain')
        all_importances['XGBoost Gain'] = _map_importance_to_features(xgb_gain, feature_names)
    
    if 'LightGBM' in models:
        all_importances['LightGBM Split'] = models['LightGBM'].feature_importances_
        lgbm_gain = models['LightGBM'].booster_.feature_importance(importance_type='gain')
        all_importances['LightGBM Gain'] = lgbm_gain
    
    if 'CatBoost' in models:
        all_importances['CatBoost'] = models['CatBoost'].feature_importances_
    
    if 'RandomForest' in models:
        all_importances['RandomForest'] = models['RandomForest'].feature_importances_
    
    _normalize_importances(all_importances)
    
    return all_importances

def _map_importance_to_features(importance_dict, feature_names):
    importance_values = np.zeros(len(feature_names))
    
    if list(importance_dict.keys())[0].startswith('f') and list(importance_dict.keys())[0][1:].isdigit():
        for key, value in importance_dict.items():
            idx = int(key.replace('f', ''))
            importance_values[idx] = value
    else:
        for feature_name in feature_names:
            if feature_name in importance_dict:
                idx = list(feature_names).index(feature_name)
                importance_values[idx] = importance_dict[feature_name]
                
    return importance_values

def _normalize_importances(df):
    for col in df.columns:
        if df[col].sum() > 0:
            df[col] = df[col] / df[col].sum()
    
    return df


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')


train.info()


train.head()


train.describe()


display(train.corr(numeric_only=True))



def prepare_data(data):
    copy = data.copy()
    copy['Cv_Load'] = copy['Heart_Rate'] * copy['Duration']
    copy['Cv_Demand'] = copy['Weight'] * copy['Duration']
    copy['Thermal_Stress'] = copy['Body_Temp'] * copy['Duration']
    return copy


train = prepare_data(train)


label_encoder = LabelEncoder()

train['Sex'] = label_encoder.fit_transform(train['Sex'])
train["Sex"] = train["Sex"].astype("category")

labels = np.log1p(train["Calories"])
train = train.drop(columns=["id", "Calories"])


X_train, X_test, y_train, y_test = train_test_split(train, labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=(0.25), random_state=42)


corr = train.corr()

plt.figure(figsize=(10, 8))

mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, 
            mask=mask,
            annot=True, 
            fmt='.2f',
            cmap='coolwarm',
            vmin=-1, vmax=1, 
            linewidths=0.5,
            annot_kws={"size": 7},
            square=True)

plt.title('Feature Correlation Heatmap', fontsize=18, pad=20)
plt.xticks(fontsize=10, rotation=45, ha='right')
plt.yticks(fontsize=10)

plt.tight_layout()
plt.show()


xgb_config = {
    'n_estimators': 1100,
    'learning_rate': 0.013716909843542555,
    'max_depth': 11,
    'min_child_weight': 3,
    'subsample': 0.6474544779422277,
    'colsample_bytree': 0.6756263234265003,
    'gamma': 2.9760049606720922e-05,
    'reg_lambda': 0.7456837622464084,
    'reg_alpha': 7.130044917976876e-06,
    'objective': 'reg:squaredlogerror',
    'enable_categorical': True,
    'eval_set': [(X_val, y_val)],
}


xgb_model = xgb.XGBRegressor(**xgb_config)
xgb_model.fit(X=X_train, y=y_train)

xgb1_predictions = xgb_model.predict(X_test)
print(f"{np.sqrt(mean_squared_error(y_test, xgb1_predictions)):.6f}")


lgb_config = {
    'learning_rate': 0.011974973795927277,
    'n_estimators': 900,
    'num_leaves': 191,
    'max_depth': 27,
    'min_child_weight': 0.3723518241236667,
    'min_child_samples': 7,
    'subsample': 0.8654774420162681,
    'subsample_freq': 6,
    'colsample_bytree': 0.8400402205580921,
    'reg_alpha': 0.002468873033000459,
    'reg_lambda': 4.6462056410378543e-07,
    'objective': 'regression',
    'random_state': 42,
    'force_col_wise': True,
    'n_jobs': -1,
    'verbose': -1,
}

lgb_model = lgb.LGBMRegressor(**lgb_config)
lgb_model.fit(
    X=X_train,
    y=y_train,
    eval_set=(X_val, y_val),
    callbacks= [lgb.early_stopping(25)]
)

lgb_predictions = lgb_model.predict(X_test)
print(f"{np.sqrt(mean_squared_error(y_test, lgb_predictions)):.6f}")


cat_config = {
    'learning_rate': 0.01928353485682738, 
    'depth': 9, 
    'l2_leaf_reg': 0.0009825798445035239, 
    'random_strength': 1.4800703979936343e-08, 
    'bagging_temperature': 2.9976035010300635, 
    'border_count': 201, 
    'grow_policy': 'SymmetricTree', 
    'min_data_in_leaf': 70, 
    'subsample': 0.7665975430759543, 
    'max_ctr_complexity': 6
}

cat_model = CatBoostRegressor(**cat_config)
cat_model.fit(
    X=X_train, 
    y=y_train, 
    cat_features=['Sex'],
    eval_set=(X_val, y_val),
    early_stopping_rounds=50,
    verbose=False
)

cat_predictions = cat_model.predict(X_test)
print(f"{np.sqrt(mean_squared_error(y_test, cat_predictions)):.6f}")


rf_config = {
    'n_estimators': 1000, 
    'max_depth': 44, 
    'min_samples_split': 18, 
    'min_samples_leaf': 1, 
    'max_features': 0.5, 
    'bootstrap': True
}

rf_model = RandomForestRegressor(**rf_config)


rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# Evaluate
display(f"{np.sqrt(mean_squared_error(y_test, rf_predictions)):.6f}")


predictions = [xgb1_predictions, lgb_predictions, rf_predictions, cat_predictions]
best_weights, best_rsmle = find_best_ensemble_weights(predictions, y_test)


test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
test = prepare_data(test)

test['Sex'] = label_encoder.transform(test['Sex'])
test["Sex"] = test["Sex"].astype("category")
test = test.drop(columns=["id"])


xgb_predictions = xgb_model.predict(test)
lgb_predictions = lgb_model.predict(test)
rf_predictions = rf_model.predict(test)
cat_predictions = cat_model.predict(test)

w_xgb, w_lgb, w_rf, w_cat = best_weights
predictions = (w_xgb * xgb_predictions) + (w_lgb * lgb_predictions) + (w_rf * rf_predictions) + (w_cat * cat_predictions)

submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
submission['Calories'] = np.expm1(predictions)

submission.to_csv('submission.csv', index=False)

print('submission.csv saved')


display(submission.describe())


models = {
    'XGBoost': xgb_model,
    'LightGBM': lgb_model,
    'CatBoost': cat_model,
    'RandomForest': rf_model
}

feature_names = train.columns

importances_df = extract_feature_importances(models, feature_names)
print("Feature Importance DataFrame")
display(importances_df)

