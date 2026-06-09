import seaborn as sns
import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import tensorflow as tf
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error 
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

from xgboost import XGBRegressor 
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
df_test=pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')


df.head()


df.info()


#fig, axes = plt.subplots(1, 2, figsize=(12, 5))

#sns.barplot(x=df.weather, y=df.accident_risk, ax=axes[0])
#axes[0].set_title("Accident risk selon la météo")

#sns.barplot(x=df.road_type, y=df.accident_risk, ax=axes[1])
#axes[1].set_title("Accident risk selon le type de route")


#plt.tight_layout()
#plt.show()


#fig, axes = plt.subplots(1, 2, figsize=(12, 5))
#sns.barplot(x=df.lighting, y=df.accident_risk, ax=axes[0])
#axes[0].set_title("Accident risk selon la lumière")

#sns.barplot(x=df.time_of_day, y=df.accident_risk, ax=axes[1])
#axes[1].set_title("Accident risk selon le moment de la journée")
#plt.tight_layout()
#plt.show()


#plt.figure (figsize=(8,8))
#sns.scatterplot(x=df.curvature, y=df.accident_risk)

#plt.show()


#sns.boxplot(x=df.accident_risk)


print("the correlation is: ",(df.speed_limit).corr(df.accident_risk))
print("the correlation is: ",(df.num_lanes).corr(df.accident_risk))
print("the correlation is: ",(df.curvature).corr(df.accident_risk))
print("the correlation is: ",(df.num_reported_accidents).corr(df.accident_risk))


df.drop(['num_lanes', 'time_of_day','road_type',"road_signs_present"],axis=1)


OHE= OneHotEncoder(handle_unknown='ignore', sparse_output=False)



s = (df.dtypes == 'object') | (df.dtypes == 'bool') | (df.dtypes == 'category')
object_cols = list(s[s].index)



df[object_cols]


OH_cols = pd.DataFrame(OHE.fit_transform(df[object_cols]))

# One-hot encoding removed index; put it back
OH_cols.index = df.index

# Remove categorical columns (will replace with one-hot encoding)
num_df = df.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_df = pd.concat([num_df, OH_cols], axis=1)


X=OH_df.drop(['id', 'accident_risk'],axis=1)
y=OH_df.pop('accident_risk')


train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)


model_XGB = XGBRegressor(n_estimators=1000, learning_rate=0.06,max_depth=8,early_stopping_rounds=20, subsample=0.8 )

model_XGB.fit(train_X, train_y,
           eval_set=[(val_X, val_y)]
)

pred_y_XGB = model_XGB.predict(val_X) 
XGB_score= np.sqrt(mean_squared_error(val_y, pred_y_XGB))

print(mean_squared_error(val_y,pred_y_XGB,squared=False))


model_LGB= LGBMRegressor(num_leaves=50, max_depth=20, learning_rate=0.06, n_estimators=400, force_row_wise= True, subsample=0.8) 

model_LGB.fit(train_X, train_y,
           eval_set=[(val_X, val_y)]
)

pred_y_LGB = model_LGB.predict(val_X) 

print(mean_squared_error(val_y,pred_y_LGB,squared=False))
LGB_score= np.sqrt(mean_squared_error(val_y, pred_y_LGB))


model_cat= CatBoostRegressor(iterations= 1000, depth=8, learning_rate=0.05)

model_cat.fit(train_X, train_y,
           eval_set=[(val_X, val_y)]
)

pred_y_cat = model_cat.predict(val_X) 

print(mean_squared_error(val_y,pred_y_cat,squared=False))
cat_score= np.sqrt(mean_squared_error(val_y, pred_y_cat))



from sklearn.linear_model import LinearRegression


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Empile les prédictions des 3 modèles en colonnes
X_stack = np.column_stack([pred_y_XGB, pred_y_LGB, pred_y_cat])

# --- 1) Régression linéaire (sans intercept) : y ≈ X_stack @ w
lr = LinearRegression(fit_intercept=False)
lr.fit(X_stack, val_y)
w_ols = lr.coef_.astype(float)

# --- 2) Variante "simplexe" : poids ≥ 0 et somme = 1
w_simplex = np.maximum(w_ols, 0.0)
s = w_simplex.sum()
if s == 0:
    w_simplex = np.array([1/3, 1/3, 1/3], dtype=float)
else:
    w_simplex = w_simplex / s

# Évalue les deux solutions sur la validation
ensemble_oof_ols = X_stack @ w_ols
ensemble_oof_simplex = X_stack @ w_simplex
rmse_ols = rmse(val_y, ensemble_oof_ols)
rmse_simplex = rmse(val_y, ensemble_oof_simplex)

# Choix final = le meilleur des deux
if rmse_simplex < rmse_ols:
    best_w = w_simplex
    ensemble_oof = ensemble_oof_simplex
    best_label = "Linear Regression (constrained to simplex)"
else:
    best_w = w_ols
    ensemble_oof = ensemble_oof_ols
    best_label = "Linear Regression (unconstrained OLS)"

ensemble_weights = {
    'xgb': float(best_w[0]),
    'lgb': float(best_w[1]),
    'cat': float(best_w[2]),
}

ensemble_cv = rmse(val_y, ensemble_oof)

print("="*50)
print("FINAL MODEL COMPARISON")
print("="*50)
print(f"XGBoost CV RMSE:     {XGB_score:.5f}")
print(f"LightGBM CV RMSE:    {LGB_score:.5f}")
print(f"CatBoost CV RMSE:    {cat_score:.5f}")
print("-"*50)
print("Weighting via Linear Regression:")
print(f"  Method: {best_label}")
print(f"  XGB: {ensemble_weights['xgb']:.6f}")
print(f"  LGB: {ensemble_weights['lgb']:.6f}")
print(f"  CAT: {ensemble_weights['cat']:.6f}")
print("-"*50)
print(f"Ensemble CV RMSE:    {ensemble_cv:.5f}")
print("="*50)



df_test.drop(['num_lanes', 'time_of_day','road_type',"road_signs_present"],axis=1)


OH_cols_test = pd.DataFrame(OHE.fit_transform(df_test[object_cols]))

# One-hot encoding removed index; put it back
OH_cols_test.index = df_test.index

# Remove categorical columns (will replace with one-hot encoding)
num_df_test = df_test.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_df_test = pd.concat([num_df_test, OH_cols_test], axis=1)


id=OH_df_test.pop('id')


predictions_xgb= model_XGB.predict(OH_df_test)
predictions_lgb= model_LGB.predict(OH_df_test)
predictions_cat=model_cat.predict(OH_df_test)






predictions= predictions_xgb * ensemble_weights['xgb'] + predictions_lgb * ensemble_weights['lgb'] + predictions_cat * ensemble_weights['cat']


predictions = predictions.flatten()


output = pd.DataFrame({ 'id':id,
                       'Target': predictions})


output.set_index('id')


output.to_csv('submission.csv', index=False)

