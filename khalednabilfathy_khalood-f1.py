import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


train_df = pd.read_csv('/kaggle/input/f-1-racer-diet-planning/train.csv')


train_df.sample(5)


print(f"Missing values:\n {train_df.isna().sum()}")


print("Some info about data")
train_df.info()
print("Data distribution")
train_df.describe()


plt.figure(figsize=(15,10))

plt.subplot(2, 3, 1)
sns.histplot(train_df['Calories'], kde=True)

plt.subplot(2, 3, 2)
sns.histplot(train_df['Heart_Rate'], kde=True)

plt.subplot(2, 3, 3)
sns.histplot(train_df['Body_Temp'], kde=True)

plt.subplot(2, 3, 4)
sns.histplot(train_df['Duration'], kde=True)

plt.subplot(2, 3, 5)
sns.histplot(train_df['Age'], kde=True)

plt.tight_layout()
plt.show()


# sex distribution
plt.figure(figsize=(8, 5))

sns.countplot(x='Sex', data=train_df)
plt.title("SEX DISTRIBUTION")
plt.show()


#correlation matrix
plt.figure(figsize=(12,10))
correlation = train_df.drop('Sex',axis=1).corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title("Correlation matrix")
plt.show()


#some relations between features and calories.
plt.figure(figsize=(20, 15))
features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
for i, feature in enumerate(features):
    plt.subplot(3, 2, i+1)
    sns.scatterplot(x=feature, y='Calories', data=train_df)
    plt.title(f'{feature} vs Calories')
plt.tight_layout()
plt.show()


#check outliers using boxplots
plt.figure(figsize=(20, 15))
features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories']
for i, feature in enumerate(features):
    plt.subplot(3, 3, i+1)
    sns.boxplot(y=train_df[feature])
    plt.title(f'{feature}')
plt.tight_layout()
plt.show()


df_train_pre = train_df.copy()

#BMI is derived from weight and height. kg/m
df_train_pre['BMI'] = df_train_pre['Weight']/((df_train_pre['Height']/100)**2)

#encode Sex to numeric.
df_train_pre['Sex'] = df_train_pre['Sex'].map({'female': 0, 'male': 1})

#some new features
df_train_pre['Duration_HeartRate'] = df_train_pre['Duration']*df_train_pre['Heart_Rate']
df_train_pre['Duration_BodyTemp'] = df_train_pre['Duration']*df_train_pre['Body_Temp']
df_train_pre['Heart_BodyTemp'] = df_train_pre['Heart_Rate']*df_train_pre['Body_Temp']

#create some polynomials for key predictors

df_train_pre['Duration_squared'] = df_train_pre['Duration']**2
df_train_pre['BodyTemp_squared'] = df_train_pre['Body_Temp']**2
df_train_pre['HeartRate_squared'] = df_train_pre['Heart_Rate']**2

#standarize our numerical features
from sklearn.preprocessing import StandardScaler

feature_col = [col for col in df_train_pre.columns if col not in ['Sex', 'id', 'Calories']]

scaler = StandardScaler()
df_train_pre[feature_col] = scaler.fit_transform(df_train_pre[feature_col])


from sklearn.model_selection import train_test_split

X = df_train_pre.drop(['id', 'Calories'], axis=1)
y = df_train_pre['Calories']

X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2,random_state=42)
print(f'Training set shape: {X_train.shape}, Validation set shape: {X_val.shape}')


from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(model, X_train, X_val, y_train, y_val):
    #train / predict 
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)

    #eva. metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    train_r2 = r2_score(y_train, y_pred_train)
    val_r2 = r2_score(y_val, y_pred_val)

    print(f"Training RMSE: {train_rmse:.4f}")
    print(f"Validation RMSE: {val_rmse:.4f}")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Validation R²: {val_r2:.4f}")

    return val_rmse, model, y_pred_val


from sklearn.linear_model import LinearRegression

print("Training Linear Regression Model:")
lr = LinearRegression()
lr_rmse, lr_model, lr_pred = evaluate_model(lr, X_train, X_val, y_train, y_val)


import lightgbm as lgb

print("Training LGBM Model:")
lgbmodel = lgb.LGBMRegressor(n_estimators=200, random_state=42)
lgb_rmse, lgb_model, lgb_pred = evaluate_model(lgbmodel, X_train, X_val, y_train, y_val)

print()

import xgboost as xgb
print("Training XGBOOST Model:")
xgbmodel = xgb.XGBRegressor(n_estimators=200, random_state=42)
xgb_rmse, xgb_model, xgb_pred = evaluate_model(xgbmodel, X_train, X_val, y_train, y_val)


import optuna
from sklearn.model_selection import cross_val_score

def objective_lgb(trial):
    param = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 200)
    }

    model = lgb.LGBMRegressor(**param, random_state=42)
    score = -cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
    ).mean()

    return score      


def objective_xgb(trial):
    param = {  # Fixed indentation here (4 spaces instead of 5)
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'booster': 'gbtree',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 200)
    }
    
    model = xgb.XGBRegressor(**param, random_state=42)
    score = -cross_val_score(
        model, 
        X_train,
        y_train, 
        cv=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    ).mean()
    return score


print("Bayesian Optimization For LightGBM:")
lgb_study = optuna.create_study(direction='minimize')
lgb_study.optimize(objective_lgb, n_trials=20)

print(f'Best Params:\n{lgb_study.best_params}')
print(f'Best RMSE:\n{lgb_study.best_value}')


print("Bayesian Optimization for XGBOOST:")
xgb_study = optuna.create_study(direction='minimize')
xgb_study.optimize(objective_xgb, n_trials=20)

print("\nBest XGBoost parameters:", xgb_study.best_params)
print("Best XGBoost RMSE:", xgb_study.best_value)


print("Training LGBM Model:")
best_lgb = lgb.LGBMRegressor(**lgb_study.best_params, random_state=42)
lgb_rmse, lgb_model, lgb_pred = evaluate_model(best_lgb, X_train, X_val, y_train, y_val)

print()

print("Training XGBOOST Model:")
best_xgb = xgb.XGBRegressor(**xgb_study.best_params, random_state=42)
xgb_rmse, xgb_model, xgb_pred = evaluate_model(best_xgb, X_train, X_val, y_train, y_val)


#1- optimal weighting between models
best_rmse = float('inf')
best_weight = 0
weights = np.arange(0.1, 1.01, 0.01)

for weight in weights:
    ensemble_pred = weight*lgb_pred + (1-weight)*xgb_pred
    ensemble_rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))

    if ensemble_rmse < best_rmse:
        best_rmse = ensemble_rmse
        best_weight = weight

ensemble_pred = best_weight*lgb_pred + (1-best_weight)*xgb_pred
ensemble_r2 = r2_score(y_val, ensemble_pred)

print(f"Best weight for LightGBM is: {best_weight:.2f}\nBest weight for XGBoost is: {(1-best_weight):.2f}")
print(f"Weighted ensemble RMSE: {best_rmse:.4f},  R2: {ensemble_r2:.4f}")


#2- stacking folds
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import Ridge

def get_oof_predictions(model_constructor, X, y, cv=5):
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"  Processing fold {fold+1}/{cv}")
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        
        # Create a new model instance for this fold
        model = model_constructor()
        model.fit(X_train_fold, y_train_fold)
        oof_preds[val_idx] = model.predict(X_val_fold)
    
    return oof_preds

print("Generating oof predictions for stacking")

lgb_constructor = lambda: lgb.LGBMRegressor(**lgb_study.best_params, random_state=42)
xgb_constructor = lambda: xgb.XGBRegressor(**xgb_study.best_params, random_state=42)


print("LightGBM:")
lgb_oof = get_oof_predictions(lgb_constructor, X_train, y_train)
print("XGBoost:")
xgb_oof = get_oof_predictions(xgb_constructor, X_train, y_train)

meta_features = np.column_stack([lgb_oof, xgb_oof])
alphas = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
grid = GridSearchCV(
    Ridge(),
    {'alpha': alphas},
    cv=5,
    scoring='neg_root_mean_squared_error'
)
grid.fit(meta_features, y_train)
best_alpha = grid.best_params_['alpha']
print(f"Best alpha: {best_alpha}")
meta_model = Ridge(alpha=best_alpha)
meta_model.fit(meta_features, y_train)

print(f"Meta-model coefficients: LightGBM = {meta_model.coef_[0]:.4f}, XGBoost = {meta_model.coef_[1]:.4f}")

meta_features_val = np.column_stack([lgb_pred, xgb_pred])
stacked_pred = meta_model.predict(meta_features_val)
stacked_rmse = np.sqrt(mean_squared_error(y_val, stacked_pred))
stacked_r2 = r2_score(y_val, stacked_pred)

print(f"Stacked ensemble RMSE: {stacked_rmse:.4f}, R²: {stacked_r2:.4f}")


#3- average ensemble
avg_pred = (lgb_pred+xgb_pred)/2
avg_rmse = np.sqrt(mean_squared_error(y_val, avg_pred))
avg_r2 = r2_score(y_val, avg_pred)
print(f"Average: RMSE = {avg_rmse:.4f}, R² = {avg_r2:.4f}")


#getting the best method
methods = {
    'average': avg_rmse,
    'weighted': best_rmse,
    'stacked': stacked_rmse
}
best_method = min(methods, key=methods.get)
print(f"Best Method is {best_method} with RMSE {methods[best_method]}")


#Generate predictions
test_df = pd.read_csv('/kaggle/input/f-1-racer-diet-planning/test.csv')
id_col = test_df['id'].copy()

#same preprocessing for our test set
df_test_pre = test_df.copy()


df_test_pre['BMI'] = df_test_pre['Weight']/((df_test_pre['Height']/100)**2)

df_test_pre['Sex'] = df_test_pre['Sex'].map({'female': 0, 'male': 1})

df_test_pre['Duration_HeartRate'] = df_test_pre['Duration']*df_test_pre['Heart_Rate']
df_test_pre['Duration_BodyTemp'] = df_test_pre['Duration']*df_test_pre['Body_Temp']
df_test_pre['Heart_BodyTemp'] = df_test_pre['Heart_Rate']*df_test_pre['Body_Temp']

df_test_pre['Duration_squared'] = df_test_pre['Duration']**2
df_test_pre['BodyTemp_squared'] = df_test_pre['Body_Temp']**2
df_test_pre['HeartRate_squared'] = df_test_pre['Heart_Rate']**2

feature_col = [col for col in df_test_pre.columns if col not in ['Sex', 'id']]
df_test_pre[feature_col] = scaler.transform(df_test_pre[feature_col])

X_test = df_test_pre.drop(['id'], axis=1, errors='ignore')

print(f"Test data preprocessed. Shape: {X_test.shape}")


#generate predictions
lgb_test_pred = lgb_model.predict(X_test)
xgb_test_pred = xgb_model.predict(X_test)

if best_method == "average":
    final_pred = (lgb_test_pred + xgb_test_pred) / 2
    method_description = "Simple Average (50/50)"
    
elif best_method == "weighted":
    final_pred = best_weight * lgb_test_pred + (1-best_weight) * xgb_test_pred
    method_description = f"Weighted (LGB: {best_weight:.2f}, XGB: {1-best_weight:.2f})"
    
else:  # stacked
    meta_features_test = np.column_stack([lgb_test_pred, xgb_test_pred])
    final_pred = meta_model.predict(meta_features_test)
    method_description = f"Stacked (LGB: {meta_model.coef_[0]:.4f}, XGB: {meta_model.coef_[1]:.4f})"

print(f"Using {method_description} for final predictions")
print(f"Min predicted calories: {final_pred.min():.2f}")
print(f"Max predicted calories: {final_pred.max():.2f}")
print(f"Mean predicted calories: {final_pred.mean():.2f}")


submission_df = pd.read_csv('/kaggle/input/f-1-racer-diet-planning/sample_submission.csv')
submission_df['Calories'] = final_pred
submission_df['Calories'] = submission_df['Calories'].clip(0) #no negative pred
submission_df.to_csv('submission.csv', index=False)

print(submission_df.head())

