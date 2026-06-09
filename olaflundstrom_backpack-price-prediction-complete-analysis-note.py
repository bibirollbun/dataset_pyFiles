# 1. Import packages and modules
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import StackingRegressor
import optuna
from optuna.samplers import TPESampler

# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')

# Feature engineering
def create_features(df):
    df = df.copy()
    df['Capacity_per_Compartment'] = df['Weight Capacity (kg)'] / (df['Compartments'] + 1)
    df['Brand_Material'] = df['Brand'] + '_' + df['Material']
    df['Size_Weight_Ratio'] = np.where(df['Size'] == 'Small', df['Weight Capacity (kg)'] * 0.8,
                                      np.where(df['Size'] == 'Medium', df['Weight Capacity (kg)'] * 1.0,
                                               df['Weight Capacity (kg)'] * 1.2))
    return df

train = create_features(train)
test = create_features(test)

# Preprocessing
num_features = ['Compartments', 'Weight Capacity (kg)',
               'Capacity_per_Compartment', 'Size_Weight_Ratio']
cat_features = ['Brand', 'Material', 'Size', 'Laptop Compartment',
               'Waterproof', 'Style', 'Color', 'Brand_Material']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])

# Hyperparameter tuning with Optuna
def objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100)
    }

    scores = []
    kf = KFold(n_splits=5)
    for train_idx, val_idx in kf.split(train):
        X_train, X_val = train.iloc[train_idx], train.iloc[val_idx]
        y_train, y_val = X_train['Price'], X_val['Price']

        train_data = lgb.Dataset(preprocessor.fit_transform(X_train), label=y_train)
        val_data = lgb.Dataset(preprocessor.transform(X_val), label=y_val, reference=train_data)

        model = lgb.train(params, train_data, valid_sets=[val_data], num_boost_round=1000,
                         callbacks=[lgb.early_stopping(50), lgb.log_evaluation(False)])

        scores.append(model.best_score['valid_0']['rmse'])

    return np.mean(scores)

study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=30)

# Best model from Optuna
best_params = study.best_params

# 3-Layer Stacking Model
base_models = [
    ('lgbm1', lgb.LGBMRegressor(**best_params)),
    ('lgbm2', lgb.LGBMRegressor(boosting_type='dart', **best_params)),
    ('catboost', cb.CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=6, verbose=0)),
    ('xgb', xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=5))
]

meta_model = Ridge(alpha=0.5)

stack = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5,
    passthrough=True,
    n_jobs=-1
)

# Train final model
X = train.drop(['id', 'Price'], axis=1)
y = train['Price']
test_X = test.drop('id', axis=1)

stack_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', stack)
])

stack_pipeline.fit(X, y)

# Generate predictions with blending
preds1 = stack_pipeline.predict(test_X)

# Secondary predictions with CatBoost
cat_model = cb.CatBoostRegressor(iterations=1500, learning_rate=0.03, depth=7)
cat_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', cat_model)
])
cat_pipeline.fit(X, y)
preds2 = cat_pipeline.predict(test_X)

# Blend predictions
final_preds = 0.7 * preds1 + 0.3 * preds2

# Create submission
submission = pd.DataFrame({'id': test['id'], 'Price': final_preds})
submission.to_csv('submission.csv', index=False)

