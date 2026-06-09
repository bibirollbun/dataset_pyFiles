import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import VotingRegressor

from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor
import optuna

from matplotlib import pyplot as plt
import seaborn as sns


train_data = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')

train_df = train_data.copy()
train_df.head()


train_df.info()


train_df.isna().sum()


train_df.describe().T


cat_cols = train_df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
num_cols = train_df.drop(columns=['id']).select_dtypes(include=["int64","float64"]).columns.tolist()
for col in train_df.select_dtypes(include=['bool']).columns:
    train_df[col] = train_df[col].astype(str)


for col in num_cols:
    plt.figure(figsize=(6,4));
    sns.histplot(data=train_df, x=col, kde=True);
    plt.title(f'Distribution of {col}');
    plt.show();


for col in cat_cols:
    plt.figure(figsize=(10,4))
    sns.countplot(data=train_df, x=col)
    plt.title(f'Countplot of {col}')
    plt.xticks(rotation=45)
    plt.show()


train_df.head(1)


train_df = train_data.copy()
X = train_df.drop(columns=['accident_risk', 'id'])
y = train_df[['accident_risk']]

cat_features = X.select_dtypes(include=['object', 'bool']).columns.tolist()
num_features = X.select_dtypes(include=['int', 'float']).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


numeric_transformer = Pipeline([('scaler', StandardScaler())])

categorical_transformer = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, num_features),
    ('cat', categorical_transformer, cat_features)])


def objective(trial):
    # LightGBM parameters
    lgb_params = {
        'n_estimators': trial.suggest_int('lgb_n_estimators', 300, 1000),
        'learning_rate': trial.suggest_float('lgb_learning_rate', 0.01, 0.2, log=True),
        'num_leaves': trial.suggest_int('lgb_num_leaves', 20, 80),
        'max_depth': trial.suggest_int('lgb_max_depth', 3, 10),
        'subsample': trial.suggest_float('lgb_subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('lgb_colsample_bytree', 0.6, 1.0),
        'reg_lambda': trial.suggest_float('lgb_reg_lambda', 1e-3, 10.0, log=True),
        'random_state': 42,
        'n_jobs': -1,
        'verbosity' : -1
    }

    # CatBoost parameters
    cat_params = {
        'depth': trial.suggest_int('cat_depth', 4, 10),
        'learning_rate': trial.suggest_float('cat_learning_rate', 0.01, 0.2, log=True),
        'iterations': trial.suggest_int('cat_iterations', 300, 1000),
        'l2_leaf_reg': trial.suggest_float('cat_l2_leaf_reg', 1, 10),
        'verbose': 0,
        'random_state': 42
    }
    # Base models
    lgb_model = LGBMRegressor(**lgb_params)
    cat_model = CatBoostRegressor(**cat_params)

    # Ensemble weights
    w_cat = trial.suggest_float("w_cat", 0.0, 1.0)
    w_lgb = 1 - w_cat


    # Ensemble
    voter = VotingRegressor(
        estimators=[('cat', cat_model), ('lgb', lgb_model)],
        weights=[w_cat, w_lgb],
        n_jobs=-1
    )

    # Pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('ensemble', voter)
    ])
    
    # Fit + Evaluate
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return rmse


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=25, show_progress_bar=True)


best = study.best_params
best_w_cat = best["w_cat"]
best_w_lgb = 1 - best_w_cat


cat_final = CatBoostRegressor(
    depth=best["cat_depth"],
    learning_rate=best["cat_learning_rate"],
    iterations=best["cat_iterations"],
    verbose=0,
    random_state=42
)

lgb_final = LGBMRegressor(
    n_estimators=best["lgb_n_estimators"],
    learning_rate=best["lgb_learning_rate"],
    num_leaves=best["lgb_num_leaves"],
    subsample=best["lgb_subsample"],
    colsample_bytree=best["lgb_colsample_bytree"],
    random_state=42
)

final_voter = VotingRegressor(
    estimators=[('cat', cat_final), ('lgb', lgb_final)],
    weights=[best_w_cat, best_w_lgb],
    n_jobs=-1
)

final_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('ensemble', final_voter)
])



final_pipeline.fit(X, y)
final_preds = final_pipeline.predict(X_test)
final_rmse = np.sqrt(mean_squared_error(y_test, final_preds))

print(f"\n✅ Final Ensemble RMSE: {final_rmse:.4f}")
print(f"CatBoost Weight: {best_w_cat:.3f}, LightGBM Weight: {best_w_lgb:.3f}")


test_df = test_data.copy()
submission = pd.DataFrame()
submission['id'] = test_df['id']
predicts = final_pipeline.predict(test_df.drop(columns=['id']))
submission['accident_risk'] = predicts
submission.to_csv('submission.csv', index=False)

