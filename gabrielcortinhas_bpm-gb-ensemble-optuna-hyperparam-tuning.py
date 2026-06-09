pip install optuna-integration[lightgbm]


import pandas as pd
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt





from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.decomposition import PCA



from scipy.optimize import minimize


import lightgbm as lgb
import xgboost as xgb
import catboost as cb


import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
from optuna.integration import LightGBMPruningCallback, XGBoostPruningCallback, CatBoostPruningCallback
import warnings
warnings.filterwarnings('ignore')

random_state = 7


train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")

original = pd.read_csv("/kaggle/input/bpm-prediction-challenge/Train.csv") # We will wait to see if this improves model performance
test_original = pd.read_csv("/kaggle/input/bpm-prediction-challenge/Test.csv")
submission_original = pd.read_csv("/kaggle/input/bpm-prediction-challenge/Submission.csv")


bpm = train['BeatsPerMinute']
train_df = train.drop('id',axis=1)
test_df = test.drop('id',axis=1)

print(original.isna().sum())


print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Working Test Shape: {test_df.shape}")
print(f"Working Train shape: {train_df.shape}")

train.head()


print(f"Train Missing values: \n {train.isna().sum()}")
print(f"\n Test Missing values: \n {test.isna().sum()}")


print("Data Types: ")
print(train.dtypes)


test.describe()


plt.figure(figsize = (8,4)) # Figure size
# Histogram plot
sns.histplot(train['BeatsPerMinute'],kde=True)
plt.title("Target Distribution (BPM)")
plt.xlabel("BeatsPerMinute")
plt.ylabel("Count")
plt.show()


# Display the correlations between our features and our target using a heatmap
sns.set(font_scale=1.1)
plt.figure(figsize=(20,20))
correlation_train = train_df.corr()
mask = np.triu(correlation_train.corr())

sns.heatmap(correlation_train,
           annot=True,
           fmt='.2f',
           cmap='coolwarm',
           square= True,
           mask=mask,
           linewidths=1,
           cbar=False)


rel_features = [feature for feature in train_df.columns if not feature == "BeatsPerMinute"]
target = "BeatsPerMinute"

sampled_df = train_df.sample(frac = 0.001)
fig, ax = plt.subplots(3,3,figsize=(40,40))

for var, subplot in zip(rel_features,ax.flatten()):
    sns.scatterplot(x=var,y=target, data = sampled_df,ax=subplot,hue=target)


y_sampled = sampled_df.BeatsPerMinute

mutual_df = sampled_df[rel_features]

mutual_info = mutual_info_regression(mutual_df, y_sampled,random_state= random_state)

mutual_info = pd.Series(mutual_info)
mutual_info.index = mutual_df.columns
mutual_info = pd.DataFrame(mutual_info.sort_values(ascending=False),columns = ["Feature_MI"])

mutual_info.style.background_gradient("cool")


"""X = train_aug.drop(target, axis=1)
y = train_aug[target]
X_test = test_df"""

X = train_df.drop(target, axis=1)
y = train_df[target]
X_test = test_df


num_features = X.columns.tolist()

preprocessor = Pipeline([
    ("scaler", StandardScaler()),
    ("power", PowerTransformer())
])



feature_engineering = FeatureUnion([
    ("base", "drop"),
    ("pca", PCA(n_components=5, random_state=random_state))
])

pipeline = Pipeline([
    ("pre", preprocessor),
    ("features", FeatureUnion([
        ("base", "passthrough"),
        ("pca", PCA(n_components=5, random_state=random_state))
    ]))
])



X_transformed = pipeline.fit_transform(X)
X_test_transformed = pipeline.transform(X_test)

pca_features = [f"PCA_{i+1}" for i in range(5)]
all_features = num_features + pca_features


X = pd.DataFrame(X_transformed, columns = all_features, index= X.index)
X_test = pd.DataFrame(X_test_transformed, columns = all_features, index= X_test.index)

print(f"Train shape: {X.shape}")
print(f"Test shape: {X_test.shape}")

print(X.head())


kf = KFold(n_splits=5, shuffle=True, random_state = random_state)

# LightGBM hyperoptimisation
def opt_lgb(dummy):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity':-1,
        'boosting_type': 'gbdt',
        'learning_rate': dummy.suggest_float('learning_rate',0.01,0.2),
        'num_leaves': dummy.suggest_int('num_leaves',20,150),
        'max_depth': dummy.suggest_int('max_depth',3,12),
        'min_data_in_leaf': dummy.suggest_int('min_data_in_leaf',10,100),
        'feature_fraction': dummy.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': dummy.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': dummy.suggest_int('bagging_freq', 1, 10),
        'seed': random_state,
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0
    }
    preds = np.zeros(len(y))
    rmse_list = []
    for train_idx, val_idx in kf.split(X,y):
        train = lgb.Dataset(X.iloc[train_idx],y.iloc[train_idx])
        val = lgb.Dataset(X.iloc[val_idx],y.iloc[val_idx])
        model = lgb.train(params,train,valid_sets=[val],
                         num_boost_round = 1000,
                         callbacks = [lgb.early_stopping(50,verbose=False),
                                     lgb.log_evaluation(100),
                                     LightGBMPruningCallback(dummy,"rmse")]
                        )
        preds[val_idx] = model.predict(X.iloc[val_idx])
        rmse= mean_squared_error(y,preds,squared=False)
        rmse_list.append(rmse)
    return np.mean(rmse_list)
    


# XGBoost hyperoptimsation 

def opt_xgb(dummy):
    params = {
        'objective': 'reg:squarederror',
        'learning_rate': dummy.suggest_float('learning_rate',0.01,0.2),
        'max_depth': dummy.suggest_int('max_depth', 3, 12),
        'subsample': dummy.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': dummy.suggest_float('colsample_bytree', 0.6, 1.0),
        'n_estimators': 1000,
        'random_state': random_state,
        'tree_method': 'gpu_hist',
        'predictor': 'gpu_predictor'
    }
    preds = np.zeros(len(y))
    rmse_list = []
    for train_idx, val_idx in kf.split(X,y):
        model = xgb.XGBRegressor(**params,eval_metric="rmse")
        model.fit(X.iloc[train_idx], y.iloc[train_idx],
                 eval_set = [(X.iloc[val_idx], y.iloc[val_idx])],
                 early_stopping_rounds = 50,
                 verbose=False,
                 callbacks=[XGBoostPruningCallback(dummy,"validation_0-rmse")])
        preds[val_idx] = model.predict(X.iloc[val_idx])
        rmse= mean_squared_error(y,preds,squared=False)
        rmse_list.append(rmse)
    return np.mean(rmse_list)
    


# CatBoost hyperparameter optimsation
def opt_cb(dummy):
    params = {
    'iterations': 1000,
    'learning_rate': dummy.suggest_float('learning_rate', 0.01, 0.2),
    'depth': dummy.suggest_int('depth', 4, 10),
    'l2_leaf_reg': dummy.suggest_float('l2_leaf_reg', 1, 10),
    'random_seed': random_state,
    'loss_function': 'RMSE',
    'verbose': False,
    'task_type': 'GPU'
        
    }
    preds = np.zeros(len(y))
    for train_idx, val_idx in kf.split(X, y):
        model = cb.CatBoostRegressor(**params)
        model.fit(X.iloc[train_idx], y.iloc[train_idx],
                eval_set=(X.iloc[val_idx], y.iloc[val_idx]),
                early_stopping_rounds=50,
                  use_best_model=True,
                 verbose=False,
                 #callbacks= [CatBoostPruningCallback(dummy,"RMSE")]
                 )
        preds[val_idx] = model.predict(X.iloc[val_idx])
        rmse= mean_squared_error(y,preds,squared=False)
        
    return rmse


print("---LightGBM---")
lgb_study = optuna.create_study(direction="minimize",pruner=optuna.pruners.MedianPruner())
lgb_study.optimize(opt_lgb,n_trials=30,show_progress_bar=True)
lgb_best = lgb_study.best_params

print("---XGBoost---")
xgb_study = optuna.create_study(direction="minimize",pruner=optuna.pruners.MedianPruner())
xgb_study.optimize(opt_xgb,n_trials=30,show_progress_bar=True)
xgb_best = xgb_study.best_params

print("---CatBoost---")
cat_study = optuna.create_study(direction="minimize")
cat_study.optimize(opt_cb,n_trials=30,show_progress_bar=True)
cat_best = cat_study.best_params

print("---Best Params--")
print(f"LightGBM Best Params: {lgb_best}")
print(f"XGBoost Best Params: {xgb_best}")
print(f"CatBoost Best Params: {cat_best}")



# LGB study stats
plot_optimization_history(lgb_study).show()
plot_param_importances(lgb_study).show()

# XGBoost study stats
plot_optimization_history(xgb_study).show()
plot_param_importances(xgb_study).show()

# CatBoost study stats
plot_optimization_history(cat_study).show()
plot_param_importances(cat_study).show()



# placeholders for oof and test preds
oof_preds_lgb = np.zeros(len(X))
oof_preds_xgb = np.zeros(len(X))
oof_preds_cb = np.zeros(len(X))
oof_preds_ridge = np.zeros(len(X))

test_preds_lgb = np.zeros(len(X_test))
test_preds_xgb = np.zeros(len(X_test))
test_preds_cb = np.zeros(len(X_test))
test_preds_ridge = np.zeros(len(X_test))


for fold, (train_idx,valid_idx) in enumerate(kf.split(X,y)):
    print(f"\n Fold {fold+1}")

    X_train, X_val = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[valid_idx]
    
    lgb_train = lgb.Dataset(X_train,y_train)
    lgb_val = lgb.Dataset(X_val,y_val,reference=lgb_train)
    
    lgb_model = lgb.train(
        lgb_best,
        lgb_train,
        valid_sets = [lgb_val],
        num_boost_round =1000,
        callbacks = [
            lgb.early_stopping(50),
            lgb.log_evaluation(100)
        ]
    )

    oof_preds_lgb[valid_idx] = lgb_model.predict(X_val)
    test_preds_lgb += lgb_model.predict(X_test)/kf.n_splits

    # XGBoost model
    xgb_model = xgb.XGBRegressor(**xgb_best)
    xgb_model.fit(
        X_train,y_train,
        eval_set=[(X_val,y_val)],
        early_stopping_rounds = 50,
        verbose=100
    )
    oof_preds_xgb[valid_idx] = xgb_model.predict(X_val)
    test_preds_xgb += xgb_model.predict(X_test)/kf.n_splits


    # CatBoost model
    cb_model = cb.CatBoostRegressor(**cat_best)
    cb_model.fit(
        X_train,y_train,
        eval_set = (X_val,y_val),
        early_stopping_rounds=100, verbose=False
    )
    oof_preds_cb[valid_idx]=cb_model.predict(X_val)
    test_preds_cb += cb_model.predict(X_test)/kf.n_splits



# Evaluate our performance 
rmse_lgb = mean_squared_error(y,oof_preds_lgb,squared=False)
rmse_xgb = mean_squared_error(y,oof_preds_xgb,squared=False)
rmse_cb = mean_squared_error(y,oof_preds_cb,squared=False)


print("\nOOF RMSE:")
print(f"LightGBM: {rmse_lgb:.4f}")
print(f"XGBoost: {rmse_xgb:.4f}")
print(f"CatBoost: {rmse_cb:.4f}")


stacked_oof = np.vstack([oof_preds_lgb,oof_preds_xgb,oof_preds_cb])

def blend_rmse(weights):
    weights = np.array(weights)
    blended = np.dot(weights, stacked_oof)
    return mean_squared_error(y,blended, squared=False)

constraints = {"type":"eq",
              "fun": lambda w: np.sum(w)-1}
bounds = [(0,1)] * stacked_oof.shape[0]

first_guess = [1/stacked_oof.shape[0]] * stacked_oof.shape[0]

result = minimize(blend_rmse, first_guess, method="SLSQP", bounds=bounds, constraints = constraints)
best_weights = result.x
best_rmse = result.fun


print(f"Best Weights: LGB={best_weights[0]:.2f}, XGB={best_weights[1]:.2f}, CB={best_weights[2]:.2f}")
print(f"Best Blended OOF RMSE: {best_rmse:.4f}")


test_stack = np.vstack([test_preds_lgb, test_preds_xgb, test_preds_cb])
test_blend = np.dot(best_weights, test_stack)


# Another option - linspace search
"""best_rmse = float("inf")
best_weights = (1/3,1/3,1/3)

for w1 in np.linspace(0,1,101):
    for w2 in np.linspace(0,1-w1,101):
        w3 = 1-w1-w2
        if w3<0:
            continue
        oof_blend = w1*oof_preds_lgb + w2*oof_preds_xgb + w3*oof_preds_cb
        rmse = mean_squared_error(y,oof_blend,squared=False)

        if rmse<best_rmse:
            best_rmse = rmse
            best_weights = (w1,w2,w3)

print(f"\nBest Weights: LGB={best_weights[0]:.2f}, XGB={best_weights[1]:.2f}, CB={best_weights[2]:.2f}")
print(f"Best Blended OOF RMSE: {best_rmse:.4f}")

test_blend = (
    best_weights[0] * test_preds_lgb +
    best_weights[1] * test_preds_xgb +
    best_weights[2] * test_preds_cb
)"""


submission = pd.DataFrame({
    "id": test["id"],
    "BeatsPerMinute": test_blend
})

submission.to_csv("submission.csv",index=False)




