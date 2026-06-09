import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import GridSearchCV
import optuna
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVC
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, median_absolute_error
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import KFold
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import catboost as cb
from scipy.optimize import minimize
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from cuml.preprocessing import TargetEncoder


train_data = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
train_ex_data = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")


train_data.drop(columns=['id'],inplace=True)
test_data.drop(columns=['id'],inplace=True)
train_ex_data.drop(columns=['id'],inplace=True)


train_data = pd.concat([train_data,  train_ex_data], axis=0).reset_index(drop=True)


# Define imputation strategies
categorical_features = ["Brand", "Material", "Size", "Laptop Compartment", "Waterproof", "Style", "Color"]
numerical_features = ["Weight Capacity (kg)"]

# Fill categorical missing values with mode (most frequent value)
for col in categorical_features:
    train_data[col].fillna('missing', inplace=True)
    test_data[col].fillna('missing', inplace=True)

# Fill numerical missing values with median
for col in numerical_features:
    train_data[col].fillna(train_data[col].median(), inplace=True)
    test_data[col].fillna(test_data[col].median(), inplace=True)


TE = TargetEncoder(n_folds=25, smooth=20, split_method='random', stat='mean')
features = test_data.columns.tolist()
for col in features:
    TE.fit(train_data[col], train_data['Price'])
    train_data[col] = TE.transform(train_data[col])
    test_data[col] = TE.transform(test_data[col])


y = train_data['Price']
X = train_data.drop(['Price'], axis=1)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

def explore_boosting_hyperparam_vs_rmse(model, X_train, y_train, X_test, y_test, param_name, param_values, tuned_params=None, early_stopping_rounds=50):
    if tuned_params is None:
        tuned_params = {}

    tuned_params = tuned_params.copy()
    if param_name in tuned_params:
        del tuned_params[param_name]

    rmse_values = []


    for val in param_values:
        params = {param_name: val, **tuned_params}
        model_instance = model.__class__(**params)

        if hasattr(model_instance, "fit"):
            model_instance.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)]            )
        else:
            raise AttributeError("must include 'fit' function")

        y_pred = model_instance.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rmse_values.append(rmse)
        print(f"{param_name}: {val}, RMSE: {rmse:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(param_values, rmse_values, marker='o', linestyle='-')
    plt.title(f"{param_name} vs RMSE")
    plt.xlabel(param_name)
    plt.ylabel("RMSE")
    plt.grid(True)
    plt.show()

    return rmse_values


random_forest_params = {
    'n_estimators': 360, 
    'max_depth': 7,      
    'min_samples_split': 2, 
    'min_samples_leaf': 1,   
    'max_features': 'sqrt', 
    'random_state': 42,      
    'n_jobs': -1            
}

xgbm_params = {'tree_method': 'hist',
                         'n_estimators': 2000,
                         'objective': 'reg:squarederror',
                         'random_state': 42,
                         'enable_categorical': True,
                         'verbosity': 0,
                         'early_stopping_rounds': 50,
                         'eval_metric': 'rmse',
                         'booster': 'gbtree',
                         'max_depth':6,
                         'min_child_weight': 13,
                         'subsample': 0.6791128709665586,
                         'reg_alpha': 0.0912837368699186,
                         'reg_lambda': 0.8697948615643227,
                         'colsample_bytree': 0.9592753750165163,
                         'n_jobs': -1,
                         'learning_rate': 0.01,
                         'max_bin': 8000,
                         "device": "cuda",
                       }
lgbm_params = {'random_state': 42,
                'early_stopping_round': 50,
                'verbose': -1,
                'boosting_type': 'gbdt',
                'n_estimators': 2000,
                'eval_metric': 'rmse',
                'objective': 'regression_l2',
                'max_depth': 4,
                'num_leaves': 7,
                'min_child_samples': 2,
                'min_child_weight': 2,
                'colsample_bytree': 0.4759506289207658, 
                'reg_alpha': 0.28461417683987383, 
                'reg_lambda': 0.6555944495127437,
                'max_bin': 255,
                'learning_rate': 0.01,
                'device': 'gpu' 
              }

catboost_params =  {'verbose': 0,
               'random_state': 42,
               'early_stopping_rounds': 50,
               'eval_metric': "RMSE",
               'n_estimators' : 3000,
               'objective': 'RMSE', 
               'depth': 3,
               'min_data_in_leaf': 20,
               'l2_leaf_reg': 0.3349374242775052,
               'bagging_temperature': 0.8315027960954179, 
               'random_strength': 0.309798135191685,
               'learning_rate': 0.01,
               'max_bin': 8000,
               'bootstrap_type': 'Poisson',
               "task_type": "GPU",
            }


models = {
        #"LightGBM": (LGBMRegressor, lgbm_params),
       # "CatBoost": (CatBoostRegressor, catboost_params),
    "XGBoost": (XGBRegressor, xgbm_params),
    #"RandomForest": (RandomForestRegressor, random_forest_params),
}


X_train , X_val , y_train , y_val = train_test_split(X, y, test_size=0.2, random_state=42)


xgb_model = XGBRegressor(tree_method="hist") 
explore_boosting_hyperparam_vs_rmse(
    xgb_model, X_train, y_train, X_val, y_val, 
    param_name='max_depth', 
    param_values=list(range(4, 9, 1)), 
    tuned_params=xgbm_params, 
    early_stopping_rounds=50
)


kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = {model_name: {"rmse_scores": [], "mae_scores": [], "test_preds": np.zeros(len(test_data))} for model_name in models.keys()}

# 手动实现 5 折交叉验证
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"Fold {fold + 1}")
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    for model_name, (model_class, params) in models.items():
        print(f"Training {model_name} on Fold {fold + 1}...")

        # 初始化模型
        model = model_class(**params)

        # 特殊处理 RandomForest（不支持 early_stopping_rounds 和 eval_set）
        if model_name == "RandomForest":
            model.fit(X_train, y_train)
        elif model_name == "LightGBM":
            model = model_class(**params)
        

        
            # 训练模型
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='rmse'
            )
        else:
            # 其他模型支持 early_stopping_rounds 和 eval_set
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=0  # 禁用日志输出
            )

        # 验证集预测
        val_preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        mae = mean_absolute_error(y_val, val_preds)

        # 存储验证集分数
        results[model_name]["rmse_scores"].append(rmse)
        results[model_name]["mae_scores"].append(mae)

        print(f"{model_name} Fold {fold + 1} RMSE: {rmse:.4f}, MAE: {mae:.4f}")

        # 测试集预测并取平均
        results[model_name]["test_preds"] += model.predict(test_data) / kf.get_n_splits()

# 对所有模型的测试集预测结果再取平均
final_test_preds = np.zeros(len(test_data))
for model_name in models.keys():
    final_test_preds += results[model_name]["test_preds"] / len(models)

# 输出每种模型的平均 RMSE 和 MAE
for model_name, metrics in results.items():
    avg_rmse = np.mean(metrics["rmse_scores"])
    avg_mae = np.mean(metrics["mae_scores"])
    print(f"{model_name} Mean RMSE: {avg_rmse:.4f}, Mean MAE: {avg_mae:.4f}")

# 输出最终的测试集预测结果
print("Final test predictions (averaged across models and folds):")
print(final_test_preds)


submission = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")
submission['Price'] = final_test_preds
submission.to_csv('submission.csv', index=False)




