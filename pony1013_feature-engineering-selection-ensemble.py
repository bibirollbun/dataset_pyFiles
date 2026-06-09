# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
from sklearn.model_selection import train_test_split 

path = "/kaggle/input/playground-series-s5e5/train.csv"
path_test = "/kaggle/input/playground-series-s5e5/test.csv"
train_df = pd.read_csv(path)
test_df = pd.read_csv(path_test)
ID = test_df['id']
train_df = train_df.drop(columns='id')
test_df = test_df.drop(columns='id')


# 1. 处理分类变量（Sex）
train_df['Sex'] = train_df['Sex'].map({'male':1, 'female':0})

# 2. 特征工程
# 创建BMI特征（Body Mass Index）
train_df['BMI'] = train_df['Weight'] / (train_df['Height']/100)**2

# 创建运动强度特征（心率与持续时间的交互）
train_df['HRI'] = train_df['Heart_Rate'] * train_df['Duration']
# 代谢率相关特征
train_df['BMR'] = (10 * train_df['Weight']) + (6.25 * train_df['Height']) - (5 * train_df['Age']) + (train_df['Sex'] * 166 - 161)
# 解释：当Sex=1时，166*1-161=5；当Sex=0时，166*0-161=-161

# 运动强度分级
train_df['Intensity'] = train_df['Heart_Rate'] / (220 - train_df['Age'])  # 最大心率百分比

# 体重身高比
train_df['Weight_Height_Ratio'] = train_df['Weight'] / train_df['Height']

# 体温与心率的交互
train_df['Temp_HR_Interaction'] = train_df['Body_Temp'] * train_df['Heart_Rate']
train_df['Age_HR_Interaction'] = train_df['Age'] * train_df['Heart_Rate']
train_df['Weight_Duration_Interaction'] = train_df['Weight'] * train_df['Duration']
train_df['Temp_Duration_Interaction'] = train_df['Body_Temp'] * train_df['Duration']

# 心率储备百分比
train_df['HR_Reserve'] = (train_df['Heart_Rate'] - 60) / (220 - train_df['Age'] - 60)

# 体温变化特征
train_df['Temp_Change'] = train_df['Body_Temp'] - 37  # 与正常体温差值

# 最大摄氧量(VO2Max)估算, Balke方程改良式（适用于跑步机测试）
train_df['VO2Max'] = 15.1 + (21.8 * train_df['Duration']) - (0.32 * train_df['Age']) - (1.9 * train_df['Sex'])

# 代谢当量(METs)
train_df['METs'] = train_df['Heart_Rate'] / (220 - train_df['Age']) * 10
train_df['log_duration'] = np.log1p(train_df['Duration'])
# 每分钟能量消耗估算
train_df['Energy_Expenditure'] = (train_df['METs'] * train_df['Weight'] * 3.5) / 200

# 活动因子校准（基于METs计算）
train_df['Activity_Factor'] = 1.2 + (train_df['METs'] * 0.1)
train_df['TDEE'] = train_df['BMR'] * train_df['Activity_Factor']
# 女性使用226-age，男性使用220-age

train_df['Max_HR'] = np.where(train_df['Sex']==0, 226-train_df['Age'], 220-train_df['Age'])
train_df['Intensity_Adj'] = train_df['Heart_Rate'] / train_df['Max_HR']

# 心血管应变指数
train_df['CVSI'] = (train_df['Heart_Rate']/train_df['Max_HR']) + (train_df['Temp_Change']/2.5)
# 3. 异常值处理（示例）
# 过滤不合理BMI值（成人BMI正常范围18.5-24.9）
train_df = train_df[(train_df['BMI'] > 15) & (train_df['BMI'] < 35)]


# 2. 应用所有特征工程转换
# 基本特征
test_df['Sex'] = test_df['Sex'].map({'male':1, 'female':0})
test_df['BMI'] = test_df['Weight'] / (test_df['Height']/100)**2
test_df['HRI'] = test_df['Heart_Rate'] * test_df['Duration']

# 代谢相关特征
test_df['BMR'] = (10 * test_df['Weight']) + (6.25 * test_df['Height']) - (5 * test_df['Age']) + (test_df['Sex'] * 166 - 161)
test_df['Intensity'] = test_df['Heart_Rate'] / (220 - test_df['Age'])
test_df['Weight_Height_Ratio'] = test_df['Weight'] / test_df['Height']

# 交互特征
test_df['Temp_HR_Interaction'] = test_df['Body_Temp'] * test_df['Heart_Rate']
test_df['Age_HR_Interaction'] = test_df['Age'] * test_df['Heart_Rate']
test_df['Weight_Duration_Interaction'] = test_df['Weight'] * test_df['Duration']
test_df['Temp_Duration_Interaction'] = test_df['Body_Temp'] * test_df['Duration']

# 新增高级特征
test_df['HR_Reserve'] = (test_df['Heart_Rate'] - 60) / (220 - test_df['Age'] - 60)
test_df['Temp_Change'] = test_df['Body_Temp'] - 37
test_df['VO2Max_Est'] = (test_df['Heart_Rate'] * test_df['Duration']) / (test_df['Weight'] * 0.01)
test_df['METs'] = test_df['Heart_Rate'] / (220 - test_df['Age']) * 10
test_df['log_duration'] = np.log1p(test_df['Duration'])

# 每分钟能量消耗估算
test_df['Energy_Expenditure'] = (test_df['METs'] * test_df['Weight'] * 3.5) / 200

# 活动因子校准（基于METs计算）
test_df['Activity_Factor'] = 1.2 + (test_df['METs'] * 0.1)
test_df['TDEE'] = test_df['BMR'] * test_df['Activity_Factor']

# 女性使用226-age，男性使用220-age
test_df['Max_HR'] = np.where(test_df['Sex'] == 0, 226 - test_df['Age'], 220 - test_df['Age'])
test_df['Intensity_Adj'] = test_df['Heart_Rate'] / test_df['Max_HR']

# 心血管应变指数
test_df['CVSI'] = (test_df['Heart_Rate'] / test_df['Max_HR']) + (test_df['Temp_Change'] / 2.5)
# 3. 异常值处理（与训练集一致）
# 注意：测试集不应删除行，只应做数值裁剪
test_df['BMI'] = test_df['BMI'].clip(15, 35)  # 裁剪而非过滤

# 4. 其他可能需要的数据保护
# 防止除零错误
test_df['Intensity'] = np.where((220 - test_df['Age']) == 0, 0, test_df['Intensity'])
test_df['HR_Reserve'] = np.where((220 - test_df['Age'] - 60) == 0, 0, test_df['HR_Reserve'])

# 5. 确保所有特征列一致
# 获取训练集有但测试集可能没有的列
missing_cols = set(train_df.columns) - set(test_df.columns)
for col in missing_cols:
    test_df[col] = 0  # 用0填充缺失特征列

# 确保列顺序一致
test_df = test_df[train_df.columns]
y = train_df['Calories']
X = train_df.drop(columns='Calories')

X_test = test_df
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
assert (y_train > 0).all()


print(X_train.head())
print(X_train.info())
print(X_test.info())
print(X_train.describe())
print(y_train.info())


# 1. Initial Filter: Remove Low-Variance Features
var_threshold = VarianceThreshold(threshold=0.02)  # Adjust based on feature scaling
X_train_var = var_threshold.fit_transform(X_train)
selected_features = X_train.columns[var_threshold.get_support()]
print(f"Step 1 of feature selection : After variance threshold: {len(selected_features)} features remaining")


# 2. Correlation Filter (for linear relationships)
corr_threshold = 0.1
correlations = pd.Series(np.corrcoef(X_train_var.T, y_train)[:-1, -1].T, 
                 index=selected_features)
significant_corr = correlations[abs(correlations) > corr_threshold].index.tolist()
print(f"Step 2 of feature selection : Features with |corr| > {corr_threshold}: {len(significant_corr)}")


# 3. Tree-Based Importance Consensus
models = {
    "XGBoost": XGBRegressor(random_state=42, verbose=2),
    "LightGBM": LGBMRegressor(random_state=42, verbose=1),
    "RF": RandomForestRegressor(n_estimators=10, random_state=42, verbose=1)
}

importance_data = []
for name, model in models.items():
    m = clone(model).fit(X_train_var, y_train)
    if hasattr(m, 'feature_importances_'):
        imp = m.feature_importances_
    else:  # LightGBM uses different attribute
        imp = m.booster_.feature_importance(importance_type='gain')
    
    # Normalize importance scores
    imp = (imp - imp.min()) / (imp.max() - imp.min())
    importance_data.append(pd.Series(imp, index=selected_features, name=name))

# Create consensus importance
importance_df = pd.concat(importance_data, axis=1)
importance_df['consensus'] = importance_df.mean(axis=1)
importance_df = importance_df.sort_values('consensus', ascending=False)




# 4. Permutation Importance Validation (on validation set)
base_model = XGBRegressor(random_state=42).fit(X_train_var, y_train)
perm_importance = permutation_importance(
    base_model, X_val[selected_features], y_val, 
    n_repeats=3, random_state=42, n_jobs=-1
)
# Combine all metrics
feature_selection_report = pd.DataFrame({
    'feature': selected_features,
    'correlation': correlations,
    'xgb_imp': importance_df['XGBoost'],
    'lgbm_imp': importance_df['LightGBM'],
    'rf_imp': importance_df['RF'],
    'perm_importance': perm_importance.importances_mean
})
print("----------------------------- Feature selection report -------------------------------")
print(feature_selection_report)


# 5. Final Selection Criteria (adjust thresholds as needed)
final_features = feature_selection_report[
    ((feature_selection_report['perm_importance'] > 0.005) |  # Has validation impact
     (feature_selection_report['correlation'].abs() > corr_threshold)) &
    ((feature_selection_report[['xgb_imp', 'lgbm_imp', 'rf_imp']] > 0.02).any(axis=1))  # Important to any model
]['feature'].tolist()

print(f"\nFinal selected features ({len(final_features)}):")
print(final_features)

# Visualization
plt.figure(figsize=(12, 8))
for idx, metric in enumerate(['correlation', 'xgb_imp', 'lgbm_imp', 'rf_imp', 'perm_importance']):
    plt.subplot(2, 3, idx+1)
    plt.barh(feature_selection_report['feature'][:15], 
             feature_selection_report[metric][:15])
    plt.title(metric.capitalize())
plt.tight_layout()
plt.show()

# Return filtered datasets
X_train_selected = X_train[final_features]
X_test_selected = X_test[final_features]
X_val_selected = X_val[final_features]


# Define the hyperparameters
params = {

    'n_estimators': 1500,
    'learning_rate': 0.01,
    'max_depth': 6,
    
    'subsample': 0.95,
    'colsample_bytree': 0.9,
    'reg_alpha': 0.1,
    'reg_lambda': 0.3,
    'min_child_weight': 5,
    'gamma': 0.2,
    
    'objective': 'reg:squaredlogerror',
    'eval_metric': 'rmsle',
    'tree_method': 'hist',
    'importance_type': 'gain',
    'single_precision_histogram': False,
    'n_jobs': -1,
    'random_state': 42,
    'max_bin' : 256,
    
    'device': 'gpu',
    'grow_policy': 'depthwise',
    'sampling_method': 'uniform',
    'booster': 'gbtree'
}

# Initialize the XGBoost model with the defined hyperparameters
xgb_model = XGBRegressor(**params)

# Fit the model on the training data
xgb_model.fit(X_train_selected, y_train, verbose=100,
              eval_set=[(X_train_selected, y_train), (X_val_selected, y_val)])


features = X_train_selected.columns
importance = xgb_model.feature_importances_
print("----------------------- Feature importance table ------------------------------------")

feature_importance = pd.DataFrame(
    {
        'features' : features,
        'feature_importance' : importance
    }
).sort_values('feature_importance', ascending=False)

print(feature_importance)
selected_features = feature_importance[feature_importance['feature_importance'] > 0.001]['features']
print("------------------------- Selected features -----------------------------------------")
print(selected_features)
X_train = X_train[selected_features]
X_test = X_test[selected_features]
X_val = X_val[selected_features]


# ======================
# 1. Define Models with Parameters
# ======================
models = {
    "xgb": XGBRegressor(
        objective="reg:squaredlogerror",
        eval_metric="rmsle",
        learning_rate=0.01,
        max_depth=9,
        subsample=0.9,
        colsample_bytree=0.8,
        n_estimators=1000,
        reg_lambda=0.5,
        random_state=42,
        device='gpu'
    ),
    "lgbm": LGBMRegressor(
        boosting_type="gbdt",
        objective="regression",
        num_leaves=31,
        learning_rate=0.01,
        n_estimators=2000,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=0.5,
        random_state=42,
        device='gpu'
    ),
    "rf": RandomForestRegressor(
        n_estimators=1000,
        max_depth=10,
        max_features="sqrt",
        min_samples_split=5,
        bootstrap=True,
        random_state=42
    )
}


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

# Custom RMSLE metric
def rmsle(y_true, y_pred):
    y_pred = np.clip(y_pred, a_min=0, a_max=None)  # Ensure non-negative predictions
    return np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred))**2))

def ensemble_kfold_predict(X_train, y_train, X_val, X_test, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_predictions = np.zeros(X_train.shape[0])
    val_preds = np.zeros((n_splits, X_val.shape[0]))
    test_preds = np.zeros((n_splits, X_test.shape[0]))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"Processing Fold {fold+1}/{n_splits}")
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_ho, y_ho = X_train.iloc[val_idx], y_train.iloc[val_idx]
        
        # Dictionary to store model predictions
        fold_preds = {"val": [], "test": [], "oof": []}
        
        # Train and predict with each model
        for name, model in models.items():
            model.fit(X_tr, y_tr)
            
            # Get predictions for all sets
            oof_pred = model.predict(X_ho)
            val_pred = model.predict(X_val)
            test_pred = model.predict(X_test)
            
            fold_preds["oof"].append(oof_pred)
            fold_preds["val"].append(val_pred)
            fold_preds["test"].append(test_pred)
        
        # Average predictions across models
        oof_predictions[val_idx] = np.mean(fold_preds["oof"], axis=0)
        val_preds[fold] = np.mean(fold_preds["val"], axis=0)
        test_preds[fold] = np.mean(fold_preds["test"], axis=0)
    
    # Calculate final metrics and predictions
    final_val_preds = np.mean(val_preds, axis=0)
    final_test_preds = np.mean(test_preds, axis=0)
    
    return {
        "oof_rmsle": rmsle(y_train, oof_predictions),
        "val_rmsle": rmsle(y_val, final_val_preds),
        "val_predictions": final_val_preds,
        "test_predictions": final_test_preds
    }

y_pred = ensemble_kfold_predict(
    X_train, y_train, X_val, X_test, n_splits=3
)




result = y_pred['test_predictions']
print(y_pred)
submission = pd.DataFrame(
    {
        'id' : ID,
        'Calories' : result
    }
)
submission.to_csv("/kaggle/working/submission.csv", index=False)

