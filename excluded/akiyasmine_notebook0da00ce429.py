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










import pandas as pd
df_train = pd.read_csv("/kaggle/input/dataset/train.csv")
df_test = pd.read_csv("/kaggle/input/dataset/test.csv")
from sklearn.impute import SimpleImputer
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
# 提取需要整合的变量
hla_vars = [
    'hla_high_res_10', 'hla_high_res_6', 'hla_high_res_8',
    'hla_low_res_10', 'hla_low_res_6', 'hla_low_res_8',
    'hla_match_a_high', 'hla_match_a_low',
    'hla_match_b_high', 'hla_match_b_low',
    'hla_match_c_high', 'hla_match_c_low',
    'hla_match_dqb1_high', 'hla_match_dqb1_low',
    'hla_match_drb1_high', 'hla_match_drb1_low'
]

hla_data = df_train[hla_vars]
from sklearn.impute import KNNImputer

# 初始化 KNNImputer
imputer = KNNImputer(n_neighbors=5)  # n_neighbors 是 K 值，可以根据需要调整

# 填补缺失值
hla_data_filled = imputer.fit_transform(hla_data)

# 将填补后的数据转换为 DataFrame
hla_data_filled = pd.DataFrame(hla_data_filled, columns=hla_vars)
print(hla_data_filled.isnull().sum())
# 将填补后的数据合并回原始数据集
df_train_filled = df_train.copy()
df_train_filled[hla_vars] = hla_data_filled

# 检查填补后的数据集
print(df_train_filled.isnull().sum())
# 标准化数据
scaler = StandardScaler()
hla_scaled = scaler.fit_transform(df_train_filled[hla_vars])

# 使用因子分析提取潜在因子
fa = FactorAnalysis(n_components=1)  # 提取 1 个潜在因子
df_train_filled['hla_summary'] = fa.fit_transform(hla_scaled)

# 查看整合后的结果
print(df_train_filled['hla_summary'].describe())
from sklearn.impute import KNNImputer
hla_data_test = df_test[hla_vars]
# 初始化 KNNImputer
imputer = KNNImputer(n_neighbors=5)  # n_neighbors 是 K 值，可以根据需要调整

# 填补缺失值
hla_data_filled = imputer.fit_transform(hla_data)

# 将填补后的数据转换为 DataFrame
hla_data_filled = pd.DataFrame(hla_data_filled, columns=hla_vars)
print(hla_data_filled.isnull().sum())
# 将填补后的数据合并回原始数据集
df_test_filled = df_test.copy()
df_test_filled[hla_vars] = hla_data_filled
# 标准化数据
scaler = StandardScaler()
hla_scaled = scaler.fit_transform(df_test_filled[hla_vars])

# 使用因子分析提取潜在因子
fa = FactorAnalysis(n_components=1)  # 提取 1 个潜在因子
df_test_filled['hla_summary'] = fa.fit_transform(hla_scaled)

# 查看整合后的结果
print(df_test_filled['hla_summary'].describe())

df_train["y"] = df_train.efs_time.values
mx = df_train.loc[df_train.efs==1,"efs_time"].max()
mn = df_train.loc[df_train.efs==0,"efs_time"].min()
df_train.loc[df_train.efs==0,"y"] = df_train.loc[df_train.efs==0,"y"] + mx - mn
df_train.y = df_train.y.rank()
df_train.loc[df_train.efs==0,"y"] += len(df_train)//2
df_train.y = df_train.y / df_train.y.max()

RMV = ["ID","efs","efs_time","y"]
FEATURES = [c for c in df_train.columns if not c in RMV]
print(f"There are {len(FEATURES)} FEATURES: {FEATURES}")
CATS = []
for c in FEATURES:
    if df_train_filled[c].dtype == 'object':  # 直接检测 object 类型
        CATS.append(c)
        df_train_filled[c] = df_train_filled[c].fillna("NAN")
        df_test_filled[c] = df_test_filled[c].fillna("NAN")
combined = pd.concat([df_train_filled,df_test_filled],axis=0,ignore_index=True)
#print("Combined data shape:", combined.shape )

# LABEL ENCODE CATEGORICAL FEATURES
print("We LABEL ENCODE the CATEGORICAL FEATURES: ",end="")
for c in FEATURES:

    # LABEL ENCODE CATEGORICAL AND CONVERT TO INT32 CATEGORY
    if c in CATS:
        print(f"{c}, ",end="")
        combined[c],_ = combined[c].factorize()
        combined[c] -= combined[c].min()
        combined[c] = combined[c].astype("int32")
        combined[c] = combined[c].astype("category")
        
    # REDUCE PRECISION OF NUMERICAL TO 32BIT TO SAVE MEMORY
    else:
        if combined[c].dtype=="float64":
            combined[c] = combined[c].astype("float32")
        if combined[c].dtype=="int64":
            combined[c] = combined[c].astype("int32")
    
df_train_filled = combined.iloc[:len(df_train_filled)].copy()
df_test_filled = combined.iloc[len(df_train_filled):].reset_index(drop=True).copy()
from sklearn.model_selection import KFold
from xgboost import XGBRegressor, XGBClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
df_train_filled["y"] = df_train_filled.efs_time.values
mx = df_train_filled.loc[df_train_filled.efs==1,"efs_time"].max()
mn = df_train_filled.loc[df_train_filled.efs==0,"efs_time"].min()
df_train_filled.loc[df_train.efs==0,"y"] = df_train_filled.loc[df_train.efs==0,"y"] + mx - mn
df_train_filled.y = df_train_filled.y.rank()
df_train_filled.loc[df_train_filled.efs==0,"y"] += len(df_train_filled)//2
df_train_filled.y = df_train_filled.y / df_train_filled.y.max()
# 定义特征列
FEATURES = [
    'dri_score', 'psych_disturb', 'cyto_score', 'diabetes', 'tbi_status', 'arrhythmia', 'graft_type', 
    'vent_hist', 'renal_issue', 'pulm_severe', 'prim_disease_hct', 'cmv_status', 'tce_imm_match', 
    'rituximab', 'prod_type', 'cyto_score_detail', 'conditioning_intensity', 'ethnicity', 'year_hct', 
    'obesity', 'mrd_hct', 'in_vivo_tcd', 'tce_match', 'hepatic_severe', 'donor_age', 'prior_tumor', 
    'peptic_ulcer', 'age_at_hct', 'gvhd_proph', 'rheum_issue', 'sex_match', 'race_group', 
    'comorbidity_score', 'karnofsky_score', 'hepatic_mild', 'tce_div_match', 'donor_related', 
    'melphalan_dose', 'cardiac', 'pulm_moderate', 'hla_summary'
]



# 确保数据已填补缺失值
df_train_filled = df_train_filled.copy()  # 假设 df_train_filled 是已填补缺失值的数据集
df_test_filled = df_test_filled.copy()  # 假设 df_test 是测试集

# 定义交叉验证参数
FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

# 初始化 OOF 和测试集预测
oof_xgb = np.zeros(len(df_train_filled))
pred_xgb = np.zeros(len(df_test))

# 交叉验证训练
for i, (train_index, test_index) in enumerate(kf.split(df_train_filled)):
    print("#" * 25)
    print(f"### Fold {i + 1}")
    print("#" * 25)

    # 划分训练集和验证集
    x_train = df_train_filled.loc[train_index, FEATURES].copy()
    y_train = df_train_filled.loc[train_index, "y"]  # 使用正确的目标变量列名
    x_valid = df_train_filled.loc[test_index, FEATURES].copy()
    y_valid = df_train_filled.loc[test_index, "y"]  # 使用正确的目标变量列名
    x_test = df_test_filled[FEATURES].copy()

    # 初始化 XGBoost 模型
    model_xgb = XGBRegressor(
        
        max_depth=3,
        colsample_bytree=0.5,
        subsample=0.8,
        n_estimators=10_000,
        learning_rate=0.1,
        eval_metric="mae",
        early_stopping_rounds=25,
        objective='reg:logistic',
        enable_categorical=True,
        min_child_weight=5,
        
    )

    # 训练模型
    model_xgb.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],
        verbose=100
    )

    # 输出特征重要性
    feature_importance = pd.DataFrame({
        'Feature': FEATURES,
        'Importance': model_xgb.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    print(f"Fold {i + 1} Feature Importance:\n", feature_importance)

    # 验证集预测
    oof_xgb[test_index] = model_xgb.predict(x_valid)

    # 测试集预测
    pred_xgb += model_xgb.predict(x_test)

    # 输出当前折的验证集性能
    fold_mae = np.mean(np.abs(y_valid - oof_xgb[test_index]))
    print(f"Fold {i + 1} Validation MAE: {fold_mae}")

# 计算平均测试集预测
pred_xgb /= FOLDS

# 输出整体 OOF 性能
oof_mae = np.mean(np.abs(df_train_filled["y"] - oof_xgb))
print(f"Overall OOF MAE: {oof_mae}")

# 将测试集预测结果添加到 df_test_filled
df_test_filled['prediction'] = pred_xgb

# 输出整体 OOF 性能
oof_mae = np.mean(np.abs(df_train_filled["y"] - oof_xgb))
print(f"Overall OOF MAE: {oof_mae}")

# 输出 df_test_filled 的预测结果
print("df_test_filled 的预测结果：")
print(df_test_filled[['prediction']])




!rm -rf /kaggle/working/*







submission = df_test_filled[["ID","prediction"]]
# 保存提交文件
submission.to_csv('/kaggle/working/submission.csv',index=False)



print("Sub shape:",submission.shape)
submission.head()
print(submission)

