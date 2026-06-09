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


from copy import deepcopy
import numpy as np
import pandas as pd
import gc

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import optuna

# 设置随机种子以保证结果可复现
np.random.seed(42)

# --- 内存优化函数 ---
def reduce_mem_usage(df):
    """ 遍历数据帧的所有列并修改数据类型以减少内存使用 """
    start_mem = df.memory_usage().sum() / 1024**2
    print(f'原始内存占用: {start_mem:.2f} MB')
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object and col_type.name != 'category':
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32) # float16 有时在某些计算中不稳定，这里保守用 float32
                else:
                    df[col] = df[col].astype(np.float32)

    end_mem = df.memory_usage().sum() / 1024**2
    print(f'优化后内存占用: {end_mem:.2f} MB (减少了 {100 * (start_mem - end_mem) / start_mem:.1f}%)')
    return df

# --- 竞赛评价指标 (pAUC) ---
# 参考: kaggle.com/code/metric/isic-pauc-abovetpr
def partial_auc_score(true, preds, min_tpr=0.80):
    v_gt = abs(np.asarray(true) - 1)
    v_pred = 1.0 - np.asarray(preds)
    max_fpr = abs(1 - min_tpr)
    
    partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    # 将比例从 [0.5, 1.0] 变换到 [0.5 * max_fpr**2, max_fpr]
    partial_auc = 0.5 * max_fpr**2 + ((max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5)) * (partial_auc_scaled - 0.5)
    
    return partial_auc

# --- 数据加载 ---
root_folder = '/kaggle/input/isic-2024-challenge'
# 仅读取必要的列可以进一步优化，但为了通用性这里读取全部
train = pd.read_csv(f'{root_folder}/train-metadata.csv', low_memory=False)
test = pd.read_csv(f'{root_folder}/test-metadata.csv', low_memory=False)

# --- 数据预处理与分层 ---
def get_bin(p):
    count = p.count()
    return int(np.floor(np.log10(count)))

# 根据每个病人的图片数量进行分桶，用于分层交叉验证
bins = train.groupby('patient_id')['isic_id'].transform(get_bin)
stratify = train['target'].astype(str) + '_' + bins.astype(str)

nfolds = 5
sgkf = StratifiedGroupKFold(n_splits=nfolds, shuffle=True, random_state=42)
splits = list(sgkf.split(train, stratify, groups=train['patient_id']))

test_indices = test['isic_id']

# --- 缺失值填充 ---
median_age = train['age_approx'].median()
train['age_approx'] = train['age_approx'].fillna(median_age)
test['age_approx'] = test['age_approx'].fillna(median_age)
# 将最小年龄限制为 15 岁
train.loc[train['age_approx'] < 15, 'age_approx'] = 15
test.loc[test['age_approx'] < 15, 'age_approx'] = 15

# 性别缺失填充为女性
train['sex'] = train['sex'].fillna('female')
test['sex'] = test['sex'].fillna('female')

# --- 类别特征编码 ---
categorical_features = ['source', 'sex_male', 'tile_type']
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=np.int32)

# 注意：这里假设列名存在，如果列名不一致需调整
cols_to_encode = ['attribution', 'sex', 'tbp_tile_type']
train[categorical_features] = encoder.fit_transform(train[cols_to_encode])
test[categorical_features] = encoder.transform(test[cols_to_encode])

train[categorical_features] = train[categorical_features].astype('category')
test[categorical_features] = test[categorical_features].astype('category')

# --- 特征工程 ---
def add_numerical_features(df):
    err = 1e-5
    # 颜色特征
    df['hue_contrast'] = (df['tbp_lv_H'] - df['tbp_lv_Hext']).abs()
    df['border_color_hmean'] = df['tbp_lv_norm_border'] * df['tbp_lv_norm_color'] / (df['tbp_lv_norm_border'] + df['tbp_lv_norm_color'] + err)
    df['color_uniformity'] = df['tbp_lv_color_std_mean'] / (df['tbp_lv_radial_color_std_max'] + err)
    df['color_consistency'] = df['tbp_lv_stdL'] / (df['tbp_lv_Lext'] + err)
    df['hue_color_std_interaction'] = df['tbp_lv_H'] * df['tbp_lv_color_std_mean']
    df['color_variance_ratio'] = df['tbp_lv_color_std_mean'] / (df['tbp_lv_stdLExt'] + err)
    df['color_range'] = df['tbp_lv_deltaL'].abs() + df['tbp_lv_deltaA'].abs() + df['tbp_lv_deltaB'].abs()

    # 形状/大小特征
    df['lesion_size_ratio'] = df['tbp_lv_minorAxisMM'] / (df['clin_size_long_diam_mm'] + err)
    df['area_perim_ratio'] = df['tbp_lv_areaMM2'] / (df['tbp_lv_perimeterMM'] + err)
    df['symmetry_perimeter_interaction'] = df['tbp_lv_symm_2axis'] * df['tbp_lv_perimeterMM']

    # 位置特征
    df['lesion_distance'] = np.sqrt(df['tbp_lv_x']**2 + df['tbp_lv_y']**2 + df['tbp_lv_z']**2)
    df['lesion_orientation'] = np.arctan2(df['tbp_lv_y'], df['tbp_lv_x'])
    df['xy'] = np.sqrt(df['tbp_lv_x']**2 + df['tbp_lv_y']**2)
    df['yz'] = np.sqrt(df['tbp_lv_y']**2 + df['tbp_lv_z']**2)
    df['zx'] = np.sqrt(df['tbp_lv_z']**2 + df['tbp_lv_x']**2)

    # 其它交互特征
    df['size_color_contrast_ratio'] = df['clin_size_long_diam_mm'] / (df['tbp_lv_deltaLBnorm'] + err)
    df['diameter_age_interaction'] = np.sqrt(df['clin_size_long_diam_mm']**2 + df['age_approx']**2)
    df['age_normalized_nevi_confidence'] = df['tbp_lv_nevi_confidence'] / (df['age_approx'] + err)
    df['color_asymmetry_index'] = df['tbp_lv_radial_color_std_max'] * df['tbp_lv_symm_2axis']
    df['volume_approximation'] = df['tbp_lv_areaMM2'] * df['lesion_distance']
    df['shape_color_consistency'] = df['tbp_lv_eccentricity'] * df['tbp_lv_color_std_mean']

    # 病人层面的归一化特征 (衡量该病灶与该病人其他病灶的差异)
    norm_cols = ['clin_size_long_diam_mm', 'tbp_lv_A','tbp_lv_Aext', 'tbp_lv_B', 'tbp_lv_Bext', 'tbp_lv_C', 'tbp_lv_Cext','tbp_lv_H', 'tbp_lv_Hext', 'tbp_lv_L', 'tbp_lv_Lext', 'tbp_lv_areaMM2', 'tbp_lv_area_perim_ratio', 'tbp_lv_color_std_mean', 'tbp_lv_deltaA', 'tbp_lv_deltaB', 'tbp_lv_deltaL', 'tbp_lv_deltaLB', 'tbp_lv_deltaLBnorm', 'tbp_lv_eccentricity', 'tbp_lv_minorAxisMM', 'tbp_lv_norm_border', 'tbp_lv_norm_color','tbp_lv_perimeterMM', 'tbp_lv_radial_color_std_max', 'tbp_lv_stdL','tbp_lv_stdLExt', 'tbp_lv_symm_2axis', 'tbp_lv_symm_2axis_angle', 'tbp_lv_x', 'tbp_lv_y', 'tbp_lv_z', 'hue_contrast', 'border_color_hmean', 'color_uniformity', 'color_consistency', 'hue_color_std_interaction', 'color_variance_ratio', 'color_range', 'lesion_size_ratio', 'area_perim_ratio', 'symmetry_perimeter_interaction', 'lesion_distance', 'lesion_orientation', 'xy', 'yz', 'zx', 'size_color_contrast_ratio', 'diameter_age_interaction', 'age_normalized_nevi_confidence', 'color_asymmetry_index', 'volume_approximation', 'shape_color_consistency']
    
    # 过滤掉不在df中的列，防止报错
    norm_cols = [c for c in norm_cols if c in df.columns]
    
    df[[f'patient_norm_{col}' for col in norm_cols]] = df.groupby('patient_id')[norm_cols].transform(lambda g: (g - g.mean()) / (g.std() + err))

    # 病人病灶数量
    df['patient_num_lesions'] = df.groupby('patient_id')['isic_id'].transform('count')

    # 随访时间差
    df['patient_visit_delta'] = df.groupby('patient_id')['age_approx'].transform(lambda g: g - g.min())

print("正在生成特征...")
add_numerical_features(train)
add_numerical_features(test)

# 删除不必要的列
drop_columns = ['isic_id', 'sex', 'anatom_site_general', 'tbp_lv_location', 'tbp_lv_location_simple', 'patient_id', 'image_type', 'tbp_tile_type', 'attribution', 'copyright_license', 'lesion_id', 'iddx_full', 'iddx_1', 'iddx_2', 'iddx_3', 'iddx_4', 'iddx_5', 'mel_mitotic_index', 'mel_thick_mm', 'tbp_lv_dnn_lesion_confidence']
train.drop(columns=drop_columns, inplace=True, errors='ignore')
test.drop(columns=drop_columns, inplace=True, errors='ignore')

# 执行内存优化
train = reduce_mem_usage(train)
test = reduce_mem_usage(test)

print(f'特征数量: {train.shape[1] - 1}')

# --- 模型训练辅助函数 ---

def get_random_samples(X, y, bfrac, mfrac, seed):
    """
    对良性样本进行下采样 (bfrac)，保留所有恶性样本 (mfrac=1.0)
    """
    benign_indices = y[y == 0].sample(frac=bfrac, random_state=seed).index
    malignant_indices = y[y == 1].sample(frac=mfrac, random_state=seed).index
    
    train_indices = benign_indices.append(malignant_indices)
    # 打乱顺序
    train_indices = y[train_indices].sample(frac=1.0, random_state=seed).index

    return train_indices

def train_and_predict(model, X_train, y_train, X_valid, y_valid, nrounds=5):
    """
    训练并预测：进行 nrounds 次下采样训练，取预测平均值
    """
    preds = []
    for idx in range(nrounds):
        # 每次使用不同的种子进行下采样
        train_indices = get_random_samples(X_train, y_train, bfrac=0.1, mfrac=1.0, seed=idx)
        
        model.set_params(random_state=idx)
        model.fit(X_train.loc[train_indices], y_train.loc[train_indices])
        
        pred = model.predict_proba(X_valid)[:, 1]
        preds.append(pred)
        
        # 打印单轮验证分数（可选）
        # score = partial_auc_score(y_valid, pred)
        # print(f'轮次 {idx}: {score:.4f}', end=' | ')
    
    # 聚合预测结果
    return np.mean(np.vstack(preds), axis=0)

def get_cv_score(model, nrounds=3):
    """
    执行交叉验证并输出得分
    """
    scores = []
    preds_oof = np.zeros(train.shape[0])
    
    print(f"开始 {nfolds} 折交叉验证...")
    for fold_idx, (train_idx, valid_idx) in enumerate(splits):
        X_train = train.loc[train_idx].drop(columns=['target'])
        X_valid = train.loc[valid_idx].drop(columns=['target'])
        y_train = train.loc[train_idx, 'target']
        y_valid = train.loc[valid_idx, 'target']
        
        preds = train_and_predict(model, X_train, y_train, X_valid, y_valid, nrounds=nrounds)
        preds_oof[valid_idx] = preds

        score = partial_auc_score(y_valid, preds)
        scores.append(score)
        
        print(f'  第 {fold_idx + 1} 折: pAUC = {score:.4f}')

    score_avg = np.mean(scores)
    score_std = np.std(scores)
    score_oof = partial_auc_score(train['target'], preds_oof)
    print(f'{nfolds} 折交叉验证结果: {score_avg:.4f} ± {score_std:.4f} (OOF: {score_oof:.4f})\n')

    return score_oof, preds_oof

# --- 获取训练好的模型列表 (用于最终预测) ---
def get_fitted_models(model, X_train, y_train, nrounds=5):
    fitted_models = []
    print(f"正在全量数据上训练 {nrounds} 个下采样模型...")
    for idx in range(nrounds):
        train_indices = get_random_samples(X_train, y_train, bfrac=0.1, mfrac=1.0, seed=idx)
    
        model.set_params(random_state=idx)
        model.fit(X_train.loc[train_indices], y_train.loc[train_indices])
        
        fitted_models.append(deepcopy(model))
    
    return fitted_models

def rank(values):
    return pd.Series(values).rank(pct=True).values

# --- 模型参数配置 ---

# XGBoost 参数
base_model_xgb = XGBClassifier(**{
    'tree_method': 'hist',
    'max_depth': 8,
    'n_estimators': 300, 
    'enable_categorical': True,
    'learning_rate': 0.029331, 
    'lambda': 0.698497, 
    'alpha': 0.009640, 
    'subsample': 0.534182, 
    'colsample_bylevel': 0.553859, 
    'scale_pos_weight': 4.1,
    'n_jobs': -1
})

# LightGBM 参数
base_model_lgb = LGBMClassifier(**{
    'objective': 'binary',
    'verbosity': -1,
    'max_depth': 8,
    'n_estimators': 300,
    'bagging_freq': 1,
    'num_leaves': 98, 
    'min_data_in_leaf': 86, 
    'learning_rate': 0.015264, 
    'reg_lambda': 0.340634, 
    'bagging_fraction': 0.833931, 
    'feature_fraction': 0.615383, 
    'scale_pos_weight': 1.7,
    'n_jobs': -1
})

X_train = train.drop(columns=['target'])
y_train = train['target']

# --- (可选) 检查 CV 分数 ---
# 如果为了节省时间，可以注释掉这两行。若要验证模型效果，请取消注释。
# _ = get_cv_score(base_model_xgb, nrounds=3)
# _ = get_cv_score(base_model_lgb, nrounds=3)

# --- 最终训练与预测 ---

# 增加轮数 nrounds=5 以提高最终预测的稳定性
final_rounds = 5

fitted_models_xgb = get_fitted_models(base_model_xgb, X_train, y_train, nrounds=final_rounds)
fitted_models_lgb = get_fitted_models(base_model_lgb, X_train, y_train, nrounds=final_rounds)

print("正在进行推理...")

# XGB 推理
preds_xgb = []
for model in fitted_models_xgb:
    pred = model.predict_proba(test)[:, 1]
    preds_xgb.append(pred)
preds_xgb = np.mean(np.vstack(preds_xgb), axis=0)

# LGBM 推理
preds_lgb = []
for model in fitted_models_lgb:
    pred = model.predict_proba(test)[:, 1]
    preds_lgb.append(pred)
preds_lgb = np.mean(np.vstack(preds_lgb), axis=0)

# 融合 (Rank Averaging)
preds_final = (rank(preds_xgb) + rank(preds_lgb)) / 2

# --- 生成提交文件 ---
submission = pd.DataFrame({
    'isic_id': test_indices,
    'target': preds_final
})

print("预览提交文件:")
print(submission.head())

submission.to_csv('submission.csv', index=False)
print("submission.csv 已保存成功！")

