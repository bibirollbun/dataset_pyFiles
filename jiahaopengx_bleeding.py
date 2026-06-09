!pip install /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz
!pip install /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl

import warnings
from pathlib import Path
warnings.filterwarnings('ignore')
import numpy as np
import polars as pl
import pandas as pd
import plotly.colors as pc
import plotly.express as px
import plotly.graph_objects as go
import lightgbm as lgb
from metric import score
from scipy.stats import rankdata 
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from lifelines import CoxPHFitter, KaplanMeierFitter, NelsonAalenFitter
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

import plotly.io as pio
pio.renderers.default = 'iframe'
pd.options.display.max_columns = None


# Cell 1: Update CFG class
class CFG:
    train_path = Path('/kaggle/input/equity-post-HCT-survival-predictions/train.csv')
    test_path = Path('/kaggle/input/equity-post-HCT-survival-predictions/test.csv')
    subm_path = Path('/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv')
    colorscale = 'Sunset'
    color = '#EADDCA'
    batch_size = 32768
    early_stop = 1300
    penalizer = 0.01
    n_splits = 10

    # Original CatBoost params
    ctb_params = {
        'loss_function': 'RMSE',
        'learning_rate': 0.03,
        'random_state': 42,
        'task_type': 'CPU',
        'num_trees': 6000,
        'subsample': 0.85,
        'reg_lambda': 8.0,
        'depth': 8
    }

    # Original LightGBM params
    lgb_params = {
        'objective': 'regression',
        'min_child_samples': 20,
        'num_iterations': 6000,
        'learning_rate': 0.01,
        'extra_trees': True,
        'reg_lambda': 3.0,
        'reg_alpha': 0.1,
        'num_leaves': 64,
        'metric': 'rmse',
        'max_depth': 10,
        'device': 'cpu',
        'max_bin': 255,
        'verbose': -1,
        'seed': 42
    }

    # Cox model params
    cox1_params = {
        'grow_policy': 'Depthwise',
        'min_child_samples': 8,
        'loss_function': 'Cox',
        'learning_rate': 0.02,
        'random_state': 42,
        'task_type': 'CPU',
        'num_trees': 6000,
        'subsample': 0.6,  
        'reg_lambda': 8.0,
        'depth': 8,
    }

    cox2_params = {
        'grow_policy': 'Lossguide',
        'loss_function': 'Cox',
        'learning_rate': 0.03,
        'random_state': 42,
        'task_type': 'CPU',
        'num_trees': 6000,
        'subsample': 0.6,  
        'reg_lambda': 8.0,
        'num_leaves': 32,
        'depth': 8,
    }

    cox3_params = {
        'grow_policy': 'Depthwise',
        'min_child_samples': 16,
        'loss_function': 'Cox',
        'learning_rate': 0.01,
        'random_state': 42,
        'task_type': 'CPU',
        'num_trees': 7000,
        'subsample': 0.5,  
        'reg_lambda': 6.0,
        'depth': 10,
    }


class FE:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        # self.scaler = StandardScaler()

    def load_data(self, path):
        return pl.read_csv(path, batch_size=self.batch_size)

    def recalculate_hla_sums(self, df):     
        df = df.with_columns(
            # 添加一列表示每行的缺失值数量
            #(pl.sum_horizontal(pl.all().is_null()).alias("missing_count")),
            (pl.col("hla_match_a_low").fill_null(0) + pl.col("hla_match_b_low").fill_null(0) + 
             pl.col("hla_match_drb1_high").fill_null(0)).alias("hla_nmdp_6"),
            
            (pl.col("hla_match_a_low").fill_null(0) + pl.col("hla_match_b_low").fill_null(0) + 
             pl.col("hla_match_drb1_low").fill_null(0)).alias("hla_low_res_6"),
            
            (pl.col("hla_match_a_high").fill_null(0) + pl.col("hla_match_b_high").fill_null(0) + 
             pl.col("hla_match_drb1_high").fill_null(0)).alias("hla_high_res_6"),
            
            (pl.col("hla_match_a_low").fill_null(0) + pl.col("hla_match_b_low").fill_null(0) + 
             pl.col("hla_match_c_low").fill_null(0) + pl.col("hla_match_drb1_low").fill_null(0)
            ).alias("hla_low_res_8"),
            
            (pl.col("hla_match_a_high").fill_null(0) + pl.col("hla_match_b_high").fill_null(0) + 
             pl.col("hla_match_c_high").fill_null(0) + pl.col("hla_match_drb1_high").fill_null(0)
            ).alias("hla_high_res_8"),
            
            (pl.col("hla_match_a_low").fill_null(0) + pl.col("hla_match_b_low").fill_null(0) + 
             pl.col("hla_match_c_low").fill_null(0) + pl.col("hla_match_drb1_low").fill_null(0) +
             pl.col("hla_match_dqb1_low").fill_null(0)).alias("hla_low_res_10"),
            
            (pl.col("hla_match_a_high").fill_null(0) + pl.col("hla_match_b_high").fill_null(0) + 
             pl.col("hla_match_c_high").fill_null(0) + pl.col("hla_match_drb1_high").fill_null(0) +
             pl.col("hla_match_dqb1_high").fill_null(0)).alias("hla_high_res_10"),
            (0.4 * pl.col("hla_match_a_low").fill_null(0) +
             0.4 * pl.col("hla_match_b_low").fill_null(0) +
             0.2 * pl.col("hla_match_drb1_high").fill_null(0)).alias("nmdp_trio_score"),
            
            # pl.max_horizontal(
            #     pl.col("hla_match_dqb1_high").fill_null(0),  # 填充空缺值为 0
            #     pl.col("hla_match_drb1_high").fill_null(0)  # 填充空缺值为 0
            # ).alias("class2_combo")  # 将结果列命名为 class2_combo
        #     (pl.col("hla_match_dqb1_high").fill_null(0) - pl.col("hla_match_dqb1_low").fill_null(0)).abs().alias("hla_match_dqb1_hl"),
        )
        return df

    def cast_datatypes(self, df):
        num_cols = [
            # 'hla_high_res_8', 'hla_low_res_8', 'hla_high_res_6',
            # 'hla_low_res_6', 'hla_high_res_10', 'hla_low_res_10',
            # 'hla_match_dqb1_high', 'hla_match_dqb1_low',
            # 'hla_match_drb1_high', 'hla_match_drb1_low',
            # 'hla_nmdp_6',  'hla_match_a_high',
            # 'hla_match_a_low', 'hla_match_b_high', 'hla_match_b_low',
            # 'hla_match_c_high', 'hla_match_c_low', 
            'donor_age','year_hct',
            'age_at_hct', 'comorbidity_score', 'karnofsky_score',
            'efs', 'efs_time','nmdp_trio_score'
        ]

        for col in df.columns:
            if col in num_cols:
                df = df.with_columns(pl.col(col).fill_null(-1).cast(pl.Float32))  

            else:
                df = df.with_columns(pl.col(col).fill_null('Unknown').cast(pl.String))  

        return df.with_columns(pl.col('ID').cast(pl.Int32))

    def info(self, df):     
        print(f'\nShape of dataframe: {df.shape}')    
        mem = df.memory_usage().sum() / 1024**2
        print('Memory usage: {:.2f} MB\n'.format(mem))
        display(df.head())

    def apply_fe(self, path):
        df = self.load_data(path)
        df = self.recalculate_hla_sums(df)
        df = self.cast_datatypes(df)
        df = df.to_pandas()
        self.info(df)
        
        cat_cols = [col for col in df.columns if df[col].dtype == pl.String]
        return df, cat_cols


fe = FE(CFG.batch_size)
train_data, cat_cols = fe.apply_fe(CFG.train_path)
test_data, _ = fe.apply_fe(CFG.test_path)


class MD:
    def __init__(self, early_stop, penalizer, n_splits, color):
        self.early_stop = early_stop
        self.penalizer = penalizer
        self.n_splits = n_splits
        self.color = color

    def create_target1(self, data, cat_cols):
        cph_data = pd.get_dummies(data, columns=cat_cols, drop_first=True)
        cph = CoxPHFitter(penalizer=self.penalizer)
        cph.fit(cph_data, duration_col='efs_time', event_col='efs')
        data['target1'] = cph.predict_partial_hazard(cph_data)
        return data

    def create_target2(self, data):
        kmf = KaplanMeierFitter()
        kmf.fit(durations=data['efs_time'], event_observed=data['efs'])
        data['target2'] = kmf.survival_function_at_times(data['efs_time']).values
        return data

    def create_target3(self, data):
        naf = NelsonAalenFitter()
        naf.fit(durations=data['efs_time'], event_observed=data['efs'])
        data['target3'] = naf.cumulative_hazard_at_times(data['efs_time']).values
        data['target3'] = data['target3'] * -1
        return data

    def create_target4(self, data):
        data['target4'] = data.efs_time.copy()
        data.loc[data.efs == 0, 'target4'] *= -1
        return data

    def train_model(self, data, cat_cols, params, target, title):
        for col in cat_cols:
            data[col] = data[col].astype('category')
            
        X = data.drop(['ID', 'efs', 'efs_time', 'target1', 'target2', 'target3', 'target4'], axis=1)
        y = data[target]
        
        models, fold_scores = [], []
        
        # Use KFold 
        cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        
        oof_preds = np.zeros(len(X))
        
        for fold, (train_index, valid_index) in enumerate(cv.split(X)):
            X_train = X.iloc[train_index]
            X_valid = X.iloc[valid_index]
            y_train = y.iloc[train_index]
            y_valid = y.iloc[valid_index]
            
            if title.startswith('LightGBM'):
                model = lgb.LGBMRegressor(**params)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_valid, y_valid)],
                    eval_metric='rmse',
                    callbacks=[
                        lgb.early_stopping(self.early_stop, verbose=0),
                        lgb.log_evaluation(0)
                    ]
                )
                
            elif title.startswith('CatBoost'):
                model = CatBoostRegressor(**params, verbose=0, cat_features=cat_cols)
                model.fit(
                    X_train, y_train,
                    eval_set=(X_valid, y_valid),
                    early_stopping_rounds=self.early_stop,
                    verbose=0
                )
                
            models.append(model)
            oof_preds[valid_index] = model.predict(X_valid)
            
            y_true_fold = data.iloc[valid_index][['ID', 'efs', 'efs_time', 'race_group']].copy()
            y_pred_fold = data.iloc[valid_index][['ID']].copy()
            y_pred_fold['prediction'] = oof_preds[valid_index]
            
            fold_score = score(y_true_fold, y_pred_fold, 'ID')
            fold_scores.append(fold_score)
        
        y_true = data[['ID', 'efs', 'efs_time', 'race_group']].copy()
        y_pred = data[['ID']].copy()
        y_pred['prediction'] = oof_preds
        
        c_index_score = score(y_true.copy(), y_pred.copy(), 'ID')
        if target == 'target1':
            t = 'Cox Target'
        elif target == 'target2':
            t = 'Kaplan-Meier Target'
        elif target == 'target3':
            t = 'Nelson-Aalen Target'
        else:
            t = 'Cox Loss'
        print(f'\nOverall C-Index for {title} {t}: {c_index_score:.6f}\n')
        
        return models, oof_preds

    def valid_test(self, data, model_dir, cat_cols):
        for col in cat_cols:
            data[col] = data[col].astype('category')

        # 初始化一个空列表来存储加载的模型
        models = []

        # 获取所有文件名并排序
        sorted_filenames = sorted(os.listdir(model_dir), key=custom_sort_key)

        # 遍历排序后的文件名
        for filename in sorted_filenames:
            # 构建完整的文件路径
            file_path = os.path.join(model_dir, filename)

            # 检查是否为文件（避免处理目录）
            if os.path.isfile(file_path):
                try:
                    # 尝试加载模型
                    model = load(file_path)
                    # 将加载的模型添加到列表中
                    models.append(model)
                    print(f"成功加载模型: {filename}")
                except Exception as e:
                    print(f"加载模型 {filename} 时出错: {e}")

        X = data.drop(['ID', 'efs', 'efs_time'], axis=1)
        y = data['efs_time']
        fold_scores = []
        cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        # cv = RepeatedKFold(n_splits=10, n_repeats=2, random_state=42)

        oof_preds = np.zeros(len(X))

        i = 0
        oob_preds = np.zeros((len(X), 10))  # 存储每个模型的OOB预测结果
        oob_counts = np.zeros(len(X))       # 记录每个样本被多少模型预测过
        val = []

        for fold, (train_index, valid_index) in enumerate(cv.split(X)):
            model = models[i]
            X_train = X.iloc[train_index]
            X_valid = X.iloc[valid_index]
            y_train = y.iloc[train_index]
            y_valid = y.iloc[valid_index]

            dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
            dvalid = xgb.DMatrix(X_valid, label=y_valid, enable_categorical=True)

            if isinstance(model, xgb.XGBModel) or isinstance(model, xgb.Booster):
                oof_preds[valid_index] = model.predict(dvalid)
            else:
                oof_preds[valid_index] = model.predict(X_valid)  # 将当前验证集的预测结果存储到 oof_preds 中，K折后将得到全部数据集的结果

                # 对验证集（OOB样本）进行预测
            if isinstance(model, xgb.XGBModel) or isinstance(model, xgb.Booster):
                oob_preds[valid_index, fold] = model.predict(dvalid)
            else:
                oob_preds[valid_index, fold] = model.predict(X_valid)

            oob_counts[valid_index] += 1

            y_true_fold = data.iloc[valid_index][['ID', 'efs', 'efs_time', 'race_group']].copy()
            y_pred_fold = data.iloc[valid_index][['ID']].copy()

            y_pred_fold['prediction'] = oob_preds[valid_index, fold]#oof_preds[valid_index]

            fold_score = score(y_true_fold, y_pred_fold, 'ID')
            fold_scores.append(fold_score)
            # print('Fold', i, ':', fold_score)
            i = i + 1

        final_oob_preds = np.zeros(len(X))
        for i in range(len(X)):
            if oob_counts[i] > 0:
                final_oob_preds[i] = np.sum(oob_preds[i]) / oob_counts[i]

        y_true = data[['ID', 'efs', 'efs_time', 'race_group']].copy()
        y_pred = data[['ID']].copy()
        y_pred['prediction'] = final_oob_preds#oof_preds

        c_index_score = score(y_true.copy(), y_pred.copy(), 'ID')

        print(f'\nOverall C-Index for : {c_index_score:.6f}\n')
        print('CV分数的方差是：',np.sqrt(np.var(fold_scores)))

        return y_pred['prediction'], fold_scores


    def infer_model(self, data, cat_cols, models, is_xgb):
        data = data.drop(['ID'], axis=1)
        for col in cat_cols:
            data[col] = data[col].astype('category')

        # predictions = []

        # for model in models:
        #     if isinstance(model, xgb.XGBModel) or isinstance(model, xgb.Booster):
        #         # 将 DataFrame 转换为 DMatrix
        #         dmatrix_data = xgb.DMatrix(data, enable_categorical=True)
        #         predictions.append(model.predict(dmatrix_data))
        #     else:
        #         predictions.append(model.predict(data))
        # return np.mean(predictions, axis=0)

        model = models[0]
        #判断是不是XGB模型
        if is_xgb == 1:
            return np.mean([model.predict(xgb.DMatrix(data, enable_categorical=True)) for model in models], axis=0)
        return np.mean([model.predict(data) for model in models], axis=0)


md = MD(CFG.early_stop, CFG.penalizer, CFG.n_splits, CFG.color)




# Create all targets
# train_data = md.create_target1(train_data, cat_cols)
# train_data = md.create_target2(train_data)
# train_data = md.create_target3(train_data)
# train_data = md.create_target4(train_data)


def custom_sort_key(name):
    # 提取名称中的数字部分
    number_part = ''.join(filter(str.isdigit, name))
    return int(number_part) if number_part else float('inf')


# model_dir = '/kaggle/input/models_nmdp_trio_score_shuzi/other/default/3/models/models/CatBoost2'
# ctb2_oof_preds, fold2_scores = md.valid_test(train_data, model_dir, cat_cols)


import os
from joblib import dump
from joblib import load

def save_catboost_models(dir, models, str):
    """
    批量保存 CatBoost 模型到指定目录

    :param dir: 保存模型的目录路径
    :param models: 包含 CatBoost 模型的列表
    :param str: 用于替换文件名中 Cat1 的字符串
    """
    # 如果目录不存在，则创建该目录
    if not os.path.exists(dir):
        os.makedirs(dir)

    # 批量保存模型
    for i, model in enumerate(models, start=1):
        model_name = f'{str}{i}'
        model_path = os.path.join(dir, model_name)
        dump(model, model_path)
        print(f"Model {model_name} saved to {model_path}")


model_dir = '/kaggle/input/models_nmdp_trio_score_shuzi/other/default/3/models/models/CatBoost1'

# 初始化一个空列表来存储加载的模型
ctb1_models = []

# 遍历目录中的所有文件
for filename in os.listdir(model_dir):
    # 构建完整的文件路径
    file_path = os.path.join(model_dir, filename)
    # 检查是否为文件（避免处理目录）
    if os.path.isfile(file_path):
        try:
            # 尝试加载模型
            model = load(file_path)
            # 将加载的模型添加到列表中
            ctb1_models.append(model)
            print(f"成功加载模型: {filename}")
        except Exception as e:
            print(f"加载模型 {filename} 时出错: {e}")


model_dir = '/kaggle/input/models_nmdp_trio_score_shuzi/other/default/3/models/models/CatBoost2'

# 初始化一个空列表来存储加载的模型
ctb2_models = []

# 遍历目录中的所有文件
for filename in os.listdir(model_dir):
    # 构建完整的文件路径
    file_path = os.path.join(model_dir, filename)
    # 检查是否为文件（避免处理目录）
    if os.path.isfile(file_path):
        try:
            # 尝试加载模型
            model = load(file_path)
            # 将加载的模型添加到列表中
            ctb2_models.append(model)
            print(f"成功加载模型: {filename}")
        except Exception as e:
            print(f"加载模型 {filename} 时出错: {e}")


model_dir = '/kaggle/input/models_nmdp_trio_score_shuzi/other/default/3/models/models/CatBoost3'

# 初始化一个空列表来存储加载的模型
ctb3_models = []

# 遍历目录中的所有文件
for filename in os.listdir(model_dir):
    # 构建完整的文件路径
    file_path = os.path.join(model_dir, filename)
    # 检查是否为文件（避免处理目录）
    if os.path.isfile(file_path):
        try:
            # 尝试加载模型
            model = load(file_path)
            # 将加载的模型添加到列表中
            ctb3_models.append(model)
            print(f"成功加载模型: {filename}")
        except Exception as e:
            print(f"加载模型 {filename} 时出错: {e}")


model_dir = '/kaggle/input/models_nmdp_trio_score_shuzi/other/default/3/models/models/LGBM1'

# 初始化一个空列表来存储加载的模型
lgb1_models = []

# 遍历目录中的所有文件
for filename in os.listdir(model_dir):
    # 构建完整的文件路径
    file_path = os.path.join(model_dir, filename)
    # 检查是否为文件（避免处理目录）
    if os.path.isfile(file_path):
        try:
            # 尝试加载模型
            model = load(file_path)
            # 将加载的模型添加到列表中
            lgb1_models.append(model)
            print(f"成功加载模型: {filename}")
        except Exception as e:
            print(f"加载模型 {filename} 时出错: {e}")


model_dir = '/kaggle/input/models_nmdp_trio_score_shuzi/other/default/3/models/models/LGBM2'

# 初始化一个空列表来存储加载的模型
lgb2_models = []

# 遍历目录中的所有文件
for filename in os.listdir(model_dir):
    # 构建完整的文件路径
    file_path = os.path.join(model_dir, filename)
    # 检查是否为文件（避免处理目录）
    if os.path.isfile(file_path):
        try:
            # 尝试加载模型
            model = load(file_path)
            # 将加载的模型添加到列表中
            lgb2_models.append(model)
            print(f"成功加载模型: {filename}")
        except Exception as e:
            print(f"加载模型 {filename} 时出错: {e}")


model_dir = '/kaggle/input/models_nmdp_trio_score_shuzi/other/default/3/models/models/LGBM3'

# 初始化一个空列表来存储加载的模型
lgb3_models = []

# 遍历目录中的所有文件
for filename in os.listdir(model_dir):
    # 构建完整的文件路径
    file_path = os.path.join(model_dir, filename)
    # 检查是否为文件（避免处理目录）
    if os.path.isfile(file_path):
        try:
            # 尝试加载模型
            model = load(file_path)
            # 将加载的模型添加到列表中
            lgb3_models.append(model)
            print(f"成功加载模型: {filename}")
        except Exception as e:
            print(f"加载模型 {filename} 时出错: {e}")


model_dir = '/kaggle/input/models_nmdp_trio_score_shuzi/other/default/3/models/models/Cox1'

# 初始化一个空列表来存储加载的模型
cox1_models = []

# 遍历目录中的所有文件
for filename in os.listdir(model_dir):
    # 构建完整的文件路径
    file_path = os.path.join(model_dir, filename)
    # 检查是否为文件（避免处理目录）
    if os.path.isfile(file_path):
        try:
            # 尝试加载模型
            model = load(file_path)
            # 将加载的模型添加到列表中
            cox1_models.append(model)
            print(f"成功加载模型: {filename}")
        except Exception as e:
            print(f"加载模型 {filename} 时出错: {e}")


model_dir = '/kaggle/input/models_nmdp_trio_score_shuzi/other/default/3/models/models/Cox2'

# 初始化一个空列表来存储加载的模型
cox2_models = []

# 遍历目录中的所有文件
for filename in os.listdir(model_dir):
    # 构建完整的文件路径
    file_path = os.path.join(model_dir, filename)
    # 检查是否为文件（避免处理目录）
    if os.path.isfile(file_path):
        try:
            # 尝试加载模型
            model = load(file_path)
            # 将加载的模型添加到列表中
            cox2_models.append(model)
            print(f"成功加载模型: {filename}")
        except Exception as e:
            print(f"加载模型 {filename} 时出错: {e}")


model_dir = '/kaggle/input/models_nmdp_trio_score_shuzi/other/default/3/models/models/Cox3'

# 初始化一个空列表来存储加载的模型
cox3_models = []

# 遍历目录中的所有文件
for filename in os.listdir(model_dir):
    # 构建完整的文件路径
    file_path = os.path.join(model_dir, filename)
    # 检查是否为文件（避免处理目录）
    if os.path.isfile(file_path):
        try:
            # 尝试加载模型
            model = load(file_path)
            # 将加载的模型添加到列表中
            cox3_models.append(model)
            print(f"成功加载模型: {filename}")
        except Exception as e:
            print(f"加载模型 {filename} 时出错: {e}")


model_dir = '/kaggle/input/models_nmdp_trio_score_shuzi/other/default/3/XGB/XGB/XGB1'

# 初始化一个空列表来存储加载的模型
xgb1_models = []

# 遍历目录中的所有文件
for filename in os.listdir(model_dir):
    # 构建完整的文件路径
    file_path = os.path.join(model_dir, filename)
    # 检查是否为文件（避免处理目录）
    if os.path.isfile(file_path):
        try:
            # 尝试加载模型
            model = load(file_path)
            # 将加载的模型添加到列表中
            xgb1_models.append(model)
            print(f"成功加载模型: {filename}")
        except Exception as e:
            print(f"加载模型 {filename} 时出错: {e}")


model_dir = '/kaggle/input/models_nmdp_trio_score_shuzi/other/default/3/XGB/XGB/XGB2'

# 初始化一个空列表来存储加载的模型
xgb2_models = []

# 遍历目录中的所有文件
for filename in os.listdir(model_dir):
    # 构建完整的文件路径
    file_path = os.path.join(model_dir, filename)
    # 检查是否为文件（避免处理目录）
    if os.path.isfile(file_path):
        try:
            # 尝试加载模型
            model = load(file_path)
            # 将加载的模型添加到列表中
            xgb2_models.append(model)
            print(f"成功加载模型: {filename}")
        except Exception as e:
            print(f"加载模型 {filename} 时出错: {e}")


model_dir = '/kaggle/input/models_nmdp_trio_score_shuzi/other/default/3/XGB/XGB/XGB3'

# 初始化一个空列表来存储加载的模型
xgb3_models = []

# 遍历目录中的所有文件
for filename in os.listdir(model_dir):
    # 构建完整的文件路径
    file_path = os.path.join(model_dir, filename)
    # 检查是否为文件（避免处理目录）
    if os.path.isfile(file_path):
        try:
            # 尝试加载模型
            model = load(file_path)
            # 将加载的模型添加到列表中
            xgb3_models.append(model)
            print(f"成功加载模型: {filename}")
        except Exception as e:
            print(f"加载模型 {filename} 时出错: {e}")


# # Train CatBoost models
# print("Training CatBoost models...")
# ctb1_models, _ = md.train_model(train_data, cat_cols, CFG.ctb_params, target='target1', title='CatBoost1')
# ctb2_models, _ = md.train_model(train_data, cat_cols, CFG.ctb_params, target='target2', title='CatBoost2')
# ctb3_models, _ = md.train_model(train_data, cat_cols, CFG.ctb_params, target='target3', title='CatBoost3')

# # Train LightGBM models
# print("\nTraining LightGBM models...")
# lgb1_models, _ = md.train_model(train_data, cat_cols, CFG.lgb_params, target='target1', title='LightGBM1')
# lgb2_models, _ = md.train_model(train_data, cat_cols, CFG.lgb_params, target='target2', title='LightGBM2')
# lgb3_models, _ = md.train_model(train_data, cat_cols, CFG.lgb_params, target='target3', title='LightGBM3')

# # Train Cox models
# print("\nTraining Cox models...")
# cox1_models, _ = md.train_model(train_data, cat_cols, CFG.cox1_params, target='target4', title='CatBoost')
# cox2_models, _ = md.train_model(train_data, cat_cols, CFG.cox2_params, target='target4', title='CatBoost')
# cox3_models, _ = md.train_model(train_data, cat_cols, CFG.cox3_params, target='target4', title='CatBoost')

# CatBoost predictions
ctb1_preds = md.infer_model(test_data, cat_cols, ctb1_models, 0)
ctb2_preds = md.infer_model(test_data, cat_cols, ctb2_models, 0)
ctb3_preds = md.infer_model(test_data, cat_cols, ctb3_models, 0)
# # LightGBM predictions
lgb1_preds = md.infer_model(test_data, cat_cols, lgb1_models, 0)
lgb2_preds = md.infer_model(test_data, cat_cols, lgb2_models, 0)
lgb3_preds = md.infer_model(test_data, cat_cols, lgb3_models, 0)
# Cox predictions
cox1_preds = md.infer_model(test_data, cat_cols, cox1_models, 0)
cox2_preds = md.infer_model(test_data, cat_cols, cox2_models, 0)
cox3_preds = md.infer_model(test_data, cat_cols, cox3_models, 0)
# XGBoost predictions
xgb1_preds = md.infer_model(test_data, cat_cols, xgb1_models, 1)
xgb2_preds = md.infer_model(test_data, cat_cols, xgb2_models, 1)
xgb3_preds = md.infer_model(test_data, cat_cols, xgb3_models, 1)


# from xgboost import XGBClassifier
# from sklearn.model_selection import KFold
# from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
# import numpy as np

# # 数据准备
# FEATURES = train_data.drop(columns=['ID', 'efs', 'efs_time']).columns.tolist()
# FOLDS = 5

# # 初始化存储数组
# oof_xgb = np.zeros(len(train_data))
# pred_efs = np.zeros(len(test_data))
# cv = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

# # 特征和目标变量
# X = train_data.drop(columns=['ID', 'efs', 'efs_time'])

# #X特征工程
# X['interaction_trio_dqb1_low'] = X['nmdp_trio_score'] * X['hla_match_dqb1_low'].replace('Unknown', 0).astype(float)

# y = train_data["efs"].astype('category')

# test_data_processed = test_data.drop(columns=['ID'])
# test_data_processed['interaction_trio_dqb1_low'] = test_data_processed['nmdp_trio_score'] *test_data_processed['hla_match_dqb1_low'].replace('Unknown', 0).astype(float)


# for col in cat_cols:
#     X[col] = X[col].astype('category')
#     test_data_processed[col] = test_data_processed[col].astype('category')

# for i, (train_index, valid_index) in enumerate(cv.split(X, y)):
#     print("#"*25)
#     print(f"### Fold {i+1}")
#     print("#"*25)
    
#     # 数据划分
#     X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
#     y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
    
#     # 模型配置
#     model = XGBClassifier(
#         device="cuda",
#         max_depth=6,#5,
#         colsample_bytree=0.7129400756425178, 
#         subsample=0.8185881823156917, 
#         n_estimators=20_000, 
#         learning_rate=0.04425768131771064,
#         eval_metric="auc", 
#         early_stopping_rounds=300,#50, 
#         objective='binary:logistic',
#         scale_pos_weight=1.5379160847615545,  
#         min_child_weight=100,#80,
#         enable_categorical=True,
#         gamma=3.1330719334577584,
#         tree_method = 'hist',
#         max_cat_to_onehot = 5, #第三个改的点
#         random_state=42
#     )
    
#     # 模型训练
#     model.fit(
#         X_train, y_train,
#         eval_set=[(X_valid, y_valid)],
#         verbose=100
#     )
    
#     # 验证集预测
#     oof_preds = model.predict_proba(X_valid)[:, 1]
#     oof_xgb[valid_index] = (oof_preds > 0.5).astype(int)
    
#     # 测试集预测
#     fold_preds = model.predict_proba(test_data_processed)[:, 1]
#     pred_efs += fold_preds

# # 平均预测结果
# pred_efs = (pred_efs / FOLDS > 0.5).astype(int)

# # 性能评估
# print("Final Model Performance:")
# print(f"Accuracy: {accuracy_score(y, oof_xgb):.4f}")#0.6777
# print(f"F1 Score: {f1_score(y, oof_xgb):.4f}")
# print(f"ROC AUC: {roc_auc_score(y, oof_xgb):.4f}")


#XGB第二版
# #对efs的分类
# from xgboost import XGBClassifier
# FEATURES = train_data.drop(columns=['ID', 'efs', 'efs_time']).columns.tolist()
# FOLDS = 5
# from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# oof_xgb = np.zeros(len(train_data))
# pred_efs = np.zeros(len(test_data))
# cv = KFold(n_splits=FOLDS, shuffle=True, random_state=42)



# for i, (train_index, test_index) in enumerate(cv.split(train_data, train_data["efs"])):

#     print("#"*25)
#     print(f"### Fold {i+1}")
#     print("#"*25)
    
#     x_train = train_data.loc[train_index, FEATURES].copy()
#     y_train = train_data.loc[train_index, "efs"]
#     x_valid = train_data.loc[test_index, FEATURES].copy()
#     y_valid = train_data.loc[test_index, "efs"]
#     x_test = test_data[FEATURES].copy()

#     # x_train['interaction_trio_dqb1_low'] = x_train['nmdp_trio_score'] * x_train['hla_match_dqb1_low'].replace('Unknown', 0).astype(float)
    
#     # x_valid['interaction_trio_dqb1_low'] = x_valid['nmdp_trio_score'] * x_valid['hla_match_dqb1_low'].replace('Unknown', 0).astype(float)

#     # x_test['interaction_trio_dqb1_low'] = x_test['nmdp_trio_score'] * x_test['hla_match_dqb1_low'].replace('Unknown', 0).astype(float)
    
#     for col in cat_cols:
#         x_train[col] = x_train[col].astype('category')
#         x_valid[col] = x_valid[col].astype('category')
#         x_test[col] = x_test[col].astype('category')


#     model_xgb = XGBClassifier(
#         device="cuda",
#         max_depth=6,#5,#3,
#         colsample_bytree=0.7129400756425178,
#         subsample=0.8185881823156917,
#         n_estimators=20_000,
#         learning_rate=0.04425768131771064,
#         eval_metric="auc",
#         early_stopping_rounds=300,#300,#50,
#         objective='binary:logistic',
#         scale_pos_weight=1.5379160847615545,  
#         min_child_weight=100,#80,#4,
#         enable_categorical=True,
#         gamma=3.1330719334577584,
#         # reg_lambda = 3,
#         tree_method = 'hist',
#         max_cat_to_onehot = 5,
#         # use_label_encoder=False,
#         random_state=42
#     )
#     model_xgb.fit(
#         x_train, y_train,
#         eval_set=[(x_valid, y_valid)],  
#         verbose=100
#     )

#     # INFER OOF (Probabilities -> Binary)
#     oof_xgb[test_index] = (model_xgb.predict_proba(x_valid)[:, 1] > 0.5).astype(int)
#     # INFER TEST (Probabilities -> Average Probs)
#     pred_efs += model_xgb.predict_proba(x_test)[:, 1]

# # COMPUTE AVERAGE TEST PREDS
# pred_efs = (pred_efs / FOLDS > 0.5).astype(int)

# # EVALUATE PERFORMANCE
# accuracy = accuracy_score(train_data["efs"], oof_xgb)
# f1 = f1_score(train_data["efs"], oof_xgb)
# roc_auc = roc_auc_score(train_data["efs"], oof_xgb)

# print(f"Accuracy: {accuracy:.6f}")#0.6748,   0.6766  0.6773     0.6783      0.6805
# print(f"F1 Score: {f1:.6f}")
# print(f"ROC AUC Score: {roc_auc:.6f}")


from catboost import CatBoostClassifier
model_dir = '/kaggle/input/cv0.686569/other/default/1/CTB_Class'

# 初始化一个空列表来存储加载的模型
model_ctbs = []

# 遍历目录中的所有文件
for filename in os.listdir(model_dir):
    # 构建完整的文件路径
    file_path = os.path.join(model_dir, filename)
    # 检查是否为文件（避免处理目录）
    if os.path.isfile(file_path):
        try:
            # 尝试加载模型
            model = load(file_path)
            # 将加载的模型添加到列表中
            model_ctbs.append(model)
            print(f"成功加载模型: {filename}")
        except Exception as e:
            print(f"加载模型 {filename} 时出错: {e}")

ctb_params = {
        # 'loss_function': 'Logloss',#分类任务
        'learning_rate': 0.02,
        'random_state': 42,
        'task_type': 'CPU',
        'num_trees': 7000,#6000,#
        'subsample': 0.85,
        'reg_lambda': 5.0,
        'depth': 8,
        'eval_metric':'Precision'
        # 'thread_count':32
}
FEATURES = train_data.drop(columns=['ID', 'efs', 'efs_time']).columns.tolist()
FOLDS = 10
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

oof_ctb = np.zeros(len(train_data))
pred_efs = np.zeros(len(test_data))
cv = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

for i, (train_index, test_index) in enumerate(cv.split(train_data, train_data["efs"])):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = train_data.loc[train_index, FEATURES].copy()
    # x_train = x_train.drop(columns=['interaction_trio_dqb1_low'],axis=1)
    y_train = train_data.loc[train_index, "efs"]
    x_valid = train_data.loc[test_index, FEATURES].copy()
    # x_valid = x_valid.drop(columns=['interaction_trio_dqb1_low'],axis=1)
    y_valid = train_data.loc[test_index, "efs"]
    x_test = test_data[FEATURES].copy()

    for col in cat_cols:
        x_train[col] = x_train[col].astype('category')
        x_valid[col] = x_valid[col].astype('category')
        x_test[col] = x_test[col].astype('category')

    # x_train['interaction_trio_dqb1_low'] = x_train['nmdp_trio_score'].replace(-1, 0) * x_train['hla_match_dqb1_low'].replace('Unknown', 0).astype(float)
    #
    # x_valid['interaction_trio_dqb1_low'] = x_valid['nmdp_trio_score'].replace(-1, 0) * x_valid['hla_match_dqb1_low'].replace('Unknown', 0).astype(float)

    # x_test['interaction_trio_dqb1_low'] = x_test['nmdp_trio_score'] * x_test['hla_match_dqb1_low'].replace('Unknown', 0).astype(float)


    model = model_ctbs[i]

    # INFER OOF (Probabilities -> Binary)
    oof_ctb[test_index] = model.predict(x_valid)
    
    oof_ctb = (oof_ctb > 0.5).astype(int)
    
    accuracy = accuracy_score(train_data.iloc[test_index]["efs"], oof_ctb[test_index])
    print(f"Accuracy: {accuracy:.6f}")
    
    # INFER TEST (Probabilities -> Average Probs)
    # pred_efs += model_xgb.predict_proba(x_test)[:, 1]
    pred_efs += model.predict(x_test)

# COMPUTE AVERAGE TEST PREDS
pred_efs = (pred_efs / FOLDS > 0.5).astype(int)

oof_ctb = (oof_ctb > 0.5).astype(int)
# EVALUATE PERFORMANCE
accuracy = accuracy_score(train_data["efs"], oof_ctb)
f1 = f1_score(train_data["efs"], oof_ctb)
roc_auc = roc_auc_score(train_data["efs"], oof_ctb)

print(f"Accuracy: {accuracy:.6f}")#0.6748,   0.6766  0.6773     0.6783      0.6805
print(f"F1 Score: {f1:.6f}")
print(f"ROC AUC Score: {roc_auc:.6f}")



# from catboost import CatBoostClassifier
# ctb_params = {
#         # 'loss_function': 'Logloss',#分类任务
#         'learning_rate': 0.02,
#         'random_state': 42,
#         'task_type': 'CPU',
#         'num_trees': 7000,#6000,#
#         'subsample': 0.85,
#         'reg_lambda': 5.0,
#         'depth': 8,
#         'eval_metric':'Precision'
#         # 'thread_count':32
# }
# FEATURES = train_data.drop(columns=['ID', 'efs', 'efs_time']).columns.tolist()
# FOLDS = 10
# from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# oof_ctb = np.zeros(len(train_data))
# pred_efs = np.zeros(len(test_data))
# cv = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

# model_ctbs = []

# for i, (train_index, test_index) in enumerate(cv.split(train_data, train_data["efs"])):

#     print("#"*25)
#     print(f"### Fold {i+1}")
#     print("#"*25)
    
#     x_train = train_data.loc[train_index, FEATURES].copy()
#     # x_train = x_train.drop(columns=['interaction_trio_dqb1_low'],axis=1)
#     y_train = train_data.loc[train_index, "efs"]
#     x_valid = train_data.loc[test_index, FEATURES].copy()
#     # x_valid = x_valid.drop(columns=['interaction_trio_dqb1_low'],axis=1)
#     y_valid = train_data.loc[test_index, "efs"]
#     x_test = test_data[FEATURES].copy()

#     for col in cat_cols:
#         x_train[col] = x_train[col].astype('category')
#         x_valid[col] = x_valid[col].astype('category')
#         x_test[col] = x_test[col].astype('category')

#     # x_train['interaction_trio_dqb1_low'] = x_train['nmdp_trio_score'].replace(-1, 0) * x_train['hla_match_dqb1_low'].replace('Unknown', 0).astype(float)
#     #
#     # x_valid['interaction_trio_dqb1_low'] = x_valid['nmdp_trio_score'].replace(-1, 0) * x_valid['hla_match_dqb1_low'].replace('Unknown', 0).astype(float)

#     # x_test['interaction_trio_dqb1_low'] = x_test['nmdp_trio_score'] * x_test['hla_match_dqb1_low'].replace('Unknown', 0).astype(float)


#     model = CatBoostClassifier(**ctb_params, verbose=0, cat_features=cat_cols,scale_pos_weight=2)
#     model.fit(
#         x_train, y_train,
#         eval_set=(x_valid, y_valid),
#         early_stopping_rounds=700,
#         verbose=100
#         )

#     model_ctbs.append(model)

#     # INFER OOF (Probabilities -> Binary)
#     oof_ctb[test_index] = model.predict(x_valid)
    
#     oof_ctb = (oof_ctb > 0.5).astype(int)
    
#     accuracy = accuracy_score(train_data.iloc[test_index]["efs"], oof_ctb[test_index])
#     print(f"Accuracy: {accuracy:.6f}")
    
#     # INFER TEST (Probabilities -> Average Probs)
#     # pred_efs += model_xgb.predict_proba(x_test)[:, 1]
#     pred_efs += model.predict(x_test)

# # COMPUTE AVERAGE TEST PREDS
# pred_efs = (pred_efs / FOLDS > 0.5).astype(int)

# oof_ctb = (oof_ctb > 0.5).astype(int)
# # EVALUATE PERFORMANCE
# accuracy = accuracy_score(train_data["efs"], oof_ctb)
# f1 = f1_score(train_data["efs"], oof_ctb)
# roc_auc = roc_auc_score(train_data["efs"], oof_ctb)

# print(f"Accuracy: {accuracy:.6f}")#0.6748,   0.6766  0.6773     0.6783      0.6805
# print(f"F1 Score: {f1:.6f}")
# print(f"ROC AUC Score: {roc_auc:.6f}")




indices = np.where(np.array(pred_efs) == 1)[0]


# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
#使用MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


ctb1 = np.array(ctb1_preds).reshape(-1, 1)
ctb1 = scaler.fit_transform(ctb1)
ctb1 = np.squeeze(ctb1, axis=-1)
ctb1[indices] += 0.3


ctb2 = np.array(ctb2_preds).reshape(-1, 1)
ctb2 = scaler.fit_transform(ctb2)
ctb2 = np.squeeze(ctb2, axis=-1)
ctb2[indices] += 0.3


ctb3 = np.array(ctb3_preds).reshape(-1, 1)
ctb3 = scaler.fit_transform(ctb3)
ctb3 = np.squeeze(ctb3, axis=-1)
ctb3[indices] += 0.3


lgb1 = np.array(lgb1_preds).reshape(-1, 1)
lgb1 = scaler.fit_transform(lgb1)
lgb1 = np.squeeze(lgb1, axis=-1)
lgb1[indices] += 0.3


lgb2 = np.array(lgb2_preds).reshape(-1, 1)
lgb2 = scaler.fit_transform(lgb2)
lgb2 = np.squeeze(lgb2, axis=-1)
lgb2[indices] += 0.3


lgb3 = np.array(lgb3_preds).reshape(-1, 1)
lgb3 = scaler.fit_transform(lgb3)
lgb3 = np.squeeze(lgb3, axis=-1)
lgb3[indices] += 0.3


cox1 = np.array(cox1_preds).reshape(-1, 1)
cox1 = scaler.fit_transform(cox1)
cox1 = np.squeeze(cox1, axis=-1)
cox1[indices] += 0.3


cox2 = np.array(cox2_preds).reshape(-1, 1)
cox2 = scaler.fit_transform(cox2)
cox2 = np.squeeze(cox2, axis=-1)
cox2[indices] += 0.3


cox3 = np.array(cox3_preds).reshape(-1, 1)
cox3 = scaler.fit_transform(cox3)
cox3 = np.squeeze(cox3, axis=-1)
cox3[indices] += 0.3


xgb1 = np.array(xgb1_preds).reshape(-1, 1)
xgb1 = scaler.fit_transform(xgb1)
xgb1 = np.squeeze(xgb1, axis=-1)
xgb1[indices] += 0.3


xgb2 = np.array(xgb2_preds).reshape(-1, 1)
xgb2 = scaler.fit_transform(xgb2)
xgb2 = np.squeeze(xgb2, axis=-1)
xgb2[indices] += 0.3


xgb3 = np.array(xgb3_preds).reshape(-1, 1)
xgb3 = scaler.fit_transform(xgb3)
xgb3 = np.squeeze(xgb3, axis=-1)
xgb3[indices] += 0.3


preds = [
    ctb1,
    ctb2,
    ctb3,
    lgb1,
    lgb2,
    lgb3,
    # cox1,
    # cox2,
    # cox3,
    xgb1,
    xgb2,
    xgb3,
]
# weights = [
#     10.0, 15.0, 100.0,
#     5.0, 10.0, 10.0,
#     20.0, 15.0, 10.0,
#     5.0, 35.0, 35.0,
# ]
#第二套参数
# weights = [
#     40.0, 60.0, 150.0,
#     5.0, 10.0, 10.0,
#     20.0, 15.0, 10.0,
#     1.0, 50.0, 100.0,
# ]
#第三套参数可调整
weights = [
    50.0, 20.0, 150.0,
    5.0, 10.0, 10.0,
    -5.0, -5.0, -5.0,
    1.0, 50.0, 100.0,#35
]
#第四套参数CV0.686959
weights = [ 0.07,   0.07,   0.17,  0.07,   0.07,  -0.09,
   -0.10,  0.07,   0.07 ]


# Create ranked predictions
ranked_preds = np.array([rankdata(p) for p in preds])
ensemble_preds = np.sum([w * p for w, p in zip(weights, ranked_preds)], axis=0)
#尝试二次后处理
# ensemble_preds[indices] -= 250


#尝试一下提交不排序版本然后减一些数字


# # Combine all predictions
# preds = [
#     ctb3_preds# ctb1_preds, ctb2_preds, ctb3_preds,
#     # lgb1_preds, lgb2_preds, lgb3_preds,
#     # cox1_preds, cox2_preds, cox3_preds,
#     # xgb1_preds, xgb2_preds, xgb3_preds,
# ]

# # print(ctb1_preds.shape)
# # print(ctb2_preds.shape)
# # print(ctb3_preds.shape)
# # print(lgb1_preds.shape)
# # print(lgb2_preds.shape)
# # print(lgb3_preds.shape)
# # print(cox1_preds.shape)
# # print(cox2_preds.shape)
# # print(cox3_preds.shape)
# # print(xgb1_preds.shape)
# # print(xgb2_preds.shape)
# # print(xgb3_preds.shape)

# preds = np.array(preds)
# indices = np.where(np.array(pred_efs) == 1)[0]
# preds[:, indices] += 1.1
# preds = preds.tolist()



# # Define weights based on model performance
# weights = [
#     1.0# 1.0, 8.0, 12.0,  # CatBoost weights
#     # 1.0, 1.0, 1.0,  # LightGBM weights
#     # 1.0, 1.0, 1.0,#4.0, 4.0, 4.0,   # Cox weights 
#     # 1.0, 1.0, 1.0,
# ]


# Create submission
subm_data = pd.read_csv(CFG.subm_path)
subm_data['prediction'] = ensemble_preds
subm_data.to_csv('submission.csv', index=False)
display(subm_data.head())

