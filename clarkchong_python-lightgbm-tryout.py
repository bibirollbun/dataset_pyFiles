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
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def calculate_calibration_metrics(df, true_label, pred_prob, 
                                  n_bins=10, strategy = 'quantile', 
                                  metrics_df=None, printout = True):
    """
    Evaluate the model and optionally store metrics in a DataFrame.

    Parameters:
    - df: DataFrame containing the true labels and predicted probabilities.
    - true_label: Column name for true binary labels.
    - pred_prob: Column name for predicted probabilities.
    - n_bins: Number of bins for calibration curve.
    - strategy: Strategy for binning ('uniform' or 'quantile').
    - metrics_df: Optional DataFrame to store the metrics.

    Returns:
    - DataFrame containing the calculated metrics if metrics_df is not None, otherwise None.
    """
    y_true = df[true_label]
    y_pred = df[pred_prob]

    num_rows = len(y_true)
    sum_y_true = sum(y_true)
    sum_y_pred = sum(y_pred)    
    brier_score = brier_score_loss(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    log_loss_score = log_loss(y_true, y_pred)

    # 计算校准曲线
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=n_bins, strategy=strategy)

    # 进行线性回归
    X = prob_pred.reshape(-1, 1)
    y = prob_true
    model = LinearRegression()
    model.fit(X, y)
    
    # 计算R^2
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)

    slope = model.coef_[0]
    intercept = model.intercept_

    if printout:
        print(f"Brier Score: {brier_score:.4f}", """\n"""
                f"AUC: {auc:.4f}", """\n"""
                f"Log Loss: {log_loss_score:.4f}"
                )
        
        # 绘制校准曲线
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot([0, 1], [0, 1], linestyle='--', label='Perfect Calibration')
        ax.plot(prob_pred, prob_true, marker='.', label='Calibrated Probabilities')
        ax.set_title('Calibration Curve')
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('True Probability')

        # 自动调整x轴和y轴的最大值
        max_value = max(max(prob_pred), max(prob_true))*1.2
        ax.set_xlim(0, max_value)
        ax.set_ylim(0, max_value)

        ax.legend()
        
        # 绘制回归线
        ax.plot(X, y_pred, color='red', label=f'Linear Regression (R^2 = {r2:.2f})')
        ax.legend()
        
        # 显示线性回归的参数
        print(f"Linear Regression Parameters:\nSlope: {slope:.2f}\nIntercept: {intercept:.2f}")
        
        plt.show()

    # Store metrics in a dictionary
    metrics = {
        'num_rows': num_rows,
        'sum_y_true': sum_y_true,
        'sum_y_pred': sum_y_pred,
        'auc': auc,
        'brier_score': brier_score,
        'log_loss_score': log_loss_score,
        'slope': slope,
        'intercept': intercept,
        'r2': r2
    }

    # If a DataFrame is provided, store the metrics in it
    if metrics_df is not None:
        # If the DataFrame is empty, create a new one
        if metrics_df.empty:
            metrics_df = pd.DataFrame([metrics])
        else:
            # Append the metrics to the existing DataFrame
            metrics_df = pd.concat([metrics_df, pd.DataFrame([metrics])], ignore_index=True)
        
    return metrics_df


import gc

#————————————————————————————————————取attribution加特征数据—————————————————————————————————————————————————————————
import pandas as pd
import numpy as np
import os
import shutil   ## 建立在 os 模块之上，提供了一系列用于复制、创建、移动和删除文件及目录的便捷方法
import random
import joblib   ## 专注于提供轻量级流水线（pipelines）和并行计算的 Python 库，特别适用于需要大量 I/O 或数值计算的应用场景
import itertools    ## 包含了一些用于生成和操作迭代对象的功能强大的工具，如组合、排列、笛卡尔积等，适用于处理数据集合时创建复杂的迭代器
from collections import defaultdict ## defaultdict 是 dict 类的一个子类，它重写了方法以提供一个默认值给不存在的键，这对于需要初始化值为列表、整数或其他类型的字典特别有用，避免了在访问新键之前手动初始化键值对的过程。

import time  
from pathlib import Path
from datetime import datetime

from lightgbm import LGBMClassifier
from lightgbm.callback import early_stopping, log_evaluation
from sklearn.isotonic import IsotonicRegression 
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

SEED = 7
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

class ChronicModelTrainer:
    def __init__(self, model_dir='model_artifacts'):
        self.model_dir = model_dir
        os.makedirs(os.path.join(model_dir, 'preprocessors'), exist_ok=True)
        os.makedirs(os.path.join(model_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(model_dir, 'calibrators'), exist_ok=True)
        os.makedirs(os.path.join(model_dir, 'raw_data'), exist_ok=True)
        
        self.params = {
            'max_depth': 5,
            'subsample': 0.9555525182119154,
            'colsample_bytree': 0.8524901914884458,
            'min_child_weight': 7.140609432167075,
            'min_child_samples': 3304,
            'reg_lambda': 683.0791408206187,
            'reg_alpha': 855.2397835167428,
            'num_leaves': 82,
            'learning_rate': 0.23496162012043764,
            'min_data_in_leaf': 180,
            'min_sum_hessian_in_leaf': 0.008200294010173081,
            'lambda_l1': 0.029635875292398417,
            'lambda_l2': 28.030857626453198,
            'bagging_fraction': 0.6668113663469499,
            'bagging_freq': 9,
            'feature_fraction': 0.6675777405570473,
            'objective': 'binary',
            'random_state': SEED,
            'boosting': 'gbdt',
            'n_estimators': 10000
        }

    def _preprocess_fold(self, train_data, valid_data, target_col, id_col, fold_num):
        X_train = train_data.drop(columns=[target_col,id_col])
        X_valid = valid_data.drop(columns=[target_col,id_col])
        
        label_encoders = {}
        for col in X_train.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col].astype(str))
            
            valid_vals = X_valid[col].unique()
            unseen = set(valid_vals) - set(le.classes_)
            if unseen:
                X_valid[col] = X_valid[col].apply(lambda x: '<UNK>' if x in unseen else x)
                le.classes_ = np.append(le.classes_, '<UNK>')
            X_valid[col] = le.transform(X_valid[col].astype(str))
            label_encoders[col] = le
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_valid_scaled = scaler.transform(X_valid)
        
        preprocessor = {
            'label_encoder': label_encoders,
            'scaler': scaler,
            'features': X_train.columns.tolist() 
        }
        joblib.dump(
            preprocessor, 
            os.path.join(self.model_dir, 'preprocessors', f'preprocessor_fold{fold_num}.pkl')
        )
        
        return X_train_scaled, X_valid_scaled

    def train(self, data_path, use_calibration=True, target_col='target', id_col='ID', fraction_to_keep = 1):

        import pandas as pd

        raw_data = pd.read_csv(data_path).sample(frac = fraction_to_keep, random_state = SEED)
            
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        X = raw_data.drop(columns=[target_col, id_col])
        y = raw_data[target_col]
        groups = raw_data[id_col].astype(str)
        
        fold_metrics = []
        calibrated_fold_metrics = [] 
        
        for fold_num, (train_idx, valid_idx) in enumerate(cv.split(X, y, groups)):
            print(f"\n{'='*30} 第 {fold_num+1} 折训练 {'='*30}")
            
            train_data = raw_data.iloc[train_idx]
            valid_data = raw_data.iloc[valid_idx]
            
            train_data.to_csv(os.path.join(self.model_dir, 'raw_data', f'train_fold{fold_num+1}.csv'), index=False)
            valid_data.to_csv(os.path.join(self.model_dir, 'raw_data', f'valid_fold{fold_num+1}.csv'), index=False)
            
            X_train, X_valid = self._preprocess_fold(train_data, valid_data, target_col, id_col, fold_num+1)
            
            model = LGBMClassifier(**self.params)
            model.fit(
                X_train, train_data[target_col],
                eval_set=[(X_valid, valid_data[target_col])],
                callbacks=[early_stopping(stopping_rounds=30), log_evaluation(100)]
            )
            
            valid_preds = model.predict_proba(X_valid)[:, 1]
            raw_auc = roc_auc_score(valid_data[target_col], valid_preds)
            fold_metrics.append(raw_auc)
            
            # 保存模型
            joblib.dump(model, os.path.join(self.model_dir, 'models', f'model_fold{fold_num+1}.pkl'))
            
            # 校准逻辑
            if use_calibration:
                train_preds = model.predict_proba(X_train)[:, 1]
                calibrator = IsotonicRegression(out_of_bounds='clip')
                calibrator.fit(train_preds, train_data[target_col])
                
                joblib.dump(
                    calibrator,
                    os.path.join(self.model_dir, 'calibrators', f'calibrator_fold{fold_num+1}.pkl')
                )
                
                calibrated_preds = calibrator.transform(valid_preds)
                calibrated_auc = roc_auc_score(valid_data[target_col], calibrated_preds)
                calibrated_fold_metrics.append(calibrated_auc)
                
                temp_metric_df = pd.DataFrame({
                    'true_label': valid_data[target_col],
                    'raw_pred_prob': valid_preds,
                    'cali_pred_prob': calibrated_preds
                })
                
                print("验证集原始的校准情况")
                calculate_calibration_metrics(temp_metric_df, 'true_label', 'raw_pred_prob', 
                                    n_bins=50, strategy='quantile')
                print("验证集校准后的校准情况")
                calculate_calibration_metrics(temp_metric_df, 'true_label', 'cali_pred_prob', 
                                    n_bins=50, strategy='quantile')
                
                print(f"验证集原始AUC: {raw_auc:.4f} | 校准后AUC: {calibrated_auc:.4f}")
            else:
                print(f"验证集原始AUC: {raw_auc:.4f}")
    
            # 删除不再使用的变量并释放内存
            del train_data, valid_data, X_train, X_valid
            gc.collect()
        
        print(f"\n{'='*40} 训练总结 {'='*40}")
        print(f"平均原始AUC: {np.mean(fold_metrics):.4f} (±{np.std(fold_metrics):.4f})")
        
        if use_calibration:
            print(f"平均校准后AUC: {np.mean(calibrated_fold_metrics):.4f} (±{np.std(calibrated_fold_metrics):.4f})")
        
        print(f"模型文件存储路径: {os.path.abspath(self.model_dir)}")



if __name__ == "__main__":
    # # ================== 训练模式 ==================
    trainer = ChronicModelTrainer(model_dir='current_offline_pipeline')
    trainer.train('/kaggle/input/springleaf-marketing-response/train.csv.zip',use_calibration=True, target_col='target', id_col='ID', fraction_to_keep = 0.75)  # 确保数据包含is_chronic_order_7d列


# #——————————————————————————————— 预测模块 ————————————————————————————————
# class ChronicPredictor:
#     """
#     用模型来预测，读入实际外呼时间，输出每位客户的预估转化率
#     """    
#     def __init__(self, model_dir='model_artifacts'):
#         self.models = []
#         self.preprocessors = []
#         self.calibrators = [] 
        
#         # 加载所有折的模型（保持原状）
#         for fold_num in range(1, 11):
#             try:
#                 model = joblib.load(os.path.join(model_dir, 'models', f'model_fold{fold_num}.pkl'))
#                 preprocessor = joblib.load(os.path.join(model_dir, 'preprocessors', f'preprocessor_fold{fold_num}.pkl'))
#                 calibrator = joblib.load(os.path.join(model_dir, 'calibrators', f'calibrator_fold{fold_num}.pkl'))
#                 self.models.append(model)
#                 self.preprocessors.append(preprocessor)
#                 self.calibrators.append(calibrator)
#             except FileNotFoundError:
#                 break
        
#         if not self.models:
#             raise ValueError(f"模型文件缺失，请检查目录: {model_dir}")

#     def _preprocess_data(self, data_path, preprocessor):
#         """ 移除target_col相关处理 """
#         raw_data = pd.read_csv(data_path)
        
#         # 仅保留训练时的特征列
#         processed_data = raw_data[preprocessor['features']]  # 移除target_col
        
#         # 标签编码处理（保持原状）
#         label_encoders = preprocessor['label_encoder']
#         for col in label_encoders.keys():
#             le = label_encoders[col]
#             valid_vals = processed_data[col].unique()
#             unseen = set(valid_vals) - set(le.classes_)
#             if unseen:
#                 processed_data[col] = processed_data[col].apply(lambda x: '<UNK>' if x in unseen else x)
#                 le.classes_ = np.append(le.classes_, '<UNK>')
#             processed_data[col] = le.transform(processed_data[col].astype(str))
        
#         # 特征缩放
#         scaler = preprocessor['scaler']
#         return scaler.transform(processed_data), raw_data.copy()

#     def predict(self, data_path, ensemble_method='mean'):
#         """ 移除target_col参数 """
#         all_preds = []
#         full_data = None
        
#         for fold_num, (model, preprocessor, calibrator) in enumerate(zip(self.models, self.preprocessors, self.calibrators)):
#             try:
#                 # 修改预处理调用
#                 X_scaled, raw_data = self._preprocess_data(data_path, preprocessor)
                
#                 # 预测流程
#                 raw_preds = model.predict_proba(X_scaled)[:, 1]
#                 calibrated_preds = calibrator.transform(raw_preds)
                
#                 all_preds.append(calibrated_preds)
#                 full_data = raw_data  # 保留原始数据
#                 print(f"第 {fold_num+1} 折预测完成")
#             except Exception as e:
#                 print(f"第 {fold_num+1} 折预测失败: {str(e)}")
#                 continue
        
#         # 集成预测结果（保持原状）
#         if ensemble_method == 'mean':
#             final_prob = np.mean(all_preds, axis=0)
#         elif ensemble_method == 'median':
#             final_prob = np.median(all_preds, axis=0)
#         else:
#             raise ValueError("不支持的集成方法，可选: mean/median")
        
#         # 组装结果
#         result = full_data.copy()
#         result['pred_prob'] = final_prob
#         result['pred_label'] = np.where(final_prob > 0.5, 1, 0)
        
#         return result


# import pandas as pd
# import numpy as np 

# # train =  pd.read_csv("/kaggle/input/springleaf-marketing-response/train.csv.zip") 
# y= train.target 
# X= train.iloc[: ,:-1]


# X.columns


# test = pd.read_csv('/kaggle/input/email-resposne-eng/test.csv')


# y_pred =  clf.predict_proba(test.values)[:,1]


# submission =  pd.read_csv("/kaggle/input/springleaf-marketing-response/sample_submission.csv.zip")


# submision=pd.DataFrame(y_pred,columns=['target'],index= submission.iloc[:,0])



# submision 


# submision.to_csv('/kaggle/working/submission.csv ')

