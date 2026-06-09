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


pip install "optuna-integration[xgboost]"


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score, KFold
from sklearn.metrics import log_loss, accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from optuna.integration import XGBoostPruningCallback
from optuna.integration import LightGBMPruningCallback
from optuna.integration import CatBoostPruningCallback
from sklearn.utils import shuffle
# 注意：这些变量需要在有数据后才能定义
# numerical_cols = train.select_dtypes(include=['int64','float64']).columns.tolist()
# categorical_cols = train.select_dtypes(include=['object']).columns.tolist()
# X, y = train.drop(columns=['personality'], axis=1), train['personality']
# X_original, y_original = original.drop(columns=['personality'], axis=1), original['personality']

def numerical_feature_engineering(numerical_cols, train,val, test):
    """
    取交叉积特征，
    取高次项特征
    """
    train_copy= train.copy()
    test_copy = test.copy()
    val_copy = val.copy()
    print("Before transform shapes:", train_copy.shape, val_copy.shape, test_copy.shape)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    train_poly = poly.fit_transform(train_copy[numerical_cols])
    val_poly = poly.transform(val_copy[numerical_cols])
    test_poly = poly.transform(test_copy[numerical_cols])
    train_poly = pd.DataFrame(train_poly, columns=poly.get_feature_names_out(numerical_cols))
    val_poly = pd.DataFrame(val_poly,columns=poly.get_feature_names_out(numerical_cols))
    test_poly = pd.DataFrame(test_poly, columns=poly.get_feature_names_out(numerical_cols))
    train_poly.index = train_copy.index
    val_poly.index = val_copy.index
    test_poly.index = test_copy.index
    return train_poly, val_poly,test_poly
    
def binned_numerical_engineering(numerical_cols,train,val,test):
    """
    对数值特征进行分箱处理
    """
    train_copy = train.copy()
    test_copy = test.copy()
    val_copy = val.copy()
    
    for col in numerical_cols:
        if col in train_copy.columns:
            train_copy[col+"binned"] = pd.qcut(train_copy[col], q=10, duplicates="drop", labels=False)
            test_copy[col+"binned"] = pd.qcut(test_copy[col], q=10, duplicates="drop", labels=False)
            val_copy[col+"binned"] = pd.qcut(val_copy[col], q=10, duplicates='drop',labels=False)
            train_copy = train_copy.drop(columns=col)
            test_copy = test_copy.drop(columns=col)
            val_copy = val_copy.drop(columns = col)

    return train_copy, val_copy, test_copy

def categorical_feature_engineering(train,val,test):
    """
    对分类特征编码
    """
    train_copy = train.copy()
    test_copy = test.copy()
    val_copy = val.copy()
    le = LabelEncoder()
    categorical_cols = train_copy.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        train_copy[col]=le.fit_transform(train_copy[col].astype(str))
        val_copy[col] = le.transform(val_copy[col].astype(str))
        test_copy[col] = le.transform(test_copy[col].astype(str))
    return train_copy, val_copy,test_copy

# 定义模型参数的configuration,xgboost, lightgbm,catboost
def objective_xgboost(trial, X_train, y_train, skf):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method':'hist',
        'gpu_id':0,
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42,
        'verbose': -1
    }
    model = XGBClassifier(**params)
    scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
    return scores.mean()
    
def objective_lightgbm(trial, X_train, y_train, skf):
    params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'device_type':'gpu',
    'gpu_platform_id':0,
    'gpu_id':0,
    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
    'max_depth': trial.suggest_int('max_depth', 3, 10),
    'num_leaves': trial.suggest_int('num_leaves', 20, 100),
    'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
    'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
    'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
    'random_state': 42,
        'verbose':-1
    }
    model = LGBMClassifier(**params)
    scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
    return scores.mean()

def objective_catboost(trial, X_train, y_train, skf):
    params = {
        'loss_function': 'Logloss',
        'eval_metric': 'Logloss',
        'task_type':'GPU',
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'depth': trial.suggest_int('depth', 3, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_strength': trial.suggest_float('random_strength', 1e-3, 10.0, log=True),
        'random_state': 42,
        'verbose': False
    }
    model = CatBoostClassifier(**params)
    scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
    return scores.mean()

# 定义训练主函数
def train_model(X_train, y_train, model_type, sample_weights=None):
    """
    训练模型，支持样本权重
    """
    
    # 定义交叉验证，使用分层抽样
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 定义模型
    if model_type == 'xgboost': 
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective_xgboost(trial, X_train, y_train, skf), n_trials=50)
        best_params = study.best_params
        print('Best parameters for XGBoost:', best_params)
        print('best cv score:', study.best_value)
        best_model = XGBClassifier(**best_params)
        best_model.fit(X_train, y_train, sample_weight=sample_weights)

    elif model_type == 'lightgbm':
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective_lightgbm(trial, X_train, y_train, skf), n_trials=50)
        best_params = study.best_params
        print('Best parameters for LightGBM:', best_params)
        print('best cv score:', study.best_value)
        best_model = LGBMClassifier(**best_params)
        best_model.fit(X_train, y_train, sample_weight=sample_weights)
        
    elif model_type == 'catboost':
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective_catboost(trial, X_train, y_train, skf), n_trials=50)
        best_params = study.best_params
        print('Best parameters for CatBoost:', best_params)
        print('best cv score:', study.best_value)  
        best_model = CatBoostClassifier(**best_params)
        best_model.fit(X_train, y_train, sample_weight=sample_weights)

    return best_model

from typing import List, Callable, Optional
import random
class HillClimbingOptimizer:
    def __init__(
        self,
        predictions: List[np.ndarray],  # 每个模型的预测结果
        true_labels: np.ndarray,        # 真实标签
        step_size: float = 0.01,
        max_iterations: int = 1000,
        random_restarts: int = 5,
        eval_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
    ):
        """
        初始化登山法优化器
        
        参数:
            predictions: 每个模型的预测结果列表，每个元素是一个numpy数组，形状为(n_samples, n_classes)
            true_labels: 真实标签，形状为(n_samples,)
            step_size: 每次迭代的步长
            max_iterations: 最大迭代次数
            random_restarts: 随机重启次数，用于避免局部最优
            eval_metric: 评估指标函数，如果为None则使用准确率
        """
        self.predictions = predictions
        self.true_labels = true_labels
        self.n_models = len(predictions)
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.random_restarts = random_restarts
        self.eval_metric = eval_metric or self._default_metric
        
        # 验证输入数据
        self._validate_inputs()
        
    def _validate_inputs(self) -> None:
        """验证输入数据的有效性"""
        if not self.predictions:
            raise ValueError("预测结果列表不能为空")
        
        # 检查所有预测结果的形状是否一致
        shape = self.predictions[0].shape
        for i, pred in enumerate(self.predictions):
            if pred.shape != shape:
                raise ValueError(f"模型{i}的预测结果形状与其他不一致")
        
        # 检查预测结果与真实标签的样本数是否匹配
        if shape[0] != len(self.true_labels):
            raise ValueError("预测结果与真实标签的样本数不匹配")
    
    def _normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        """归一化权重，使其和为1"""
        return weights / np.sum(weights)
    
    def _random_weights(self) -> np.ndarray:
        """生成随机初始权重"""
        weights = np.random.random(self.n_models)
        return self._normalize_weights(weights)
    
    def _default_metric(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """默认的评估指标：准确率"""
        return np.mean(np.argmax(y_pred, axis=1) == y_true)
    
    def _ensemble_predict(self, weights: np.ndarray) -> np.ndarray:
        """根据权重融合模型预测结果"""
        weighted_sum = np.zeros_like(self.predictions[0])
        for w, pred in zip(weights, self.predictions):
            weighted_sum += w * pred
        return weighted_sum
    
    def optimize(self) -> tuple[np.ndarray, float]:
        """
        使用登山法寻找最优权重
        
        返回:
            best_weights: 最优权重数组
            best_score: 最优分数
        """
        best_weights = None
        best_score = float('-inf')
        
        for _ in range(self.random_restarts):
            # 随机初始化权重
            current_weights = self._random_weights()
            current_pred = self._ensemble_predict(current_weights)
            current_score = self.eval_metric(current_pred, self.true_labels)
            
            for _ in range(self.max_iterations):
                improved = False
                
                # 对每个权重进行调整
                for i in range(self.n_models):
                    # 尝试增加权重
                    temp_weights = current_weights.copy()
                    temp_weights[i] += self.step_size
                    temp_weights = self._normalize_weights(temp_weights)
                    pred_up = self._ensemble_predict(temp_weights)
                    score_up = self.eval_metric(pred_up, self.true_labels)
                    
                    # 尝试减少权重
                    temp_weights = current_weights.copy()
                    temp_weights[i] -= self.step_size
                    temp_weights = self._normalize_weights(temp_weights)
                    pred_down = self._ensemble_predict(temp_weights)
                    score_down = self.eval_metric(pred_down, self.true_labels)
                    
                    # 选择最好的方向
                    if score_up > current_score or score_down > current_score:
                        if score_up > score_down:
                            current_weights[i] += self.step_size
                            current_score = score_up
                        else:
                            current_weights[i] -= self.step_size
                            current_score = score_down
                        current_weights = self._normalize_weights(current_weights)
                        improved = True
                
                # 如果没有改进，说明达到局部最优
                if not improved:
                    break
                    
            # 更新全局最优解
            if current_score > best_score:
                best_score = current_score
                best_weights = current_weights.copy()
                
        if best_weights is None:
            best_weights = self._random_weights()
            
        return best_weights, best_score


    # 将X和y分为训练集和验证集

def pipeline(X, y, test, original_X, original_y):
    '''
对数据进行三类预处理，分别对应不同的模型
1. 利用pinned 将所有数值数据都转化成分类数据再导入 catboost 训练
2. 利用交叉积，将所有数值转化为多项式导入 lightgbm训练
3. 不处理特征值，导入original数据， 使用xgboost 训练
'''
    # 数据划分
    y=y.map({'Extrovert':0, 'Introvert':1})
    original_y=original_y.map({'Extrovert':0,'Introvert':1})
    # 计算类别权重
    class_counts = y.value_counts()
    total_samples = len(y)
    class_weights = {
        class_label: total_samples / (len(class_counts) * count)
        for class_label, count in class_counts.items()
    }
    print("类别权重:", class_weights)
    
    # 为每个样本计算权重
    sample_weights = y.map(class_weights)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # 相应的样本权重也需要分割
    train_weights = sample_weights.loc[X_train.index]
    val_weights = sample_weights.loc[X_val.index]

    X_original, y_original = original_X.copy(), original_y.copy()


 # 分类数据处理
 # 复制训练数据，验证数据，测试数据， 进行统一的preprocessing
    X_train_cat = X_train.copy() # 复制训练数据
    X_val_cat = X_val.copy() #复制验证数据
    test_cat = test.copy() #复制测试数据
    y_train_cat = y_train.copy() #复制训练标签
    y_val_cat = y_val.copy() #复制验证标签
    # 对训练，测试，验证数据进行分箱处理
    X_train_binned,X_val_binned, test_binned = binned_numerical_engineering(numerical_cols, X_train_cat, X_val_cat, test_cat)



    # 对训练，测试，验证数据进行编码
    X_train_binned,X_val_binned, test_binned = categorical_feature_engineering(X_train_binned,X_val_binned, test_binned)
   # 训练 catboost

    cat_model = train_model(X_train_binned, y_train_cat, model_type='catboost', sample_weights=train_weights)
    cat_model_xgboost = train_model(X_train_binned, y_train_cat, model_type='xgboost', sample_weights=train_weights)
    # 数值特征处理路径
    X_train_num = X_train.copy() #复制训练数据
    X_val_num = X_val.copy()  #复制验证数据
    test_num = test.copy() #复制测试训练数据
    y_train_num = y_train.copy() #复制训练标签
    y_val_num = y_val.copy()  #复制验证标签
    print('after copy:',len(X_val_num))
    #对训练，测试，验证数据进行交叉积处理
    X_train_num_poly,X_val_num_poly, test_num_poly = numerical_feature_engineering(numerical_cols, X_train_num, X_val_num,test_num)
    # 其中还有两个非数值特征，需要编码
    print('first transform',len(X_val_num_poly))
    X_train_num_poly,X_val_num_poly, test_num_poly = categorical_feature_engineering(
        X_train_num_poly, X_val_num_poly,test_num_poly
    )
    print('second_transform',len(X_val_num_poly))
    poly_model = train_model(X_train_num_poly, y_train_num, model_type='lightgbm', sample_weights=train_weights)
    # 不处理特征，直接训练模型，加入original 数据
    X_train_original = X_train.copy()
    y_train_original = y_train.copy()
    X_val_original =X_val.copy()
    test_original = test.copy()
    # 修正：应为original_X, original_y，而不是X_original, y_original
    #repeat_times = round(len(X_train) / len(original_X))
    #for i in range(repeat_times):
       # X_train_original = pd.concat([X_train_original, original_X], axis=0)
       # y_train_original = pd.concat([y_train_original, original_y], axis=0)

   # X_train_original = X_train_original.reset_index(drop=True)
   # y_train_original = y_train_original.reset_index(drop=True)
   # X_train_original, X_val_original, test_original = categorical_feature_engineering(X_train_original,X_val_original, test_original)
   # train_weights_original = train_weights.copy()
   # for i in range(repeat_times):
   #     train_weights_original = pd.concat([train_weights_original, train_weights], axis=0)
   # train_weights_original = train_weights_original.reset_index(drop=True)

#    original_model = train_model(X_train_original, y_train_original, model_type='xgboost', sample_weights=train_weights_original)
    original_weights = original_y.map(class_weights)
# 确定重复倍数
    repeat_times = max(1, round(len(X_train) / len(original_X)))

# 拼接原始数据（多次）
    X_train_xgb = pd.concat([X_train] + [original_X] * repeat_times, axis=0).reset_index(drop=True)
    y_train_xgb = pd.concat([y_train] + [original_y] * repeat_times, axis=0).reset_index(drop=True)
    train_weights_xgb = pd.concat([train_weights] + [original_weights] * repeat_times, axis=0).reset_index(drop=True)

# 进行编码（只对categorical列，不要搞乱数值）
    X_train_xgb, X_val_xgb, test_xgb = categorical_feature_engineering(
    X_train_xgb, X_val.copy(), test.copy())

# 打乱顺序（可选但推荐）
    
    X_train_xgb, y_train_xgb, train_weights_xgb = shuffle(
        X_train_xgb, y_train_xgb, train_weights_xgb, random_state=42
    )
    original_model = train_model(X_train_xgb, y_train_xgb, model_type='xgboost', sample_weights=train_weights_xgb)

    
    # 模型融合时，使用同一个验证集X_val/y_val进行融合和评估
    # 使用登山法找最优权重
    print(len(X_val_binned),len(X_val_num_poly),len(X_val_xgb))
    predictions =[
        cat_model.predict_proba(X_val_binned),
        poly_model.predict_proba(X_val_num_poly),
        original_model.predict_proba(X_val_xgb)
    ]
    true_labels = y_val.values
    optimizer = HillClimbingOptimizer(
        predictions = predictions,
        true_labels = true_labels,
        step_size= 0.01,
        max_iterations= 800,
        random_restarts=5
    )
    best_weights, best_score = optimizer.optimize()
    print("最优权重", best_weights)
    print("最优分数", best_score)
    print(len(test_binned),len(test_num_poly),len(test_xgb))
    cat_test_pred = cat_model.predict_proba(test_binned)
    poly_test_pred = poly_model.predict_proba(test_num_poly)
    original_test_pred = original_model.predict_proba(test_xgb)
    test_predictions = [cat_test_pred,poly_test_pred,original_test_pred]
    final_predictions = np.zeros_like(test_predictions[0])
    for w, pred in zip(best_weights, test_predictions):
        final_predictions += w*pred


    
    
    return final_predictions


# 使用示例函数

def load_and_prepare_data(train_path, test_path=None):
    """
    加载和准备数据
    """
    train = pd.read_csv(train_path)
    global numerical_cols, categorical_cols, X, y
    
    X = train.drop(columns=['Personality','id'], axis=1)
    
    y = train['Personality']
    
    # 定义全局变量
    
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    

    if test_path:
        test = pd.read_csv(test_path)
        test=test.drop(columns=['id'],axis=1)
        
        return X, y, test
    else:
        return X, y, None



train_path=f"/kaggle/input/playground-series-s5e7/train.csv"
test_path= f"/kaggle/input/playground-series-s5e7/test.csv"
X,y,test =  load_and_prepare_data(train_path, test_path=test_path)




original_1=f"/kaggle/input/extrovert-vs-introvert-behavior-data/personality_datasert.csv"
original_2 = f"/kaggle/input/extrovert-vs-introvert-behavior-data/personality_dataset.csv"


df_original_1 = pd.read_csv(original_1)
df_original_2= pd.read_csv(original_2)


df_original = pd.concat([df_original_1,df_original_2],axis=0)
df_original = df_original.reset_index(drop=True)
original_X=df_original.drop(columns=['Personality'], axis=1)
original_y = df_original['Personality']


for col in numerical_cols:
    X[col].fillna(X[col].median(), inplace=True)
    test[col].fillna(X[col].median(), inplace=True)
    df_original.fillna(df_original[col].median(),inplace=True)


for col in categorical_cols:
    X[col].fillna(X[col].mode()[0], inplace=True)
    test[col].fillna(X[col].mode()[0],inplace=True)
    df_original.fillna(df_original[col].mode()[0],inplace=True)








final_predictions=pipeline(X, y, test, original_X, original_y)


submission =pd.read_csv(f'/kaggle/input/playground-series-s5e7/sample_submission.csv')


# Convert predictions back to string labels
y_pred = np.argmax(final_predictions, axis=1)
label_map = {0: 'Extrovert', 1: 'Introvert'}
predicted_labels = pd.Series(y_pred).map(label_map)

submission['Personality'] =predicted_labels
submission.to_csv('submission.csv', index=False)

print(submission.head())




