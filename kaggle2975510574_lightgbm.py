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
# train = pd.read_feather("/kaggle/input/second-code/train_concat.fth")
# test = pd.read_feather("/kaggle/input/second-code/test_concat.fth")
train=pd.read_csv("/kaggle/input/santander-customer-transaction-prediction/train.csv").drop(['ID_code'],axis=1)
test=pd.read_csv("/kaggle/input/santander-customer-transaction-prediction/test.csv").drop(['ID_code'],axis=1)


from tqdm import tqdm
import numpy as np

te_=test.values
#如果一行数据中的每一列的数据在其他行中都有出现过说明这行数据是伪造数据
unique_samples = []
unique_count = np.zeros_like(te_)
for feature in tqdm(range(te_.shape[1])):
    _, index_, count_ = np.unique(te_[:, feature], return_counts=True, return_index=True)
    unique_count[index_[count_ == 1], feature] += 1

# Samples which have unique values are real the others are fake
real_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) > 0)[:, 0]
synthetic_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) == 0)[:, 0]


test=test.iloc[real_samples_indexes,:]


train_x=train.drop(['target'],axis=1)
train_y=train['target']


#查看划分的比例
import matplotlib.pyplot as plt
import seaborn as sns

def plot_class_distribution(y, title="Class Distribution",save_path=None):
    """
    可视化二分类数据的数量和比例
    :param y: 二分类目标变量 (pd.Series 或 list)
    :param title: 图表标题
    """
    # 转换为 Series
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    # 统计数量和比例
    class_counts = y.value_counts()
    class_percent = y.value_counts(normalize=True) * 100

    # 合并到一个 DataFrame 方便展示
    df = pd.DataFrame({"Count": class_counts, "Percentage": class_percent.round(2)})

    # 绘制柱状图
    plt.figure(figsize=(6, 4))
    sns.barplot(x=df.index, y=df["Count"], palette="Blues_d")
    
    # 在柱子上显示数量和比例
    for i, (count, pct) in enumerate(zip(df["Count"], df["Percentage"])):
        plt.text(i, count + 0.5, f"{count} ({pct}%)", ha="center")

    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.show()
    if save_path:
        plt.savefig(f"{save_path}/{title}")
    

plot_class_distribution(train_y, "Train Target Distribution")


import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

def evaluate_binary_classifier(
    y_true, y_pred, y_proba, model, model_name, 
    label_names=('Class 0', 'Class 1'), title_suffix='(Validation Set)', 
    save_results=True
):
    """
    二分类评估工具函数：打印Accuracy/AUC/分类报告，并绘制混淆矩阵
    :param y_true: 真实标签（验证集）
    :param y_pred: 预测标签（由模型predict得到）
    :param y_proba: 正类概率（由模型predict_proba得到的[:,1]）
    :param model: 训练好的模型
    :param model_name: 模型名称（用于文件夹命名和保存模型）
    :param label_names: 混淆矩阵坐标轴的类别名称
    :param title_suffix: 图标题后缀，便于标注是验证集/测试集
    :param save_results: 是否保存评估结果、混淆矩阵图和模型，默认保存
    :return: 指标字典
    """
    
    # 创建保存目录
    save_dir = f"/kaggle/working/{model_name}_evaluation"
    if save_results and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 评估指标计算
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    
    # 打印数值指标
    print(f"\n=== {title_suffix} ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC: {auc:.4f}")
    print("\n分类报告:")
    print(classification_report(y_true, y_pred))
    
    # 如果需要保存评估结果
    if save_results:
        # 保存评估指标到txt文件
        with open(f"{save_dir}/{title_suffix}_evaluation_results.txt", "w") as f:
            f.write(f"=== {title_suffix} ===\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"AUC: {auc:.4f}\n")
            f.write("\n分类报告:\n")
            f.write(classification_report(y_true, y_pred))
        
        # 保存混淆矩阵图
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_names,
            yticklabels=label_names
        )
        plt.title(f'Confusion Matrix {title_suffix}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f"{save_dir}/{title_suffix}_confusion_matrix.png")
        plt.close()

        # 保存模型
        joblib.dump(model, f"{save_dir}/model_{title_suffix}.joblib")

        #保存标签比例
        result = y_pred.astype(int)
        plot_class_distribution(result, f"{model_name} {title_suffix} Target Distribution",save_dir)
        
    
    # 总是展示混淆矩阵
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=label_names,
        yticklabels=label_names
    )
    plt.title(f'Confusion Matrix {title_suffix}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


import os
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
import lightgbm as lgb

def plot_feature_importance(
    model, 
    feature_names=None, 
    top_n=20, 
    importance_type='gain', 
    title='Top Feature Importance',
    save_path=None
):
    """
    通用特征重要性可视化函数（支持XGBoost、CatBoost和LightGBM）
    :param model: 训练好的 XGBClassifier / CatBoostClassifier / LGBMClassifier
    :param feature_names: 特征名列表（默认自动生成）
    :param top_n: 展示前N个特征
    :param importance_type: 
        - XGBoost: 'weight'/'gain'/'cover'
        - LightGBM: 'split'/'gain'
    :param title: 图标题
    :param save_path: 保存路径（如 'feature_importance.png'），为 None 时不保存
    """

    # ===== XGBoost =====
    if hasattr(model, "get_booster"):  
        plt.figure(figsize=(12, 8))
        xgb.plot_importance(model, max_num_features=top_n, importance_type=importance_type)
        plt.title(title)

    # ===== CatBoost =====
    elif hasattr(model, "get_feature_importance"):  
        importances = model.get_feature_importance()
        if feature_names is None:
            feature_names = [f'feat_{i}' for i in range(len(importances))]
        idx = np.argsort(importances)[::-1][:top_n]
        plt.figure(figsize=(12, 8))
        plt.barh([feature_names[i] for i in idx][::-1], np.array(importances)[idx][::-1])
        plt.title(title)
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()

    # ===== LightGBM =====
    elif hasattr(model, "feature_importances_"):  
        importances = model.feature_importances_
        if feature_names is None:
            feature_names = [f'feat_{i}' for i in range(len(importances))]
        idx = np.argsort(importances)[::-1][:top_n]
        plt.figure(figsize=(12, 8))
        plt.barh([feature_names[i] for i in idx][::-1], np.array(importances)[idx][::-1])
        plt.title(title)
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()

    else:
        raise ValueError("Unsupported model type: only XGBoost, CatBoost and LightGBM are supported.")

    # 保存或显示
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"特征重要性图已保存到: {save_path}")
        plt.show()
        plt.close()
        
    else:
        plt.show()


import lightgbm as lgb
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

def train_lightgbm_cv(
    X, y, n_splits=5, X_test=None,
    lgb_params=None, early_stopping_rounds=50, verbose=100, random_state=42
):
    """
    X: pd.DataFrame
    y: pd.Series or np.array (二分类0/1)
    X_test: 可选 pd.DataFrame，用于生成平均的测试集预测
    lgb_params: 可选，覆盖默认参数的dict
    返回:
      models: 每折模型列表
      oof_pred: OOF概率预测 (与y同长度)
      fold_aucs: 每折AUC
      test_pred: 若提供X_test，则返回对test的平均概率预测；否则为None
    """
    # 保证 y 是一维
    if isinstance(y, (pd.Series, pd.DataFrame)):
        y_array = np.asarray(y).ravel()
    else:
        y_array = np.array(y).ravel()

    # 默认参数（与你原始代码风格一致）  
    default_params = dict(
        objective='binary',  # 二分类
        metric='auc',  # AUC评估
        num_iterations=9670,#6718
        learning_rate=0.07060558415732333,  
        max_depth=1,  # 3
        num_leaves=4,#6
        min_child_samples=6,#15
        min_child_weight=2.8723589065032327,#0.008472548669075467
        subsample=0.7054748813054393,  # 0.8
        colsample_bytree= 0.86138768141066,  # 0.9136515852120487
        reg_alpha=0.00016110079930559584,  #0.0004989217807341253
        reg_lambda= 0.5337203943951745,  #0.02616901004393538
        min_split_gain=0.39037082317718297,#0.4291593183223486
        max_bin=207,#111
        bagging_freq=10,#1
        max_delta_step=10,#6
        feature_fraction=0.8965393908036574,#0.8438775706632261
        scale_pos_weight=0.5053861561043613*9,  #  0.582510612769138
        random_state=random_state,
        n_jobs=-1,
        early_stopping_rounds=early_stopping_rounds,
        verbose=-1
    )
    
    if lgb_params is not None:
        default_params.update(lgb_params)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    oof_pred = np.zeros(len(X), dtype=float)
    test_pred = np.zeros(len(X_test), dtype=float) if X_test is not None else None
    models = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_array), 1):
        print(f"\n===== Fold {fold}/{n_splits} =====")

        # 按行号切片（关键修复点）
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_array[train_idx], y_array[val_idx]

        # 每折计算类别权重（可缓解不平衡）
        params_fold = default_params.copy()

        # 使用 LightGBM 模型
        model = lgb.LGBMClassifier(**params_fold)

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)]
        )

        # ====== 评估：调用封装的评估函数评估训练集 ======
        y_pred = model.predict(X_train)
        y_pred_proba = model.predict_proba(X_train)[:, 1]
        
        metrics = evaluate_binary_classifier(
            y_true=y_train,
            y_pred=y_pred,
            y_proba=y_pred_proba,
            model=model,
            model_name="lightgbm",
            label_names=('Class 0','Class 1'),
            title_suffix=f'(Train Set)_{fold}',
            save_results=True
        )

        #  ====== 评估：调用封装的评估函数评估验证集 ======
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
            
        metrics = evaluate_binary_classifier(
            y_true=y_val,
            y_pred=y_pred,
            y_proba=y_pred_proba,
            model=model,
            model_name="lightgbm",
            label_names=('Class 0','Class 1'),
            title_suffix=f'(Validation Set)_{fold}',
            save_results=True
        )
        models.append(model)
    
    return models


lightgbm_models = train_lightgbm_cv(train_x, train_y)


from lightgbm import create_tree_digraph
import os

def save_lgb_tree_pdfs(lgb_models, tree_index=0, out_prefix="lgb_tree", rankdir="LR"):
    """
    lgb_models: 训练好的 LGBMClassifier/LGBMRegressor 列表
    tree_index: 想导出的树索引（从 0 开始）
    out_prefix: 输出文件前缀
    rankdir: Graphviz 方向，'LR' 左→右，'TB' 上→下
    """
    os.makedirs("lgb_trees_pdf", exist_ok=True)

    for i, model in enumerate(lgb_models, start=1):
        # 兼容不同版本：booster_（新）/_Booster（旧）
        booster = getattr(model, "booster_", None) or getattr(model, "_Booster", None)
        if booster is None:
            raise RuntimeError("模型还未 fit，或未找到 booster_/_Booster。")

        # 生成 graphviz.Digraph
        dot = create_tree_digraph(
            booster,
            tree_index=tree_index,
            show_info=("split_gain", "internal_value", "internal_count", "leaf_count")
        )
        # 设置从左到右
        dot.graph_attr.update(rankdir=rankdir)

        # 渲染为 PDF
        out_path = os.path.join("lgb_trees_pdf", f"{out_prefix}_{i}_{tree_index}")
        dot.render(out_path, format="pdf", cleanup=True)
        print(f"Saved: {out_path}.pdf")
save_lgb_tree_pdfs(lightgbm_models, tree_index=0, out_prefix="lgb_tree", rankdir="LR")


# ====== 特征重要性（如需可关闭/挪到外部）======
i=0
for model in lightgbm_models:
    i+=1
    plot_feature_importance(model, top_n=20, save_path=f"/kaggle/working/lightgbm_feature_importance_{i}.png")


test_pred = np.zeros(len(test))
for model in lightgbm_models:
    test_pred += model.predict_proba(test)[:, 1]

sub = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/sample_submission.csv')
sub['target'] = 0.0
# 对于 real_samples_indexes 中的索引，填充预测值
sub.loc[real_samples_indexes, 'target'] = test_pred
sub.to_csv('/kaggle/working/submission.csv',index=False)
sub.head()


import numpy as np
import optuna
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# --- 小工具：根据是否用GPU返回合适的参数（避免使用 gpu_hist） ---
def _device_params(use_gpu: bool):
    if use_gpu:
        return {"device": "gpu"}  # 使用 GPU版本
    else:
        return {}  # CPU版本

def stratified_subsample(X, y, frac=0.1, random_state=42):
    """分层抽样：用于快速搜参（修正为只抽一次，避免不一致）"""
    X_sub, _, y_sub, _ = train_test_split(
        X, y, test_size=1 - frac, stratify=y, random_state=random_state
    )
    return X_sub, y_sub

def objective(trial, X_train, y_train, X_val, y_val, frac=0.15, use_gpu=False):
    """Optuna优化目标函数"""
    # 子样本（搜参更快）
    X_sub, y_sub = stratified_subsample(X_train, y_train, frac=frac, random_state=42)

    # 定义超参数空间
    param = {
    "num_iterations": trial.suggest_int('num_iterations', 1000, 10000),
    "learning_rate": trial.suggest_float('learning_rate', 0.01, 0.1),
    "max_depth": trial.suggest_int('max_depth', 1, 3),
    "num_leaves": trial.suggest_int('num_leaves', 2, 4),  # 关键：叶子数
    "min_child_samples": trial.suggest_int('min_child_samples', 1, 30),  # 每个叶子最小样本数
    "min_child_weight": trial.suggest_float('min_child_weight', 1e-3, 10.0, log=True),  # LGBM版本的 min_data_in_hessian
    "subsample": trial.suggest_float('subsample', 0.7, 1.0),
    "colsample_bytree": trial.suggest_float('colsample_bytree', 0.7, 1.0),
    "reg_alpha": trial.suggest_float('reg_alpha', 1e-5, 1.0, log=True),
    "reg_lambda": trial.suggest_float('reg_lambda', 1e-5, 5.0, log=True),
    "min_split_gain": trial.suggest_float('min_split_gain', 0.0, 1.0),  # 分裂最小增益
    "max_bin": trial.suggest_int('max_bin', 63, 255),  # 分箱数，精度 vs 速度
    "bagging_freq": trial.suggest_int('bagging_freq', 1, 10),  # bagging频率
    "feature_fraction": trial.suggest_float('feature_fraction', 0.7, 1.0),  # 等价于 colsample_bytree
    "scale_pos_weight": trial.suggest_float('scale_pos_weight', 0.5, 2.0),
    "max_delta_step": trial.suggest_int('max_delta_step', 0, 10),
}


    # 类别不平衡（基于子样本）
    pos = np.sum(y_sub == 1)
    neg = np.sum(y_sub == 0)
    spw = (neg / pos) if pos > 0 else 1.0
    param['scale_pos_weight'] = spw * param['scale_pos_weight']

    # 创建 LightGBM 模型
    model = lgb.LGBMClassifier(
        objective='binary',
        metric='auc',
        n_estimators=3000,
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1,
        **_device_params(use_gpu),
        **param,
        verbose=-1
    )

    # 转换为 NumPy 数组
    X_sub_np = X_sub.values if hasattr(X_sub, "values") else np.asarray(X_sub)
    X_val_np = X_val.values if hasattr(X_val, "values") else np.asarray(X_val)
    y_sub_np = np.asarray(y_sub)
    y_val_np = np.asarray(y_val)

    # 训练模型并评估 AUC
    model.fit(X_sub_np, y_sub_np, eval_set=[(X_val_np, y_val_np)])
    val_auc = roc_auc_score(y_val_np, model.predict_proba(X_val_np)[:, 1])

    return val_auc  # 目标是最大化 AUC


def tune_lgb_with_optuna(X_train, y_train, X_val, y_val, n_trials=25, use_gpu=False):
    """使用 Optuna 优化 LightGBM 超参数"""
    study = optuna.create_study(direction='maximize')  # 目标是最大化 AUC
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val, use_gpu=use_gpu), n_trials=n_trials)

    print("Best parameters:", study.best_params)
    print("Best AUC:", study.best_value)
    return study.best_params

def train_full_lgb(X_train, y_train, X_val, y_val, best_params, use_gpu=False):
    """用全量数据+早停做精修训练"""
    # 转 numpy
    X_train_np = X_train.values if hasattr(X_train, "values") else np.asarray(X_train)
    X_val_np = X_val.values if hasattr(X_val, "values") else np.asarray(X_val)
    y_train_np = np.asarray(y_train)
    y_val_np = np.asarray(y_val)

    # 创建最终模型
    model = lgb.LGBMClassifier(
        objective='binary',
        metric='auc',
        n_estimators=5000,
        early_stopping_rounds=100,
        random_state=42,
        n_jobs=-1,
        **_device_params(use_gpu),
        **best_params,
        verbose=-1
    )

    # 训练模型
    model.fit(X_train_np, y_train_np, eval_set=[(X_val_np, y_val_np)])

    # 验证集 AUC
    val_auc = roc_auc_score(y_val_np, model.predict_proba(X_val_np)[:, 1])
    print("Final model AUC:", val_auc)
    return model



#划分训练集和验证集
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(train_x, train_y, test_size=0.2, random_state=42)


best_params = tune_lgb_with_optuna(X_train=x_train, y_train=y_train, X_val=x_val, y_val=y_val, n_trials=50, use_gpu=True)

# 步骤2：使用调优后的超参数训练最终模型
final_model = train_full_lgb(X_train=x_train, y_train=y_train, X_val=x_val, y_val=y_val, best_params=best_params, use_gpu=True)

