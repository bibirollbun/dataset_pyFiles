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


import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any, Tuple

def analyze_correlations(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    methods: Tuple[str, ...] = ("pearson", "spearman"),
    top_k: int = 30,
    high_threshold: float = 0.90,
    min_non_nan: int = 3,
    compute_vif: bool = False,
    vif_threshold: Optional[float] = None,
    plot: bool = False,
    figsize: Tuple[int, int] = (12, 10),
) -> Dict[str, Any]:
    """
    针对多列（例如 200 列）数值特征做相关性分析。

    参数
    ----
    df : DataFrame
        原始数据表。
    cols : list[str] or None
        参与分析的列名；None 则自动选取 df 中的数值列。
    methods : ('pearson','spearman',...)
        相关系数类型；可同时给多个。
    top_k : int
        每种方法返回 |corr| 最大的前 top_k 对（去重后）。
    high_threshold : float
        判定“高相关”的阈值（基于绝对值），用于生成 high_pairs 与建议删除列表。
    min_non_nan : int
        计算相关时每对变量至少需要的有效样本数（过低会让结果不稳定）。
    compute_vif : bool
        是否计算方差膨胀因子（VIF）。依赖 statsmodels，如环境没有会自动跳过。
    vif_threshold : float or None
        若给定，则返回 vif_exceed 列表，包含 VIF 超阈值的变量。
    plot : bool
        是否画热力图（matplotlib）。200 列较大，图会比较密集，仅做概览。
    figsize : (w,h)
        热力图尺寸。

    返回
    ----
    result : dict
        {
          'used_columns': [参与分析的列名],
          'corr': {method: corr_matrix_df},
          'top_pairs': {method: DataFrame(columns=['feature_1','feature_2','corr','abs_corr'])},
          'high_pairs': {method: 同上，筛选 |corr|>=threshold},
          'suggest_drop': {method: [根据 high_pairs 贪心建议删除的变量名列表]},
          'vif': DataFrame or None,
          'vif_exceed': list or None
        }
    """
    # 1) 列选择与预处理
    if cols is None:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    # 只保留至少有 min_non_nan 非空的列
    good_cols = [c for c in num_cols if df[c].notna().sum() >= min_non_nan]
    data = df[good_cols].copy()

    result: Dict[str, Any] = {
        "used_columns": good_cols,
        "corr": {},
        "top_pairs": {},
        "high_pairs": {},
        "suggest_drop": {},
        "vif": None,
        "vif_exceed": None,
    }

    if len(good_cols) < 2:
        return result

    # 2) 相关矩阵计算（支持多种方法）
    for m in methods:
        corr = data.corr(method=m, min_periods=min_non_nan)
        result["corr"][m] = corr

        # 3) 展平为 pair 列表（仅上三角，去掉对角线）
        #    用向量化方式，效率较高
        cols_arr = np.array(corr.columns)
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        corr_vals = corr.where(mask).stack()
        pairs = corr_vals.rename("corr").reset_index()
        pairs.columns = ["feature_1", "feature_2", "corr"]
        pairs["abs_corr"] = np.abs(pairs["corr"])

        # 取 Top-K（按 |corr| 排序）
        top_pairs = pairs.sort_values("abs_corr", ascending=False).head(top_k).reset_index(drop=True)
        result["top_pairs"][m] = top_pairs

        # 4) 高相关对（|corr| >= threshold）
        high_pairs = pairs[pairs["abs_corr"] >= high_threshold].sort_values("abs_corr", ascending=False).reset_index(drop=True)
        result["high_pairs"][m] = high_pairs

        # 5) 基于高相关对的“建议删除列表”
        #    贪心策略：每次删掉“连接度（高相关次数）最高”的变量；
        #    若连接度相同，删掉与“全局平均|corr|”更高的变量（代表更冗余）。
        to_drop = []
        if not high_pairs.empty:
            # 建图（无向）
            deg = {}
            for _, row in high_pairs.iterrows():
                a, b = row["feature_1"], row["feature_2"]
                deg[a] = deg.get(a, 0) + 1
                deg[b] = deg.get(b, 0) + 1

            # 预先计算每个变量的“全局平均|corr|”（不含自己）
            avg_abs = corr.abs().where(~np.eye(len(corr), dtype=bool)).mean(axis=1).to_dict()

            # 工作副本
            remaining_edges = set(tuple(sorted((r["feature_1"], r["feature_2"]))) for _, r in high_pairs.iterrows())
            remaining_nodes = set(corr.columns)

            while remaining_edges:
                # 计算当前度
                curr_deg = {v: 0 for v in remaining_nodes}
                for a, b in remaining_edges:
                    if a in remaining_nodes and b in remaining_nodes:
                        curr_deg[a] += 1
                        curr_deg[b] += 1
                # 选择要移除的点：度最大；若并列，avg_abs 最大
                max_deg = max(curr_deg.values())
                candidates = [v for v, d in curr_deg.items() if d == max_deg]
                if len(candidates) > 1:
                    pick = max(candidates, key=lambda v: avg_abs.get(v, 0.0))
                else:
                    pick = candidates[0]

                to_drop.append(pick)
                remaining_nodes.discard(pick)
                # 删除包含 pick 的所有边
                remaining_edges = {e for e in remaining_edges if pick not in e}

        result["suggest_drop"][m] = to_drop

    # 6) 可选：VIF（方差膨胀因子）
    if compute_vif and len(good_cols) >= 2:
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            X = data.dropna(axis=0)  # 简单方式：完整案例
            if X.shape[0] >= 5:      # 样本过少时 VIF 不稳定
                X_ = X.assign(_const=1.0).astype(float).values
                vif_vals = []
                for i in range(X.shape[1]):
                    vif_vals.append(variance_inflation_factor(X_, i))
                vif_df = pd.DataFrame({"feature": X.columns, "VIF": vif_vals}).sort_values("VIF", ascending=False).reset_index(drop=True)
                result["vif"] = vif_df
                if vif_threshold is not None:
                    result["vif_exceed"] = vif_df.loc[vif_df["VIF"] >= vif_threshold, "feature"].tolist()
        except Exception:
            # statsmodels 不可用或运算失败时忽略
            result["vif"] = None
            result["vif_exceed"] = None

    # 7) 可选：绘图（总览热力图）
    if plot:
        import matplotlib.pyplot as plt
        for m in methods:
            corr = result["corr"][m]
            plt.figure(figsize=figsize)
            plt.imshow(corr.values, interpolation="nearest")
            plt.title(f"Correlation Heatmap ({m})")
            plt.colorbar()
            plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=8)
            plt.yticks(range(len(corr.index)), corr.index, fontsize=8)
            plt.tight_layout()
            plt.show()

    return result



# # df 为你的 DataFrame（含 ~200 数值列）
# res = analyze_correlations(
#     train,
#     methods=("pearson","spearman"),
#     top_k=200,
#     high_threshold=0.01,
#     compute_vif=True,
#     vif_threshold=10.0,
#     plot=False  # 200 列热力图很密，默认不画
# )


# 取 Pearson 的 Top 相关对
# res["top_pairs"]["pearson"].iloc[:,:]


# 看看高相关对与建议删除的列（Pearson）
# res["high_pairs"]["pearson"].head()
# res["suggest_drop"]["pearson"]


# 若需要 VIF 结果
# res["vif"]         # DataFrame
# res["vif_exceed"]  # 超过阈值的变量名


train


test


#查看缺失值
from prettytable import PrettyTable
import numpy as np

table = PrettyTable()

table.field_names = ['Feature', 'Data Type', 'Train Missing %', 'Test Missing %']
for column in train.columns:
    data_type = str(train[column].dtype)
    non_null_count_train= np.round(100-train[column].count()/train.shape[0]*100,1)
    if column!='target':
        non_null_count_test = np.round(100-test[column].count()/test.shape[0]*100,1)
    else:
        non_null_count_test="NA"
    table.add_row([column, data_type, non_null_count_train,non_null_count_test])
print(table)


train_x=train.drop(['target'],axis=1)
train_y=train['target']


train_x


train_y


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

def plot_feature_importance(
    model, 
    feature_names=None, 
    top_n=20, 
    importance_type='gain', 
    title='Top Feature Importance',
    save_path=None
):
    """
    通用特征重要性可视化函数（支持XGBoost和CatBoost）
    :param model: 训练好的 XGBClassifier 或 CatBoostClassifier
    :param feature_names: 特征名列表（默认自动生成）
    :param top_n: 展示前N个特征
    :param importance_type: XGBoost的重要性类型（'weight'/'gain'/'cover'等）
    :param title: 图标题
    :param save_path: 保存路径（如 'feature_importance.png'），为 None 时不保存
    """

    if hasattr(model, "get_booster"):  # XGBoost
        plt.figure(figsize=(12, 8))
        xgb.plot_importance(model, max_num_features=top_n, importance_type=importance_type)
        plt.title(title)

    elif hasattr(model, "get_feature_importance"):  # CatBoost
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

    else:
        raise ValueError("Unsupported model type: only XGBoost and CatBoost are supported.")

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


import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold

def train_xgboost_cv(
    X, y, n_splits=5, X_test=None,
    xgb_params=None, early_stopping_rounds=50, verbose=100, random_state=42
):
    """
    X: pd.DataFrame
    y: pd.Series or np.array (二分类0/1)
    X_test: 可选 pd.DataFrame，用于生成平均的测试集预测
    xgb_params: 可选，覆盖默认参数的dict
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

    # 默认参数（与你原始代码风格一致） 'scale_pos_weight': 0.67459337735408}
    default_params = dict(
        objective='binary:logistic',
        eval_metric='logloss',
        n_estimators=10000,
        learning_rate= 0.06013845295170628,#0.03
        max_depth=1,# 3     
        subsample=0.816954814787356,#0.8
        min_child_weight=4,#7
        colsample_bytree=0.9026884174677129,#0.7
        reg_alpha=5.39358790419384e-05,#0.1
        reg_lambda= 1.1777748682560547,#0.5
        scale_pos_weight= 0.67459337735408*9,#4.5
        gamma=3.1378647727234616,
        max_delta_step=0,
        random_state=random_state,
        n_jobs=-1,
        early_stopping_rounds=early_stopping_rounds
    )
    if xgb_params is not None:
        default_params.update(xgb_params)

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
        # pos = np.sum(y_train == 1)
        # neg = np.sum(y_train == 0)
        # scale_pos_weight = (neg / pos) if pos > 0 else 1.0

        params_fold = default_params.copy()
        # params_fold['scale_pos_weight'] = params_fold.get('scale_pos_weight', scale_pos_weight)

        model = xgb.XGBClassifier(**params_fold)

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=verbose,
        )

        # ====== 评估：调用封装的评估函数评估训练集 ======
        y_pred = model.predict(X_train)
        y_pred_proba = model.predict_proba(X_train)[:, 1]
        
        metrics = evaluate_binary_classifier(
            y_true=y_train,
            y_pred=y_pred,
            y_proba=y_pred_proba,
            model=model,
            model_name="xgboost",
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
            model_name="xgboost",
            label_names=('Class 0','Class 1'),
            title_suffix=f'(Validation Set)_{fold}',
            save_results=True
        )
        models.append(model)
    return models
xgboost_models=train_xgboost_cv(train_x,train_y)


# ====== 特征重要性（如需可关闭/挪到外部）======
i=0
for xgboost_model in xgboost_models:
    i+=1
    plot_feature_importance(xgboost_model, top_n=20, save_path=f"/kaggle/working/xgboost_evaluation/xgboost_feature_importance_{i}.png")


from xgboost import to_graphviz
i=0
for xgboost_model in xgboost_models:
    i+=1
    booster = xgboost_model.get_booster()
    
    # 导出第0棵树
    dot = to_graphviz(booster, num_trees=0, rankdir='LR')
    dot.render(f'xgb_tree_{i}_0', format='pdf')  # 会生成 xgb_tree_0.pdf


test_pred = np.zeros(len(test))
for model in xgboost_models:
    test_pred += model.predict_proba(test)[:, 1]

sub = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/sample_submission.csv')
sub['target'] = 0.0
# 对于 real_samples_indexes 中的索引，填充预测值
sub.loc[real_samples_indexes, 'target'] = test_pred
sub.to_csv('/kaggle/working/xgboost_evaluation/submission.csv',index=False)
sub.head()


import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV, PredefinedSplit
from sklearn.metrics import roc_auc_score

# --- 小工具：根据是否用GPU返回合适的参数（避免使用 gpu_hist） ---
def _device_params(use_gpu: bool):
    """
    XGBoost>=2.0 推荐：
      - CPU: tree_method='hist'
      - GPU: tree_method='hist', device='cuda'
    """
    if use_gpu:
        # 若环境无GPU，XGBoost会自行回退到CPU；这里不再使用 gpu_hist
        return {"tree_method": "hist", "device": "cuda"}
    else:
        return {"tree_method": "hist"}  # 纯CPU

def stratified_subsample(X, y, frac=0.1, random_state=42):
    """分层抽样：用于快速搜参（修正为只抽一次，避免不一致）"""
    X_sub, _, y_sub, _ = train_test_split(
        X, y, test_size=1 - frac, stratify=y, random_state=random_state
    )
    return X_sub, y_sub

def quick_tune_xgb(X_train, y_train, X_val, y_val, n_iter=25, random_state=42, use_gpu=False, frac=0.15):
    """在小样本上快速搜参，然后返回最优参数"""
    X_sub, y_sub = stratified_subsample(X_train, y_train, frac=frac, random_state=random_state)

    # 兼容 pandas
    X_sub_np = X_sub.values if hasattr(X_sub, "values") else np.asarray(X_sub)
    X_val_np = X_val.values if hasattr(X_val, "values") else np.asarray(X_val)
    y_sub_np = np.asarray(y_sub)
    y_val_np = np.asarray(y_val)

    # 固定验证折
    X_all = np.vstack([X_sub_np, X_val_np])
    y_all = np.concatenate([y_sub_np, y_val_np])
    test_fold = np.concatenate([
        -1 * np.ones(len(X_sub_np), dtype=int),
         0 * np.ones(len(X_val_np), dtype=int)
    ])
    ps = PredefinedSplit(test_fold)

    # 类别不平衡权重（基于子样本）
    pos = np.sum(y_sub_np == 1)
    neg = np.sum(y_sub_np == 0)
    spw = (neg / pos) if pos > 0 else 1.0

    base = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',           # 与 scoring 对齐
        n_estimators=3000,
        early_stopping_rounds=50,
        random_state=random_state,
        n_jobs=-1,
        **_device_params(use_gpu)
    )

    param_dist = {
        "learning_rate":    [0.03, 0.05, 0.07, 0.1, 0.15],
        "max_depth":        [3, 4, 5, 6, 8],
        "min_child_weight": [1, 2, 3, 5, 8],
        "subsample":        [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "reg_alpha":        [0.0, 0.1, 0.5, 1.0],
        "reg_lambda":       [0.5, 1.0, 2.0, 5.0],
        "scale_pos_weight": [spw * 0.5, spw, spw * 1.5, spw * 2.0]
    }

    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=ps,
        verbose=1,
        n_jobs=-1,
        random_state=random_state,
        refit=True
    )

    search.fit(X_all, y_all, eval_set=[(X_val_np, y_val_np)], verbose=False)
    print("快速搜参最优参数：", search.best_params_)
    print("快速搜参最佳AUC：", search.best_score_)
    return search.best_params_

def train_full_xgb(X_train, y_train, X_val, y_val, best_params, random_state=42, use_gpu=False):
    """用全量数据+早停做精修训练"""
    X_train_np = X_train.values if hasattr(X_train, "values") else np.asarray(X_train)
    X_val_np   = X_val.values   if hasattr(X_val, "values")   else np.asarray(X_val)
    y_train_np = np.asarray(y_train)
    y_val_np   = np.asarray(y_val)

    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        n_estimators=5000,
        early_stopping_rounds=100,
        random_state=random_state,
        n_jobs=-1,
        **_device_params(use_gpu),
        **best_params
    )
    model.fit(X_train_np, y_train_np, eval_set=[(X_val_np, y_val_np)], verbose=100)

    val_auc = roc_auc_score(y_val_np, model.predict_proba(X_val_np)[:, 1])
    print("全量精修 AUC:", val_auc)
    return model


# 1) 小样本快速搜参
# best_params = quick_tune_xgb(x_train, y_train, x_val, y_val, n_iter=25, use_gpu=False)

# # 2) 全量数据精修训练（早停）
# best_model = train_full_xgb(x_train, y_train, x_val, y_val, best_params, use_gpu=False)


import numpy as np
import optuna
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from catboost import Pool

# --- 小工具：根据是否用GPU返回合适的参数（避免使用 gpu_hist） ---
def _device_params(use_gpu: bool):
    if use_gpu:
        return {"tree_method": "hist", "device": "cuda"}  # GPU版本
    else:
        return {"tree_method": "hist"}  # CPU版本

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
        "learning_rate": trial.suggest_float('learning_rate', 0.01, 0.1),
        "max_depth": trial.suggest_int('max_depth', 1, 4),
        "min_child_weight": trial.suggest_int('min_child_weight', 1, 10),
        "subsample": trial.suggest_float('subsample', 0.7, 1.0),  # 替换为 suggest_float
        "colsample_bytree": trial.suggest_float('colsample_bytree', 0.7, 1.0),  # 替换为 suggest_float
        "reg_alpha": trial.suggest_float('reg_alpha', 1e-5, 1.0, log=True),  # 使用 log=True 替代 suggest_loguniform
        "reg_lambda": trial.suggest_float('reg_lambda', 1e-5, 5.0, log=True),  # 使用 log=True 替代 suggest_loguniform
        "gamma": trial.suggest_float('gamma', 0.0, 5.0),  # 添加 gamma 来防止过拟合
        "max_delta_step": trial.suggest_int('max_delta_step', 0, 10),  # 增加 max_delta_step
        "scale_pos_weight": trial.suggest_float('scale_pos_weight', 0.5, 2.0),  # 根据类别不平衡调整
    }

    # 类别不平衡（基于子样本）
    pos = np.sum(y_sub == 1)
    neg = np.sum(y_sub == 0)
    spw = (neg / pos) if pos > 0 else 1.0
    param['scale_pos_weight'] = spw * param['scale_pos_weight']

    # 创建 XGBoost 模型
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        n_estimators=3000,
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1,
        **_device_params(use_gpu),
        **param
    )

    # 转换为 NumPy 数组
    X_sub_np = X_sub.values if hasattr(X_sub, "values") else np.asarray(X_sub)
    X_val_np = X_val.values if hasattr(X_val, "values") else np.asarray(X_val)
    y_sub_np = np.asarray(y_sub)
    y_val_np = np.asarray(y_val)

    # 训练模型并评估 AUC
    model.fit(X_sub_np, y_sub_np, eval_set=[(X_val_np, y_val_np)], verbose=False)
    val_auc = roc_auc_score(y_val_np, model.predict_proba(X_val_np)[:, 1])

    return val_auc  # 目标是最大化 AUC


def tune_xgb_with_optuna(X_train, y_train, X_val, y_val, n_trials=25, use_gpu=False):
    """使用 Optuna 优化 XGBoost 超参数"""
    study = optuna.create_study(direction='maximize')  # 目标是最大化 AUC
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val, use_gpu=use_gpu), n_trials=n_trials)

    print("Best parameters:", study.best_params)
    print("Best AUC:", study.best_value)
    return study.best_params

def train_full_xgb(X_train, y_train, X_val, y_val, best_params, use_gpu=False):
    """用全量数据+早停做精修训练"""
    # 转 numpy
    X_train_np = X_train.values if hasattr(X_train, "values") else np.asarray(X_train)
    X_val_np = X_val.values if hasattr(X_val, "values") else np.asarray(X_val)
    y_train_np = np.asarray(y_train)
    y_val_np = np.asarray(y_val)

    # 创建最终模型
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        n_estimators=5000,
        early_stopping_rounds=100,
        random_state=42,
        n_jobs=-1,
        **_device_params(use_gpu),
        **best_params
    )

    # 训练模型
    model.fit(X_train_np, y_train_np, eval_set=[(X_val_np, y_val_np)], verbose=100)

    # 验证集 AUC
    val_auc = roc_auc_score(y_val_np, model.predict_proba(X_val_np)[:, 1])
    print("Final model AUC:", val_auc)
    return model


#划分训练集和验证集
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(train_x, train_y, test_size=0.2, random_state=42)


# best_params = tune_xgb_with_optuna(x_train, y_train, x_val, y_val, n_trials=50, use_gpu=True)
# model = train_full_xgb(x_train, y_train, x_val, y_val, best_params, use_gpu=True)

