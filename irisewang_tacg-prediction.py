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


import pandas as pd;
train_data = pd.read_csv("/kaggle/input/njtech-ml-e1-new2/train.csv")
print(f"训练集样本数: {len(train_data)}");
test_data = pd.read_csv("/kaggle/input/njtech-ml-e1-new2/test.csv")
print(f"测试集样本数: {len(test_data)}");
train_data.head()


train_data.describe()


train_data["VASN"].describe()


print(train_data.isna().sum().sum())
print(train_data.isnull().sum().sum())

dtypes = train_data.dtypes
    
# 找出非数值型列（不是int或float类型的列）
non_numeric_cols = []
for col, dtype in dtypes.items():
    # 检查列的数据类型是否为非数值型
    if not np.issubdtype(dtype, np.number):
        non_numeric_cols.append(col)

non_numeric_cols

train_data = train_data.drop(columns=['PATIENT']);
print(len(train_data))


X = train_data.drop(columns=["LABEL", "ID"]);
y = train_data["LABEL"];


import seaborn as sns;

# 类别分布分析
class_1_sample_count = (y == 1).astype(int).sum();
class_0_sample_count = (y == 0).astype(int).sum();
print(f"正类样本数: {class_1_sample_count}, 负类样本数: {class_0_sample_count}, 总数: {len(y)}");

sns.countplot(train_data, x="LABEL");


from sklearn.preprocessing import StandardScaler;
from sklearn.decomposition import PCA;
import matplotlib.pyplot as plt;

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print("解释方差比例:", pca.explained_variance_ratio_)
print("累计解释方差:", sum(pca.explained_variance_ratio_))
plt.figure(figsize=(8, 6))
colors = ['navy', 'turquoise']
lw = 2
for color, i, target_name in zip(colors, [0, 1], list(X)):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], color=color, alpha=0.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA Decomposition');
plt.xlabel('First key feature');
plt.ylabel('Second key feature');
plt.grid(True)
plt.show()


from sklearn.model_selection import train_test_split;

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
);


X_test = test_data.drop(columns=["PATIENT", "ID"]);
len(X_test)


from sklearn.feature_selection import SelectFromModel;
from imblearn.over_sampling import SMOTE;
from sklearn.preprocessing import StandardScaler;
from sklearn.ensemble import RandomForestClassifier;

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

selector = SelectFromModel(
    RandomForestClassifier(n_estimators=200, random_state=42), 
    max_features=200
)

X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_val_selected = selector.transform(X_val_scaled)
X_test_selected = selector.transform(X_test_scaled)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_selected, y_train)


# 定义Optuna优化学习器参数的函数
def optimize_rf(trial):
    """优化随机森林参数"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
        'max_depth': trial.suggest_int('max_depth', 5, 30, log=True),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'class_weight': 'balanced',
        'random_state': 42
    }
    
    model = RandomForestClassifier(**params)
    
    # 使用交叉验证评估模型
    cv_scores = cross_val_score(
        model, X_resampled, y_resampled, 
        cv=StratifiedKFold(n_splits=5), 
        scoring='roc_auc',
        n_jobs=-1
    )
    
    return np.mean(cv_scores)


def optimize_xgb(trial):
    """优化XGBoost参数"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'use_label_encoder': False,
        'random_state': 42
    }
    
    model = XGBClassifier(**params)
    
    cv_scores = cross_val_score(
        model, X_resampled, y_resampled, 
        cv=StratifiedKFold(n_splits=5), 
        scoring='roc_auc',
        n_jobs=-1
    )
    
    return np.mean(cv_scores)


def optimize_svm(trial):
    """优化SVM参数"""
    params = {
        'C': trial.suggest_float('C', 0.01, 100, log=True),
        'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
        'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly']),
        'probability': True,
        'random_state': 42
    }
    
    if params['kernel'] == 'poly':
        params['degree'] = trial.suggest_int('degree', 2, 5)
    
    model = SVC(**params)
    
    cv_scores = cross_val_score(
        model, X_resampled, y_resampled, 
        cv=StratifiedKFold(n_splits=5), 
        scoring='roc_auc',
        n_jobs=-1
    )
    
    return np.mean(cv_scores)


from sklearn.model_selection import cross_val_score


# 使用Optuna优化每个模型
best_models = {}
best_params = {}

# 优化随机森林
print("优化随机森林...")
rf_study = optuna.create_study(direction='maximize')
rf_study.optimize(optimize_rf, n_trials=20, show_progress_bar=True)
best_params["rf"] = rf_study.best_params
best_models["rf"] = RandomForestClassifier(**best_params["rf"], random_state=42)
best_models["rf"].fit(X_resampled, y_resampled)
print(f"RF 最佳参数: {best_params['rf']}")
print(f"RF 最佳AUC: {rf_study.best_value:.4f}")

# 优化XGBoost
print("\n优化XGBoost...")
xgb_study = optuna.create_study(direction='maximize')
xgb_study.optimize(optimize_xgb, n_trials=20, show_progress_bar=True)
best_params["xgb"] = xgb_study.best_params
best_params["xgb"].update({
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'use_label_encoder': False,
    'random_state': 42
})
best_models["xgb"] = XGBClassifier(**best_params["xgb"])
best_models["xgb"].fit(X_resampled, y_resampled)
print(f"XGB 最佳参数: {best_params['xgb']}")
print(f"XGB 最佳AUC: {xgb_study.best_value:.4f}")

# 优化SVM
print("\n优化SVM...")
svm_study = optuna.create_study(direction='maximize')
svm_study.optimize(optimize_svm, n_trials=20, show_progress_bar=True)
best_params["svm"] = svm_study.best_params
best_params["svm"]["probability"] = True
best_models["svm"] = SVC(**best_params["svm"], random_state=42)
best_models["svm"].fit(X_resampled, y_resampled)
print(f"SVM 最佳参数: {best_params['svm']}")
print(f"SVM 最佳AUC: {svm_study.best_value:.4f}")


# 优化投票权重
def optimize_voting_weights(trial):
    """优化投票集成的权重"""
    # 获取每个模型的权重
    w_rf = trial.suggest_float('w_rf', 0.0, 1.0)
    w_xgb = trial.suggest_float('w_xgb', 0.0, 1.0)
    w_svm = trial.suggest_float('w_svm', 0.0, 1.0)
    
    # 归一化权重
    total = w_rf + w_xgb + w_svm
    weights = [w_rf/total, w_xgb/total, w_svm/total]
    
    # 获取各模型对验证集的预测
    rf_proba = best_models["rf"].predict_proba(X_val_selected)[:, 1]
    xgb_proba = best_models["xgb"].predict_proba(X_val_selected)[:, 1]
    svm_proba = best_models["svm"].predict_proba(X_val_selected)[:, 1]
    
    # 加权组合预测
    combined_proba = (
        weights[0] * rf_proba + 
        weights[1] * xgb_proba + 
        weights[2] * svm_proba
    )
    
    # 计算AUC
    return roc_auc_score(y_val, combined_proba)

# 优化投票权重
print("\n优化投票权重...")
weight_study = optuna.create_study(direction='maximize')
weight_study.optimize(optimize_voting_weights, n_trials=100, show_progress_bar=True)



from sklearn.ensemble import VotingClassifier

# 获取最佳权重
best_weights = weight_study.best_params
w_rf = best_weights['w_rf']
w_xgb = best_weights['w_xgb']
w_svm = best_weights['w_svm']

# 归一化权重
total = w_rf + w_xgb + w_svm
w_rf /= total
w_xgb /= total
w_svm /= total

print(f"最佳投票权重: RF={w_rf:.4f}, XGB={w_xgb:.4f}, SVM={w_svm:.4f}")
print(f"最佳加权集成AUC: {weight_study.best_value:.4f}")

# 用优化权重创建VotingClassifier
ensemble = VotingClassifier(
    estimators=[
        ("rf", best_models["rf"]),
        ("xgb", best_models["xgb"]),
        ("svm", best_models["svm"])
    ],
    voting="soft",
    weights=[w_rf, w_xgb, w_svm],
    n_jobs=-1
)
ensemble.fit(X_resampled, y_resampled)

# 验证集成模型
val_proba = ensemble.predict_proba(X_val_selected)[:, 1]
auc_score = roc_auc_score(y_val, val_proba)
print(f"加权集成模型验证集AUC: {auc_score:.4f}")


# 最终训练（使用全部训练数据）
# 重新处理完整数据集
X_full_scaled = scaler.fit_transform(X)
X_full_selected = selector.transform(X_full_scaled)
X_test_final = selector.transform(X_test_scaled)

# 处理完整数据的类别不平衡
X_full_resampled, y_full_resampled = smote.fit_resample(X_full_selected, y)

# 训练最终集成模型
ensemble.fit(X_full_resampled, y_full_resampled)

print(len(X_test_final))

# 预测测试集
test_proba = ensemble.predict_proba(X_test_final)[:, 1]

# 生成提交文件
submission_data = pd.DataFrame({
    "ID": test_data["ID"],
    "LABEL": test_proba
})



submit_data = {'ID': test_id, 'LABEL': test_preds}
submission_df = pd.DataFrame(data=submission_data)
submission_df.to_csv("/kaggle/working/submission.csv", index=False);

submission_df.head()

