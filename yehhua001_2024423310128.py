import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier

# 指定支持中文的字体（以 SimSun 为例，Windows/Linux 通用）
plt.rcParams['font.sans-serif'] = ['SimSun']  
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

# 1. 数据准备与预处理
print("开始数据准备与预处理...")

train_data = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')  
test_data = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')   

# 查看数据基本信息（验证列名）
print("训练数据基本信息：")
print(train_data.info())
print("\n测试数据基本信息：")
print(test_data.info())

# 查看缺失值情况（实际无缺失，但保留流程）
print("训练数据缺失值统计：")
print(train_data.isnull().sum())
print("\n测试数据缺失值统计：")
print(test_data.isnull().sum())

# 特征工程 - 创建新的特征（基于真实列名）
# 1. 氮磷钾比例特征
train_data['N_P_ratio'] = train_data['Nitrogen'] / (train_data['Phosphorous'] + 1e-6)
train_data['N_K_ratio'] = train_data['Nitrogen'] / (train_data['Potassium'] + 1e-6)
train_data['P_K_ratio'] = train_data['Phosphorous'] / (train_data['Potassium'] + 1e-6)

test_data['N_P_ratio'] = test_data['Nitrogen'] / (test_data['Phosphorous'] + 1e-6)
test_data['N_K_ratio'] = test_data['Nitrogen'] / (test_data['Potassium'] + 1e-6)
test_data['P_K_ratio'] = test_data['Phosphorous'] / (test_data['Potassium'] + 1e-6)

# 编码类别特征
# 对目标变量（训练集的 'Fertilizer Name'）进行标签编码
label_encoder = LabelEncoder()
train_data['fertilizer_label'] = label_encoder.fit_transform(train_data['Fertilizer Name'])
fertilizer_classes = label_encoder.classes_
num_classes = len(fertilizer_classes)

# 对其他类别特征进行独热编码（列名从数据里提取）
categorical_features = ['Soil Type', 'Crop Type']  # 这两列是 object 类型，且测试集也有
train_encoded = pd.get_dummies(train_data, columns=categorical_features)
test_encoded = pd.get_dummies(test_data, columns=categorical_features)

# 确保测试集和训练集的特征一致（处理独热编码后的列差异）
missing_cols = set(train_encoded.columns) - set(test_encoded.columns)
for col in missing_cols:
    if col != 'fertilizer_label' and col != 'Fertilizer Name':  # 测试集没有这两列，跳过
        test_encoded[col] = 0
test_encoded = test_encoded[train_encoded.columns.drop(['Fertilizer Name', 'fertilizer_label'])]  # 测试集无需目标列

# 划分训练集和验证集（从训练数据里拆分）
X = train_encoded.drop(['Fertilizer Name', 'fertilizer_label'], axis=1)
y = train_encoded['fertilizer_label']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("数据准备与预处理完成！")

# 2. 模型训练与评估
print("\n开始模型训练与评估...")

# 2.1 XGBoost模型
print("\n训练XGBoost模型...")
model = xgb.XGBClassifier(
    objective='multi:softprob',
    eval_metric=['mlogloss'], 
    num_classes=num_classes, 
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.01,
    reg_lambda=0.01,
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    early_stopping_rounds=50,
    verbose=10
)

# 验证集预测 & MAP@5 计算
y_val_proba = model.predict_proba(X_val)
y_val_true = y_val.values

def calculate_map_at_k(y_true, y_pred_proba, k=5):
    map_scores = []
    for true_label, pred_proba in zip(y_true, y_pred_proba):
        pred_indices = np.argsort(pred_proba)[::-1][:k]
        precision_scores = []
        relevant = 0
        for i, pred_idx in enumerate(pred_indices):
            if pred_idx == true_label:
                relevant += 1
                precision_scores.append(relevant / (i + 1))
        map_scores.append(np.mean(precision_scores) if precision_scores else 0)
    return np.mean(map_scores)

xgb_map_at_5 = calculate_map_at_k(y_val_true, y_val_proba)
print(f"XGBoost模型在验证集上的MAP@5分数: {xgb_map_at_5:.4f}")

# 2.2 LightGBM模型
print("\n训练LightGBM模型...")
lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

lgb_params = {
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'num_classes': num_classes,
    'learning_rate': 0.05,
    'early_stopping_rounds': 50,  # 早停轮数
    'n_estimators': 1000,
    'max_depth': 5,
    'num_leaves': 31,
    'min_child_samples': 20,
    'reg_alpha': 0.01,
    'reg_lambda': 0.01,
    'random_state': 42
}

lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

# 调用 train 函数，通过 valid_sets 指定验证集，结合参数里的 early_stopping_rounds 实现早停
lgb_model = lgb.train(
    lgb_params,
    lgb_train,
    valid_sets=[lgb_train, lgb_val]
)

y_val_lgb_proba = lgb_model.predict(X_val)
lgb_map_at_5 = calculate_map_at_k(y_val_true, y_val_lgb_proba)
print(f"LightGBM模型在验证集上的MAP@5分数: {lgb_map_at_5:.4f}")

# 2.3 CatBoost模型
print("\n训练CatBoost模型...")
cat_features = [X.columns.tolist().index(col) for col in categorical_features if col in X.columns]

cat_model = CatBoostClassifier(
    objective='MultiClass',
    learning_rate=0.05,
    n_estimators=1000,
    max_depth=5,
    random_state=42,
    verbose=10
)

cat_model.fit(
    X_train, y_train,
    cat_features=cat_features,
    eval_set=(X_val, y_val),
    early_stopping_rounds=50
)

y_val_cat_proba = cat_model.predict_proba(X_val)
cat_map_at_5 = calculate_map_at_k(y_val_true, y_val_cat_proba)
print(f"CatBoost模型在验证集上的MAP@5分数: {cat_map_at_5:.4f}")

# 2.4 模型集成
print("\n构建集成模型...")
ensemble_proba = (y_val_proba + y_val_lgb_proba + y_val_cat_proba) / 3
ensemble_map_at_5 = calculate_map_at_k(y_val_true, ensemble_proba)
print(f"集成模型在验证集上的MAP@5分数: {ensemble_map_at_5:.4f}")

# 选择最佳模型
best_model = model
best_map = xgb_map_at_5

if lgb_map_at_5 > best_map:
    best_model = lgb_model
    best_map = lgb_map_at_5
if cat_map_at_5 > best_map:
    best_model = cat_model
    best_map = cat_map_at_5
if ensemble_map_at_5 > best_map:
    best_model = "ensemble"
    best_map = ensemble_map_at_5

print(f"\n选择最佳模型: {best_model} (MAP@5 = {best_map:.4f})")

# 3. 特征重要性分析（仅对非集成模型）
print("\n开始特征重要性分析...")

if best_model != "ensemble":
    if isinstance(best_model, XGBClassifier):
        # XGBClassifier 类型的处理逻辑
        feature_importance = best_model.feature_importances_  # 注意这里用 feature_importances_
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False).head(20)
    elif isinstance(best_model, xgb.Booster):
        # XGBoost Booster 类型的处理逻辑
        feature_importance = best_model.get_score(importance_type='gain')
        features = X.columns
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': [feature_importance.get(col, 0) for col in features]
        }).sort_values('Importance', ascending=False).head(20)
    elif isinstance(best_model, lgb.Booster):
        # LightGBM Booster 类型的处理逻辑
        feature_importance = best_model.feature_importance(importance_type='gain')
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False).head(20)
    elif isinstance(best_model, CatBoostClassifier):
        # CatBoostClassifier 类型的处理逻辑
        feature_importance = best_model.get_feature_importance()
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False).head(20)
    else:
        print("未识别的模型类型，无法进行特征重要性分析")
        importance_df = None

    if importance_df is not None:
        # 绘图和打印特征重要性的代码...
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title('Feature Importance Analysis')
        plt.xlabel('Importance Score')
        plt.ylabel('Feature Name')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.show()

        print("最重要的10个特征:")
        for i, row in importance_df.head(10).iterrows():
            print(f"{i+1}. {row['Feature']}: {row['Importance']:.4f}")
    else:
        print("特征重要性分析未执行或执行失败")

# 4. 预测结果处理与输出
print("\n开始生成预测结果...")
X_test = test_encoded  # 测试集特征（已对齐训练集）

if best_model == "ensemble":
    xgb_proba = model.predict_proba(X_test)
    lgb_proba = lgb_model.predict(X_test)
    cat_proba = cat_model.predict_proba(X_test)
    test_proba = (xgb_proba + lgb_proba + cat_proba) / 3
else:
    if isinstance(best_model, lgb.Booster):
        test_proba = best_model.predict(X_test)
    else:
        test_proba = best_model.predict_proba(X_test)

# 获取前5预测并转换为肥料名称
top5_predictions = np.argsort(test_proba, axis=1)[:, -5:][:, ::-1]
top5_fertilizers = []
for pred_indices in top5_predictions:
    fertilizers = [fertilizer_classes[idx] for idx in pred_indices]
    top5_fertilizers.append(fertilizers)

# 生成提交文件（按竞赛要求格式）
submission = pd.DataFrame({
    'id': test_data['id'], 
    'Fertilizer Name': [' '.join(top3) for top3 in [fertilizers[:3] for fertilizers in top5_fertilizers]]
})

submission.to_csv('submission.csv', index=False)
print("提交文件已生成: submission.csv")
print("\n全部流程已完成！")  

