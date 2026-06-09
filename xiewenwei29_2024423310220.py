import pandas as pd
train_path = "/kaggle/input/playground-series-s5e6/train.csv"
test_path = "/kaggle/input/playground-series-s5e6/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# 打印列名以检查
print("训练集列名:", train_df.columns.tolist())
print("测试集列名:", test_df.columns.tolist())


import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.utils.class_weight import compute_class_weight

# 设置随机种子以确保结果可重现
np.random.seed(42)

# 修复的MAP@5评估函数
def map5(y_true, y_pred_top5):
    """
    y_true: 真实标签数组 (n_samples,)
    y_pred_top5: 预测的top5标签 (n_samples, 5)
    """
    ap_scores = []
    for true, pred in zip(y_true, y_pred_top5):
        if true in pred:
            # 计算首次命中位置的倒数
            rank = np.where(pred == true)[0][0] + 1
            ap_scores.append(1.0 / rank)
        else:
            ap_scores.append(0.0)
    return np.mean(ap_scores)

# 数据加载与增强特征工程
def load_and_preprocess_data():
    # 从Kaggle加载数据集
    train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
    test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
    
    # 重命名列以匹配实际数据集
    column_rename = {
        'Temparature': 'Temperature',
        'Phosphorous': 'Phosphorus',
        'Fertilizer Name': 'fertilizer'
    }
    train = train.rename(columns=column_rename)
    test = test.rename(columns=column_rename)
    
    # 删除ID列
    train = train.drop('id', axis=1)
    test_ids = test['id']
    test = test.drop('id', axis=1)
    
    # 目标变量编码
    le = LabelEncoder()
    train['fertilizer'] = le.fit_transform(train['fertilizer'])
    
    # 分离特征和目标
    X = train.drop('fertilizer', axis=1)
    y = train['fertilizer']
    
    # 特征工程
    # 1. 基础养分比例
    X['N/P_ratio'] = (X['Nitrogen'] + 1) / (X['Phosphorus'] + 1)
    test['N/P_ratio'] = (test['Nitrogen'] + 1) / (test['Phosphorus'] + 1)
    
    X['N/K_ratio'] = (X['Nitrogen'] + 1) / (X['Potassium'] + 1)
    test['N/K_ratio'] = (test['Nitrogen'] + 1) / (test['Potassium'] + 1)
    
    X['P/K_ratio'] = (X['Phosphorus'] + 1) / (X['Potassium'] + 1)
    test['P/K_ratio'] = (test['Phosphorus'] + 1) / (test['Potassium'] + 1)
    
    # 2. 综合养分指数
    X['nutrient_index'] = X['Nitrogen'] * 0.4 + X['Phosphorus'] * 0.3 + X['Potassium'] * 0.3
    test['nutrient_index'] = test['Nitrogen'] * 0.4 + test['Phosphorus'] * 0.3 + test['Potassium'] * 0.3
    
    # 3. 环境因子交互
    X['temp_humidity'] = X['Temperature'] * X['Humidity']
    test['temp_humidity'] = test['Temperature'] * test['Humidity']
    
    # 4. 养分平衡指数 (NB = N/(P+K))
    X['Nutrient_Balance'] = X['Nitrogen'] / (X['Phosphorus'] + X['Potassium'] + 1e-5)
    test['Nutrient_Balance'] = test['Nitrogen'] / (test['Phosphorus'] + test['Potassium'] + 1e-5)
    
    # 5. 温度-湿度压力因子
    X['Temp_Humidity_Stress'] = np.where(
        (X['Temperature'] > 30) & (X['Humidity'] < 40), 1, 0)
    test['Temp_Humidity_Stress'] = np.where(
        (test['Temperature'] > 30) & (test['Humidity'] < 40), 1, 0)
    
    # 6. 温度适应性指数
    X['Temp_Adaptation'] = np.abs(X['Temperature'] - 22)  # 22°C为多数作物最适温度
    test['Temp_Adaptation'] = np.abs(test['Temperature'] - 22)
    
    # 7. 养分阈值特征
    X['N_Deficiency'] = (X['Nitrogen'] < 25).astype(int)
    test['N_Deficiency'] = (test['Nitrogen'] < 25).astype(int)
    
    X['P_Deficiency'] = (X['Phosphorus'] < 15).astype(int)
    test['P_Deficiency'] = (test['Phosphorus'] < 15).astype(int)
    
    X['K_Deficiency'] = (X['Potassium'] < 20).astype(int)
    test['K_Deficiency'] = (test['Potassium'] < 20).astype(int)
    
    # 8. 季节性特征
    X['Season'] = np.where(X['Temperature'] > 25, 'Summer', 
                          np.where(X['Temperature'] < 10, 'Winter', 'Transition'))
    test['Season'] = np.where(test['Temperature'] > 25, 'Summer', 
                             np.where(test['Temperature'] < 10, 'Winter', 'Transition'))
    
    # 9. 土壤-作物组合特征
    X['Soil_Crop_Combo'] = X['Soil Type'].astype(str) + '_' + X['Crop Type'].astype(str)
    test['Soil_Crop_Combo'] = test['Soil Type'].astype(str) + '_' + test['Crop Type'].astype(str)
    
    # 类别特征列表
    cat_features = ['Soil Type', 'Crop Type', 'Season', 'Soil_Crop_Combo']
    
    return X, y, test, test_ids, le, cat_features

# 目标编码函数（在交叉验证中安全使用）
def target_encode(train, val, test, cat_features, target):
    # 创建数据副本
    train_encoded = train.copy()
    val_encoded = val.copy()
    test_encoded = test.copy()
    
    for col in cat_features:
        # 计算训练集上的目标均值（平滑处理）
        target_mean = train.groupby(col)[target].mean()
        
        # 应用到训练集
        train_encoded[col + '_target'] = train[col].map(target_mean)
        
        # 应用到验证集和测试集（使用训练集的统计量）
        val_encoded[col + '_target'] = val[col].map(target_mean).fillna(target_mean.mean())
        test_encoded[col + '_target'] = test[col].map(target_mean).fillna(target_mean.mean())
    
    # 删除原始类别特征
    train_encoded = train_encoded.drop(cat_features, axis=1)
    val_encoded = val_encoded.drop(cat_features, axis=1)
    test_encoded = test_encoded.drop(cat_features, axis=1)
    
    return train_encoded, val_encoded, test_encoded

# 特征重要性分析
def analyze_feature_importance(models, feature_names, model_names):
    plt.figure(figsize=(15, 10))
    for i, model in enumerate(models):
        plt.subplot(2, 2, i+1)
        
        if isinstance(model, xgb.Booster):
            # XGBoost模型
            importance = model.get_score(importance_type='weight')
            importance = pd.Series(importance).sort_values(ascending=False)
            importance = importance.reindex(feature_names, fill_value=0)
        elif isinstance(model, lgb.Booster):
            # LightGBM模型
            importance = pd.Series(model.feature_importance(importance_type='gain'))
            importance.index = feature_names
            importance = importance.sort_values(ascending=False)
        elif isinstance(model, cb.CatBoostClassifier):
            # CatBoost模型
            importance = pd.Series(model.get_feature_importance())
            importance.index = feature_names
            importance = importance.sort_values(ascending=False)
        
        sns.barplot(x=importance.values[:15], y=importance.index[:15])
        plt.title(f'{model_names[i]} - Top 15 Features')
        plt.xlabel('Importance Score')
    
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.show()
    
    print("\n农学特征分析:")
    print("1. 氮磷钾比例是决定肥料类型的关键因素")
    print("2. 土壤类型和作物类型直接影响养分需求")
    print("3. 温湿度交互作用影响肥料吸收效率")
    print("4. 养分缺乏指标对特定肥料类型有强预测性")

# 训练XGBoost模型
def train_xgboost(X_train, y_train, X_val, y_val, class_weights):
    # 转换为DMatrix格式
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # 设置优化后的参数
    params = {
        'objective': 'multi:softprob',
        'num_class': len(np.unique(y_train)),
        'eval_metric': 'mlogloss',
        'tree_method': 'gpu_hist',
        'gpu_id': 0,
        'max_depth': 10,
        'learning_rate': 0.03,
        'subsample': 0.7,
        'colsample_bytree': 0.6,
        'min_child_weight': 5,
        'gamma': 0.1,
        'alpha': 0.5,
        'lambda': 0.8,
        'seed': 42,
        # 处理类别不平衡
        'scale_pos_weight': len(y_train) / (len(np.unique(y_train)) * np.bincount(y_train))
    }
    
    print("训练XGBoost模型...")
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1500,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=100,
        verbose_eval=100
    )
    
    return model

# 训练LightGBM模型
def train_lightgbm(X_train, y_train, X_val, y_val, class_weights):
    # 创建数据集
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # 设置优化后的参数
    params = {
        'objective': 'multiclass',
        'num_class': len(np.unique(y_train)),
        'metric': 'multi_logloss',
        'boosting_type': 'goss',
        'num_leaves': 127,
        'learning_rate': 0.02,
        'feature_fraction': 0.6,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 20,
        'lambda_l1': 0.6,
        'lambda_l2': 0.8,
        'seed': 42,
        # 处理类别不平衡
        'class_weight': 'balanced',
        'verbose': -1
    }
    
    print("训练LightGBM模型...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1500,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=True),
            lgb.log_evaluation(100)
        ]
    )
    
    return model

# 训练CatBoost模型
def train_catboost(X_train, y_train, X_val, y_val, class_weights):
    # 设置优化后的参数
    params = {
        'iterations': 2000,
        'learning_rate': 0.025,
        'depth': 10,
        'l2_leaf_reg': 5,
        'border_count': 128,
        'loss_function': 'MultiClass',
        'auto_class_weights': 'Balanced',
        'task_type': 'GPU',
        'random_seed': 42,
        'verbose': 100
    }
    
    print("训练CatBoost模型...")
    model = cb.CatBoostClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=100,
        verbose=100
    )
    
    return model

# 集成预测（基于模型性能加权）
def ensemble_predictions(models, model_weights, X_test):
    test_preds = []
    
    for model in models:
        if isinstance(model, xgb.Booster):
            dtest = xgb.DMatrix(X_test)
            preds = model.predict(dtest)
        elif isinstance(model, lgb.Booster):
            preds = model.predict(X_test)
        elif isinstance(model, cb.CatBoostClassifier):
            preds = model.predict_proba(X_test)
        test_preds.append(preds)
    
    # 加权平均（根据模型性能）
    avg_preds = np.zeros_like(test_preds[0])
    
    for i, pred in enumerate(test_preds):
        avg_preds += model_weights[i] * pred
    
    # 获取Top5预测
    test_pred_top5 = np.argsort(-avg_preds, axis=1)[:, :5]
    
    return test_pred_top5

# 结果解释
def interpret_results(submission, test, le, test_ids, num_samples=5):
    # 解码肥料名称
    inv_labels = le.inverse_transform(np.arange(len(le.classes_)))
    
    # 选择随机样本
    sample_indices = np.random.choice(len(submission), num_samples, replace=False)
    
    print("\n预测结果解释（随机样本分析）:")
    for idx in sample_indices:
        sample_id = submission.iloc[idx]['id']
        # 从预测列获取预测结果
        predictions = [submission.iloc[idx][f'Fertilizer Name_{i+1}'] for i in range(5)]
        
        # 获取样本特征
        sample = test.loc[test.index[idx]]
        
        print(f"\n样本ID: {sample_id}")
        print(f"作物类型: {sample['Crop Type']}, 土壤类型: {sample['Soil Type']}")
        print(f"氮: {sample['Nitrogen']:.2f}, 磷: {sample['Phosphorus']:.2f}, 钾: {sample['Potassium']:.2f}")
        print(f"温度: {sample['Temperature']:.2f}, 湿度: {sample['Humidity']:.2f}")
        print(f"预测肥料: {', '.join(predictions)}")
        
        # 农学解释
        print("农学解释:")
        if sample['Nitrogen'] < 25:
            print(" - 土壤氮含量低，推荐氮基肥料")
        if sample['Phosphorus'] < 15:
            print(" - 土壤磷含量低，推荐含磷肥料")
        if sample['Potassium'] < 20:
            print(" - 土壤钾含量低，推荐钾基肥料")
        if sample['Temperature'] > 28:
            print(" - 高温条件下推荐缓释肥料")
        if sample['Humidity'] < 30:
            print(" - 低湿度环境建议增加施肥频率")

# 生成提交文件
def create_submission(test_ids, test_pred_top5, le):
    # 将预测的索引转换为肥料名称
    inv_labels = le.inverse_transform(test_pred_top5.flatten())
    inv_labels = inv_labels.reshape(test_pred_top5.shape)
    
    # 创建符合竞赛要求的提交格式
    submission = pd.DataFrame({
        'id': test_ids
    })
    
    # 添加5个预测列
    for i in range(5):
        submission[f'Fertilizer Name_{i+1}'] = inv_labels[:, i]
    
    # 创建符合竞赛要求的预测列（空格分隔的5个预测值）
    submission['Fertilizer Name'] = submission[[f'Fertilizer Name_{i+1}' for i in range(5)]].apply(
        lambda x: ' '.join(x), axis=1
    )
    
    # 保存提交文件
    submission[['id', 'Fertilizer Name']].to_csv('submission.csv', index=False)
    print("\n提交文件已保存为 submission.csv")
    
    return submission

# 主函数（使用5折交叉验证）
def main():
    start_time = time.time()
    
    # 加载和预处理数据
    print("加载和预处理数据...")
    X, y, test, test_ids, le, cat_features = load_and_preprocess_data()
    
    # 初始化交叉验证
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # 存储模型和预测结果
    models = {'XGBoost': [], 'LightGBM': [], 'CatBoost': []}
    model_weights = {'XGBoost': [], 'LightGBM': [], 'CatBoost': []}
    test_preds = np.zeros((test.shape[0], len(le.classes_)))
    
    # 计算类别权重
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(y), 
        y=y
    )
    class_weights = {i: weight for i, weight in enumerate(class_weights)}
    
    print(f"\n开始{n_folds}折交叉验证训练...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n========== 折叠 {fold+1}/{n_folds} ==========")
        
        # 划分训练集和验证集
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # 目标编码
        X_train_enc, X_val_enc, test_enc = target_encode(
            X_train, X_val, test, cat_features, 'fertilizer'
        )
        
        # 训练XGBoost
        print("\n--- 训练XGBoost模型 ---")
        xgb_model = train_xgboost(X_train_enc, y_train, X_val_enc, y_val, class_weights)
        models['XGBoost'].append(xgb_model)
        
        # 评估XGBoost
        dval = xgb.DMatrix(X_val_enc)
        xgb_preds = xgb_model.predict(dval)
        xgb_pred_top5 = np.argsort(-xgb_preds, axis=1)[:, :5]
        xgb_map5 = map5(y_val.values, xgb_pred_top5)
        print(f"XGBoost MAP@5: {xgb_map5:.5f}")
        model_weights['XGBoost'].append(xgb_map5)
        
        # 训练LightGBM
        print("\n--- 训练LightGBM模型 ---")
        lgb_model = train_lightgbm(X_train_enc, y_train, X_val_enc, y_val, class_weights)
        models['LightGBM'].append(lgb_model)
        
        # 评估LightGBM
        lgb_preds = lgb_model.predict(X_val_enc)
        lgb_pred_top5 = np.argsort(-lgb_preds, axis=1)[:, :5]
        lgb_map5 = map5(y_val.values, lgb_pred_top5)
        print(f"LightGBM MAP@5: {lgb_map5:.5f}")
        model_weights['LightGBM'].append(lgb_map5)
        
        # 训练CatBoost
        print("\n--- 训练CatBoost模型 ---")
        cb_model = train_catboost(X_train_enc, y_train, X_val_enc, y_val, class_weights)
        models['CatBoost'].append(cb_model)
        
        # 评估CatBoost
        cb_preds = cb_model.predict_proba(X_val_enc)
        cb_pred_top5 = np.argsort(-cb_preds, axis=1)[:, :5]
        cb_map5 = map5(y_val.values, cb_pred_top5)
        print(f"CatBoost MAP@5: {cb_map5:.5f}")
        model_weights['CatBoost'].append(cb_map5)
        
        # 集成预测测试集
        fold_weights = [
            model_weights['XGBoost'][-1],
            model_weights['LightGBM'][-1],
            model_weights['CatBoost'][-1]
        ]
        fold_weights = [w / sum(fold_weights) for w in fold_weights]
        
        fold_test_preds = ensemble_predictions(
            [xgb_model, lgb_model, cb_model],
            fold_weights,
            test_enc
        )
        
        # 转换为概率并累加
        for i in range(test_enc.shape[0]):
            for j in range(5):
                test_preds[i, fold_test_preds[i, j]] += 1 / (j + 1)
    
    # 最终预测（取概率最高的5个）
    final_test_pred_top5 = np.argsort(-test_preds, axis=1)[:, :5]
    
    # 生成提交文件
    print("\n生成提交文件...")
    submission = create_submission(test_ids, final_test_pred_top5, le)
    
    # 结果解释
    interpret_results(submission, test, le, test_ids)
    
    # 特征重要性分析（使用最后一折的模型）
    print("\n分析特征重要性...")
    feature_names = X_train_enc.columns.tolist()
    analyze_feature_importance(
        [models['XGBoost'][-1], models['LightGBM'][-1], models['CatBoost'][-1]],
        feature_names,
        ['XGBoost', 'LightGBM', 'CatBoost']
    )
    
    # 计算平均模型性能
    avg_map5 = {
        'XGBoost': np.mean(model_weights['XGBoost']),
        'LightGBM': np.mean(model_weights['LightGBM']),
        'CatBoost': np.mean(model_weights['CatBoost'])
    }
    
    end_time = time.time()
    print(f"\n总运行时间: {end_time - start_time:.2f}秒")
    print(f"模型平均MAP@5分数:")
    for model, score in avg_map5.items():
        print(f"{model}: {score:.5f}")
    print(f"集成模型最终MAP@5估计: {np.mean(list(avg_map5.values())):.5f}")

if __name__ == "__main__":
    main()

