# 学号:2024423320229  姓名:曾智涛
# 数学建模作业：肥料类型预测竞赛

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import warnings
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
warnings.filterwarnings('ignore')

# 全局设置
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于正确显示中文
plt.rcParams['axes.unicode_minus'] = False  # 用于正确显示负号
sns.set_style("whitegrid")  # 设置 seaborn 绘图风格

# ====================== 特征工程函数 ======================
def build_features(df):
    """
    对输入的 DataFrame 进行特征工程处理，包括列名标准化和构造新特征
    :param df: 输入的 DataFrame
    :return: 经过特征工程处理后的 DataFrame
    """
    df = df.copy()
    # 列名标准化，将特定列名映射为更规范的名称
    rename_dict = {
        'Temparature': 'temperature',
        'Phosphorous': 'phosphorus',
        'Nitrogen': 'nitrogen',
        'Potassium': 'potassium',
        'Moisture': 'moisture'
    }
    df.rename(columns=rename_dict, inplace=True)
    
    # 构造基础营养比例相关特征
    df['n_p_ratio'] = df['nitrogen'] / (df['phosphorus'] + 1e-6)  # 氮磷比，加小值避免除零
    df['n_k_ratio'] = df['nitrogen'] / (df['potassium'] + 1e-6)  # 氮钾比
    df['p_k_ratio'] = df['phosphorus'] / (df['potassium'] + 1e-6)  # 磷钾比
    df['nutrient_sum'] = df['nitrogen'] + df['phosphorus'] + df['potassium']  # 营养元素总和
    
    # 构造环境交互相关特征
    df['temp_humidity_prod'] = df['temperature'] * df['Humidity'] / 100  # 温度和湿度的交互项（缩放后）
    df['temp_moisture_diff'] = df['temperature'] - df['moisture']  # 温度和湿度的差值
    df['humidity_moisture_ratio'] = df['Humidity'] / (df['moisture'] + 1e-6)  # 湿度和水分的比值，加小值避免除零
    
    # 构造营养平衡特征
    df['nutrient_balance'] = (df['nitrogen'] + df['phosphorus'] + df['potassium']) / 3  # 营养元素的平均值，体现平衡
    
    return df

# ====================== MAP@5 评估函数 ======================
def compute_map5(y_actual, y_probs):
    """
    计算 Mean Average Precision @ 5 (MAP@5) 指标
    :param y_actual: 真实标签的一维数组
    :param y_probs: 模型预测的各类别概率的二维数组，形状为 [n_samples, n_classes]
    :return: MAP@5 得分
    """
    # 获取每个样本预测概率最高的前 5 个类别索引
    top5_indices = np.argsort(-y_probs, axis=1)[:, :5]
    ap_list = []
    for i in range(len(y_actual)):
        true_label = y_actual[i]  # 当前样本的真实标签
        pred_group = top5_indices[i]  # 当前样本预测的前 5 个类别索引
        score_val = 0.0
        hit_count = 0.0
        for pos in range(min(5, len(pred_group))):  # 遍历前 5 个预测结果
            if pred_group[pos] == true_label:
                hit_count += 1
                score_val += hit_count / (pos + 1)  # 计算精确率并累加
        # 计算当前样本的 AP 并添加到列表中
        ap_list.append(score_val / hit_count if hit_count > 0 else 0.0)
    return np.mean(ap_list)  # 返回所有样本的 MAP@5 均值

# ====================== 特征重要性可视化 ======================
def plot_feature_importance(model, feature_names, top_n=10):
    """
    绘制特征重要性条形图并保存
    :param model: 训练好的 XGBoost 模型
    :param feature_names: 特征名称的列表
    :param top_n: 展示最重要的前 N 个特征，默认 10
    """
    try:
        bst = model
        # 获取特征重要性（基于增益）
        importance = bst.get_score(importance_type='gain')
        # 转换为 DataFrame 方便处理和绘图
        imp_df = pd.DataFrame({
            'Feature': list(importance.keys()),
            'Importance': list(importance.values())
        })
        # 选取重要性最高的 top_n 个特征
        imp_df = imp_df.nlargest(top_n, 'Importance')
        
        # 绘制条形图
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=imp_df)
        plt.title(f'Top {top_n} 特征重要性')  # 设置标题
        plt.xlabel('重要性得分')  # 设置 x 轴标签
        plt.tight_layout()  # 优化布局
        plt.savefig('feature_importance.png', dpi=300)  # 保存图片，设置分辨率
        plt.close()  # 关闭绘图窗口
        print("✅ 特征重要性图已保存")
    except Exception as e:
        # 捕获异常并打印错误信息
        print(f"⚠️ 特征图生成失败: {str(e)}")

# ====================== 主流程 ======================
def main():
    print("=== 肥料类型预测系统 ===")
    start_time = time.time()  # 记录开始时间
    
    # 1. 数据加载
    print("Step 1: 加载数据集...")
    train_path = '/kaggle/input/playground-series-s5e6/train.csv'
    test_path = '/kaggle/input/playground-series-s5e6/test.csv'
    # 读取训练集和测试集数据
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    # 打印数据集大小信息
    print(f"训练集大小: {len(train_data)} 样本, 测试集大小: {len(test_data)} 样本")
    
    # 2. 数据预处理
    print("Step 2: 数据预处理...")
    # 分离特征和标签，以及测试集的 ID 和特征
    X = train_data.drop(['id', 'Fertilizer Name'], axis=1)
    y = train_data['Fertilizer Name']
    test_ids = test_data['id']
    X_test = test_data.drop('id', axis=1)
    
    # 3. 特征工程
    print("Step 3: 特征工程...")
    # 对训练集和测试集特征进行工程处理
    X_engineered = build_features(X)
    X_test_engineered = build_features(X_test)
    
    # 4. 划分特征类型
    cat_cols = ['Soil Type', 'Crop Type']  # 类别型特征列名
    # 数值型特征列名（排除类别型特征列名）
    num_cols = [col for col in X_engineered.columns if col not in cat_cols]  
    
    # 5. 数据预处理
    print("Step 4: 数据预处理...")
    # 构建 ColumnTransformer 进行预处理，数值型特征直接传递，类别型特征进行 OneHot 编码
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ])
    
    # 对训练集和测试集特征进行预处理
    X_processed = preprocessor.fit_transform(X_engineered)
    X_test_processed = preprocessor.transform(X_test_engineered)
    # 打印预处理后的数据集形状
    print(f"预处理后训练集形状: {X_processed.shape}, 测试集形状: {X_test_processed.shape}")
    
    # 6. 标签编码
    le = LabelEncoder()  # 初始化 LabelEncoder
    y_encoded = le.fit_transform(y)  # 对标签进行编码
    class_names = le.classes_  # 获取类别名称
    n_classes = len(class_names)  # 获取类别数量
    print(f"识别到 {n_classes} 种肥料类型")
    
    # 7. 交叉验证训练
    print("Step 5: 交叉验证训练...")
    n_folds = 5  # 设置交叉验证的折数
    # 初始化 StratifiedKFold，保证每个折中的类别分布相似
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=42)
    # 初始化数组存储 Out-of-Fold 预测概率和测试集预测概率
    oof_probs = np.zeros((len(X_processed), n_classes))
    test_probs = np.zeros((len(X_test_processed), n_classes))
    fold_maps = []  # 存储每折的 MAP@5 得分
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_processed, y_encoded)):
        print(f"\n=== 第 {fold+1}/{n_folds} 折 ===")
        # 划分训练集和验证集的特征和标签
        X_train, X_val = X_processed[train_idx], X_processed[val_idx]
        y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]
        
        # 构建 XGBoost 的 DMatrix 数据格式
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(X_test_processed)
        
        # 设置 XGBoost 模型参数
        params = {
            'objective': 'multi:softprob',  # 多分类任务，输出各类别概率
            'num_class': n_classes,  # 类别数量
            'eval_metric': 'mlogloss',  # 评估指标为多分类对数损失
            'learning_rate': 0.06,  # 学习率
            'max_depth': 6,  # 树的最大深度
            'subsample': 0.8,  # 样本子采样比例
            'colsample_bytree': 0.8,  # 特征子采样比例
            'seed': 42,  # 随机种子，保证可复现
            'tree_method': 'hist',  # 直方图算法加速训练
            'n_estimators': 1500  # 树的数量
        }
        
        # 训练 XGBoost 模型
        model = xgb.train(
            params, dtrain,
            evals=[(dtrain, 'train'), (dval, 'val')],  # 训练集和验证集评估
            early_stopping_rounds=100,  # 早停轮数
            verbose_eval=100  # 每 100 轮打印评估信息
        )
        
        # 保存验证集和测试集的预测概率
        oof_probs[val_idx] = model.predict(dval)
        test_probs += model.predict(dtest) / n_folds  # 测试集概率平均
        
        # 计算并保存当前折的 MAP@5 得分
        fold_map5 = compute_map5(y_val, oof_probs[val_idx])
        fold_maps.append(fold_map5)
        print(f"第 {fold+1} 折 MAP@5: {fold_map5:.5f}")
        
        # 在最后一折训练完成后绘制特征重要性图
        if fold == n_folds - 1:
            # 获取特征名称（数值型特征 + 类别型特征编码后的名称）
            feature_names = num_cols + list(preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols))
            plot_feature_importance(model, feature_names)
    
    # 8. 整体评估
    # 计算整体交叉验证的 MAP@5 得分
    overall_map5 = compute_map5(y_encoded, oof_probs)
    print(f"\n整体交叉验证 MAP@5: {overall_map5:.5f}，各折MAP@5: {fold_maps}")
    
    # 9. 生成Top5预测（修正 inverse_transform 部分）
    print("Step 6: 生成预测结果...")
    # 获取测试集每个样本预测概率最高的前 5 个类别索引
    top5_indices = np.argsort(-test_probs, axis=1)[:, :5]
    
    # 修正：先展平，逆变换后再恢复形状，以适配 LabelEncoder 的 inverse_transform 要求
    top5_flat = top5_indices.flatten()  # 将二维索引数组展平为一维
    top5_labels_flat = le.inverse_transform(top5_flat)  # 对一维索引进行逆编码得到类别名称
    # 将一维类别名称数组恢复为原二维形状
    top5_labels = top5_labels_flat.reshape(top5_indices.shape)  
    
    # 将每个样本的前 5 个预测类别名称用空格连接
    top5_pred = [' '.join(labels) for labels in top5_labels]
    
    # 10. 保存提交文件
    # 构建提交结果的 DataFrame
    submission = pd.DataFrame({
        'id': test_ids,
        'Fertilizer Name': top5_pred
    })
    submission.to_csv('submission.csv', index=False)  # 保存为 CSV 文件
    print("✅ 提交文件已保存: submission.csv")
    
    # 11. 分类报告
    # 获取 Out-of-Fold 预测的类别（取概率最高的类别）
    y_oof = np.argmax(oof_probs, axis=1)
    print("\n交叉验证最后一折分类报告:")
    # 打印分类报告
    print(classification_report(y_encoded, y_oof, target_names=class_names))
    
    # 12. 训练耗时
    total_time = time.time() - start_time  # 计算总耗时
    print(f"\n=== 任务完成! 总耗时: {total_time:.2f} 秒 ===")

if __name__ == '__main__':
    main()

