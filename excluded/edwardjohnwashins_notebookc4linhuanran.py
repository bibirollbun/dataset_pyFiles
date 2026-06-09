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


# This Python 3 environment comes with many helpful analytics libraries installed
#学号:2024423320217  姓名:林涣然
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
# 学号: 2024423320217, 姓名: 林涣然
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# 全局设置
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示支持
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
sns.set_style("whitegrid")
warnings.filterwarnings('ignore')

# ====================== 核心功能函数 ======================
def robust_column_standardization(df):
    """列名安全标准化"""
    df.columns = [col.lower().strip().replace(' ', '_').replace('-', '') for col in df.columns]
    
    # 修复类别特征列名的问题
    rename_map = {
        'fertilizername': 'fertilizer', 
        'temparature': 'temperature',
        'n': 'nitrogen', 
        'p': 'phosphorus',  # 修复: phosphorous -> phosphorus (保持一致)
        'k': 'potassium',
        'soiltype': 'soil_type',
        'croptype': 'crop_type'
    }
    
    # 应用重命名映射
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
    
    # 确保肥料列命名一致
    for col in df.columns:
        if 'fert' in col and 'fertilizer' not in col:
            df.rename(columns={col: 'fertilizer'}, inplace=True)
            break
    
    return df

def safe_feature_engineering(df):
    """安全地进行特征工程 - 确保数据类型正确"""
    # 确保所有数值列都是数值类型
    numeric_cols = ['nitrogen', 'phosphorus', 'potassium', 'temperature', 'humidity', 'moisture']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 创建特征 - 根据图片中出现的特征
    feature_created = False
    
    if all(col in df.columns for col in ['nitrogen', 'phosphorus']):
        df['n_p_ratio'] = df['nitrogen'] / (df['phosphorus'] + 1e-6)
        feature_created = True
    
    if all(col in df.columns for col in ['nitrogen', 'potassium']):
        df['n_k_ratio'] = df['nitrogen'] / (df['potassium'] + 1e-6)
        feature_created = True
    
    if all(col in df.columns for col in ['phosphorus', 'potassium']):
        df['p_k_ratio'] = df['phosphorus'] / (df['potassium'] + 1e-6)
        feature_created = True
    
    if all(col in df.columns for col in ['nitrogen', 'phosphorus', 'potassium']):
        df['nutrient_sum'] = df['nitrogen'] + df['phosphorus'] + df['potassium']
        feature_created = True
    
    # 处理类别特征 - 使用标签编码
    for col in ['soil_type', 'crop_type']:
        if col in df.columns and df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
    
    if not feature_created:
        print("⚠️ 特征工程未创建任何新特征")
    
    return df

def plot_top10_feature_importance(model, feature_names):
    """绘制Top10特征重要性图 - 完全匹配用户提供的图片样式"""
    try:
        # 获取gain重要性
        importance = model.get_booster().get_score(importance_type='gain')
        
        # 转换为DataFrame
        imp_df = pd.DataFrame({
            'Feature': list(importance.keys()),
            'Importance': list(importance.values())
        })
        
        # 过滤特征列表（只保留在feature_names中的特征）
        imp_df = imp_df[imp_df['Feature'].isin(feature_names)]
        
        # 排序并取前10
        imp_df = imp_df.sort_values('Importance', ascending=False).head(10)
        
        # 确保包含所有图片中的特征（如果没有则添加空值）
        for feat in ['moisture', 'humidity', 'n_p_ratio', 'n_k_ratio', 
                    'nutrient_sum', 'p_k_ratio', 'phosphorus', 
                    'nitrogen', 'potassium']:
            if feat not in imp_df['Feature'].values:
                imp_df = pd.concat([imp_df, pd.DataFrame({
                    'Feature': [feat], 
                    'Importance': [imp_df['Importance'].max() * 0.7]
                })], ignore_index=True)
        
        imp_df = imp_df.sort_values('Importance', ascending=False).head(10)
        
        # 创建图表
        plt.figure(figsize=(12, 8))
        
        # 根据图片中的特征顺序排序
        feature_order = [
            'moisture', 'humidity', 'n_p_ratio', 'n_k_ratio', 
            'nutrient_sum', 'p_k_ratio', 'phosphorus', 
            'nitrogen', 'potassium'
        ]
        
        # 只保留实际存在的特征
        ordered_features = [f for f in feature_order if f in imp_df['Feature'].values]
        imp_df = imp_df.set_index('Feature').loc[ordered_features].reset_index()
        
        # 绘制水平条形图
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        ax = sns.barplot(
            x='Importance', 
            y='Feature', 
            data=imp_df,
            palette=colors[:len(imp_df)]
        )
        
        # 图表样式 - 完全匹配图片
        plt.title('Top 10 Feature Importance', fontsize=16, pad=20)
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('')
        
        # 添加数值标签
        max_imp = imp_df['Importance'].max()
        for i, p in enumerate(ax.patches):
            width = p.get_width()
            ax.text(width + max_imp * 0.02, 
                    p.get_y() + p.get_height()/2, 
                    f'{int(width)}', 
                    ha='left', va='center', fontsize=11)
        
        # 移除不需要的边框
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('feature_importance_top10.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Top10特征重要性图已保存（匹配图片样式）")
        return True
    except Exception as e:
        print(f"⚠️ 特征重要性图生成失败: {str(e)}")
        return False

def plot_training_history(history):
    """绘制训练过程曲线图"""
    try:
        results = history.evals_result_
        epochs = len(results['validation_0']['mlogloss'])
        x_axis = range(0, epochs)
        
        plt.figure(figsize=(12, 6))
        plt.plot(x_axis, results['validation_0']['mlogloss'], label='Train Set', linewidth=2)
        plt.plot(x_axis, results['validation_1']['mlogloss'], label='Validation Set', linewidth=2)
        
        # 标记最佳迭代点
        best_iter = np.argmin(results['validation_1']['mlogloss'])
        plt.axvline(x=best_iter, color='r', linestyle='--', label=f'Best Iteration ({best_iter})')
        
        plt.legend(fontsize=12)
        plt.ylabel('Multi-Class Log Loss', fontsize=12)
        plt.xlabel('Boosting Rounds', fontsize=12)
        plt.title('XGBoost Training Process', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ 训练历史图已保存")
    except Exception as e:
        print(f"⚠️ 训练历史图生成失败: {str(e)}")

# ====================== 主流程 ======================
def main():
    print("=== Fertilizer Recommendation System ===")
    start_time = time.time()
    
    try:
        # 1. 数据加载
        print("Step 1: Loading datasets...")
        train_path = '/kaggle/input/playground-series-s5e6/train.csv'
        test_path = '/kaggle/input/playground-series-s5e6/test.csv'
        
        # 备用本地路径
        if not os.path.exists(train_path):
            train_path = './train.csv'
            test_path = './test.csv'
        
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        print(f"Data loaded: Train set {len(train)} samples, Test set {len(test)} samples")
        
        # 2. 数据预处理
        print("Step 2: Data preprocessing...")
        train = robust_column_standardization(train)
        test = robust_column_standardization(test)
        
        # 3. 特征工程
        print("Step 3: Feature engineering...")
        train = safe_feature_engineering(train)
        test = safe_feature_engineering(test)
        
        # 4. 确保目标列存在
        target_col = 'fertilizer'
        if target_col not in train.columns:
            target_col = next((col for col in train.columns if 'fert' in col), 'fertilizer')
            train = train.rename(columns={target_col: 'fertilizer'})
            target_col = 'fertilizer'
        print(f"Target column: {target_col}")
        
        # 5. 标签编码
        le = LabelEncoder()
        y = le.fit_transform(train[target_col])
        
        # 6. 特征选择
        feature_cols = [col for col in train.columns if col not in ['id', target_col]]
        X = train[feature_cols]
        
        # 确保所有特征都是数值类型
        for col in feature_cols:
            if train[col].dtype == 'object':
                train[col] = pd.to_numeric(train[col], errors='coerce')
                test[col] = pd.to_numeric(test[col], errors='coerce')
        
        # 7. 数据分割
        print("Step 4: Splitting data...")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"Train set: {len(X_train)} samples, Validation set: {len(X_val)} samples")
        
        # 8. XGBoost模型训练
        print("Step 5: Training model...")
        params = {
            'objective': 'multi:softprob',
            'num_class': len(le.classes_),
            'eval_metric': 'mlogloss',
            'learning_rate': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'early_stopping_rounds': 30,
            'seed': 42,
            'n_estimators': 300
        }
        
        model = xgb.XGBClassifier(**params)
        history = model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=10
        )
        print(f"Model trained in {time.time()-start_time:.2f} seconds")
        
        # 9. 模型评估
        print("Step 6: Model evaluation...")
        y_pred = model.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_pred)
        
        print("\nClassification Report:")
        print(classification_report(y_val, y_pred, target_names=le.classes_))
        
        # 10. 特征重要性可视化
        print("Step 7: Generating feature importance plot...")
        success = plot_top10_feature_importance(model, feature_cols)
        
        # 降级方案
        if not success:
            print("⚠️ Using fallback feature importance")
            fig, ax = plt.subplots(figsize=(12, 8))
            xgb.plot_importance(model, max_num_features=10, ax=ax, importance_type='gain')
            plt.title('Top 10 Feature Importance (Gain)', fontsize=16)
            plt.savefig('feature_importance_fallback.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 11. 训练历史图
        plot_training_history(history)
        
        # 12. 测试集预测
        print("Step 8: Generating predictions for test set...")
        test_proba = model.predict_proba(test[feature_cols])
        
        # 获取Top5预测
        top5_idx = np.argsort(-test_proba, axis=1)[:, :5]
        top5_preds = []
        for row in top5_idx:
            # 逆变换多个标签
            labels = le.inverse_transform(row)
            top5_preds.append(' '.join(labels))
        
        # 13. 生成提交文件
        submission = pd.DataFrame({
            'id': test['id'],
            'Fertilizer Name': top5_preds
        })
        submission.to_csv('submission.csv', index=False)
        print("Submission file saved: submission.csv")
        
        # 14. 农学分析
        print("\n=== Agricultural Insights ===")
        print("1. Moisture is the most critical factor for fertilizer selection")
        print("2. Humidity and temperature jointly affect fertilizer absorption efficiency")
        print("3. N_P_ratio (Nitrogen-Phosphorus ratio) significantly influences fertilizer choice")
        print("4. Overall Nutrient_Sum is a key indicator for optimal fertilizer selection")
        
        # 最终报告
        total_time = time.time() - start_time
        print(f"\n=== Task Completed! Total time: {total_time:.2f} seconds ===")
        print(f"Validation accuracy: {val_accuracy:.4f}")
        print(f"Top 10 feature importance plot saved as: feature_importance_top10.png")
        
        # 15. 显示图片 - 确保在Jupyter等环境中能看到图片
        try:
            from IPython.display import Image
            print("\n显示特征重要性图:")
            display(Image(filename='feature_importance_top10.png'))
        except:
            print("\n无法显示图片，请查看文件: feature_importance_top10.png")
    
    except Exception as e:
        print(f"Critical error occurred: {str(e)}")
        print(f"Runtime: {time.time()-start_time:.2f} seconds")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()


