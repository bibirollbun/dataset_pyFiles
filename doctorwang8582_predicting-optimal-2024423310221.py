# 学号: 2024423310221, 姓名: 谢昕旭

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import lightgbm as lgb

# ===================== 数据加载模块 =====================
def load_data():
    """加载训练集和测试集数据"""
    print("正在加载数据...")
    train_data = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
    test_data = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
    print("数据加载完成!")
    print(f"训练数据形状: {train_data.shape}")
    print(f"测试数据形状: {test_data.shape}\n")
    return train_data, test_data

# ===================== 数据预处理模块 =====================
def preprocess_data(train_data, test_data):
    """对类别特征和目标变量进行编码"""
    print("开始数据预处理...")
    cat_cols = ['Soil Type', 'Crop Type']
    
    for col in cat_cols:
        print(f"正在编码类别特征: {col}")
        le = LabelEncoder()
        train_data[col] = le.fit_transform(train_data[col])
        test_data[col] = le.transform(test_data[col])
    
    print("正在编码目标变量...")
    le_y = LabelEncoder()
    train_data['Fertilizer Name'] = le_y.fit_transform(train_data['Fertilizer Name'])
    print(f"目标变量类别数量: {len(le_y.classes_)}")
    print("数据预处理完成!\n")
    
    return train_data, test_data, le_y

# ===================== 特征工程模块 =====================
def feature_engineering(df):
    """添加自定义特征"""
    print("开始特征工程...")
    df['N_P_ratio'] = df['Nitrogen'] / (df['Phosphorous'] + 1e-6)
    df['N_K_ratio'] = df['Nitrogen'] / (df['Potassium'] + 1e-6)
    df['Total_NPK'] = df['Nitrogen'] + df['Phosphorous'] + df['Potassium']
    
    print("添加的特征:")
    print(" - N_P_ratio: 氮磷比")
    print(" - N_K_ratio: 氮钾比")
    print(" - Total_NPK: 总NPK含量")
    print("特征工程完成!\n")
    
    return df

# ===================== 数据准备模块 =====================
def prepare_data(train_data, test_data):
    """准备特征和目标变量，划分训练验证集"""
    print("定义特征和目标变量...")
    features = ['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 
                'Nitrogen', 'Potassium', 'Phosphorous', 'N_P_ratio', 'N_K_ratio', 'Total_NPK']
    target = 'Fertilizer Name'
    
    X = train_data[features]
    y = train_data[target]
    
    print("划分训练集和验证集...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"训练集大小: {X_train.shape[0]}")
    print(f"验证集大小: {X_val.shape[0]}\n")
    
    print("准备LightGBM数据集...")
    train_data_lgb = lgb.Dataset(X_train, label=y_train)
    val_data_lgb = lgb.Dataset(X_val, label=y_val, reference=train_data_lgb)
    print("数据集准备完成!\n")
    
    return X_train, X_val, y_train, y_val, train_data_lgb, val_data_lgb, features, test_data

# ===================== 模型训练模块 =====================
def train_lightgbm_model(train_data_lgb, val_data_lgb):
    """训练LightGBM多分类模型"""
    print("开始训练LightGBM模型...")
    params = {
        'objective': 'multiclass',
        'num_class': len(np.unique(y_train)),
        'metric': 'multi_logloss',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': -1,
        'random_state': 42
    }
    
    print("\n模型参数:")
    for key, value in params.items():
        print(f"{key}: {value}")
    
    model = lgb.train(
        params,
        train_data_lgb,
        num_boost_round=1000,
        valid_sets=[val_data_lgb],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(50)
        ]
    )
    
    print("\n模型训练完成!")
    print(f"最佳迭代次数: {model.best_iteration}")
    return model

# ===================== 预测与结果生成模块 =====================
def generate_predictions(model, test_data, features, le_y):
    """生成测试集预测并创建提交文件"""
    print("\n开始在测试集上进行预测...")
    test_probs = model.predict(test_data[features])
    print(f"预测完成! 共生成 {len(test_probs)} 条预测结果")
    
    print("\n生成Top5预测结果...")
    top5_preds = []
    for i, probs in enumerate(test_probs):
        top5_idx = np.argsort(probs)[::-1][:5]
        top5_labels = le_y.inverse_transform(top5_idx)
        top5_preds.append(" ".join(top5_labels))
        
        if i < 5:
            print(f"样本{i+1}预测结果: {top5_labels}")
    
    print("\n生成提交文件...")
    submission = pd.DataFrame({
        'id': test_data['id'],
        'Fertilizer Name': top5_preds
    })
    submission.to_csv('submission.csv', index=False)
    print("提交文件已保存为 submission.csv")
    
    print("\n=== 处理完成 ===")
    print(f"总样本数: {len(submission)}")
    print("前5条预测结果:")
    print(submission.head())
    return submission

# ===================== 主程序流程 =====================
if __name__ == "__main__":
    # 1. 加载数据
    train_data, test_data = load_data()
    
    # 2. 数据预处理
    train_data, test_data, le_y = preprocess_data(train_data, test_data)
    
    # 3. 特征工程
    train_data = feature_engineering(train_data)
    test_data = feature_engineering(test_data)
    
    # 4. 数据准备
    X_train, X_val, y_train, y_val, train_data_lgb, val_data_lgb, features, test_data = prepare_data(train_data, test_data)
    
    # 5. 模型训练
    model = train_lightgbm_model(train_data_lgb, val_data_lgb)
    
    # 6. 预测与结果生成
    submission = generate_predictions(model, test_data, features, le_y)


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

