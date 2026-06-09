# 学号: 2024423310221, 姓名: 李国铭

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import lightgbm as lgb

class FertilizerPredictor:
    """肥料类型预测模型"""
    
    def __init__(self):
        """初始化模型参数"""
        self.models = []
        self.label_encoders = {}
        self.features = None
        self.target = None
        
    def load_data(self, train_path, test_path):
        """加载训练集和测试集数据"""
        print("正在加载数据...")
        self.train_data = pd.read_csv(train_path)
        self.test_data = pd.read_csv(test_path)
        print("数据加载完成!")
        print(f"训练数据形状: {self.train_data.shape}")
        print(f"测试数据形状: {self.test_data.shape}\n")
        return self
        
    def preprocess_data(self):
        """数据预处理和特征工程"""
        print("开始数据预处理...")
        
        # 编码类别特征
        cat_cols = ['Soil Type', 'Crop Type']
        for col in cat_cols:
            print(f"正在编码类别特征: {col}")
            le = LabelEncoder()
            self.train_data[col] = le.fit_transform(self.train_data[col])
            self.test_data[col] = le.transform(self.test_data[col])
            self.label_encoders[col] = le
        
        # 编码目标变量
        print("正在编码目标变量...")
        le_y = LabelEncoder()
        self.train_data['Fertilizer Name'] = le_y.fit_transform(self.train_data['Fertilizer Name'])
        self.label_encoders['Fertilizer Name'] = le_y
        print(f"目标变量类别数量: {len(le_y.classes_)}")
        
        # 特征工程
        print("\n开始特征工程...")
        self.train_data = self._add_features(self.train_data)
        self.test_data = self._add_features(self.test_data)
        
        print("添加的特征:")
        print(" - N_P_ratio: 氮磷比")
        print(" - N_K_ratio: 氮钾比")
        print(" - Total_NPK: 总NPK含量")
        print("数据预处理完成!\n")
        return self
    
    def _add_features(self, df):
        """添加自定义特征"""
        # 计算氮磷比，加1e-6避免除零错误
        df['N_P_ratio'] = df['Nitrogen'] / (df['Phosphorous'] + 1e-6)
        # 计算氮钾比，加1e-6避免除零错误
        df['N_K_ratio'] = df['Nitrogen'] / (df['Potassium'] + 1e-6)
        # 计算总NPK含量
        df['Total_NPK'] = df['Nitrogen'] + df['Phosphorous'] + df['Potassium']
        return df
    
    def prepare_model_data(self):
        """准备模型训练数据"""
        print("准备模型训练数据...")
        
        # 定义特征和目标
        self.features = [
            'Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 
            'Nitrogen', 'Potassium', 'Phosphorous', 'N_P_ratio', 'N_K_ratio', 'Total_NPK'
        ]
        self.target = 'Fertilizer Name'
        
        # 划分训练集和验证集
        print("划分训练集和验证集...")
        X = self.train_data[self.features]
        y = self.train_data[self.target]
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"训练集大小: {self.X_train.shape[0]}")
        print(f"验证集大小: {self.X_val.shape[0]}\n")
        
        # 转换为LightGBM数据集格式
        print("准备LightGBM数据集...")
        self.train_data_lgb = lgb.Dataset(self.X_train, label=self.y_train)
        self.val_data_lgb = lgb.Dataset(self.X_val, label=self.y_val, reference=self.train_data_lgb)
        print("数据集准备完成!\n")
        return self
    
    def train_model(self):
        """训练LightGBM模型"""
        print("开始训练LightGBM模型...")
        
        # 模型参数设置
        params = {
            'objective': 'multiclass',
            'num_class': len(np.unique(self.y_train)),
            'metric': 'multi_logloss',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': -1,
            'random_state': 42
        }
        
        print("\n模型参数:")
        for key, value in params.items():
            print(f"{key}: {value}")
        
        # 训练模型
        self.model = lgb.train(
            params,
            self.train_data_lgb,
            num_boost_round=1000,
            valid_sets=[self.val_data_lgb],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=True),
                lgb.log_evaluation(50)
            ]
        )
        
        print("\n模型训练完成!")
        print(f"最佳迭代次数: {self.model.best_iteration}")
        return self
    
    def predict(self):
        """在测试集上进行预测"""
        print("\n开始在测试集上进行预测...")
        self.test_probs = self.model.predict(self.test_data[self.features])
        print(f"预测完成! 共生成 {len(self.test_probs)} 条预测结果")
        return self
    
    def generate_submission(self, output_path='submission.csv'):
        """生成提交文件"""
        print("\n生成Top5预测结果...")
        top5_preds = []
        
        for i, probs in enumerate(self.test_probs):
            # 对概率排序，取前5个类别索引（降序）
            top5_idx = np.argsort(probs)[::-1][:5]
            # 将索引转换为原始类别标签
            top5_labels = self.label_encoders['Fertilizer Name'].inverse_transform(top5_idx)
            top5_preds.append(" ".join(top5_labels))
            
            # 显示前5条预测结果示例
            if i < 5:
                print(f"样本{i+1}预测结果: {top5_labels}")
        
        print("\n生成提交文件...")
        submission = pd.DataFrame({
            'id': self.test_data['id'],
            'Fertilizer Name': top5_preds
        })
        
        # 保存提交文件
        submission.to_csv(output_path, index=False)
        print(f"提交文件已保存为 {output_path}")
        
        print("\n=== 处理完成 ===")
        print(f"总样本数: {len(submission)}")
        print("前5条预测结果:")
        print(submission.head())
        return submission

# 主程序
if __name__ == "__main__":
    # 创建并训练模型
    predictor = FertilizerPredictor()
    predictor.load_data(
        train_path="/kaggle/input/playground-series-s5e6/train.csv",
        test_path="/kaggle/input/playground-series-s5e6/test.csv"
    )
    predictor.preprocess_data()
    predictor.prepare_model_data()
    predictor.train_model()
    predictor.predict()
    predictor.generate_submission()

