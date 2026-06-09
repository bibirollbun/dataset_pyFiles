import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
import xgboost as xgb
from typing import Tuple, List, Dict, Any

# 定义常量
N_FOLDS = 5
SEED = 42
N_ESTIMATORS = 2000

class FertilizerPredictor:
    def __init__(self, train_path: str, test_path: str):
        self.train_path = train_path
        self.test_path = test_path
        self.label_encoder = LabelEncoder()
        self.preprocessor = None
        self.models = []
        self.classes = None
        
    def load_data(self) -> None:
        """加载训练和测试数据"""
        print("加载数据...")
        self.train_data = pd.read_csv(self.train_path)
        self.test_data = pd.read_csv(self.test_path)
        
        # 提取特征和标签
        self.X = self.train_data.drop(['id', 'Fertilizer Name'], axis=1)
        self.y = self.train_data['Fertilizer Name']
        self.test_ids = self.test_data['id']
        self.X_test = self.test_data.drop('id', axis=1)
        
        # 编码标签
        self.y_encoded = self.label_encoder.fit_transform(self.y)
        self.classes = self.label_encoder.classes_
        self.num_classes = len(self.classes)
        
        print(f"\n肥料类别数量: {self.num_classes}")
        print("肥料类别映射:")
        for idx, class_name in enumerate(self.classes):
            print(f"{idx}: {class_name}")
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """特征工程处理"""
        print("\n执行特征工程...")
        df_copy = df.copy()
        
        # 重命名列
        rename_dict = {
            'Temparature': 'temperature',
            'Phosphorous': 'P',
            'Nitrogen': 'N',
            'Potassium': 'K',
            'Moisture': 'moisture'
        }
        df_copy.rename(columns=rename_dict, inplace=True)
        
        # 添加特征
        df_copy['N_P_ratio'] = df_copy['N'] / (df_copy['P'] + 1e-6)
        df_copy['N_K_ratio'] = df_copy['N'] / (df_copy['K'] + 1e-6)
        df_copy['P_K_ratio'] = df_copy['P'] / (df_copy['K'] + 1e-6)
        df_copy['nutrient_total'] = df_copy['N'] + df_copy['P'] + df_copy['K']
        df_copy['nutrient_balance'] = (df_copy['N'] + df_copy['P'] + df_copy['K']) / 3
        
        df_copy['temp_humidity_inter'] = df_copy['temperature'] * df_copy['Humidity']
        df_copy['temp_moisture_inter'] = df_copy['temperature'] * df_copy['moisture']
        df_copy['humidity_moisture_inter'] = df_copy['Humidity'] * df_copy['moisture']
        df_copy['env_total_inter'] = df_copy['temperature'] * df_copy['Humidity'] * df_copy['moisture']
        
        return df_copy
    
    def preprocess_data(self) -> None:
        """预处理数据"""
        print("\n预处理数据...")
        # 特征工程
        self.X_eng = self.engineer_features(self.X)
        self.X_test_eng = self.engineer_features(self.X_test)
        
        # 分离类别和数值特征
        self.cat_features = ['Soil Type', 'Crop Type']
        self.num_features = [col for col in self.X_eng.columns if col not in self.cat_features]
        
        # 创建预处理管道
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', self.num_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.cat_features)
            ])
        
        # 应用预处理
        self.X_processed = self.preprocessor.fit_transform(self.X_eng)
        self.X_test_processed = self.preprocessor.transform(self.X_test_eng)
        
        print("预处理后训练集形状:", self.X_processed.shape)
        print("预处理后测试集形状:", self.X_test_processed.shape)
    
    def get_model_params(self) -> Dict[str, Any]:
        """获取XGBoost模型参数"""
        return {
            'objective': 'multi:softprob',
            'num_class': self.num_classes,
            'eval_metric': 'mlogloss',
            'learning_rate': 0.05,
            'max_depth': 7,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': SEED,
            'tree_method': 'hist',
            'n_estimators': N_ESTIMATORS
        }
    
    def train_model(self) -> None:
        """使用StratifiedKFold训练模型"""
        print(f"\n开始 {N_FOLDS} 折交叉验证训练...")
        self.oof_probs = np.zeros((len(self.X_processed), self.num_classes))
        self.test_probs = np.zeros((len(self.X_test_processed), self.num_classes))
        
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.X_processed, self.y_encoded)):
            print(f"\n=== Fold {fold + 1}/{N_FOLDS} ===")
            
            # 准备数据
            X_train, X_val = self.X_processed[train_idx], self.X_processed[val_idx]
            y_train, y_val = self.y_encoded[train_idx], self.y_encoded[val_idx]
            
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)
            dtest = xgb.DMatrix(self.X_test_processed)
            
            # 训练模型
            model = xgb.train(
                params=self.get_model_params(),
                dtrain=dtrain,
                num_boost_round=N_ESTIMATORS,
                evals=[(dtrain, 'train'), (dval, 'valid')],
                early_stopping_rounds=100,
                verbose_eval=100
            )
            
            # 预测和保存结果
            self.oof_probs[val_idx] = model.predict(dval)
            self.test_probs += model.predict(dtest) / N_FOLDS
            self.models.append(model)
            
            fold_map5 = self.compute_map5(y_val, self.oof_probs[val_idx])
            print(f"Fold {fold + 1} MAP@5: {fold_map5:.5f}")
        
        overall_map5 = self.compute_map5(self.y_encoded, self.oof_probs)
        print(f"\n整体交叉验证 MAP@5: {overall_map5:.5f}")
    
    def compute_map5(self, y_true: np.ndarray, y_pred_probs: np.ndarray) -> float:
        """计算Mean Average Precision @ 5"""
        top5_indices = np.argsort(-y_pred_probs, axis=1)[:, :5]
        ap_list = []
        
        for i in range(len(y_true)):
            true_label = y_true[i]
            pred_group = top5_indices[i]
            score = 0.0
            hit_count = 0.0
            
            for pos in range(min(5, len(pred_group))):
                if pred_group[pos] == true_label:
                    hit_count += 1
                    score += hit_count / (pos + 1)
            
            ap_list.append(score / hit_count if hit_count > 0 else 0.0)
        
        return np.mean(ap_list)
    
    def predict_and_save(self) -> None:
        """生成预测结果并保存到CSV"""
        print("\n生成预测结果...")
        # 获取top5预测索引
        top5_indices = np.argsort(-self.test_probs, axis=1)[:, :5]
        
        # 解码预测结果
        flat_top5 = top5_indices.ravel()
        flat_top5_decoded = self.label_encoder.inverse_transform(flat_top5)
        top5_decoded = flat_top5_decoded.reshape(top5_indices.shape)
        
        # 创建提交文件
        submission = pd.DataFrame({
            'id': self.test_ids,
            'Fertilizer Name': [' '.join(row) for row in top5_decoded]
        })
        
        submission.to_csv('submission.csv', index=False)
        print("提交文件已保存: submission.csv")

if __name__ == "__main__":
    # 数据路径
    TRAIN_CSV = "/kaggle/input/playground-series-s5e6/train.csv"
    TEST_CSV = "/kaggle/input/playground-series-s5e6/test.csv"
    
    # 创建并运行预测器
    predictor = FertilizerPredictor(TRAIN_CSV, TEST_CSV)
    predictor.load_data()
    predictor.preprocess_data()
    predictor.train_model()
    predictor.predict_and_save()    

