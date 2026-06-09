# å­¦å�·:2024423310104 å§“å��:é™ˆæ˜�è�ª
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')

print(f"âœ… XGBoost sÃ¼rÃ¼mÃ¼: {xgb.__version__}")

# æ•°æ�®åŠ è½½ä¸�é¢„å¤„ç�†ç±»
class FertilizerDataProcessor:
    def __init__(self):
        self.label_encoders = {}
        self.target_encoder = LabelEncoder()
        self.features = []
        
    def find_data_directory(self):
        """æŸ¥æ‰¾æ•°æ�®é›†ç›®å½•"""
        data_dir = None
        for dirname, _, filenames in os.walk('/kaggle/input'):
            if 'train.csv' in filenames and 'test.csv' in filenames:
                data_dir = dirname
                break
                
        if data_dir is None:
            raise FileNotFoundError("train.csv ve test.csv bulunamadÄ±.")
            
        print(f"ğŸ“‚ Veriler bulundu: {data_dir}")
        return data_dir
    
    def load_data(self, data_dir):
        """åŠ è½½è®­ç»ƒé›†å’Œæµ‹è¯•é›†"""
        train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
        test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
        return train, test
    
    def encode_categorical_features(self, train, test):
        """ç¼–ç �åˆ†ç±»ç‰¹å¾�"""
        for col in ['Soil Type', 'Crop Type']:
            le = LabelEncoder()
            train[col] = le.fit_transform(train[col])
            test[col] = le.transform(test[col])
            self.label_encoders[col] = le
            
        train['Fertilizer Name'] = self.target_encoder.fit_transform(train['Fertilizer Name'])
        return train, test
    
    def engineer_features(self, df):
        """ç‰¹å¾�å·¥ç¨‹"""
        # å�Ÿå§‹ç‰¹å¾�ç»„å�ˆ
        df['NPK_Ratio'] = df['Nitrogen'] / (df['Potassium'] + df['Phosphorous'] + 1e-6)
        df['N_Moisture'] = df['Nitrogen'] * df['Moisture']
        df['P_Humidity'] = df['Phosphorous'] * df['Humidity']
        df['K_Temp'] = df['Potassium'] * df['Temparature']
        df['NP_sum'] = df['Nitrogen'] + df['Phosphorous']
        df['Temp_Humidity'] = df['Temparature'] * df['Humidity']
        df['SoilCrop_Interaction'] = df['Soil Type'] * 100 + df['Crop Type'] * 10
        df['Moisture_log'] = np.log1p(df['Moisture'])
        df['N_to_P'] = df['Nitrogen'] / (df['Phosphorous'] + 1e-6)
        df['K_to_N'] = df['Potassium'] / (df['Nitrogen'] + 1e-6)
        
        # æ–°å¢�ç‰¹å¾� - è�¥å…»å…ƒç´ ç»„å�ˆ
        df['NPK_Sum'] = df['Nitrogen'] + df['Phosphorous'] + df['Potassium']
        df['NP_Ratio'] = df['Nitrogen'] / (df['Phosphorous'] + 1e-6)
        df['NK_Ratio'] = df['Nitrogen'] / (df['Potassium'] + 1e-6)
        df['PK_Ratio'] = df['Phosphorous'] / (df['Potassium'] + 1e-6)
        
        # æ–°å¢�ç‰¹å¾� - ç�¯å¢ƒå› ç´ ç»„å�ˆ
        df['Temp_Moisture'] = df['Temparature'] * df['Moisture']
        df['Humidity_Moisture'] = df['Humidity'] * df['Moisture']
        df['Temp_Humidity_Moisture'] = df['Temparature'] * df['Humidity'] * df['Moisture']
        
        # æ–°å¢�ç‰¹å¾� - åˆ†ç±»ç‰¹å¾�ä¸�æ•°å€¼ç‰¹å¾�ç»„å�ˆ
        for col in ['Soil Type', 'Crop Type']:
            for nutrient in ['Nitrogen', 'Phosphorous', 'Potassium']:
                df[f'{col}_{nutrient}_mean'] = df.groupby(col)[nutrient].transform('mean')
                df[f'{col}_{nutrient}_std'] = df.groupby(col)[nutrient].transform('std')
        
        # æ–°å¢�ç‰¹å¾� - å¤šé¡¹å¼�ç‰¹å¾�
        df['Temparature_sq'] = df['Temparature'] ** 2
        df['Humidity_sq'] = df['Humidity'] ** 2
        df['Moisture_sq'] = df['Moisture'] ** 2
        
        # å¤„ç�†å�¯èƒ½çš„NaNå€¼
        df.fillna(0, inplace=True)
        df.replace([np.inf, -np.inf], 0, inplace=True)
        
        return df
    
    def prepare_features(self, train, test):
        """å‡†å¤‡ç‰¹å¾�åˆ—è¡¨"""
        # æ�’é™¤IDå’Œç›®æ ‡åˆ—
        exclude_cols = ['id', 'Fertilizer Name'] if 'Fertilizer Name' in train.columns else ['id']
        self.features = [col for col in train.columns if col not in exclude_cols]
        return self.features

# æ¨¡å�‹è®­ç»ƒä¸�è¯„ä¼°ç±»
class FertilizerModel:
    def __init__(self, features, target_encoder):
        self.features = features
        self.target_encoder = target_encoder
        self.val_preds = None
        self.test_preds = None
        self.models = []
        
    def train(self, X, y, X_test, n_splits=5):
        """ä½¿ç”¨Stratified K-Foldè®­ç»ƒæ¨¡å�‹"""
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        self.val_preds = np.zeros((len(X), len(np.unique(y))))
        self.test_preds = np.zeros((len(X_test), len(np.unique(y))))
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"ğŸ“˜ Fold {fold + 1} eÄŸitiliyor...")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # ä¼˜åŒ–çš„XGBoostå�‚æ•°
            model = XGBClassifier(
                objective='multi:softprob',
                num_class=len(self.target_encoder.classes_),
                random_state=42,
                learning_rate=0.03,
                max_depth=5,
                n_estimators=1000,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=3.0,
                reg_alpha=1.5,
                gamma=0.1,
                min_child_weight=5,
                tree_method='hist',
                eval_metric='mlogloss',
                use_label_encoder=False
            )
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=50,
                verbose=0
            )
            
            self.val_preds[val_idx] = model.predict_proba(X_val)
            self.test_preds += model.predict_proba(X_test) / skf.n_splits
            self.models.append(model)
            
        return self.val_preds, self.test_preds
    
    def get_top_k_predictions(self, probs, k=3):
        """è�·å�–top-ké¢„æµ‹ç»“æ�œ"""
        top_k_indices = np.argsort(probs, axis=1)[:, -k:][:, ::-1]
        top_k_labels = [self.target_encoder.inverse_transform(indices) for indices in top_k_indices]
        return [' '.join(str(label) for label in labels) for labels in top_k_labels]
    
    def create_submission(self, test_df, k=3):
        """åˆ›å»ºæ��äº¤æ–‡ä»¶"""
        submission = pd.DataFrame({
            'id': test_df['id'],
            'Fertilizer Name': self.get_top_k_predictions(self.test_preds, k=k)
        })
        submission.to_csv('/kaggle/working/submission.csv', index=False)
        print("âœ… 'submission.csv' baÅŸarÄ±yla oluÅŸturuldu!")
        return submission
    
    def evaluate_map_at_k(self, y_true, k=3):
        """è®¡ç®—MAP@kè¯„ä¼°æŒ‡æ ‡"""
        score = 0.0
        for i in range(len(y_true)):
            true_label = self.target_encoder.inverse_transform([y_true[i]])[0]
            top_k_preds = self.target_encoder.inverse_transform(np.argsort(self.val_preds[i])[::-1][:k])
            for j in range(k):
                if top_k_preds[j] == true_label:
                    score += 1.0 / (j + 1)
                    break
        return score / len(y_true)
    
    def plot_feature_importance(self, num_features=15):
        """ç»˜åˆ¶ç‰¹å¾�é‡�è¦�æ€§å›¾"""
        feature_importance = pd.DataFrame()
        for i, model in enumerate(self.models):
            fold_importance = pd.DataFrame({
                'feature': self.features,
                'importance': model.feature_importances_,
                'fold': i
            })
            feature_importance = pd.concat([feature_importance, fold_importance])
        
        # è®¡ç®—å¹³å�‡é‡�è¦�æ€§å¹¶æ�’åº�
        avg_importance = feature_importance.groupby('feature')['importance'].mean().sort_values(ascending=False)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(x=avg_importance.values[:num_features], y=avg_importance.index[:num_features])
        plt.title('Ã–zellik Ã–nemi (Ortalama)')
        plt.xlabel('Ã–nem Skoru')
        plt.ylabel('Ã–zellik')
        plt.tight_layout()
        plt.show()
        
        return avg_importance
    
    def plot_prediction_distribution(self):
        """ç»˜åˆ¶é¢„æµ‹åˆ†å¸ƒ"""
        # è�·å�–é¢„æµ‹çš„ç±»åˆ«æ¦‚ç�‡
        pred_classes = np.argmax(self.test_preds, axis=1)
        pred_labels = self.target_encoder.inverse_transform(pred_classes)
        
        plt.figure(figsize=(10, 6))
        sns.countplot(y=pred_labels)
        plt.title('Test Verisi iÃ§in Tahmin Edilen GÃ¼bre DaÄŸÄ±lÄ±mÄ±')
        plt.xlabel('Ã–rnek SayÄ±sÄ±')
        plt.ylabel('GÃ¼bre TÃ¼rÃ¼')
        plt.tight_layout()
        plt.show()

# ä¸»å‡½æ•°
def main():
    # æ•°æ�®å¤„ç�†
    processor = FertilizerDataProcessor()
    data_dir = processor.find_data_directory()
    train, test = processor.load_data(data_dir)
    train, test = processor.encode_categorical_features(train, test)
    train = processor.engineer_features(train)
    test = processor.engineer_features(test)
    features = processor.prepare_features(train, test)
    
    # å‡†å¤‡è®­ç»ƒæ•°æ�®
    X = train[features]
    y = train['Fertilizer Name']
    X_test = test[features]
    
    # æ¨¡å�‹è®­ç»ƒä¸�è¯„ä¼°
    model = FertilizerModel(features, processor.target_encoder)
    val_preds, test_preds = model.train(X, y, X_test)
    
    # åˆ›å»ºæ��äº¤æ–‡ä»¶
    submission = model.create_submission(test)
    
    # è¯„ä¼°æ¨¡å�‹
    val_score = model.evaluate_map_at_k(y, k=3)
    print(f"ğŸ“Š Validation MAP@3: {val_score:.4f}")
    
    # å�¯è§†åŒ–
    model.plot_feature_importance(num_features=20)
    model.plot_prediction_distribution()
    
    return model, submission

if __name__ == "__main__":
    model, submission = main()

