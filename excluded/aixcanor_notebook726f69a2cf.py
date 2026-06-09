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


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, label_ranking_average_precision_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

# è®¾ç½®éš�æœºç§�å­�ç¡®ä¿�ç»“æ�œå�¯å¤�ç�°
np.random.seed(42)

class FertilizerPredictor:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.imputer = SimpleImputer(strategy='mean')
        self.scaler = StandardScaler()
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_importances = {}
        self.feature_importance = None
        self.class_mapping = None
        
    def load_data(self, train_path, test_path):
        """åŠ è½½å¹¶æ£€æŸ¥æ•°æ�®"""
        try:
            self.train_df = pd.read_csv(train_path)
            self.test_df = pd.read_csv(test_path)
            print(f"è®­ç»ƒé›†å½¢çŠ¶: {self.train_df.shape}, æµ‹è¯•é›†å½¢çŠ¶: {self.test_df.shape}")
            
            # è‡ªåŠ¨è¯†åˆ«ç›®æ ‡åˆ—
            possible_targets = ['Fertilizer Name', 'fertilizer', 'target', 'label']
            self.target_column = None
            for col in possible_targets:
                if col in self.train_df.columns:
                    self.target_column = col
                    break
            
            if self.target_column is None:
                print("æœªè‡ªåŠ¨è¯†åˆ«åˆ°ç›®æ ‡åˆ—ï¼Œè¯·æ‰‹åŠ¨æŒ‡å®š:")
                print(self.train_df.columns.tolist())
                self.target_column = input("è¯·è¾“å…¥ç›®æ ‡åˆ—å��ç§°: ").strip()
                
            print(f"ä½¿ç”¨ç›®æ ‡åˆ—: {self.target_column}")
            return True
        except Exception as e:
            print(f"æ•°æ�®åŠ è½½å¤±è´¥: {e}")
            return False
    
    def explore_data(self):
        """æ•°æ�®æ�¢ç´¢åˆ†æ��"""
        print("\n=== æ•°æ�®æ�¢ç´¢åˆ†æ�� ===")
        
        # ç›®æ ‡å�˜é‡�åˆ†å¸ƒ
        plt.figure(figsize=(12, 6))
        sns.countplot(x=self.target_column, data=self.train_df)
        plt.title('è‚¥æ–™ç±»å�‹åˆ†å¸ƒ')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('target_distribution.png')
        plt.close()
        
        # æ•°å€¼ç‰¹å¾�åˆ†å¸ƒ
        num_features = ['Nitrogen', 'Phosphorous', 'Potassium', 
                        'Temparature', 'Humidity', 'Moisture']
        available_num_features = [f for f in num_features if f in self.train_df.columns]
        
        if available_num_features:
            self.train_df[available_num_features].hist(bins=20, figsize=(15, 10))
            plt.tight_layout()
            plt.savefig('numerical_features_distribution.png')
            plt.close()
        
        # ç±»åˆ«ç‰¹å¾�åˆ†å¸ƒ
        cat_features = ['Soil Type', 'Crop Type']
        available_cat_features = [f for f in cat_features if f in self.train_df.columns]
        
        if available_cat_features:
            n_cols = len(available_cat_features)
            plt.figure(figsize=(5 * n_cols, 5))
            for i, feature in enumerate(available_cat_features):
                plt.subplot(1, n_cols, i+1)
                sns.countplot(x=feature, data=self.train_df)
                plt.title(f'{feature}åˆ†å¸ƒ')
                plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('categorical_features_distribution.png')
            plt.close()
        
        # ç‰¹å¾�ä¸�ç›®æ ‡å�˜é‡�çš„å…³ç³»
        nutrient_features = ['Nitrogen', 'Phosphorous', 'Potassium']
        available_nutrients = [f for f in nutrient_features if f in self.train_df.columns]
        
        if available_nutrients and self.target_column:
            plt.figure(figsize=(5 * len(available_nutrients), 5))
            for i, feature in enumerate(available_nutrients):
                plt.subplot(1, len(available_nutrients), i+1)
                sns.boxplot(x=self.target_column, y=feature, data=self.train_df)
                plt.xticks(rotation=45)
                plt.title(f'{feature}åˆ†å¸ƒ')
            plt.tight_layout()
            plt.savefig('nutrient_distribution_by_fertilizer.png')
            plt.close()
    
    def preprocess_data(self):
        """æ•°æ�®é¢„å¤„ç�†"""
        print("\n=== æ•°æ�®é¢„å¤„ç�† ===")
        
        # åˆ†ç¦»ç‰¹å¾�å’Œç›®æ ‡å�˜é‡�
        self.soil_features = ['Nitrogen', 'Phosphorous', 'Potassium', 
                             'Temparature', 'Humidity', 'Moisture', 
                             'Soil Type', 'Crop Type']
        available_features = [f for f in self.soil_features if f in self.train_df.columns]
        
        self.X = self.train_df[available_features].copy()
        self.y_str = self.train_df[self.target_column].copy()
        
        # ç¼–ç �ç›®æ ‡å�˜é‡�
        self.y = self.label_encoder.fit_transform(self.y_str)
        self.num_classes = len(self.label_encoder.classes_)
        print(f"ç±»åˆ«æ•°é‡�: {self.num_classes}")
        
        # ä¿�å­˜ç±»åˆ«æ˜ å°„
        self.class_mapping = pd.DataFrame({
            'encoded_label': range(self.num_classes),
            'fertilizer_type': self.label_encoder.classes_
        })
        self.class_mapping.to_csv('class_mapping.csv', index=False)
        print("ç±»åˆ«æ˜ å°„å·²ä¿�å­˜ä¸º 'class_mapping.csv'")
        
        # åˆ†ç¦»æ•°å€¼å’Œç±»åˆ«ç‰¹å¾�
        self.num_features = [f for f in available_features if self.X[f].dtype != 'object']
        self.cat_features = [f for f in available_features if self.X[f].dtype == 'object']
        
        # å¤„ç�†ç¼ºå¤±å€¼
        if self.num_features:
            self.X[self.num_features] = self.imputer.fit_transform(self.X[self.num_features])
            print("æ•°å€¼ç‰¹å¾�ç¼ºå¤±å€¼å¤„ç�†å®Œæˆ�")
        
        # ç¼–ç �ç±»åˆ«ç‰¹å¾�
        if self.cat_features:
            self.X = pd.get_dummies(self.X, columns=self.cat_features)
            print(f"ç¼–ç �å��ç‰¹å¾�æ•°é‡�: {self.X.shape[1]}")
        
        # æ ‡å‡†åŒ–æ•°å€¼ç‰¹å¾�
        if self.num_features:
            self.X[self.num_features] = self.scaler.fit_transform(self.X[self.num_features])
            print("æ•°å€¼ç‰¹å¾�æ ‡å‡†åŒ–å®Œæˆ�")
        
        # åˆ’åˆ†è®­ç»ƒé›†å’ŒéªŒè¯�é›†
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y)
        print(f"è®­ç»ƒé›†å½¢çŠ¶: {self.X_train.shape}, éªŒè¯�é›†å½¢çŠ¶: {self.X_val.shape}")
        
        # å‡†å¤‡æµ‹è¯•é›†
        self.X_test = self.test_df[available_features].copy()
        
        # å¤„ç�†æµ‹è¯•é›†ç¼ºå¤±å€¼
        if self.num_features:
            self.X_test[self.num_features] = self.imputer.transform(self.X_test[self.num_features])
        
        # ç¼–ç �ç±»åˆ«ç‰¹å¾�
        if self.cat_features:
            self.X_test = pd.get_dummies(self.X_test, columns=self.cat_features)
            
            # ç¡®ä¿�æµ‹è¯•é›†æœ‰ç›¸å�Œçš„åˆ—
            for col in self.X.columns:
                if col not in self.X_test.columns:
                    self.X_test[col] = 0
            
            # æŒ‰è®­ç»ƒé›†çš„åˆ—é¡ºåº�æ�’åº�
            self.X_test = self.X_test[self.X.columns]
        
        # æ ‡å‡†åŒ–æ•°å€¼ç‰¹å¾�
        if self.num_features:
            self.X_test[self.num_features] = self.scaler.transform(self.X_test[self.num_features])
        
        print(f"æµ‹è¯•é›†å½¢çŠ¶: {self.X_test.shape}")
        print("æ•°æ�®é¢„å¤„ç�†å®Œæˆ�!")
    
    def train_models(self):
        """è®­ç»ƒå¤šä¸ªæ¨¡å�‹å¹¶è¯„ä¼°"""
        print("\n=== æ¨¡å�‹è®­ç»ƒä¸�è¯„ä¼° ===")
        
        # å®šä¹‰æ¨¡å�‹
        self.models = {
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
            "XGBoost": XGBClassifier(n_estimators=300, learning_rate=0.1, random_state=42, n_jobs=-1),
            "LightGBM": LGBMClassifier(n_estimators=300, learning_rate=0.1, random_state=42, n_jobs=-1),
            "CatBoost": CatBoostClassifier(iterations=300, learning_rate=0.1, random_state=42, verbose=False)
        }
        
        # å­˜å‚¨ç‰¹å¾�é‡�è¦�æ€§çš„å­—å…¸
        self.feature_importances = {}
        
        best_score = 0
        results = []
        
        for name, model in self.models.items():
            print(f"\nè®­ç»ƒ {name}...")
            
            # è®­ç»ƒæ¨¡å�‹
            model.fit(self.X_train, self.y_train)
            
            # åœ¨éªŒè¯�é›†ä¸Šé¢„æµ‹
            y_pred = model.predict(self.X_val)
            y_pred_proba = model.predict_proba(self.X_val)
            
            # è®¡ç®—è¯„ä¼°æŒ‡æ ‡
            accuracy = accuracy_score(self.y_val, y_pred)
            map5 = self.calculate_map5(self.y_val, y_pred_proba)
            
            results.append({
                'Model': name,
                'Accuracy': accuracy,
                'MAP@5': map5
            })
            
            print(f"{name} è¯„ä¼°ç»“æ�œ:")
            print(f"  å‡†ç¡®ç�‡: {accuracy:.4f}")
            print(f"  MAP@5: {map5:.4f}")
            
            # ä¿�å­˜ç‰¹å¾�é‡�è¦�æ€§
            if hasattr(model, 'feature_importances_'):
                self.feature_importances[name] = model.feature_importances_
            
            # è®°å½•æœ€ä½³æ¨¡å�‹
            if map5 > best_score:
                best_score = map5
                self.best_model = model
                self.best_model_name = name  # ä¿�å­˜æœ€ä½³æ¨¡å�‹å��ç§°
        
        # æ˜¾ç¤ºç»“æ�œè¡¨æ ¼
        results_df = pd.DataFrame(results).sort_values('MAP@5', ascending=False)
        print("\n=== æ¨¡å�‹æ¯”è¾ƒç»“æ�œ ===")
        print(results_df)
        
        # ä¿�å­˜æœ€ä½³æ¨¡å�‹çš„ç‰¹å¾�é‡�è¦�æ€§
        if self.best_model and self.best_model_name in self.feature_importances:
            self.feature_importance = pd.DataFrame({
                'Feature': self.X.columns,
                'Importance': self.feature_importances[self.best_model_name]
            }).sort_values('Importance', ascending=False)
            self.feature_importance.to_csv('feature_importance.csv', index=False)
            print(f"\n{self.best_model_name} ç‰¹å¾�é‡�è¦�æ€§å·²ä¿�å­˜ä¸º 'feature_importance.csv'")
    
    def calculate_map5(self, y_true, y_pred_proba):
        """è®¡ç®—MAP@5æŒ‡æ ‡"""
        # è�·å�–æ¯�ä¸ªæ ·æœ¬çš„å‰�5ä¸ªé¢„æµ‹ç±»åˆ«
        top5_preds = np.argsort(y_pred_proba, axis=1)[:, -5:][:, ::-1]
        
        map_score = 0.0
        for i in range(len(y_true)):
            true_label = y_true[i]
            preds = top5_preds[i]
            
            # è®¡ç®—æ¯�ä¸ªæ ·æœ¬çš„AP@5
            score = 0.0
            hits = 0
            for j, pred in enumerate(preds):
                if pred == true_label:
                    hits += 1
                    score += hits / (j + 1)
            
            if hits > 0:
                map_score += score / 1.0
        
        return map_score / len(y_true)
    
    def generate_submission(self):
        """ç”Ÿæˆ�æ��äº¤æ–‡ä»¶"""
        if self.best_model is None:
            print("è¯·å…ˆè®­ç»ƒæ¨¡å�‹!")
            return
        
        print("\n=== ç”Ÿæˆ�æ��äº¤æ–‡ä»¶ ===")
        
        # é¢„æµ‹æµ‹è¯•é›†
        test_proba = self.best_model.predict_proba(self.X_test)
        
        # è�·å�–æ¯�ä¸ªæ ·æœ¬çš„å‰�5ä¸ªé¢„æµ‹
        top5_indices = np.argsort(test_proba, axis=1)[:, -5:][:, ::-1]
        
        # è½¬æ�¢ä¸ºè‚¥æ–™å��ç§°
        top5_fertilizers = []
        for indices in top5_indices:
            fertilizers = [self.label_encoder.classes_[idx] for idx in indices]
            top5_fertilizers.append(fertilizers)
        
        # åˆ›å»ºæ��äº¤æ–‡ä»¶
        submission = self.test_df[['id']].copy()
        submission['Fertilizer'] = [' '.join(ferts) for ferts in top5_fertilizers]
        submission.to_csv('submission.csv', index=False)
        
        print("æ��äº¤æ–‡ä»¶å·²ç”Ÿæˆ�: submission.csv")
        print(f"é¢„æµ‹ç±»åˆ«åˆ†å¸ƒ: {pd.Series([f[0] for f in top5_fertilizers]).value_counts()}")
    
    def run_pipeline(self, train_path, test_path):
        """è¿�è¡Œå®Œæ•´çš„å¤„ç�†æµ�ç¨‹"""
        if self.load_data(train_path, test_path):
            self.explore_data()
            self.preprocess_data()
            self.train_models()
            self.generate_submission()
            print("\nâœ… å…¨éƒ¨æµ�ç¨‹å®Œæˆ�!")

# ä¸»ç¨‹åº�
if __name__ == "__main__":
    # è®¾ç½®Kaggleæ•°æ�®è·¯å¾„
    kaggle_path = '/kaggle/input'
    
    # æ˜¾ç¤ºKaggleè¾“å…¥ç›®å½•ä¸­çš„æ‰€æœ‰æ•°æ�®é›†
    print("Kaggleè¾“å…¥ç›®å½•ä¸­çš„æ‰€æœ‰æ•°æ�®é›†ï¼š")
    print("=" * 50)
    
    try:
        datasets = os.listdir(kaggle_path)
        for dataset in datasets:
            dataset_path = os.path.join(kaggle_path, dataset)
            print(f"\næ•°æ�®é›†: {dataset}")
            print(f"è·¯å¾„: {dataset_path}")
            
            # åˆ—å‡ºæ•°æ�®é›†ä¸­çš„æ–‡ä»¶
            try:
                files = os.listdir(dataset_path)
                print(f"åŒ…å�« {len(files)} ä¸ªæ–‡ä»¶:")
                for file in files:
                    file_path = os.path.join(dataset_path, file)
                    file_size = os.path.getsize(file_path) / 1024  # KB
                    print(f"  - {file} ({file_size:.2f} KB)")
            except Exception as e:
                print(f"  æ— æ³•è¯»å�–æ•°æ�®é›†å†…å®¹: {e}")
        
        print("\n" + "=" * 50)
        print(f"æ€»å…±æ‰¾åˆ° {len(datasets)} ä¸ªæ•°æ�®é›†")
    except:
        print("æ— æ³•è®¿é—®Kaggleè¾“å…¥ç›®å½•ï¼Œä½¿ç”¨é»˜è®¤è·¯å¾„...")
    
    # è®¾ç½®è®­ç»ƒå’Œæµ‹è¯•æ•°æ�®è·¯å¾„
    train_path = '/kaggle/input/playground-series-s5e6/train.csv'
    test_path = '/kaggle/input/playground-series-s5e6/test.csv'
    
    print(f"\nè®­ç»ƒæ•°æ�®è·¯å¾„: {train_path}")
    print(f"æµ‹è¯•æ•°æ�®è·¯å¾„: {test_path}")
    
    # è¿�è¡Œå®Œæ•´æµ�ç¨‹
    predictor = FertilizerPredictor()
    predictor.run_pipeline(train_path, test_path)

