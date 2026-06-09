import os
# è®¾ç½®ç�¯å¢ƒå�˜é‡�ä»¥å�¯ç”¨GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # ä½¿ç”¨ç¬¬ä¸€ä¸ªGPU

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.metrics import classification_report, confusion_matrix
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
import optuna

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    import torch
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False
warnings.filterwarnings('ignore')


class FertilizerMultiClassPredictor:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.feature_encoders = {}
        self.models = {}
        self.feature_importance = {}
        self.scaler = StandardScaler()

    def load_and_explore_data(self, train_path='/kaggle/input/playground-series-s5e6/train.csv',
                              test_path='/kaggle/input/playground-series-s5e6/test.csv'):
        """åŠ è½½æ•°æ�®å¹¶è¿›è¡Œåˆ�æ­¥æ�¢ç´¢"""
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)

        print("ğŸ“Š æ•°æ�®åŸºæœ¬ä¿¡æ�¯")
        print(f"è®­ç»ƒé›†å½¢çŠ¶: {self.train_df.shape}")
        print(f"æµ‹è¯•é›†å½¢çŠ¶: {self.test_df.shape}")
        print(f"è‚¥æ–™ç±»åˆ«æ•°é‡�: {self.train_df['Fertilizer Name'].nunique()}")

        # æ˜¾ç¤ºè‚¥æ–™ç±»å�‹åˆ†å¸ƒ
        plt.figure(figsize=(12, 6))
        fertilizer_counts = self.train_df['Fertilizer Name'].value_counts()

        plt.subplot(1, 2, 1)
        fertilizer_counts.plot(kind='bar')
        plt.title('ğŸŒ± è‚¥æ–™ç±»å�‹åˆ†å¸ƒ')
        plt.xticks(rotation=45)
        plt.ylabel('æ•°é‡�')

        plt.subplot(1, 2, 2)
        fertilizer_counts.plot(kind='pie', autopct='%1.1f%%')
        plt.title('ğŸ¥§ è‚¥æ–™ç±»å�‹å� æ¯”')
        plt.ylabel('')

        plt.tight_layout()
        plt.show()

        print(f"\nğŸ”� è‚¥æ–™ç±»å�‹è¯¦æƒ…:")
        for i, (fertilizer, count) in enumerate(fertilizer_counts.items(), 1):
            print(f"{i:2d}. {fertilizer:<20} : {count:4d} ({count / len(self.train_df) * 100:.1f}%)")

        return self.train_df, self.test_df

    def advanced_feature_engineering(self):
        """é«˜çº§ç‰¹å¾�å·¥ç¨‹"""
        print("\nğŸ”§ å¼€å§‹ç‰¹å¾�å·¥ç¨‹...")

        # å�ˆå¹¶æ•°æ�®è¿›è¡Œä¸€è‡´æ€§å¤„ç�†
        all_data = pd.concat([self.train_df, self.test_df], ignore_index=True)

        # 1. NPKç‰¹å¾�å·¥ç¨‹
        if all(col in all_data.columns for col in ['Nitrogen', 'Phosphorus', 'Potassium']):
            print("âœ… åˆ›å»ºNPKç‰¹å¾�...")
            all_data['NPK_total'] = all_data['Nitrogen'] + all_data['Phosphorus'] + all_data['Potassium']
            all_data['N_ratio'] = all_data['Nitrogen'] / (all_data['NPK_total'] + 1e-8)
            all_data['P_ratio'] = all_data['Phosphorus'] / (all_data['NPK_total'] + 1e-8)
            all_data['K_ratio'] = all_data['Potassium'] / (all_data['NPK_total'] + 1e-8)

            # NPKå¹³è¡¡æŒ‡æ ‡
            all_data['NPK_balance'] = all_data[['N_ratio', 'P_ratio', 'K_ratio']].std(axis=1)

            # NPKä¸»å¯¼å…ƒç´ 
            npk_cols = ['Nitrogen', 'Phosphorus', 'Potassium']
            all_data['dominant_nutrient'] = all_data[npk_cols].idxmax(axis=1)

            # æ–°å¢�: NPKäºŒé˜¶äº¤äº’ç‰¹å¾�
            all_data['NP_interaction'] = all_data['Nitrogen'] * all_data['Phosphorus']
            all_data['NK_interaction'] = all_data['Nitrogen'] * all_data['Potassium']
            all_data['PK_interaction'] = all_data['Phosphorus'] * all_data['Potassium']

            # æ–°å¢�: NPKå·®å€¼ç‰¹å¾�
            all_data['NP_diff'] = all_data['Nitrogen'] - all_data['Phosphorus']
            all_data['NK_diff'] = all_data['Nitrogen'] - all_data['Potassium']
            all_data['PK_diff'] = all_data['Phosphorus'] - all_data['Potassium']

            # æ–°å¢�: NPKæ¯”ç�‡ç‰¹å¾�
            all_data['NP_ratio'] = all_data['Nitrogen'] / (all_data['Phosphorus'] + 1e-8)
            all_data['NK_ratio'] = all_data['Nitrogen'] / (all_data['Potassium'] + 1e-8)
            all_data['PK_ratio'] = all_data['Phosphorus'] / (all_data['Potassium'] + 1e-8)

        # 2. ç�¯å¢ƒç‰¹å¾�äº¤äº’
        if all(col in all_data.columns for col in ['Temperature', 'Humidity']):
            print("âœ… åˆ›å»ºç�¯å¢ƒäº¤äº’ç‰¹å¾�...")
            all_data['Temp_Humidity_interaction'] = all_data['Temperature'] * all_data['Humidity']
            all_data['Temp_Humidity_ratio'] = all_data['Temperature'] / (all_data['Humidity'] + 1e-8)

            # æ–°å¢�: æ¸©æ¹¿åº¦å¹³æ–¹ç‰¹å¾�
            all_data['Temperature_squared'] = all_data['Temperature'] ** 2
            all_data['Humidity_squared'] = all_data['Humidity'] ** 2

            # æ–°å¢�: æ¸©æ¹¿åº¦ä¸�NPKäº¤äº’
            if all(col in all_data.columns for col in ['Nitrogen', 'Phosphorus', 'Potassium']):
                all_data['Temp_N_interaction'] = all_data['Temperature'] * all_data['Nitrogen']
                all_data['Humidity_N_interaction'] = all_data['Humidity'] * all_data['Nitrogen']
                all_data['Temp_P_interaction'] = all_data['Temperature'] * all_data['Phosphorus']
                all_data['Humidity_P_interaction'] = all_data['Humidity'] * all_data['Phosphorus']
                all_data['Temp_K_interaction'] = all_data['Temperature'] * all_data['Potassium']
                all_data['Humidity_K_interaction'] = all_data['Humidity'] * all_data['Potassium']

        # 3. pHç‰¹å¾�å·¥ç¨‹
        if 'pH' in all_data.columns:
            print("âœ… åˆ›å»ºpHç‰¹å¾�...")
            all_data['pH_category'] = pd.cut(all_data['pH'],
                                             bins=[0, 5.5, 6.5, 7.5, 8.5, 14],
                                             labels=['Very_Acidic', 'Acidic', 'Neutral', 'Alkaline', 'Very_Alkaline'])
            all_data['pH_distance_from_neutral'] = abs(all_data['pH'] - 7.0)

            # æ–°å¢�: pHå¹³æ–¹ç‰¹å¾�
            all_data['pH_squared'] = all_data['pH'] ** 2

            # æ–°å¢�: pHä¸�NPKäº¤äº’
            if all(col in all_data.columns for col in ['Nitrogen', 'Phosphorus', 'Potassium']):
                all_data['pH_N_interaction'] = all_data['pH'] * all_data['Nitrogen']
                all_data['pH_P_interaction'] = all_data['pH'] * all_data['Phosphorus']
                all_data['pH_K_interaction'] = all_data['pH'] * all_data['Potassium']

        # 4. å¾®é‡�å…ƒç´ ç‰¹å¾�ï¼ˆå¦‚æ�œå­˜åœ¨ï¼‰
        micro_nutrients = ['Calcium', 'Magnesium', 'Sulfur', 'Iron', 'Manganese', 'Zinc']
        existing_micro = [col for col in micro_nutrients if col in all_data.columns]

        if existing_micro:
            print(f"âœ… åˆ›å»ºå¾®é‡�å…ƒç´ ç‰¹å¾�: {existing_micro}")
            all_data['micro_nutrients_sum'] = all_data[existing_micro].sum(axis=1)
            all_data['micro_nutrients_mean'] = all_data[existing_micro].mean(axis=1)

            # æ–°å¢�: å¾®é‡�å…ƒç´ æ¯”ç�‡ç‰¹å¾�
            for i, nutrient1 in enumerate(existing_micro):
                for nutrient2 in existing_micro[i + 1:]:
                    ratio_name = f"{nutrient1}_{nutrient2}_ratio"
                    all_data[ratio_name] = all_data[nutrient1] / (all_data[nutrient2] + 1e-8)

        # 5. ç»Ÿè®¡ç‰¹å¾�
        numeric_cols = all_data.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['id']]

        if len(numeric_cols) > 3:
            print("âœ… åˆ›å»ºç»Ÿè®¡ç‰¹å¾�...")
            all_data['feature_sum'] = all_data[numeric_cols].sum(axis=1)
            all_data['feature_mean'] = all_data[numeric_cols].mean(axis=1)
            all_data['feature_std'] = all_data[numeric_cols].std(axis=1)
            all_data['feature_max'] = all_data[numeric_cols].max(axis=1)
            all_data['feature_min'] = all_data[numeric_cols].min(axis=1)
            all_data['feature_range'] = all_data['feature_max'] - all_data['feature_min']

        # 6. æ–°å¢�: è�šç±»ç‰¹å¾�
        if len(numeric_cols) > 3:
            print("âœ… åˆ›å»ºè�šç±»ç‰¹å¾�...")
            kmeans_features = all_data[numeric_cols].fillna(all_data[numeric_cols].mean())
            scaled_features = StandardScaler().fit_transform(kmeans_features)

            for n_clusters in [3, 5, 8]:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                all_data[f'kmeans_cluster_{n_clusters}'] = kmeans.fit_predict(scaled_features)

        # 7. æ–°å¢�: PCAç‰¹å¾�
        if len(numeric_cols) > 5:
            print("âœ… åˆ›å»ºPCAç‰¹å¾�...")
            pca_features = all_data[numeric_cols].fillna(all_data[numeric_cols].mean())
            scaled_features = StandardScaler().fit_transform(pca_features)

            pca = PCA(n_components=min(5, len(numeric_cols)))
            pca_result = pca.fit_transform(scaled_features)

            for i in range(pca_result.shape[1]):
                all_data[f'pca_component_{i + 1}'] = pca_result[:, i]

        # 8. æ–°å¢�: å¤šé¡¹å¼�ç‰¹å¾�
        if len(numeric_cols) > 3:
            print("âœ… åˆ›å»ºå¤šé¡¹å¼�ç‰¹å¾�...")
            poly_features = all_data[numeric_cols].fillna(all_data[numeric_cols].mean())

            # æ£€æŸ¥NPKåˆ—æ˜¯å�¦éƒ½å­˜åœ¨
            npk_cols = []
            for col in ['Nitrogen', 'Phosphorus', 'Potassium']:
                if col in poly_features.columns:
                    npk_cols.append(col)

            # å�ªæœ‰å½“è‡³å°‘æœ‰ä¸¤ä¸ªNPKåˆ—å­˜åœ¨æ—¶æ‰�åˆ›å»ºå¤šé¡¹å¼�ç‰¹å¾�
            if len(npk_cols) >= 2:
                poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
                poly_result = poly.fit_transform(poly_features[npk_cols])

                poly_cols = poly.get_feature_names_out(npk_cols)
                for i in range(1, len(poly_cols)):  # è·³è¿‡å�Ÿå§‹ç‰¹å¾�
                    all_data[f'poly_{poly_cols[i]}'] = poly_result[:, i]

        # 9. æ–°å¢�: æ›´å¤šNPKæ¯”ç�‡ç‰¹å¾�
        if all(col in all_data.columns for col in ['Nitrogen', 'Phosphorus', 'Potassium']):
            print("âœ… åˆ›å»ºé«˜çº§NPKæ¯”ç�‡ç‰¹å¾�...")
            # ä¸‰å…ƒæ¯”ç�‡
            all_data['NPK_ratio'] = all_data['Nitrogen'] / (all_data['Phosphorus'] + all_data['Potassium'] + 1e-8)
            all_data['PNK_ratio'] = all_data['Phosphorus'] / (all_data['Nitrogen'] + all_data['Potassium'] + 1e-8)
            all_data['KNP_ratio'] = all_data['Potassium'] / (all_data['Nitrogen'] + all_data['Phosphorus'] + 1e-8)

            # å¹³æ–¹æ ¹æ¯”ç�‡
            all_data['N_sqrt_PK'] = all_data['Nitrogen'] / (
                np.sqrt(all_data['Phosphorus'] * all_data['Potassium'] + 1e-8))
            all_data['P_sqrt_NK'] = all_data['Phosphorus'] / (
                np.sqrt(all_data['Nitrogen'] * all_data['Potassium'] + 1e-8))
            all_data['K_sqrt_NP'] = all_data['Potassium'] / (
                np.sqrt(all_data['Nitrogen'] * all_data['Phosphorus'] + 1e-8))

        # 10. æ–°å¢�: åœŸå£¤ç‰¹å¾�å·¥ç¨‹
        if all(col in all_data.columns for col in ['Soil Type', 'Soil pH']):
            print("âœ… åˆ›å»ºåœŸå£¤ç›¸å…³ç‰¹å¾�...")
            # åœŸå£¤ç±»å�‹ä¸�pHäº¤äº’
            all_data['Soil_pH_interaction'] = all_data['Soil Type'].astype(str) + "_" + all_data['Soil pH'].round(
                1).astype(str)

        # 11. æ–°å¢�: ä½œç‰©ç›¸å…³ç‰¹å¾�
        if 'Crop Type' in all_data.columns:
            print("âœ… åˆ›å»ºä½œç‰©ç›¸å…³ç‰¹å¾�...")
            # ä½œç‰©ä¸�NPKäº¤äº’
            if all(col in all_data.columns for col in ['Nitrogen', 'Phosphorus', 'Potassium']):
                all_data['Crop_N_interaction'] = all_data['Crop Type'].astype(str) + "_" + all_data['Nitrogen'].round(
                    0).astype(str)
                all_data['Crop_P_interaction'] = all_data['Crop Type'].astype(str) + "_" + all_data['Phosphorus'].round(
                    0).astype(str)
                all_data['Crop_K_interaction'] = all_data['Crop Type'].astype(str) + "_" + all_data['Potassium'].round(
                    0).astype(str)

        # 12. æ–°å¢�: å­£èŠ‚æ€§ç‰¹å¾�
        if 'Season' in all_data.columns:
            print("âœ… åˆ›å»ºå­£èŠ‚æ€§ç‰¹å¾�...")
            # å­£èŠ‚ä¸�æ¸©æ¹¿åº¦äº¤äº’
            if all(col in all_data.columns for col in ['Temperature', 'Humidity']):
                all_data['Season_Temp_interaction'] = all_data['Season'].astype(str) + "_" + all_data[
                    'Temperature'].round(0).astype(str)
                all_data['Season_Humidity_interaction'] = all_data['Season'].astype(str) + "_" + all_data[
                    'Humidity'].round(0).astype(str)

        # åˆ†ç¦»å›�å�Ÿå§‹æ•°æ�®
        self.train_df = all_data[:len(self.train_df)].copy()
        self.test_df = all_data[len(self.train_df):].copy()

        print(f"ğŸ�¯ ç‰¹å¾�å·¥ç¨‹å®Œæˆ�! æ–°ç‰¹å¾�æ•°: {self.train_df.shape[1]}")
        return self.train_df.columns.tolist()

    def prepare_features(self):
        """å‡†å¤‡å»ºæ¨¡ç‰¹å¾�"""
        print("\nğŸ“‹ å‡†å¤‡å»ºæ¨¡ç‰¹å¾�...")

        # åˆ†ç¦»ç‰¹å¾�å’Œç›®æ ‡
        X = self.train_df.drop(['Fertilizer Name'], axis=1)
        y = self.train_df['Fertilizer Name']
        X_test = self.test_df.copy()

        # å¤„ç�†IDåˆ—
        if 'id' in X.columns:
            X = X.drop(['id'], axis=1)
        if 'id' in X_test.columns:
            self.test_ids = X_test['id'].copy()
            X_test = X_test.drop(['id'], axis=1)

        # å¤„ç�†åˆ†ç±»ç‰¹å¾�
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns

        for col in categorical_columns:
            print(f"  ğŸ”¤ ç¼–ç �åˆ†ç±»ç‰¹å¾�: {col}")
            if col not in self.feature_encoders:
                self.feature_encoders[col] = LabelEncoder()
                # å�ˆå¹¶è®­ç»ƒå’Œæµ‹è¯•é›†çš„æ‰€æœ‰å€¼
                all_values = pd.concat([X[col], X_test[col]]).astype(str).fillna('missing')
                self.feature_encoders[col].fit(all_values)

            X[col] = self.feature_encoders[col].transform(X[col].astype(str).fillna('missing'))
            X_test[col] = self.feature_encoders[col].transform(X_test[col].astype(str).fillna('missing'))

        # ç¼–ç �ç›®æ ‡å�˜é‡�
        y_encoded = self.label_encoder.fit_transform(y)

        # å¼‚å¸¸å€¼å¤„ç�†
        print("  ğŸ”� å¤„ç�†å¼‚å¸¸å€¼...")
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            # è®¡ç®—IQR
            Q1 = X[col].quantile(0.01)
            Q3 = X[col].quantile(0.99)
            IQR = Q3 - Q1

            # å®šä¹‰å¼‚å¸¸å€¼è¾¹ç•Œ
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # æ›¿æ�¢å¼‚å¸¸å€¼
            X.loc[X[col] < lower_bound, col] = lower_bound
            X.loc[X[col] > upper_bound, col] = upper_bound

            # å¯¹æµ‹è¯•é›†åº”ç”¨ç›¸å�Œçš„å¤„ç�†
            X_test.loc[X_test[col] < lower_bound, col] = lower_bound
            X_test.loc[X_test[col] > upper_bound, col] = upper_bound

        # æ ‡å‡†åŒ–æ•°å€¼ç‰¹å¾�
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        X_scaled = X.copy()
        X_test_scaled = X_test.copy()

        X_scaled[numeric_columns] = self.scaler.fit_transform(X[numeric_columns])
        X_test_scaled[numeric_columns] = self.scaler.transform(X_test[numeric_columns])

        # ç‰¹å¾�é€‰æ‹© - æ·»åŠ åœ¨æ­¤å¤„
        if hasattr(self, 'feature_importance') and 'lgb' in self.feature_importance:
            importance_threshold = np.percentile(self.feature_importance['lgb'], 20)  # å�»é™¤æœ€ä¸�é‡�è¦�çš„20%ç‰¹å¾�
            important_features = X.columns[self.feature_importance['lgb'] > importance_threshold]
            X_scaled = X_scaled[important_features]
            X_test_scaled = X_test_scaled[important_features]
            print(f"   - ç‰¹å¾�é€‰æ‹©å��ç‰¹å¾�æ•°é‡�: {X_scaled.shape[1]}")

        print(f"âœ… ç‰¹å¾�å‡†å¤‡å®Œæˆ�!")
        print(f"   - ç‰¹å¾�æ•°é‡�: {X.shape[1]}")
        print(f"   - æ ·æœ¬æ•°é‡�: {X.shape[0]}")
        print(f"   - ç±»åˆ«æ•°é‡�: {len(self.label_encoder.classes_)}")

        return X_scaled, y_encoded, X_test_scaled

    def optimize_and_train_models(self, X, y):
        """ä¼˜åŒ–å�‚æ•°å¹¶è®­ç»ƒæ¨¡å�‹"""
        print("\nğŸ”„ å¼€å§‹è¶…å�‚æ•°ä¼˜åŒ–...")

        # ä¼˜åŒ–LightGBMå�‚æ•° (è®¾ç½®è¾ƒå°�çš„n_trialsä»¥èŠ‚çœ�æ—¶é—´ï¼Œç”Ÿäº§ç�¯å¢ƒå�¯å¢�åŠ )
        best_lgb_params = optimize_lightgbm_params(X, y, n_trials=10)

        # ä¼˜åŒ–XGBoostå�‚æ•°
        best_xgb_params = optimize_xgboost_params(X, y, n_trials=8)

        # ä¼˜åŒ–CatBoostå�‚æ•°
        best_cb_params = optimize_catboost_params(X, y, n_trials=5)

        # ä½¿ç”¨ä¼˜åŒ–å��çš„å�‚æ•°è®­ç»ƒæ¨¡å�‹
        print("\nğŸŒŸ ä½¿ç”¨ä¼˜åŒ–å�‚æ•°è®­ç»ƒæ¨¡å�‹...")
        lgb_oof = self.train_lightgbm(X, y, params=best_lgb_params)
        lgb_map5 = self.calculate_mapk(y, lgb_oof, k=5)
        print(f"ğŸ“ˆ ä¼˜åŒ–å��çš„ LightGBM MAP@5: {lgb_map5:.4f}")

        xgb_oof = self.train_xgboost(X, y, params=best_xgb_params)
        xgb_map5 = self.calculate_mapk(y, xgb_oof, k=5)
        print(f"ğŸ“ˆ ä¼˜åŒ–å��çš„ XGBoost MAP@5: {xgb_map5:.4f}")

        cb_oof = self.train_catboost(X, y, params=best_cb_params)
        cb_map5 = self.calculate_mapk(y, cb_oof, k=5)
        print(f"ğŸ“ˆ ä¼˜åŒ–å��çš„ CatBoost MAP@5: {cb_map5:.4f}")

        return best_lgb_params, best_xgb_params, best_cb_params, lgb_oof, xgb_oof, cb_oof

    def train_stacking_model(self, X, y, base_models_oof):
        """è®­ç»ƒå¢�å¼ºçš„å †å� æ¨¡å�‹"""
        print("\nğŸ”„ è®­ç»ƒé«˜çº§å †å� æ¨¡å�‹...")

        # å‡†å¤‡å †å� ç‰¹å¾�
        stacking_features = np.hstack(base_models_oof)

        # æ·»åŠ å�Ÿå§‹ç‰¹å¾�ä¸�å †å� ç‰¹å¾�çš„ç»„å�ˆ
        X_numeric = X.select_dtypes(include=[np.number])
        combined_features = np.hstack([stacking_features, X_numeric])

        # ä½¿ç”¨ä¸¤å±‚å †å� æ�¶æ�„
        # ç¬¬ä¸€å±‚: LightGBMä½œä¸ºå…ƒå­¦ä¹ å™¨
        meta_lgb_params = {
            'objective': 'multiclass',
            'num_class': len(self.label_encoder.classes_),
            'metric': 'multi_logloss',
            'learning_rate': 0.03,
            'num_leaves': 31,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0
        }

        # ç¬¬äºŒå±‚: XGBoostä½œä¸ºæœ€ç»ˆé›†æˆ�å™¨
        meta_xgb_params = {
            'objective': 'multi:softprob',
            'num_class': len(self.label_encoder.classes_),
            'eval_metric': 'mlogloss',
            'learning_rate': 0.03,
            'max_depth': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'tree_method': 'gpu_hist',
            'gpu_id': 0

        }

        # 8æŠ˜äº¤å�‰éªŒè¯�
        kf = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
        oof_predictions = np.zeros((len(X), len(self.label_encoder.classes_)))

        # ç¬¬ä¸€å±‚å…ƒå­¦ä¹ å™¨çš„é¢„æµ‹
        meta_lgb_preds = np.zeros((len(X), len(self.label_encoder.classes_)))

        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            print(f"  ğŸ“� è®­ç»ƒç¬¬ {fold + 1}/8 æŠ˜...")

            # å‡†å¤‡è®­ç»ƒå’ŒéªŒè¯�æ•°æ�®
            X_train_fold, X_val_fold = stacking_features[train_idx], stacking_features[val_idx]
            X_train_combined, X_val_combined = combined_features[train_idx], combined_features[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            # ç¬¬ä¸€å±‚: LightGBM
            train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
            val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

            lgb_model = lgb.train(
                meta_lgb_params,
                train_data,
                valid_sets=[train_data, val_data],
                num_boost_round=1000,
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
            )

            # é¢„æµ‹éªŒè¯�é›†
            lgb_val_pred = lgb_model.predict(X_val_fold, num_iteration=lgb_model.best_iteration)
            meta_lgb_preds[val_idx] = lgb_val_pred

            # ç¬¬äºŒå±‚: XGBoost (ä½¿ç”¨å�Ÿå§‹ç‰¹å¾�+ç¬¬ä¸€å±‚é¢„æµ‹)
            X_train_meta = np.hstack([X_train_combined, meta_lgb_preds[train_idx]])
            X_val_meta = np.hstack([X_val_combined, lgb_val_pred])

            xgb_model = xgb.XGBClassifier(**meta_xgb_params)
            xgb_model.fit(
                X_train_meta, y_train_fold,
                eval_set=[(X_val_meta, y_val_fold)],
                verbose=False
            )

            # æœ€ç»ˆé¢„æµ‹
            val_pred = xgb_model.predict_proba(X_val_meta)
            oof_predictions[val_idx] = val_pred

            # ä¿�å­˜ç¬¬ä¸€ä¸ªæŠ˜çš„æ¨¡å�‹
            if fold == 0:
                self.models['stacking_lgb'] = lgb_model
                self.models['stacking'] = xgb_model

        stacking_map5 = self.calculate_mapk(y, oof_predictions, k=5)
        print(f"ğŸ“ˆ é«˜çº§å †å� æ¨¡å�‹ MAP@5: {stacking_map5:.4f}")

        return oof_predictions, stacking_map5

    def predict_with_stacking(self, X_test, base_models_preds):
        """ä½¿ç”¨å¢�å¼ºçš„å †å� æ¨¡å�‹è¿›è¡Œé¢„æµ‹"""
        if 'stacking' not in self.models or 'stacking_lgb' not in self.models:
            print("âš ï¸� å †å� æ¨¡å�‹æœªè®­ç»ƒï¼Œæ— æ³•ä½¿ç”¨")
            return None

        # å‡†å¤‡å †å� ç‰¹å¾�
        stacking_features = np.hstack(base_models_preds)

        # æ£€æŸ¥ç‰¹å¾�æ•°é‡�æ˜¯å�¦åŒ¹é…�
        expected_features = self.models['stacking_lgb'].num_feature()
        # ç‰¹å¾�æ•°é‡�ä¸�åŒ¹é…�æ—¶çš„æ›´ç²¾ç»†å¤„ç�†
        if stacking_features.shape[1] != expected_features:
            print(f"âš ï¸� ç‰¹å¾�æ•°é‡�ä¸�åŒ¹é…�: é¢„æœŸ{expected_features}ä¸ªï¼Œå®�é™…{stacking_features.shape[1]}ä¸ª")

            # è®°å½•å�Ÿå§‹ç‰¹å¾�å��ç§°
            feature_names = list(self.models['stacking_lgb'].feature_name())

            # å¦‚æ�œç‰¹å¾�ä¸�è¶³ï¼Œç”¨é›¶å¡«å……å¹¶è®°å½•æ—¥å¿—
            if stacking_features.shape[1] < expected_features:
                padding = np.zeros((stacking_features.shape[0], expected_features - stacking_features.shape[1]))
                stacking_features = np.hstack([stacking_features, padding])
                print(f"âš ï¸� æ·»åŠ äº†{expected_features - stacking_features.shape[1]}ä¸ªé›¶å¡«å……ç‰¹å¾�")
            else:
                # å¦‚æ�œç‰¹å¾�è¿‡å¤šï¼Œè£�å‰ªå¹¶è®°å½•æ—¥å¿—
                stacking_features = stacking_features[:, :expected_features]
                print(f"âš ï¸� è£�å‰ªäº†{stacking_features.shape[1] - expected_features}ä¸ªå¤šä½™ç‰¹å¾�")

        # æ·»åŠ å�Ÿå§‹ç‰¹å¾�ä¸�å †å� ç‰¹å¾�çš„ç»„å�ˆ
        X_test_numeric = X_test.select_dtypes(include=[np.number])
        combined_features = np.hstack([stacking_features, X_test_numeric])

        # ç¬¬ä¸€å±‚é¢„æµ‹
        lgb_pred = self.models['stacking_lgb'].predict(stacking_features,
                                                       num_iteration=self.models['stacking_lgb'].best_iteration,
                                                       predict_disable_shape_check=True)

        # ç¬¬äºŒå±‚é¢„æµ‹
        X_test_meta = np.hstack([combined_features, lgb_pred])
        final_pred = self.models['stacking'].predict_proba(X_test_meta)

        return final_pred

    def predict_topk(self, X_test, k=5, weights=None):
        """ç”Ÿæˆ� Top-K é¢„æµ‹ï¼Œæ”¯æŒ�åŠ æ�ƒå¹³å�‡"""
        print(f"\nğŸ�¯ ç”Ÿæˆ� Top-{k} é¢„æµ‹...")

        # ç¡®ä¿�kæ˜¯æ•´æ•°å¹¶ä¿�å­˜åˆ°ä¸“ç”¨å�˜é‡�
        try:
            top_k = int(k)
        except (TypeError, ValueError):
            print(f"âš ï¸� kå�‚æ•° '{k}' æ— æ•ˆï¼Œä½¿ç”¨é»˜è®¤å€¼5")
            top_k = 5

        try:
            # åˆ›å»ºX_testçš„æ·±æ‹·è´�
            X_test = X_test.copy()

            # è°ƒè¯•ä¿¡æ�¯
            print(f"è°ƒè¯•: X_teståˆ—å��: {X_test.columns.tolist()}")

            # ç§»é™¤ç›®æ ‡åˆ—(å¦‚æ�œå­˜åœ¨)
            if 'Fertilizer Name' in X_test.columns:
                X_test = X_test.drop(['Fertilizer Name'], axis=1)
                print(f"ğŸ”„ ä»�æµ‹è¯•æ•°æ�®ä¸­ç§»é™¤ç›®æ ‡åˆ— 'Fertilizer Name'")

            print(f"è°ƒè¯•: ç§»é™¤å��X_teståˆ—å��: {X_test.columns.tolist()}")

            # ä¿�å­˜IDåˆ—
            test_ids = None
            if 'id' in X_test.columns:
                test_ids = X_test['id'].values
                X_test = X_test.drop(['id'], axis=1)

            # å�ªä¿�ç•™æ•°å€¼åˆ—
            X_test_numeric = X_test.select_dtypes(include=[np.number])
            if X_test_numeric.shape[1] < X_test.shape[1]:
                print(f"é—®é¢˜æ•°æ�®ç±»å�‹: {X_test.dtypes}")
                X_test = X_test_numeric
                print(f"å�ªä¿�ç•™æ•°å€¼åˆ—å��çš„å½¢çŠ¶: {X_test.shape}")

            # å¡«å……ç¼ºå¤±å€¼
            if X_test.isnull().values.any():
                X_test = X_test.fillna(0)

            # æ ‡å‡†åŒ–å¤„ç�†
            numeric_columns = X_test.columns
            X_test[numeric_columns] = self.scaler.transform(X_test[numeric_columns])
            print("âœ… åº”ç”¨æ ‡å‡†åŒ–å¤„ç�†å®Œæˆ�")

            # æ¨¡å�‹é¢„æµ‹
            predictions = {}
            model_preds = []

            # LightGBMé¢„æµ‹
            if 'lgb' in self.models:
                lgb_pred = self.models['lgb'].predict(X_test, num_iteration=self.models['lgb'].best_iteration)
                predictions['lgb'] = lgb_pred
                print("  âœ… LightGBM é¢„æµ‹å®Œæˆ�")

            # XGBoosté¢„æµ‹
            if 'xgb' in self.models:
                xgb_pred = self.models['xgb'].predict_proba(X_test)
                predictions['xgb'] = xgb_pred
                print("  âœ… XGBoost é¢„æµ‹å®Œæˆ�")

            # CatBoosté¢„æµ‹
            if 'catboost' in self.models:
                cb_pred = self.models['catboost'].predict_proba(X_test)
                predictions['catboost'] = cb_pred
                print("  âœ… CatBoost é¢„æµ‹å®Œæˆ�")

            # TabNeté¢„æµ‹
            if 'tabnet' in self.models and TABNET_AVAILABLE:
                tabnet_pred = self.models['tabnet'].predict_proba(X_test.values)
                predictions['tabnet'] = tabnet_pred
                print("  âœ… TabNet é¢„æµ‹å®Œæˆ�")

            # å †å� æ¨¡å�‹é¢„æµ‹
            if 'stacking' in self.models and 'stacking_lgb' in self.models:
                # æ”¶é›†åŸºç¡€æ¨¡å�‹é¢„æµ‹
                base_preds = []
                if 'lgb' in predictions: base_preds.append(predictions['lgb'])
                if 'xgb' in predictions: base_preds.append(predictions['xgb'])
                if 'catboost' in predictions: base_preds.append(predictions['catboost'])
                if 'tabnet' in predictions: base_preds.append(predictions['tabnet'])

                if base_preds:
                    stacking_pred = self.predict_with_stacking(X_test, base_preds)
                    if stacking_pred is not None:
                        predictions['stacking'] = stacking_pred
                        print("  âœ… å †å� æ¨¡å�‹é¢„æµ‹å®Œæˆ�")

            # å¦‚æ�œæ²¡æœ‰ä»»ä½•æ¨¡å�‹é¢„æµ‹æˆ�åŠŸï¼Œä½¿ç”¨LightGBMå�•ç‹¬é¢„æµ‹
            if not predictions:
                raise Exception("æ²¡æœ‰å�¯ç”¨çš„æ¨¡å�‹é¢„æµ‹")

            # ä½¿ç”¨åŠ æ�ƒå¹³å�‡
            if weights is None:
                weights = {model: 1.0 for model in predictions.keys()}

            # å�ªä½¿ç”¨å­˜åœ¨çš„æ¨¡å�‹
            available_weights = {k: v for k, v in weights.items() if k in predictions}

            if not available_weights:
                available_weights = {k: 1.0 for k in predictions.keys()}

            # å½’ä¸€åŒ–æ�ƒé‡�
            total_weight = sum(available_weights.values())
            for k in available_weights:
                available_weights[k] /= total_weight

            # åˆ›å»ºåŠ æ�ƒå¹³å�‡é¢„æµ‹
            first_model = list(predictions.keys())[0]  # æ”¹å��é�¿å…�å��é�¢è¦†ç›–
            ensemble_pred = np.zeros_like(predictions[first_model], dtype=np.float64)

            for model_key, weight in available_weights.items():  # ä½¿ç”¨ä¸�å�Œçš„å�˜é‡�å��
                ensemble_pred += predictions[model_key] * weight

            # è�·å�–Top-kç´¢å¼•
            top_k_indices = np.argsort(ensemble_pred, axis=1)[:, -top_k:][:, ::-1]

            # è½¬æ�¢ä¸ºè‚¥æ–™å��ç§°
            top_k_fertilizers = []
            top_k_probs = []

            for i, indices in enumerate(top_k_indices):
                fertilizers = [self.label_encoder.classes_[idx] for idx in indices]
                probs = [ensemble_pred[i, idx] for idx in indices]
                top_k_fertilizers.append(fertilizers)
                top_k_probs.append(probs)

            return top_k_fertilizers, ensemble_pred

        except Exception as e:
            print(f"â�Œ é¢„æµ‹è¿‡ç¨‹å‡ºé”™: {str(e)}")
            try:
                # å°�è¯•ä½¿ç”¨å�•ä¸ªæ¨¡å�‹é¢„æµ‹
                print(f"âš ï¸� é¢„æµ‹å¤±è´¥: {str(e)}ï¼Œå°�è¯•ä½¿ç”¨å�•ä¸ªæ¨¡å�‹é¢„æµ‹")
                if 'lgb' in self.models:
                    lgb_pred = self.models['lgb'].predict(X_test, num_iteration=self.models['lgb'].best_iteration)
                    top_k_indices = np.argsort(lgb_pred, axis=1)[:, -top_k:][:, ::-1]
                    #è½¬æ�¢ä¸ºè‚¥æ–™å��ç§°
                    top_k_fertilizers = []
                    for indices in top_k_indices:
                        fertilizers = [self.label_encoder.classes_[idx] for idx in indices]
                        top_k_fertilizers.append(fertilizers)

                    return top_k_fertilizers, lgb_pred
                else:
                    raise Exception("æ²¡æœ‰å�¯ç”¨çš„å¤‡ç”¨æ¨¡å�‹")
            except Exception as e2:
                print(f"â�Œ å¤‡ç”¨é¢„æµ‹ä¹Ÿå¤±è´¥: {str(e2)}")

                # åˆ›å»ºé»˜è®¤é¢„æµ‹
                n_samples = len(X_test)
                n_classes = len(self.label_encoder.classes_)

                # ä½¿ç”¨å‰�kä¸ªç±»åˆ«ä½œä¸ºé»˜è®¤é¢„æµ‹
                default_classes = self.label_encoder.classes_[:top_k].tolist()
                default_fertilizers = [default_classes for _ in range(n_samples)]

                # åˆ›å»ºå�‡åŒ€æ¦‚ç�‡åˆ†å¸ƒ
                default_preds = np.ones((n_samples, n_classes)) / n_classes

                return default_fertilizers, default_preds

def optimize_lightgbm_params(X, y, n_trials=30):
    """ä¼˜åŒ– LightGBM è¶…å�‚æ•°"""

    def objective(trial):
        try:
            boosting_type = trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss'])
            params = {
                'objective': 'multiclass',
                'num_class': len(np.unique(y)),
                'metric': 'multi_logloss',
                'boosting_type': boosting_type,
                'num_leaves': trial.suggest_int('num_leaves', 10, 200),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.3, 1.0),
                'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 20.0, log=True),
                'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 20.0, log=True),
                'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 2.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 150),
                'verbose': -1,
                'random_state': 42,
                'device': 'gpu',
                'gpu_platform_id': 0,
                'gpu_device_id': 0
            }

            # ä»…å½“boosting_typeä¸�æ˜¯'goss'æ—¶æ·»åŠ baggingå�‚æ•°
            if boosting_type != 'goss':
                params['bagging_fraction'] = trial.suggest_float('bagging_fraction', 0.3, 1.0)
                params['bagging_freq'] = trial.suggest_int('bagging_freq', 1, 15)

            # å®‰å…¨åœ°æ£€æŸ¥GPUå�¯ç”¨æ€§
            try:
                import importlib.util
                if importlib.util.find_spec("torch") is not None:
                    import torch
                    if torch.cuda.is_available():
                        params['device_type'] = 'gpu'
                        params['gpu_platform_id'] = 0
                        params['gpu_device_id'] = 0
                        print("ä½¿ç”¨GPUåŠ é€ŸLightGBMè®­ç»ƒ")
                # å°�è¯•ä½¿ç”¨LightGBMå�Ÿç”Ÿçš„GPUæ£€æµ‹
                elif hasattr(lgb, 'gpu_version_satisfied') and lgb.gpu_version_satisfied():
                    params['device_type'] = 'gpu'
                    params['gpu_platform_id'] = 0
                    params['gpu_device_id'] = 0
                    print("ä½¿ç”¨LightGBMå�Ÿç”ŸGPUæ”¯æŒ�")
            except Exception as e:
                print(f"GPUæ£€æŸ¥æ—¶å‡ºé”™: {str(e)}ï¼Œå°†ä½¿ç”¨CPU")

            # ä½¿ç”¨æ›´å°‘çš„æŠ˜æ•°åŠ é€Ÿä¼˜åŒ–è¿‡ç¨‹
            n_folds = 3 if n_trials > 10 else 5
            kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            scores = []

            # è®¾ç½®æ¯�æŠ˜è®­ç»ƒçš„æœ€å¤§è¿­ä»£æ¬¡æ•°
            max_boost_rounds = 100 if n_trials > 15 else 200

            for train_idx, val_idx in kf.split(X, y):
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]

                train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
                val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

                # ä½¿ç”¨æ—©å�œæ�¥åŠ é€Ÿè®­ç»ƒ
                model = lgb.train(
                    params,
                    train_data,
                    valid_sets=[val_data],
                    num_boost_round=max_boost_rounds,
                    callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
                )

                # é¢„æµ‹éªŒè¯�é›†
                val_pred = model.predict(X_val_fold, num_iteration=model.best_iteration)

                # è®¡ç®—MAP@5
                from sklearn.preprocessing import label_binarize
                y_val_bin = label_binarize(y_val_fold, classes=np.unique(y))
                if y_val_bin.shape[1] == 1:  # äºŒåˆ†ç±»æƒ…å†µ
                    y_val_bin = np.hstack([1 - y_val_bin, y_val_bin])

                # è®¡ç®—æ¯�ä¸ªæ ·æœ¬çš„top-5é¢„æµ‹
                top_k_pred = np.argsort(val_pred, axis=1)[:, -5:][:, ::-1]

                # è®¡ç®—MAP@5
                map5_score = 0
                for i in range(len(y_val_fold)):
                    actual = y_val_fold[i]
                    predicted = top_k_pred[i]
                    score = 1 if actual in predicted else 0
                    map5_score += score
                map5_score /= len(y_val_fold)

                scores.append(map5_score)

            return np.mean(scores)
        except Exception as e:
            print(f"ä¼˜åŒ–è¿‡ç¨‹ä¸­å‡ºé”™: {str(e)}")
            return 0.0  # è¿”å›�æœ€ä½�åˆ†æ•°

    # è®¾ç½®è¶…æ—¶æœºåˆ¶
    timeout = 600  # 10åˆ†é’Ÿè¶…æ—¶

    # åˆ›å»ºä¼˜åŒ–ç ”ç©¶
    study = optuna.create_study(direction='maximize')

    try:
        # ç§»é™¤ä¸�å…¼å®¹çš„å›�è°ƒï¼Œä»…ä½¿ç”¨è¶…æ—¶
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
    except Exception as e:
        print(f"ä¼˜åŒ–è¿‡ç¨‹è¢«ä¸­æ–­: {str(e)}")

    # æ£€æŸ¥æ˜¯å�¦æœ‰ä»»ä½•æˆ�åŠŸçš„è¯•éªŒ
    if len(study.trials) == 0:
        print("âš ï¸� ä¼˜åŒ–å¤±è´¥ï¼Œä½¿ç”¨é»˜è®¤å�‚æ•°")
        return {
            'objective': 'multiclass',
            'num_class': len(np.unique(y)),
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'min_split_gain': 0.0,
            'min_child_samples': 20,
            'verbose': -1,
            'random_state': 42,
            'bagging_fraction': 0.8,
            'bagging_freq': 5
        }

    # æœ€å��è¿”å›�æ—¶ï¼Œç¡®ä¿�å¿…è¦�å�‚æ•°å­˜åœ¨
    best_params = study.best_params
    best_params['objective'] = 'multiclass'
    best_params['num_class'] = len(np.unique(y))
    best_params['verbose'] = -1
    best_params['random_state'] = 42

    # åœ¨optimize_lightgbm_paramså‡½æ•°çš„è¿”å›�éƒ¨åˆ†
    best_params['device'] = 'gpu'
    best_params['gpu_platform_id'] = 0
    best_params['gpu_device_id'] = 0

    print(f"ğŸ�¯ æœ€ä½³LightGBMå�‚æ•°: {best_params}")
    print(f"ğŸ�† æœ€ä½³å¾—åˆ†: {study.best_value:.4f}")

    return best_params

def optimize_xgboost_params(X, y, n_trials=8):
    """ä¼˜åŒ– XGBoost è¶…å�‚æ•°"""

    def objective(trial):
        try:
            params = {
                'objective': 'multi:softprob',
                'num_class': len(np.unique(y)),
                'eval_metric': 'mlogloss',
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'random_state': 42,
                'tree_method': 'gpu_hist',
                'gpu_id': 0

            }

            # ä½¿ç”¨æ›´å°‘çš„æŠ˜æ•°åŠ é€Ÿä¼˜åŒ–è¿‡ç¨‹
            n_folds = 3 if n_trials > 5 else 5
            kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            scores = []

            # è®¾ç½®æ¯�æŠ˜è®­ç»ƒçš„æœ€å¤§è¿­ä»£æ¬¡æ•°
            max_boost_rounds = 100 if n_trials > 5 else 200
            early_stopping_rounds = 20

            for train_idx, val_idx in kf.split(X, y):
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]

                # ä½¿ç”¨æ—©å�œæ�¥åŠ é€Ÿè®­ç»ƒ
                model = xgb.XGBClassifier(
                    **params,
                    n_estimators=max_boost_rounds,
                    early_stopping_rounds=early_stopping_rounds,
                    verbose=0
                )

                model.fit(
                    X_train_fold, y_train_fold,
                    eval_set=[(X_val_fold, y_val_fold)],
                    verbose=False
                )

                # é¢„æµ‹éªŒè¯�é›†
                val_pred = model.predict_proba(X_val_fold)

                # è®¡ç®—MAP@5
                top_k_pred = np.argsort(val_pred, axis=1)[:, -5:][:, ::-1]

                # è®¡ç®—MAP@5
                map5_score = 0
                for i in range(len(y_val_fold)):
                    actual = y_val_fold[i]
                    predicted = top_k_pred[i]
                    score = 1 if actual in predicted else 0
                    map5_score += score
                map5_score /= len(y_val_fold)

                scores.append(map5_score)

            return np.mean(scores)
        except Exception as e:
            print(f"XGBoostä¼˜åŒ–è¿‡ç¨‹ä¸­å‡ºé”™: {str(e)}")
            return 0.0  # è¿”å›�æœ€ä½�åˆ†æ•°

    # è®¾ç½®è¶…æ—¶æœºåˆ¶
    timeout = 600  # 10åˆ†é’Ÿè¶…æ—¶

    # åˆ›å»ºä¼˜åŒ–ç ”ç©¶
    study = optuna.create_study(direction='maximize')

    try:
        # ç§»é™¤ä¸�å…¼å®¹çš„å›�è°ƒï¼Œä»…ä½¿ç”¨è¶…æ—¶
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
    except Exception as e:
        print(f"XGBoostä¼˜åŒ–è¿‡ç¨‹è¢«ä¸­æ–­: {str(e)}")

    # æ£€æŸ¥æ˜¯å�¦æœ‰ä»»ä½•æˆ�åŠŸçš„è¯•éªŒ
    if len(study.trials) == 0:
        print("âš ï¸� XGBoostä¼˜åŒ–å¤±è´¥ï¼Œä½¿ç”¨é»˜è®¤å�‚æ•°")
        return {
            'objective': 'multi:softprob',
            'num_class': len(np.unique(y)),
            'eval_metric': 'mlogloss',
            'learning_rate': 0.1,
            'max_depth': 6,
            'min_child_weight': 1,
            'gamma': 0,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'random_state': 42
        }

    # æœ€å��è¿”å›�æ—¶ï¼Œç¡®ä¿�å¿…è¦�å�‚æ•°å­˜åœ¨
    best_params = study.best_params
    best_params['objective'] = 'multi:softprob'
    best_params['num_class'] = len(np.unique(y))
    best_params['eval_metric'] = 'mlogloss'
    best_params['random_state'] = 42

    # åœ¨optimize_xgboost_paramså‡½æ•°çš„è¿”å›�éƒ¨åˆ†
    best_params['tree_method'] = 'gpu_hist'
    best_params['gpu_id'] = 0

    print(f"ğŸ�¯ æœ€ä½³XGBoostå�‚æ•°: {best_params}")
    print(f"ğŸ�† æœ€ä½³å¾—åˆ†: {study.best_value:.4f}")

    return best_params


def optimize_catboost_params(X, y, n_trials=5):
    """ä¼˜åŒ– CatBoost è¶…å�‚æ•°"""

    def objective(trial):
        try:
            params = {
                'objective': 'MultiClass',
                'eval_metric': 'MultiClass',
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'depth': trial.suggest_int('depth', 4, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
                'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 10.0),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'random_seed': 42,
                'verbose': 0,
                'task_type': 'GPU',
                'devices': '0'
            }

            # ä½¿ç”¨æ›´å°‘çš„æŠ˜æ•°åŠ é€Ÿä¼˜åŒ–è¿‡ç¨‹
            n_folds = 3 if n_trials > 3 else 5
            kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            scores = []

            # è®¾ç½®æ¯�æŠ˜è®­ç»ƒçš„æœ€å¤§è¿­ä»£æ¬¡æ•°
            max_boost_rounds = 100 if n_trials > 3 else 200
            early_stopping_rounds = 20

            for train_idx, val_idx in kf.split(X, y):
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]

                # ä½¿ç”¨æ—©å�œæ�¥åŠ é€Ÿè®­ç»ƒ
                model = cb.CatBoostClassifier(
                    **params,
                    iterations=max_boost_rounds,
                    early_stopping_rounds=early_stopping_rounds
                )

                # æ�•è�·è®­ç»ƒè¿‡ç¨‹ä¸­çš„æ½œåœ¨é”™è¯¯
                try:
                    model.fit(
                        X_train_fold, y_train_fold,
                        eval_set=[(X_val_fold, y_val_fold)],
                        verbose=False
                    )
                except Exception as inner_e:
                    print(f"CatBoostè®­ç»ƒå¤±è´¥: {str(inner_e)}")
                    # å°�è¯•ä½¿ç”¨æ›´ç®€å�•çš„å�‚æ•°é‡�æ–°è®­ç»ƒ
                    simple_params = {
                        'objective': 'MultiClass',
                        'iterations': 50,
                        'learning_rate': 0.1,
                        'depth': 6,
                        'verbose': 0
                    }
                    model = cb.CatBoostClassifier(**simple_params)
                    model.fit(X_train_fold, y_train_fold, verbose=False)

                # é¢„æµ‹éªŒè¯�é›†
                val_pred = model.predict_proba(X_val_fold)

                # è®¡ç®—MAP@5
                top_k_pred = np.argsort(val_pred, axis=1)[:, -5:][:, ::-1]

                # è®¡ç®—MAP@5
                map5_score = 0
                for i in range(len(y_val_fold)):
                    actual = y_val_fold[i]
                    predicted = top_k_pred[i]
                    score = 1 if actual in predicted else 0
                    map5_score += score
                map5_score /= len(y_val_fold)

                scores.append(map5_score)

            return np.mean(scores)
        except Exception as e:
            print(f"CatBoostä¼˜åŒ–è¿‡ç¨‹ä¸­å‡ºé”™: {str(e)}")
            return 0.0  # è¿”å›�æœ€ä½�åˆ†æ•°

    # è®¾ç½®è¶…æ—¶æœºåˆ¶
    timeout = 600  # 10åˆ†é’Ÿè¶…æ—¶

    # åˆ›å»ºä¼˜åŒ–ç ”ç©¶
    study = optuna.create_study(direction='maximize')

    try:
        # ç§»é™¤ä¸�å…¼å®¹çš„å›�è°ƒï¼Œä»…ä½¿ç”¨è¶…æ—¶
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
    except Exception as e:
        print(f"CatBoostä¼˜åŒ–è¿‡ç¨‹è¢«ä¸­æ–­: {str(e)}")

    # æ£€æŸ¥æ˜¯å�¦æœ‰ä»»ä½•æˆ�åŠŸçš„è¯•éªŒ
    if len(study.trials) == 0:
        print("âš ï¸� CatBoostä¼˜åŒ–å¤±è´¥ï¼Œä½¿ç”¨é»˜è®¤å�‚æ•°")
        return {
            'objective': 'MultiClass',
            'eval_metric': 'MultiClass',
            'learning_rate': 0.1,
            'depth': 6,
            'l2_leaf_reg': 3,
            'random_strength': 1,
            'bagging_temperature': 1,
            'border_count': 128,
            'random_seed': 42,
            'verbose': 0
        }

    # æœ€å��è¿”å›�æ—¶ï¼Œç¡®ä¿�å¿…è¦�å�‚æ•°å­˜åœ¨
    best_params = study.best_params
    best_params['objective'] = 'MultiClass'
    best_params['eval_metric'] = 'MultiClass'
    best_params['random_seed'] = 42
    best_params['verbose'] = 0

    # åœ¨optimize_catboost_paramså‡½æ•°çš„è¿”å›�éƒ¨åˆ†
    best_params['task_type'] = 'GPU'
    best_params['devices'] = '0'

    print(f"ğŸ�¯ æœ€ä½³CatBoostå�‚æ•°: {best_params}")
    print(f"ğŸ�† æœ€ä½³å¾—åˆ†: {study.best_value:.4f}")

    return best_params

# ä¿®æ”¹ç»§æ‰¿æ–¹å¼�ï¼Œé�¿å…�è‡ªæˆ‘ç»§æ‰¿
class FertilizerMultiClassPredictorV5(FertilizerMultiClassPredictor):
    # ç»§æ‰¿å�Ÿæœ‰ç±»å¹¶æ·»åŠ /ä¿®æ”¹æ–¹æ³•

    def train_lightgbm(self, X, y, params=None):
        """è®­ç»ƒ LightGBM å¤šåˆ†ç±»æ¨¡å�‹"""
        print("\nğŸŒŸ è®­ç»ƒ LightGBM æ¨¡å�‹...")

        try:
            if params is None:
                params = {
                    'objective': 'multiclass',
                    'num_class': len(self.label_encoder.classes_),
                    'metric': 'multi_logloss',
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.9,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'verbose': -1,
                    'random_state': 42,
                    'device': 'gpu',  # å�¯ç”¨GPU
                    'gpu_platform_id': 0,
                    'gpu_device_id': 0
                }

                # ç¡®ä¿�GPUå�‚æ•°å­˜åœ¨
            if 'device' not in params:
                    params['device'] = 'gpu'

            # ç¡®ä¿�å�‚æ•°åŒ…å�«å¤šåˆ†ç±»è®¾ç½®
            if 'objective' not in params or params['objective'] != 'multiclass':
                params['objective'] = 'multiclass'
            if 'num_class' not in params:
                params['num_class'] = len(self.label_encoder.classes_)

            # æ£€æŸ¥æ•°æ�®æœ‰æ•ˆæ€§
            if X.isnull().values.any():
                print("âš ï¸� è¾“å…¥æ•°æ�®åŒ…å�«NaNå€¼ï¼Œå°†è¿›è¡Œå¡«å……")
                X = X.fillna(X.mean())

            # 5æŠ˜äº¤å�‰éªŒè¯�
            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            oof_predictions = np.zeros((len(X), len(self.label_encoder.classes_)))
            feature_importance = np.zeros(X.shape[1])

            # é‡Šæ”¾å†…å­˜
            import gc
            gc.collect()

            for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
                print(f"  ğŸ“� è®­ç»ƒç¬¬ {fold + 1}/5 æŠ˜...")

                try:
                    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                    y_train_fold, y_val_fold = y[train_idx], y[val_idx]

                    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
                    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

                    # æ·»åŠ å­¦ä¹ ç�‡è°ƒåº¦å›�è°ƒå‡½æ•°
                    callbacks = [
                        lgb.early_stopping(100),
                        lgb.log_evaluation(0),
                        lgb.reset_parameter(learning_rate=lambda iter: params['learning_rate'] * (0.99 ** iter))
                    ]

                    model = lgb.train(
                        params,
                        train_data,
                        valid_sets=[train_data, val_data],
                        num_boost_round=1000,
                        callbacks=callbacks
                    )

                    # é¢„æµ‹éªŒè¯�é›†
                    val_pred = model.predict(X_val_fold, num_iteration=model.best_iteration)
                    oof_predictions[val_idx] = val_pred

                    # ç´¯è®¡ç‰¹å¾�é‡�è¦�æ€§
                    feature_importance += model.feature_importance(importance_type='gain')

                    if fold == 0:
                        self.models['lgb'] = model

                except Exception as e:
                    print(f"âš ï¸� ç¬¬{fold + 1}æŠ˜è®­ç»ƒå¤±è´¥: {str(e)}")
                    # ä½¿ç”¨é›¶å¡«å……é¢„æµ‹
                    oof_predictions[val_idx] = np.zeros((len(val_idx), len(self.label_encoder.classes_)))

            # å¦‚æ�œæ‰€æœ‰æŠ˜éƒ½å¤±è´¥ï¼Œå°�è¯•è®­ç»ƒä¸€ä¸ªç®€å�•æ¨¡å�‹
            if 'lgb' not in self.models:
                print("âš ï¸� æ‰€æœ‰æŠ˜è®­ç»ƒå¤±è´¥ï¼Œå°�è¯•è®­ç»ƒç®€åŒ–æ¨¡å�‹")
                try:
                    simple_params = {
                        'objective': 'multiclass',
                        'num_class': len(self.label_encoder.classes_),
                        'metric': 'multi_logloss',
                        'boosting_type': 'gbdt',
                        'num_leaves': 15,
                        'learning_rate': 0.1,
                        'feature_fraction': 0.7,
                        'verbose': -1,
                        'random_state': 42
                    }

                    train_data = lgb.Dataset(X, label=y)
                    self.models['lgb'] = lgb.train(simple_params, train_data, num_boost_round=100)

                    # ç”Ÿæˆ�OOFé¢„æµ‹
                    oof_predictions = self.models['lgb'].predict(X)
                except Exception as e:
                    print(f"âš ï¸� ç®€åŒ–æ¨¡å�‹è®­ç»ƒå¤±è´¥: {str(e)}")
                    # è¿”å›�å�‡åŒ€åˆ†å¸ƒé¢„æµ‹
                    oof_predictions = np.ones((len(X), len(self.label_encoder.classes_))) / len(
                        self.label_encoder.classes_)

            self.feature_importance['lgb'] = feature_importance / 5

            # é‡Šæ”¾å†…å­˜
            gc.collect()

            return oof_predictions

        except Exception as e:
            print(f"âš ï¸� LightGBMè®­ç»ƒè¿‡ç¨‹å‡ºé”™: {str(e)}")
            # è¿”å›�å�‡åŒ€åˆ†å¸ƒé¢„æµ‹
            return np.ones((len(X), len(self.label_encoder.classes_))) / len(self.label_encoder.classes_)

    def train_xgboost(self, X, y, params=None):
        """è®­ç»ƒ XGBoost å¤šåˆ†ç±»æ¨¡å�‹"""
        print("\nğŸš€ è®­ç»ƒ XGBoost æ¨¡å�‹...")

        if params is None:
            params = {
                'objective': 'multi:softprob',
                'num_class': len(self.label_encoder.classes_),
                'eval_metric': 'mlogloss',
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'verbosity': 0,
                'tree_method': 'gpu_hist',  # ä½¿ç”¨GPUåŠ é€Ÿ
                'gpu_id': 0
            }

            # ç¡®ä¿�GPUå�‚æ•°å­˜åœ¨
        if 'tree_method' not in params:
            params['tree_method'] = 'gpu_hist'

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof_predictions = np.zeros((len(X), len(self.label_encoder.classes_)))

        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            print(f"  ğŸ“� è®­ç»ƒç¬¬ {fold + 1}/5 æŠ˜...")

            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            model = xgb.XGBClassifier(**params)
            # å®Œå…¨ç§»é™¤early_stopping_roundså�‚æ•°
            model.fit(
                X_train_fold, y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                verbose=False
            )

            val_pred = model.predict_proba(X_val_fold)
            oof_predictions[val_idx] = val_pred

            if fold == 0:
                self.models['xgb'] = model

        return oof_predictions

    def train_catboost(self, X, y, params=None):
        """è®­ç»ƒ CatBoost å¤šåˆ†ç±»æ¨¡å�‹"""
        print("\nğŸ�± è®­ç»ƒ CatBoost æ¨¡å�‹...")

        if params is None:
            params = {
                'objective': 'MultiClass',
                'eval_metric': 'MultiClass',
                'learning_rate': 0.05,
                'iterations': 1000,
                'depth': 6,
                'random_seed': 42,
                'verbose': 0,
                'task_type': 'GPU',  # ä½¿ç”¨GPU
                'devices': '0'  # ä½¿ç”¨ç¬¬ä¸€ä¸ªGPUè®¾å¤‡
            }

        # ç¡®ä¿�GPUå�‚æ•°å­˜åœ¨
        if 'task_type' not in params:
            params['task_type'] = 'GPU'

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof_predictions = np.zeros((len(X), len(self.label_encoder.classes_)))

        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            print(f"  ğŸ“� è®­ç»ƒç¬¬ {fold + 1}/5 æŠ˜...")

            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            model = cb.CatBoostClassifier(**params)
            model.fit(
                X_train_fold, y_train_fold,
                eval_set=(X_val_fold, y_val_fold),
                early_stopping_rounds=100,
                verbose=False
            )

            val_pred = model.predict_proba(X_val_fold)
            oof_predictions[val_idx] = val_pred

            if fold == 0:
                self.models['catboost'] = model

        return oof_predictions

    def train_tabnet(self, X, y, params=None):
        """è®­ç»ƒ TabNet å¤šåˆ†ç±»æ¨¡å�‹"""
        if not TABNET_AVAILABLE:
            print("âš ï¸� TabNetæœªå®‰è£…ï¼Œè¯·ä½¿ç”¨pip install pytorch-tabnetå®‰è£…")
            return None
        print("\nğŸ“Š è®­ç»ƒ TabNet æ¨¡å�‹...")

        # æ£€æŸ¥CUDAå�¯ç”¨æ€§
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"  ä½¿ç”¨è®¾å¤‡: {device}")

        if params is None:
            params = {
                'n_d': 16,
                'n_a': 16,
                'n_steps': 3,
                'gamma': 1.5,
                'n_independent': 2,
                'n_shared': 2,
                'lambda_sparse': 1e-4,
                'optimizer_params': {'lr': 0.02},
                'mask_type': 'entmax',
                'scheduler_params': {'step_size': 50, 'gamma': 0.9},
                'scheduler_fn': torch.optim.lr_scheduler.StepLR,
                'seed': 42,
                'verbose': 0,
                'device_name': device  # ä½¿ç”¨CUDAè®¾å¤‡
            }
        else:
            params['device_name'] = device

        if params is None:
            params = {
                'n_d': 16,
                'n_a': 16,
                'n_steps': 3,
                'gamma': 1.5,
                'lambda_sparse': 1e-3,
                'optimizer_fn': torch.optim.Adam,
                'optimizer_params': dict(lr=2e-2),
                'mask_type': 'entmax',
                'scheduler_params': dict(mode="min",
                                         patience=5,
                                         min_lr=1e-5,
                                         factor=0.5),
                'scheduler_fn': torch.optim.lr_scheduler.ReduceLROnPlateau,
                'verbose': 0,
                'device_name': 'cuda' if torch.cuda.is_available() else 'cpu'
            }

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof_predictions = np.zeros((len(X), len(self.label_encoder.classes_)))

        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            print(f"  ğŸ“� è®­ç»ƒç¬¬ {fold + 1}/5 æŠ˜...")

            X_train_fold, X_val_fold = X.iloc[train_idx].values, X.iloc[val_idx].values
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            model = TabNetClassifier(**params)
            model.fit(
                X_train=X_train_fold, y_train=y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                max_epochs=100,
                patience=15,
                batch_size=1024,
                virtual_batch_size=128,
                eval_metric=['logloss']
            )

            val_pred = model.predict_proba(X_val_fold)
            oof_predictions[val_idx] = val_pred

            if fold == 0:
                self.models['tabnet'] = model

        return oof_predictions

    def train_diverse_stacking(self, X, y, base_models_oof):
        """è®­ç»ƒå¤šæ ·åŒ–å †å� æ¨¡å�‹"""
        print("\nğŸ”„ è®­ç»ƒå¤šæ ·åŒ–å †å� æ¨¡å�‹...")

        try:
            # æ£€æŸ¥è¾“å…¥æœ‰æ•ˆæ€§
            if not base_models_oof or len(base_models_oof) < 2:
                print("âš ï¸� éœ€è¦�è‡³å°‘ä¸¤ä¸ªåŸºç¡€æ¨¡å�‹è¿›è¡Œå †å� ")
                return np.zeros((len(X), len(self.label_encoder.classes_))), 0.0

            # ç¡®ä¿�æ‰€æœ‰é¢„æµ‹å½¢çŠ¶ä¸€è‡´
            shapes = [pred.shape for pred in base_models_oof]
            if len(set(shapes)) > 1:
                print(f"âš ï¸� é¢„æµ‹å½¢çŠ¶ä¸�ä¸€è‡´: {shapes}")
                return np.zeros((len(X), len(self.label_encoder.classes_))), 0.0

            # å‡†å¤‡å †å� ç‰¹å¾�
            stacking_features = np.hstack(base_models_oof)

            # æ£€æŸ¥æ˜¯å�¦æœ‰NaNæˆ–Infå€¼
            if np.isnan(stacking_features).any() or np.isinf(stacking_features).any():
                print("âš ï¸� å †å� ç‰¹å¾�åŒ…å�«NaNæˆ–Infå€¼ï¼Œå°†è¿›è¡Œæ›¿æ�¢")
                stacking_features = np.nan_to_num(stacking_features, nan=0.0, posinf=1.0, neginf=0.0)

            # æ·»åŠ å�Ÿå§‹ç‰¹å¾�ä¸�å †å� ç‰¹å¾�çš„ç»„å�ˆ
            X_numeric = X.select_dtypes(include=[np.number])
            combined_features = np.hstack([stacking_features, X_numeric])

            # ä½¿ç”¨ä¸‰å±‚å †å� æ�¶æ�„
            # ç¬¬ä¸€å±‚: LightGBMä½œä¸ºå…ƒå­¦ä¹ å™¨
            meta_lgb_params = {
                'objective': 'multiclass',
                'num_class': len(self.label_encoder.classes_),
                'metric': 'multi_logloss',
                'learning_rate': 0.03,
                'num_leaves': 31,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': 42,
                'device': 'gpu',
                'gpu_platform_id': 0,
                'gpu_device_id': 0
            }

            # ç¬¬äºŒå±‚: XGBoostä½œä¸ºå…ƒå­¦ä¹ å™¨
            meta_xgb_params = {
                'objective': 'multi:softprob',
                'num_class': len(self.label_encoder.classes_),
                'eval_metric': 'mlogloss',
                'learning_rate': 0.03,
                'max_depth': 5,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'tree_method': 'gpu_hist',
                'gpu_id': 0
            }

            # ç¬¬ä¸‰å±‚: CatBoostä½œä¸ºæœ€ç»ˆé›†æˆ�å™¨
            meta_cb_params = {
                'objective': 'MultiClass',
                'eval_metric': 'MultiClass',
                'learning_rate': 0.03,
                'depth': 5,
                'random_seed': 42,
                'verbose': False,
                'task_type': 'GPU',
                'devices': '0'
            }

            # 8æŠ˜äº¤å�‰éªŒè¯�
            kf = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
            oof_predictions = np.zeros((len(X), len(self.label_encoder.classes_)))

            # ç¬¬ä¸€å±‚å…ƒå­¦ä¹ å™¨çš„é¢„æµ‹
            meta_lgb_preds = np.zeros((len(X), len(self.label_encoder.classes_)))
            meta_xgb_preds = np.zeros((len(X), len(self.label_encoder.classes_)))

            # é‡Šæ”¾å†…å­˜
            import gc
            gc.collect()

            for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
                print(f"  ğŸ“� è®­ç»ƒç¬¬ {fold + 1}/8 æŠ˜...")

                try:
                    # å‡†å¤‡è®­ç»ƒå’ŒéªŒè¯�æ•°æ�®
                    X_train_fold, X_val_fold = stacking_features[train_idx], stacking_features[val_idx]
                    X_train_combined, X_val_combined = combined_features[train_idx], combined_features[val_idx]
                    y_train_fold, y_val_fold = y[train_idx], y[val_idx]

                    # ç¬¬ä¸€å±‚: LightGBM
                    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
                    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

                    lgb_model = lgb.train(
                        meta_lgb_params,
                        train_data,
                        valid_sets=[train_data, val_data],
                        num_boost_round=1000,
                        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
                    )

                    # é¢„æµ‹éªŒè¯�é›†
                    lgb_val_pred = lgb_model.predict(X_val_fold, num_iteration=lgb_model.best_iteration)
                    meta_lgb_preds[val_idx] = lgb_val_pred

                    # ç¬¬ä¸€å±‚: XGBoost
                    xgb_model = xgb.XGBClassifier(**meta_xgb_params)
                    xgb_model.fit(
                        X_train_fold, y_train_fold,
                        eval_set=[(X_val_fold, y_val_fold)],
                        verbose=False
                    )

                    # é¢„æµ‹éªŒè¯�é›†
                    xgb_val_pred = xgb_model.predict_proba(X_val_fold)
                    meta_xgb_preds[val_idx] = xgb_val_pred

                    # ç¬¬äºŒå±‚: CatBoost (ä½¿ç”¨å�Ÿå§‹ç‰¹å¾�+ç¬¬ä¸€å±‚é¢„æµ‹)
                    X_train_meta = np.hstack([X_train_combined, meta_lgb_preds[train_idx], meta_xgb_preds[train_idx]])
                    X_val_meta = np.hstack([X_val_combined, lgb_val_pred, xgb_val_pred])

                    cb_model = cb.CatBoostClassifier(**meta_cb_params)
                    cb_model.fit(
                        X_train_meta, y_train_fold,
                        eval_set=[(X_val_meta, y_val_fold)],
                        verbose=False
                    )

                    # æœ€ç»ˆé¢„æµ‹
                    val_pred = cb_model.predict_proba(X_val_meta)
                    oof_predictions[val_idx] = val_pred

                    # ä¿�å­˜ç¬¬ä¸€ä¸ªæŠ˜çš„æ¨¡å�‹
                    if fold == 0:
                        self.models['stacking_lgb'] = lgb_model
                        self.models['stacking_xgb'] = xgb_model
                        self.models['stacking'] = cb_model

                except Exception as e:
                    print(f"âš ï¸� ç¬¬{fold + 1}æŠ˜å †å� è®­ç»ƒå¤±è´¥: {str(e)}")
                    # ä½¿ç”¨åŸºç¡€æ¨¡å�‹çš„å¹³å�‡å€¼å¡«å……é¢„æµ‹
                    base_avg = np.mean([pred[val_idx] for pred in base_models_oof], axis=0)
                    oof_predictions[val_idx] = base_avg

            # å¦‚æ�œæ‰€æœ‰æŠ˜éƒ½å¤±è´¥ï¼Œä½¿ç”¨åŸºç¡€æ¨¡å�‹çš„å¹³å�‡å€¼
            if 'stacking' not in self.models:
                print("âš ï¸� æ‰€æœ‰å †å� æŠ˜è®­ç»ƒå¤±è´¥ï¼Œä½¿ç”¨åŸºç¡€æ¨¡å�‹å¹³å�‡")
                oof_predictions = np.mean(base_models_oof, axis=0)
                stacking_map5 = self.calculate_mapk(y, oof_predictions, k=5)
                print(f"ğŸ“ˆ åŸºç¡€æ¨¡å�‹å¹³å�‡ MAP@5: {stacking_map5:.4f}")
                return oof_predictions, stacking_map5

            stacking_map5 = self.calculate_mapk(y, oof_predictions, k=5)
            print(f"ğŸ“ˆ å¤šæ ·åŒ–å †å� æ¨¡å�‹ MAP@5: {stacking_map5:.4f}")

            # é‡Šæ”¾å†…å­˜
            gc.collect()

            return oof_predictions, stacking_map5

        except Exception as e:
            print(f"âš ï¸� å †å� æ¨¡å�‹è®­ç»ƒå¤±è´¥: {str(e)}")
            # è¿”å›�åŸºç¡€æ¨¡å�‹çš„å¹³å�‡å€¼ä½œä¸ºå¤‡é€‰
            try:
                avg_pred = np.mean(base_models_oof, axis=0)
                avg_map5 = self.calculate_mapk(y, avg_pred, k=5)
                print(f"ğŸ“ˆ ä½¿ç”¨åŸºç¡€æ¨¡å�‹å¹³å�‡ä½œä¸ºå¤‡é€‰, MAP@5: {avg_map5:.4f}")
                return avg_pred, avg_map5
            except:
                print("âš ï¸� åŸºç¡€æ¨¡å�‹å¹³å�‡ä¹Ÿå¤±è´¥")
                return np.zeros((len(X), len(self.label_encoder.classes_))), 0.0

    def train_with_grouped_cv(self, X, y, groups, params=None):
        """ä½¿ç”¨åˆ†ç»„äº¤å�‰éªŒè¯�è®­ç»ƒæ¨¡å�‹"""
        from sklearn.model_selection import GroupKFold

        print("\nğŸ”„ ä½¿ç”¨åˆ†ç»„äº¤å�‰éªŒè¯�è®­ç»ƒæ¨¡å�‹...")

        if params is None:
            params = {
                'objective': 'multiclass',
                'num_class': len(self.label_encoder.classes_),
                'metric': 'multi_logloss',
                'learning_rate': 0.05,
                'num_leaves': 31,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': 42,
                'device': 'gpu',
                'gpu_platform_id': 0,
                'gpu_device_id': 0
            }

        # åˆ†ç»„KæŠ˜äº¤å�‰éªŒè¯�
        gkf = GroupKFold(n_splits=5)
        oof_predictions = np.zeros((len(X), len(self.label_encoder.classes_)))

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            print(f"  ğŸ“� è®­ç»ƒç¬¬ {fold + 1}/5 æŠ˜...")

            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
            val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

            callbacks = [
                lgb.early_stopping(100),
                lgb.log_evaluation(0),
                lgb.reset_parameter(learning_rate=lambda iter: params['learning_rate'] * (0.99 ** iter))
            ]

            model = lgb.train(
                params,
                train_data,
                valid_sets=[train_data, val_data],
                num_boost_round=1000,
                callbacks=callbacks
            )

            # é¢„æµ‹éªŒè¯�é›†
            val_pred = model.predict(X_val_fold, num_iteration=model.best_iteration)
            oof_predictions[val_idx] = val_pred

            if fold == 0:
                self.models['grouped_lgb'] = model

        return oof_predictions

    def optimize_ensemble_weights(self, X, y, base_models_oof):
        """ä¼˜åŒ–é›†æˆ�æ¨¡å�‹çš„æ�ƒé‡�"""
        print("\nğŸ”„ ä¼˜åŒ–é›†æˆ�æ�ƒé‡�...")

        try:
            # æ£€æŸ¥è¾“å…¥æœ‰æ•ˆæ€§
            if not base_models_oof or len(base_models_oof) < 2:
                print("âš ï¸� éœ€è¦�è‡³å°‘ä¸¤ä¸ªåŸºç¡€æ¨¡å�‹è¿›è¡Œé›†æˆ�")
                return {'lgb': 0.4, 'xgb': 0.3, 'catboost': 0.3}

            # ç¡®ä¿�æ‰€æœ‰é¢„æµ‹å½¢çŠ¶ä¸€è‡´
            shapes = [pred.shape for pred in base_models_oof]
            if len(set(shapes)) > 1:
                print(f"âš ï¸� é¢„æµ‹å½¢çŠ¶ä¸�ä¸€è‡´: {shapes}")
                return {'lgb': 0.4, 'xgb': 0.3, 'catboost': 0.3}

            from scipy.optimize import minimize

            # å®šä¹‰è¦�ä¼˜åŒ–çš„ç›®æ ‡å‡½æ•° - æœ€å¤§åŒ–MAP@5
            def objective(weights):
                # å½’ä¸€åŒ–æ�ƒé‡�
                weights = weights / np.sum(weights)

                # åŠ æ�ƒå¹³å�‡é¢„æµ‹
                ensemble_pred = np.zeros_like(base_models_oof[0])
                for i, pred in enumerate(base_models_oof):
                    ensemble_pred += weights[i] * pred

                # è®¡ç®—MAP@5
                score = -self.calculate_mapk(y, ensemble_pred, k=5)  # è´Ÿå�·æ˜¯å› ä¸ºæˆ‘ä»¬è¦�æœ€å°�åŒ–
                return score

            # åˆ�å§‹æ�ƒé‡� - å¹³å�‡åˆ†é…�
            initial_weights = np.ones(len(base_models_oof)) / len(base_models_oof)

            # çº¦æ�Ÿæ�¡ä»¶ - æ�ƒé‡�ä¹‹å’Œä¸º1ï¼Œæ¯�ä¸ªæ�ƒé‡�éƒ½æ˜¯é��è´Ÿçš„
            constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
            bounds = [(0, 1) for _ in range(len(base_models_oof))]

            # æ‰§è¡Œä¼˜åŒ–
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'disp': False, 'maxiter': 1000}
            )

            # è�·å�–ä¼˜åŒ–å��çš„æ�ƒé‡�
            optimized_weights = result['x'] / np.sum(result['x'])

            # åˆ›å»ºæ�ƒé‡�å­—å…¸
            model_names = ['lgb', 'xgb', 'catboost', 'tabnet', 'stacking'][:len(base_models_oof)]
            weights_dict = {name: weight for name, weight in zip(model_names, optimized_weights)}

            print(f"âœ… ä¼˜åŒ–å��çš„æ�ƒé‡�: {weights_dict}")
            return weights_dict

        except Exception as e:
            print(f"âš ï¸� æ�ƒé‡�ä¼˜åŒ–å¤±è´¥: {str(e)}")
            # è¿”å›�é»˜è®¤æ�ƒé‡�
            default_weights = {'lgb': 0.4, 'xgb': 0.3, 'catboost': 0.3}
            if len(base_models_oof) > 3:
                default_weights = {'lgb': 0.3, 'xgb': 0.25, 'catboost': 0.2, 'tabnet': 0.25}
            if len(base_models_oof) > 4:
                default_weights = {'lgb': 0.25, 'xgb': 0.2, 'catboost': 0.15, 'tabnet': 0.2, 'stacking': 0.2}

            print(f"âš ï¸� ä½¿ç”¨é»˜è®¤æ�ƒé‡�: {default_weights}")
            return default_weights
    def focal_loss_objective(self, y_true, y_pred, gamma=2.0, alpha=0.25):
        """Focal Lossç›®æ ‡å‡½æ•°ï¼Œç”¨äº�å¤„ç�†ç±»åˆ«ä¸�å¹³è¡¡"""
        # è®¡ç®—äº¤å�‰ç†µ
        epsilon = 1e-10
        y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
        ce = -y_true * np.log(y_pred)

        # è®¡ç®—focalæ�ƒé‡�
        p_t = np.where(y_true == 1, y_pred, 1 - y_pred)
        focal_weight = np.power(1 - p_t, gamma)

        # åº”ç”¨alphaå¹³è¡¡å› å­�
        alpha_t = np.where(y_true == 1, alpha, 1 - alpha)

        # è®¡ç®—æœ€ç»ˆæ�Ÿå¤±
        loss = alpha_t * focal_weight * ce
        return np.mean(loss)

    def calculate_mapk(self, y_true, y_pred_proba, k=5):
        """è®¡ç®— MAP@K æŒ‡æ ‡"""

        def apk(actual, predicted, k):
            if len(predicted) > k:
                predicted = predicted[:k]

            score = 0.0
            num_hits = 0.0

            for i, p in enumerate(predicted):
                if p == actual:
                    num_hits += 1.0
                    score += num_hits / (i + 1.0)
                    break

            return score

        # è�·å�–æ¯�ä¸ªæ ·æœ¬çš„top-ké¢„æµ‹
        top_k_pred = np.argsort(y_pred_proba, axis=1)[:, -k:][:, ::-1]

        scores = []
        for i, true_label in enumerate(y_true):
            predicted_labels = top_k_pred[i]
            score = apk(true_label, predicted_labels, k)
            scores.append(score)

        return np.mean(scores)

    def analyze_feature_importance(self, X):
        """åˆ†æ��ç‰¹å¾�é‡�è¦�æ€§"""
        print("\nğŸ“Š ç‰¹å¾�é‡�è¦�æ€§åˆ†æ��...")

        if 'lgb' in self.feature_importance:
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': self.feature_importance['lgb']
            }).sort_values('importance', ascending=False)

            plt.figure(figsize=(10, 8))
            top_features = importance_df.head(20)
            sns.barplot(data=top_features, y='feature', x='importance')
            plt.title('ğŸ�† Top 20 ç‰¹å¾�é‡�è¦�æ€§ (LightGBM)')
            plt.xlabel('é‡�è¦�æ€§å¾—åˆ†')
            plt.tight_layout()
            plt.show()

            print("Top 10 é‡�è¦�ç‰¹å¾�:")
            for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
                print(f"{i:2d}. {row['feature']:<25} : {row['importance']:8.2f}")

        return importance_df if 'lgb' in self.feature_importance else None


def main():
    try:
        # ç›‘æ�§å†…å­˜ä½¿ç”¨
        import gc
        import psutil
        import os
        import numpy as np
        process = psutil.Process(os.getpid())

        # æ£€æµ‹GPUå�¯ç”¨æ€§
        gpu_available = False
        try:
            if TABNET_AVAILABLE:
                gpu_available = torch.cuda.is_available()
                if gpu_available:
                    print(f"ğŸ�® GPUå�¯ç”¨: {torch.cuda.get_device_name(0)}")
                    print(f"ğŸ“Š CUDAç‰ˆæœ¬: {torch.version.cuda}")
        except:
            pass

        print(f"åˆ�å§‹å†…å­˜ä½¿ç”¨: {process.memory_info().rss / 1024 / 1024:.2f} MB")

        # åˆ�å§‹åŒ–é¢„æµ‹å™¨
        predictor = FertilizerMultiClassPredictorV5()

        print("ğŸŒ± Kaggle è‚¥æ–™é¢„æµ‹ - å¤šåˆ†ç±»æ¢¯åº¦æ��å�‡æ ‘è§£å†³æ–¹æ¡ˆ (ä¼˜åŒ–ç‰ˆæœ¬6)")
        print("=" * 60)

        # 1. åŠ è½½å’Œæ�¢ç´¢æ•°æ�®
        train_df, test_df = predictor.load_and_explore_data()
        print(f"æ•°æ�®åŠ è½½å��å†…å­˜ä½¿ç”¨: {process.memory_info().rss / 1024 / 1024:.2f} MB")

        # 2. ç‰¹å¾�å·¥ç¨‹
        feature_list = predictor.advanced_feature_engineering()
        gc.collect()
        print(f"ç‰¹å¾�å·¥ç¨‹å��å†…å­˜ä½¿ç”¨: {process.memory_info().rss / 1024 / 1024:.2f} MB")

        # 3. å‡†å¤‡å»ºæ¨¡ç‰¹å¾�
        X, y, X_test = predictor.prepare_features()
        gc.collect()
        print(f"ç‰¹å¾�å‡†å¤‡å��å†…å­˜ä½¿ç”¨: {process.memory_info().rss / 1024 / 1024:.2f} MB")

        # ç¡®ä¿�åœ¨æ­¤é˜¶æ®µå¤„ç�†æµ‹è¯•æ•°æ�®ä¸­çš„é��æ•°å€¼åˆ—
        print("\nğŸ”� æ£€æŸ¥æµ‹è¯•æ•°æ�®ç±»å�‹")
        X_test_numeric = X_test.select_dtypes(include=[np.number])
        if X_test_numeric.shape[1] < X_test.shape[1]:
            print(f"âš ï¸� æµ‹è¯•æ•°æ�®åŒ…å�«{X_test.shape[1] - X_test_numeric.shape[1]}ä¸ªé��æ•°å€¼åˆ—ï¼Œå°†å�ªä¿�ç•™æ•°å€¼åˆ—")
            X_test = X_test_numeric
            print(f"å¤„ç�†å��æµ‹è¯•æ•°æ�®å½¢çŠ¶: {X_test.shape}")

        # 4. è®­ç»ƒå¤šä¸ªæ¨¡å�‹
        print("\nğŸ”¥ å¼€å§‹è®­ç»ƒå¤šåˆ†ç±»æ¨¡å�‹...")
        try:
            best_lgb_params, best_xgb_params, best_cb_params, lgb_oof, xgb_oof, cb_oof = predictor.optimize_and_train_models(
                X, y)
        except Exception as e:
            print(f"âš ï¸� ä¼˜åŒ–æ¨¡å�‹è®­ç»ƒå¤±è´¥: {str(e)}ï¼Œä½¿ç”¨é»˜è®¤å�‚æ•°")
            lgb_oof = predictor.train_lightgbm(X, y)
            xgb_oof = predictor.train_xgboost(X, y)
            cb_oof = predictor.train_catboost(X, y)

        # 5. è®­ç»ƒTabNetæ¨¡å�‹ (å¦‚æ�œå·²å®‰è£…)
        base_models_oof = [lgb_oof, xgb_oof, cb_oof]
        tabnet_map5 = 0
        try:
            if TABNET_AVAILABLE:
                tabnet_oof = predictor.train_tabnet(X, y)
                tabnet_map5 = predictor.calculate_mapk(y, tabnet_oof, k=5)
                print(f"ğŸ“ˆ TabNet MAP@5: {tabnet_map5:.4f}")
                base_models_oof.append(tabnet_oof)
        except Exception as e:
            print(f"âš ï¸� TabNetè®­ç»ƒå¤±è´¥: {str(e)}")

        gc.collect()
        print(f"TabNetè®­ç»ƒå��å†…å­˜ä½¿ç”¨: {process.memory_info().rss / 1024 / 1024:.2f} MB")

        # 6. è®­ç»ƒå¤šæ ·åŒ–å †å� æ¨¡å�‹
        stacking_map5 = 0
        try:
            stacking_oof, stacking_map5 = predictor.train_diverse_stacking(X, y, base_models_oof)
        except Exception as e:
            print(f"âš ï¸� å †å� æ¨¡å�‹è®­ç»ƒå¤±è´¥: {str(e)}ï¼Œè·³è¿‡æ­¤æ­¥éª¤")

        gc.collect()
        print(f"å †å� æ¨¡å�‹è®­ç»ƒå��å†…å­˜ä½¿ç”¨: {process.memory_info().rss / 1024 / 1024:.2f} MB")

        # 7. è®¡ç®—å�„æ¨¡å�‹æ€§èƒ½
        lgb_map5 = predictor.calculate_mapk(y, lgb_oof, k=5)
        xgb_map5 = predictor.calculate_mapk(y, xgb_oof, k=5)
        cb_map5 = predictor.calculate_mapk(y, cb_oof, k=5)

        # 8. ç”Ÿæˆ�æœ€ç»ˆé¢„æµ‹ - ä½¿ç”¨åŠ æ�ƒå¹³å�‡é›†æˆ�
        test_ids = predictor.test_ids if hasattr(predictor, 'test_ids') else range(len(X_test))

        # æ ¹æ�®å�„æ¨¡å�‹çš„æ€§èƒ½è°ƒæ•´æ�ƒé‡�
        weights = {
            'lgb': lgb_map5 * 1.2,  # å¢�åŠ LightGBMæ�ƒé‡�
            'xgb': xgb_map5,
            'catboost': cb_map5
        }

        # å¦‚æ�œTabNetæˆ�åŠŸè®­ç»ƒï¼Œæ·»åŠ å…¶æ�ƒé‡�
        if 'tabnet' in predictor.models and tabnet_map5 > 0:
            weights['tabnet'] = tabnet_map5

        # å¦‚æ�œå †å� æ¨¡å�‹æˆ�åŠŸè®­ç»ƒï¼Œæ·»åŠ å…¶æ�ƒé‡�
        if 'stacking' in predictor.models and stacking_map5 > 0:
            weights['stacking'] = stacking_map5 * 1.5  # å¢�åŠ å †å� æ¨¡å�‹æ�ƒé‡�

        # å½’ä¸€åŒ–æ�ƒé‡�
        total_weight = sum(weights.values())
        if total_weight > 0:
            for k in weights:
                weights[k] /= total_weight
        else:
            weights = {'lgb': 0.5, 'xgb': 0.3, 'catboost': 0.2}

        # æ‰“å�°æ�ƒé‡�ä¿¡æ�¯
        print(f"\nğŸ“Š æ¨¡å�‹é›†æˆ�æ�ƒé‡�:")
        for model, weight in sorted(weights.items()):
            print(f"  - {model}: {weight:.4f}")

        # å®‰å…¨é¢„æµ‹æµ�ç¨‹
        try:
            print("\nğŸ”® ç”Ÿæˆ�æœ€ç»ˆé¢„æµ‹...")
            # ç¡®ä¿�æ²¡æœ‰å­—ç¬¦ä¸²åˆ—
            if X_test.select_dtypes(include=['object']).shape[1] > 0:
                print("âš ï¸� ç§»é™¤æµ‹è¯•æ•°æ�®ä¸­çš„é��æ•°å€¼åˆ—")
                X_test = X_test.select_dtypes(include=[np.number])

            # ç¡®ä¿�æ²¡æœ‰NaNå€¼
            if X_test.isnull().values.any():
                print("âš ï¸� å¡«å……æµ‹è¯•æ•°æ�®ä¸­çš„NaNå€¼")
                X_test = X_test.fillna(X_test.mean())

            # ç¡®ä¿�kæ˜¯æ•´æ•°
            k = 5

            # è°ƒç”¨predict_topk
            top5_predictions, probabilities = predictor.predict_topk(X_test.copy(), k=int(5), weights=weights)

            # ç”Ÿæˆ�æ��äº¤æ–‡ä»¶
            submission = pd.DataFrame({
                'id': test_ids,
                'Fertilizer Name': [' '.join(preds) for preds in top5_predictions]
            })

            submission_file = 'submission_v6.csv'
            submission.to_csv(submission_file, index=False)
            print(f"âœ… å·²ä¿�å­˜æ��äº¤æ–‡ä»¶: {submission_file}")

        except Exception as e:
            print(f"â�Œ é¢„æµ‹è¿‡ç¨‹å‡ºé”™: {str(e)}")

            # åº”æ€¥æ��äº¤
            try:
                print("âš ï¸� åˆ›å»ºåº”æ€¥æ��äº¤...")
                # è�·å�–ç±»åˆ«åˆ—è¡¨
                class_labels = list(predictor.label_encoder.classes_)
                # å�–å‰�5ä¸ªç±»åˆ«ä½œä¸ºé»˜è®¤é¢„æµ‹
                default_preds = class_labels[:5]

                # åˆ›å»ºæ��äº¤
                submission = pd.DataFrame({
                    'id': test_ids,
                    'Fertilizer Name': [' '.join(default_preds) for _ in range(len(X_test))]
                })

                emergency_file = 'emergency_submission_v6.csv'
                submission.to_csv(emergency_file, index=False)
                print(f"âœ… å·²ä¿�å­˜åº”æ€¥æ��äº¤æ–‡ä»¶: {emergency_file}")
            except Exception as final_e:
                print(f"â�Œ åº”æ€¥æ��äº¤ä¹Ÿå¤±è´¥: {str(final_e)}")
                print("è¯·æ‰‹åŠ¨æ£€æŸ¥æ•°æ�®å’Œæ¨¡å�‹")

    except Exception as global_e:
        print(f"â�Œ ç¨‹åº�æ‰§è¡Œè¿‡ç¨‹ä¸­å�‘ç”Ÿé”™è¯¯: {str(global_e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

