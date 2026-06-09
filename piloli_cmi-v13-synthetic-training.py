import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import warnings
import os
import joblib

warnings.filterwarnings("ignore")

print("Starting CMI V13 Synthetic Data Training...")
print("GPU available:", os.system("nvidia-smi") == 0)


class PhysicsBasedSyntheticDataGenerator:
    """物理法則に基づく自然な合成データ生成器"""
    
    def __init__(self, train_df):
        self.train_df = train_df.copy()
        self.sensor_correlations = self._calculate_sensor_correlations()
        self.gesture_distributions = self._analyze_gesture_distributions()
        
    def _calculate_sensor_correlations(self):
        """センサー間の物理的相関を計算"""
        correlations = {}
        
        # 加速度-ジャイロ相関
        acc_cols = [c for c in self.train_df.columns if c.startswith("acc_")]
        gyro_cols = [c for c in self.train_df.columns if c.startswith("gyro_")]
        quat_cols = [c for c in self.train_df.columns if c.startswith("quat_")]
        
        if acc_cols and gyro_cols:
            correlations["acc_gyro"] = self.train_df[acc_cols + gyro_cols].corr()
        
        if quat_cols:
            correlations["quat"] = self.train_df[quat_cols].corr()
            
        return correlations
    
    def _analyze_gesture_distributions(self):
        """ジェスチャーごとのセンサーパターン分析"""
        distributions = {}
        
        for gesture in self.train_df["gesture"].unique():
            gesture_data = self.train_df[self.train_df["gesture"] == gesture]
            
            # 各センサーの統計的特性
            sensor_stats = {}
            for col in ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]:
                if col in gesture_data.columns:
                    sensor_stats[col] = {
                        "mean": gesture_data[col].mean(),
                        "std": gesture_data[col].std(),
                        "min": gesture_data[col].min(),
                        "max": gesture_data[col].max(),
                        "q25": gesture_data[col].quantile(0.25),
                        "q75": gesture_data[col].quantile(0.75),
                    }
            
            distributions[gesture] = sensor_stats
            
        return distributions
    
    def generate_realistic_sequence(self, base_sequence, gesture, noise_factor=0.08):
        """物理的に自然な合成シーケンス生成"""
        synthetic_seq = base_sequence.copy()
        
        # ジェスチャー固有の統計的特性を取得
        if gesture in self.gesture_distributions:
            gesture_stats = self.gesture_distributions[gesture]
            
            # 各センサーに物理的制約を保持したノイズを追加
            for col in ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]:
                if col in synthetic_seq.columns and col in gesture_stats:
                    stats = gesture_stats[col]
                    
                    # 元データの範囲内で自然なノイズを追加
                    current_values = synthetic_seq[col].values
                    noise_range = stats["std"] * noise_factor
                    noise = np.random.normal(0, noise_range, len(current_values))
                    
                    # 物理的制限内に制約
                    synthetic_seq[col] = np.clip(
                        current_values + noise, stats["min"], stats["max"]
                    )
        
        # クォータニオンの正規化（物理的制約）
        quat_cols = ["quat_w", "quat_x", "quat_y", "quat_z"]
        if all(col in synthetic_seq.columns for col in quat_cols):
            for i in range(len(synthetic_seq)):
                quat = synthetic_seq.iloc[i][quat_cols].values
                quat_norm = quat / np.linalg.norm(quat)
                synthetic_seq.iloc[i, synthetic_seq.columns.get_indexer(quat_cols)] = quat_norm
        
        # 加速度マグニチュードの物理的整合性
        if all(col in synthetic_seq.columns for col in ["acc_x", "acc_y", "acc_z"]):
            acc_magnitude = np.sqrt(
                synthetic_seq["acc_x"] ** 2
                + synthetic_seq["acc_y"] ** 2
                + synthetic_seq["acc_z"] ** 2
            )
            # 重力付近（9.8m/s²）の範囲で制約
            acc_magnitude = np.clip(acc_magnitude, 5.0, 15.0)
            
            # 元の方向を保持しつつマグニチュードを調整
            current_magnitude = np.sqrt(
                synthetic_seq["acc_x"] ** 2
                + synthetic_seq["acc_y"] ** 2
                + synthetic_seq["acc_z"] ** 2
            )
            ratio = acc_magnitude / (current_magnitude + 1e-10)
            
            synthetic_seq["acc_x"] *= ratio
            synthetic_seq["acc_y"] *= ratio
            synthetic_seq["acc_z"] *= ratio
        
        return synthetic_seq
    
    def generate_synthetic_dataset(self, target_ratio=0.15, max_per_gesture=50):
        """指定比率の合成データセット生成"""
        print(f"Generating synthetic data with {target_ratio * 100:.1f}% ratio...")
        
        synthetic_data = []
        real_count = len(self.train_df["sequence_id"].unique())
        target_synthetic_count = int(real_count * target_ratio)
        
        gestures = self.train_df["gesture"].unique()
        samples_per_gesture = max(1, target_synthetic_count // len(gestures))
        samples_per_gesture = min(samples_per_gesture, max_per_gesture)
        
        print(f"Target synthetic sequences: {target_synthetic_count}")
        print(f"Samples per gesture: {samples_per_gesture}")
        
        for gesture in gestures:
            gesture_data = self.train_df[self.train_df["gesture"] == gesture]
            unique_sequences = gesture_data["sequence_id"].unique()
            
            # ジェスチャーごとに指定数の合成シーケンス生成
            for i in range(samples_per_gesture):
                # ランダムに実シーケンスを選択
                base_seq_id = np.random.choice(unique_sequences)
                base_sequence = gesture_data[
                    gesture_data["sequence_id"] == base_seq_id
                ].copy()
                
                # 物理的に自然な合成シーケンス生成
                synthetic_sequence = self.generate_realistic_sequence(
                    base_sequence, gesture, noise_factor=0.08
                )
                
                # 新しいsequence_idを割り当て
                new_seq_id = f"synthetic_{gesture}_{i:03d}"
                synthetic_sequence["sequence_id"] = new_seq_id
                
                synthetic_data.append(synthetic_sequence)
        
        if synthetic_data:
            synthetic_df = pd.concat(synthetic_data, ignore_index=True)
            print(
                f"Generated {len(synthetic_df['sequence_id'].unique())} synthetic sequences"
            )
            return synthetic_df
        else:
            return pd.DataFrame()

print("PhysicsBasedSyntheticDataGenerator defined.")


def extract_advanced_features(df):
    """高度な特徴量抽出（V12準拠）"""
    features = []
    
    for seq_id in df["sequence_id"].unique():
        seq_data = df[df["sequence_id"] == seq_id].copy()
        feat = {}
        
        # Basic info
        feat["sequence_id"] = seq_id
        feat["subject_id"] = (
            seq_data["subject_id"].iloc[0] if "subject_id" in seq_data else 0
        )
        feat["seq_length"] = len(seq_data)
        feat["seq_length_log"] = np.log1p(len(seq_data))
        
        # Sensor features
        sensor_cols = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
        for col in sensor_cols:
            if col in seq_data.columns:
                values = seq_data[col].values
                feat[f"{col}_mean"] = np.mean(values)
                feat[f"{col}_std"] = np.std(values)
                feat[f"{col}_max"] = np.max(values)
                feat[f"{col}_min"] = np.min(values)
                feat[f"{col}_range"] = np.max(values) - np.min(values)
                feat[f"{col}_skew"] = pd.Series(values).skew()
                feat[f"{col}_kurt"] = pd.Series(values).kurtosis()
                feat[f"{col}_q50"] = np.median(values)
                
                # FFT features
                if len(values) > 1:
                    fft = np.abs(np.fft.fft(values))
                    feat[f"{col}_fft_max"] = np.max(fft)
                    feat[f"{col}_fft_mean"] = np.mean(fft)
                    feat[f"{col}_fft_std"] = np.std(fft)
                    feat[f"{col}_fft_energy"] = np.sum(fft**2)
                    feat[f"{col}_dominant_freq"] = np.argmax(fft)
                
                # Jerk features
                if len(values) > 1:
                    jerk = np.diff(values)
                    feat[f"{col}_jerk_mean"] = np.mean(jerk)
                    feat[f"{col}_jerk_std"] = np.std(jerk)
                    feat[f"{col}_jerk_max"] = np.max(np.abs(jerk))
                
                # Zero crossing
                feat[f"{col}_zero_cross"] = np.sum(np.diff(np.sign(values)) != 0)
        
        # Magnitude features
        if all(col in seq_data.columns for col in ["acc_x", "acc_y", "acc_z"]):
            acc_mag = np.sqrt(
                seq_data["acc_x"] ** 2 + seq_data["acc_y"] ** 2 + seq_data["acc_z"] ** 2
            )
            feat["acc_magnitude_mean"] = np.mean(acc_mag)
            feat["acc_magnitude_std"] = np.std(acc_mag)
            feat["acc_magnitude_max"] = np.max(acc_mag)
            feat["acc_magnitude_energy"] = np.sum(acc_mag**2)
            
            # Linear acceleration (gravity removed)
            gravity = np.array(
                [
                    np.mean(seq_data["acc_x"]),
                    np.mean(seq_data["acc_y"]),
                    np.mean(seq_data["acc_z"]),
                ]
            )
            linear_acc = acc_mag - np.linalg.norm(gravity)
            feat["linear_acc_mean"] = np.mean(linear_acc)
            feat["linear_acc_std"] = np.std(linear_acc)
            feat["linear_acc_max"] = np.max(np.abs(linear_acc))
        
        # Target
        feat["gesture"] = seq_data["gesture"].iloc[0]
        
        features.append(feat)
    
    return pd.DataFrame(features)

print("Feature extraction function defined.")


def train_hybrid_model_v13(synthetic_ratio=0.15):
    """V13ハイブリッドモデル（実データ + 物理ベース合成データ）"""
    print(
        f"=== CMI V13 Hybrid Model Training (Synthetic: {synthetic_ratio * 100:.1f}%) ==="
    )
    
    # データロード
    print("Loading data...")
    train_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv")
    print(f"Original data shape: {train_df.shape}")
    print(f"Unique sequences: {len(train_df['sequence_id'].unique())}")
    print(f"Unique gestures: {len(train_df['gesture'].unique())}")
    
    # 合成データ生成
    generator = PhysicsBasedSyntheticDataGenerator(train_df)
    synthetic_df = generator.generate_synthetic_dataset(target_ratio=synthetic_ratio)
    
    # データ結合
    if len(synthetic_df) > 0:
        combined_df = pd.concat([train_df, synthetic_df], ignore_index=True)
        print(
            f"Combined dataset: {len(train_df)} real + {len(synthetic_df)} synthetic = {len(combined_df)} total"
        )
        print(f"Synthetic ratio achieved: {len(synthetic_df) / len(combined_df) * 100:.1f}%")
    else:
        combined_df = train_df
        print("No synthetic data generated, using real data only")
    
    # 特徴量抽出
    print("Extracting features...")
    features_df = extract_advanced_features(combined_df)
    
    # 特徴量準備
    feature_cols = [
        col for col in features_df.columns if col not in ["sequence_id", "gesture"]
    ]
    X = features_df[feature_cols].fillna(0)
    y = features_df["gesture"].values
    
    print(f"Feature shape: {X.shape}")
    print(f"Feature columns: {len(feature_cols)}")
    print(f"Unique gestures: {len(np.unique(y))}")
    
    # 5-fold GroupKFold validation
    subject_ids = (
        features_df["subject_id"].values
        if "subject_id" in features_df.columns
        else np.arange(len(X))
    )
    gkf = GroupKFold(n_splits=5)
    
    val_scores = []
    models = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, subject_ids)):
        print(f"\n--- Fold {fold + 1}/5 ---")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        print(f"Train size: {len(X_train)}, Val size: {len(X_val)}")
        
        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # LightGBM
        lgb_model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=18,
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42 + fold,
            verbosity=-1,
        )
        lgb_model.fit(
            X_train_scaled,
            y_train,
            eval_set=[(X_val_scaled, y_val)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
        
        # XGBoost
        xgb_model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=18,
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            random_state=42 + fold,
            verbosity=0,
        )
        xgb_model.fit(
            X_train_scaled,
            y_train,
            eval_set=[(X_val_scaled, y_val)],
            early_stopping_rounds=50,
            verbose=False,
        )
        
        # CatBoost
        cat_model = CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            random_state=42 + fold,
            verbose=False,
        )
        cat_model.fit(
            X_train_scaled,
            y_train,
            eval_set=(X_val_scaled, y_val),
            early_stopping_rounds=50,
            verbose=False,
        )
        
        # Ensemble prediction
        pred_lgb = lgb_model.predict_proba(X_val_scaled)
        pred_xgb = xgb_model.predict_proba(X_val_scaled)
        pred_cat = cat_model.predict_proba(X_val_scaled)
        
        ensemble_pred = 0.4 * pred_lgb + 0.3 * pred_xgb + 0.3 * pred_cat
        val_pred = np.argmax(ensemble_pred, axis=1)
        
        # Validation accuracy
        accuracy = np.mean(val_pred == y_val)
        val_scores.append(accuracy)
        
        print(f"Fold {fold + 1} Validation Accuracy: {accuracy:.4f}")
        
        models.append(
            {"lgb": lgb_model, "xgb": xgb_model, "cat": cat_model, "scaler": scaler}
        )
    
    mean_cv_score = np.mean(val_scores)
    std_cv_score = np.std(val_scores)
    
    print(f"\n=== Cross-Validation Results ===")
    print(f"Mean CV Accuracy: {mean_cv_score:.4f} ± {std_cv_score:.4f}")
    print(f"Individual fold scores: {[f'{score:.4f}' for score in val_scores]}")
    
    # Return best performing fold model
    best_fold = np.argmax(val_scores)
    print(f"Best fold: {best_fold + 1} (accuracy: {val_scores[best_fold]:.4f})")
    
    return models[best_fold], feature_cols, mean_cv_score

print("Training function defined.")


# Test different synthetic ratios
best_ratio = 0.10
best_score = 0.0
best_model = None
best_features = None

for ratio in [0.05, 0.10, 0.15, 0.20]:
    print(f"\n{'=' * 60}")
    print(f"Testing synthetic ratio: {ratio * 100:.1f}%")
    print(f"{'=' * 60}")
    
    try:
        model, features, cv_score = train_hybrid_model_v13(synthetic_ratio=ratio)
        print(f"Final CV Score for {ratio * 100:.1f}% synthetic: {cv_score:.4f}")
        
        # Track best model
        if cv_score > best_score:
            best_score = cv_score
            best_ratio = ratio
            best_model = model
            best_features = features
            print(f"NEW BEST SCORE: {cv_score:.4f} at ratio {ratio * 100:.0f}%")
        
        # Save model if performance is good
        if cv_score > 0.55:  # Better than V12's 55-65% range
            model_name = f"hybrid_v13_synthetic_{ratio * 100:.0f}pct"
            print(f"Saving model: {model_name}")
            
            # Save models and metadata
            os.makedirs(model_name, exist_ok=True)
            
            # Save each component
            joblib.dump(model["scaler"], f"{model_name}/scaler.pkl")
            joblib.dump(features, f"{model_name}/feature_names.pkl")
            
            with open(f"{model_name}/cv_score.txt", "w") as f:
                f.write(f"{cv_score:.6f}")
            
            print(f"Model saved to {model_name}/")
    
    except Exception as e:
        print(f"Error with ratio {ratio}: {e}")
        continue

print(f"\n{'=' * 60}")
print(f"FINAL RESULTS")
print(f"{'=' * 60}")
print(f"Best synthetic ratio: {best_ratio * 100:.1f}%")
print(f"Best CV score: {best_score:.4f}")
print(f"Previous best (V12): 0.55-0.65")
print(f"Improvement: {'+' if best_score > 0.60 else '-'}{abs(best_score - 0.60):.4f}")


# Save best model with detailed info
if best_model is not None:
    final_model_name = f"v13_best_synthetic_{best_ratio * 100:.0f}pct"
    print(f"Saving final best model: {final_model_name}")
    
    os.makedirs(final_model_name, exist_ok=True)
    
    # Save all components
    joblib.dump(best_model["lgb"], f"{final_model_name}/lgb_model.pkl")
    joblib.dump(best_model["xgb"], f"{final_model_name}/xgb_model.pkl")
    joblib.dump(best_model["cat"], f"{final_model_name}/cat_model.pkl")
    joblib.dump(best_model["scaler"], f"{final_model_name}/scaler.pkl")
    joblib.dump(best_features, f"{final_model_name}/feature_names.pkl")
    
    # Save metadata
    metadata = {
        "cv_score": best_score,
        "synthetic_ratio": best_ratio,
        "model_version": "V13",
        "strategy": "Physics-based synthetic data",
        "feature_count": len(best_features),
        "ensemble_weights": {"lgb": 0.4, "xgb": 0.3, "cat": 0.3}
    }
    
    with open(f"{final_model_name}/metadata.json", "w") as f:
        import json
        json.dump(metadata, f, indent=2)
    
    print(f"Final model saved to {final_model_name}/")
    print(f"Files: lgb_model.pkl, xgb_model.pkl, cat_model.pkl, scaler.pkl, feature_names.pkl, metadata.json")
else:
    print("No best model found - all experiments failed")

print("\nV13 Synthetic Data Training Complete!")

