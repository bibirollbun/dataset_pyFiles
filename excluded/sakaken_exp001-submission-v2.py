import numpy as np
import polars as pl
import joblib
import traceback

# モデルとエンコーダーの読み込み（エラーハンドリング追加）
try:
    models = []
    for fold in range(5):
        model = joblib.load(f'/kaggle/input/exp001-training/model_lgb{fold}.joblib')
        models.append(model)
    le = joblib.load('/kaggle/input/exp001-training/le.joblib')
    print(f"Successfully loaded {len(models)} models and label encoder")
except Exception as e:
    print(f"Error loading models: {e}")
    raise

from sklearn.base import BaseEstimator, TransformerMixin
import polars as pl
import numpy as np
from tqdm import tqdm

class PolarsBasicStatsExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, sensor_cols, group_col="sequence_id"):
        self.sensor_cols = sensor_cols
        self.group_col = group_col
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        try:
            if not isinstance(X, pl.DataFrame):
                X = pl.DataFrame(X)
            
            aggs = []
            for col in self.sensor_cols:
                # NaNやInfの処理を追加
                aggs.extend([
                    pl.col(col).drop_nans().mean().alias(f"{col}_mean"),
                    pl.col(col).drop_nans().std().alias(f"{col}_std"), 
                    pl.col(col).drop_nans().min().alias(f"{col}_min"),
                    pl.col(col).drop_nans().max().alias(f"{col}_max"),
                    pl.col(col).drop_nans().median().alias(f"{col}_median"),
                    pl.col(col).drop_nans().skew().alias(f"{col}_skew"),
                ])
            
            result = X.group_by(self.group_col).agg(aggs)
            # NaNを0で埋める
            return result.fill_nan(0).fill_null(0)
        except Exception as e:
            print(f"Error in PolarsBasicStatsExtractor: {e}")
            raise

class PolarsDiffRateExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, sensor_cols, group_col="sequence_id"):
        self.sensor_cols = sensor_cols
        self.group_col = group_col
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        try:
            X_sorted = X.sort(by=[self.group_col])
            
            for col in self.sensor_cols:
                X_sorted = X_sorted.with_columns([
                    pl.col(col).diff().over(self.group_col).fill_nan(0).alias(f"{col}_diff"),
                    pl.col(col).pct_change().over(self.group_col).fill_nan(0).alias(f"{col}_rate")
                ])
            return X_sorted
        except Exception as e:
            print(f"Error in PolarsDiffRateExtractor: {e}")
            raise

class PolarsFFTFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, sensor_cols, group_col="sequence_id"):
        self.sensor_cols = sensor_cols
        self.group_col = group_col
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        try:
            features = []
            groups = list(X.group_by(self.group_col, maintain_order=True))
            
            for seq_id, group in groups:
                if isinstance(seq_id, tuple):
                    seq_id = seq_id[0]
                
                row = {self.group_col: seq_id}
                for col in self.sensor_cols:
                    try:
                        signal = group[col].to_numpy()
                        # NaNや無限値をチェック
                        signal = signal[~np.isnan(signal)]
                        if len(signal) == 0:
                            # データがない場合のデフォルト値
                            row[f"{col}_fft_peak_freq"] = 0
                            row[f"{col}_fft_energy"] = 0.0
                        else:
                            fft_vals = np.fft.fft(signal)
                            fft_abs = np.abs(fft_vals)
                            
                            # エラーハンドリング強化
                            if len(fft_abs) > 1:
                                peak_freq = int(np.argmax(fft_abs[1:]) + 1)
                                energy = float(np.sum(fft_abs ** 2))
                            else:
                                peak_freq = 0
                                energy = 0.0
                            
                            # 無限値チェック
                            if np.isinf(energy) or np.isnan(energy):
                                energy = 0.0
                            
                            row[f"{col}_fft_peak_freq"] = peak_freq
                            row[f"{col}_fft_energy"] = energy
                    except Exception as e:
                        print(f"Error processing FFT for {col}: {e}")
                        row[f"{col}_fft_peak_freq"] = 0
                        row[f"{col}_fft_energy"] = 0.0
                        
                features.append(row)
            return pl.DataFrame(features)
        except Exception as e:
            print(f"Error in PolarsFFTFeatureExtractor: {e}")
            raise

class PolarsSegmentStatsExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, sensor_cols, group_col="sequence_id", n_segments=3):
        self.sensor_cols = sensor_cols
        self.group_col = group_col
        self.n_segments = n_segments
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        try:
            rows = []
            groups = list(X.group_by(self.group_col, maintain_order=True))
            
            for seq_id, group in groups:
                if isinstance(seq_id, tuple):
                    seq_id = seq_id[0]
                    
                segment_size = max(1, len(group) // self.n_segments)
                feature = {self.group_col: seq_id}
                
                for i in range(self.n_segments):
                    start_idx = i * segment_size
                    end_idx = min((i + 1) * segment_size, len(group))
                    seg = group[start_idx:end_idx]
                    
                    for col in self.sensor_cols:
                        try:
                            mean_val = seg[col].mean()
                            std_val = seg[col].std()
                            
                            # NaN/Inf チェック
                            feature[f"{col}_seg{i}_mean"] = 0.0 if (mean_val is None or np.isnan(mean_val) or np.isinf(mean_val)) else float(mean_val)
                            feature[f"{col}_seg{i}_std"] = 0.0 if (std_val is None or np.isnan(std_val) or np.isinf(std_val)) else float(std_val)
                        except:
                            feature[f"{col}_seg{i}_mean"] = 0.0
                            feature[f"{col}_seg{i}_std"] = 0.0
                            
                rows.append(feature)
            return pl.DataFrame(rows)
        except Exception as e:
            print(f"Error in PolarsSegmentStatsExtractor: {e}")
            raise

# 特徴量エンジニアリングパイプライン
def fe_pipeline(df, sensor_cols):
    try:
        print("Starting feature engineering pipeline...")
        if not isinstance(df, pl.DataFrame):
            df = pl.DataFrame(df)
        
        print("\n=== Step 1/4: Basic Statistics ===")
        basic_stats = PolarsBasicStatsExtractor(sensor_cols).transform(df)
        
        print("\n=== Step 2/4: Diff/Rate Features ===")
        df_diff = PolarsDiffRateExtractor(sensor_cols).transform(df)
        
        print("\n=== Step 3/4: FFT Features ===")
        fft_features = PolarsFFTFeatureExtractor(sensor_cols).transform(df_diff)
        
        print("\n=== Step 4/4: Segment Statistics ===")
        segment_stats = PolarsSegmentStatsExtractor(sensor_cols).transform(df)
        
        print("\n=== Joining all features ===")
        result = (
            basic_stats
            .join(fft_features, on="sequence_id", how="inner")
            .join(segment_stats, on="sequence_id", how="inner")
        )
        print(f"Feature engineering completed! Shape: {result.shape}")
        return result
    except Exception as e:
        print(f"Error in fe_pipeline: {e}")
        print(traceback.format_exc())
        raise

sensor_cols = [
    "acc_x", "acc_y", "acc_z",
    "rot_w", "rot_x", "rot_y", "rot_z"
]

def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    try:
        print(f"Processing sequence with shape: {sequence.shape}")
        
        # 特徴量抽出
        cleaned_data = fe_pipeline(sequence, sensor_cols)
        
        # Pandasに変換して予測
        pdf = cleaned_data.to_pandas().drop(columns=["sequence_id"])
        print(f"Features shape for prediction: {pdf.shape}")
        
        # アンサンブル予測
        predictions = []
        for i, model in enumerate(models):
            try:
                pred = model.predict(pdf)
                if isinstance(pred, np.ndarray):
                    pred = pred[0]
                predictions.append(int(pred))
                print(f"Model {i} prediction: {pred}")
            except Exception as e:
                print(f"Error with model {i}: {e}")
                # デフォルト予測を追加
                predictions.append(0)
        
        # 多数決
        if predictions:
            predicted_label_id = max(set(predictions), key=predictions.count)
        else:
            predicted_label_id = 0  # デフォルト
        
        # ラベルを文字列に変換
        try:
            predicted_gesture_str = le.inverse_transform([predicted_label_id])[0]
        except:
            # エラーの場合はデフォルトラベルを返す
            predicted_gesture_str = le.classes_[0] if hasattr(le, 'classes_') else "unknown"
        
        print(f"Final prediction: {predicted_gesture_str}")
        return predicted_gesture_str
        
    except Exception as e:
        print(f"Error in predict function: {e}")
        print(traceback.format_exc())
        # エラーの場合はデフォルト値を返す
        try:
            return le.classes_[0] if hasattr(le, 'classes_') else "unknown"
        except:
            return "unknown"

# 推論サーバーの実行
import kaggle_evaluation.cmi_inference_server
import os

try:
    inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)
    
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        print("Running in competition mode")
        inference_server.serve()
    else:
        print("Running in local test mode")
        inference_server.run_local_gateway(
            data_paths=(
                '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
                '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv',
            )
        )
except Exception as e:
    print(f"Error running inference server: {e}")
    print(traceback.format_exc())
    raise

